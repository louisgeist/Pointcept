r"""
MALiBU3D preprocessing: labeled LiDAR HD PLY tiles -> Pointcept scene folders.

The tile CSV is the single source of truth: the MALiBU3D ``tiles.csv`` or an older
``scene_split_manifest.csv`` (only ``LIDARHD=True`` rows), read via ``tile_catalog.py``.
The input trees are never globbed; each tile produces the folder
``<output_root>/<split>/<dept_year>_LIDARHD/<roi>/<tile_id>/`` containing:

- meta.json             (date_gap_days, label_definitions, natural_habitat_layout)
- color.npy             (uint8 RGB; zeros when the PLY has no color)
- segment.npy           (uint8, PLY ``semantic`` remapped via ``--segment_definition``)
- strength.npy          (intensity clipped to [0, 60000] and divided by 60000)
- natural_habitat.npy   (only when has_natural_habitat=True: CarHab ids sampled from the
                         NATURAL_HABITAT GeoTIFF, stored as ``uint8 (N, 4)`` ecological axes,
                         see ``nathab_axes.py``)
- elevation.npy         (only when has_elevation=True: z minus the DTM band of DEM_ELEV)
- coord_translation.npy (float64 XYZ of the first point)
- coord.npy             (float32 xyz - coord_translation; written LAST, it marks the
                         scene as complete for resume)

forest_2d.npy and network.npy are added afterwards by rasterize_forest.py and
rasterize_network.py.

Required columns (tiles.csv names; older aliases in parentheses; extra columns ignored):
    split, dept_year, roi, scene_i_j, tile_id (patch_id),
    has_natural_habitat (NATURAL_HABITAT), has_elevation (DEM_ELEV), date_gap_days

Path conventions:
    tile_id  = f"{dept_year}_{roi}_{scene_i_j}"
    PLY path = {ply_root}/LIDARHD/{dept_year}_LIDARHD/{roi}/
               {dept_year}_LIDARHD_{roi}_{scene_i_j}.ply
    rasters  = {dataset_root}/{MODALITY}/{dept_year}_{MODALITY}/{roi}/
               {dept_year}_{MODALITY}_{roi}_{scene_i_j}.tif   (see unzip_raw_archives.py)

Reports written under ``--output_root``:
- ``missing_ply_preflight.txt``: written right after the PLY existence check, before
  any scene is processed; lists the tiles whose ``.ply`` is missing under ``--ply_root``.
- ``missing_scenes.txt``: written at the end; missing PLY, missing rasters and failed tiles.
- ``preprocess_malibu3d.log``: full log.

Resume: tiles whose ``coord.npy`` exists are skipped. Use ``--force`` to reprocess
everything after changing the modality flags, the label definitions or the PLY labels.

Example::

    python pointcept/datasets/preprocessing/malibu3d/preprocess_malibu3d.py \
        --ply_root data/malibu3d/ply \
        --dataset_root data/malibu3d/raw \
        --output_root data/malibu3d \
        --split_manifest_csv data/malibu3d/tiles.csv \
        --num_workers 16
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from plyfile import PlyData
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from malibu3d_label_remap import (  # noqa: E402
    DEFAULT_LABEL_DEFINITION_NAMES,
    PreprocessLabelDefinitions,
    apply_remap,
    build_preprocess_label_definitions,
    supported_definitions,
)
from tile_catalog import iter_tiles, tile_csv_columns  # noqa: E402
from nathab_axes import (  # noqa: E402
    NATHAB_LAYOUT_META,
    NATHAB_ONDISK_DEFINITION,
    carhab_to_nathab_axes,
    is_nathab_carhab_array,
)

# Natural-habitat rasters are sampled with the CarHab definition ``default`` (ids 0-43);
# ``_save_scene`` then converts them to the ecological axes ``(N, 4)``.
PREPROCESS_LABEL_DEFINITION_DEFAULTS = {
    **DEFAULT_LABEL_DEFINITION_NAMES,
    "natural_habitat": "default",
}

# Columns (tiles.csv names, older aliases accepted) that must be present in
# --split_manifest_csv; others may be present and are ignored.
REQUIRED_MANIFEST_COLUMNS = frozenset(
    {
        "split",
        "dept_year",
        "roi",
        "scene_i_j",
        "tile_id",
        "has_natural_habitat",
        "has_elevation",
        "date_gap_days",
    }
)


@dataclass(frozen=True)
class PatchTask:
    """One patch to preprocess, fully specified by its manifest row."""

    split: str
    dept_year: str
    roi: str
    scene_i_j: str
    patch_id: str
    has_natural_habitat: bool
    has_dem_elev: bool
    date_gap_days: Optional[float]


def _segment_from_ply(
    attributes: Dict[str, np.ndarray],
    ply_path: str,
    definition,
) -> np.ndarray:
    """Read the PLY ``semantic`` labels and apply the segment label definition."""
    if "semantic" not in attributes:
        raise KeyError(f"Missing 'semantic' in {ply_path}")
    raw = attributes["semantic"]
    return apply_remap(raw, definition)


def read_ply_binary(filepath: str) -> Dict[str, np.ndarray]:
    """
    Read a PLY file using plyfile (supports binary and ASCII formats).

    Returns a dict mapping attribute names to numpy arrays.
    """

    ply_data = PlyData.read(filepath)
    if len(ply_data.elements) == 0:
        raise ValueError(f"No element found in PLY file: {filepath}")
    vertex_data = ply_data.elements[0].data
    return {name: np.asarray(vertex_data[name]) for name in vertex_data.dtype.names}


def sample_raster_to_points(
    raster_path: str,
    xy: np.ndarray,
    fill_value: int = -1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Sample one raster value per XY point (nearest pixel, no interpolation).

    Args:
        raster_path: Path to the GeoTIFF file.
        xy: Array of shape (N, 2) with point coordinates in raster CRS.
        fill_value: Value assigned to points outside raster extent or nodata.

    Returns:
        values: Array (N,) with sampled values.
        stats: Sampling statistics dict for optional downstream checks.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be of shape (N, 2), got {xy.shape}")

    # Lazy import to keep this script importable even when raster support is unused.
    import rasterio  # type: ignore[import-not-found]
    values = np.full(xy.shape[0], fill_value=fill_value, dtype=np.int16)
    with rasterio.open(raster_path) as src:
        raster_nodata = src.nodata


        # Vectorized XY -> pixel conversion with the inverse affine transform (same
        # result as rasterio.transform.rowcol, much faster), in float64 because
        # projected coordinates are large.
        inv = ~src.transform
        x = xy[:, 0].astype(np.float64, copy=False)
        y = xy[:, 1].astype(np.float64, copy=False)
        cols = np.floor(inv.a * x + inv.b * y + inv.c).astype(np.int64, copy=False)
        rows = np.floor(inv.d * x + inv.e * y + inv.f).astype(np.int64, copy=False)
        inside_mask = (
            (rows >= 0)
            & (rows < src.height)
            & (cols >= 0)
            & (cols < src.width)
        )
        outside_count = int((~inside_mask).sum())

        if inside_mask.any():
            band1 = src.read(1)
            sampled = band1[rows[inside_mask], cols[inside_mask]]

            if raster_nodata is not None:
                nodata_mask = sampled == raster_nodata
                sampled = sampled.astype(np.int16, copy=True)
                sampled[nodata_mask] = fill_value
                nodata_points = int(nodata_mask.sum())
            else:
                nodata_points = 0

            values[inside_mask] = sampled.astype(np.int16, copy=False)
        else:
            nodata_points = 0

    stats = {
        "raster_nodata": raster_nodata,
        "num_points": int(xy.shape[0]),
        "num_points_outside_raster": outside_count,
        "num_points_nodata": nodata_points,
    }
    return values, stats


def sample_raster_to_points_float(
    raster_path: str,
    xy: np.ndarray,
    fill_value: float = np.nan,
    band_index: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Sample one float raster value per XY point (nearest pixel, no interpolation).

    Returns:
        values: Array (N,) float32 with sampled values.
        stats: Sampling statistics dict for optional downstream checks.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be of shape (N, 2), got {xy.shape}")

    import rasterio  # type: ignore[import-not-found]

    values = np.full(xy.shape[0], fill_value=fill_value, dtype=np.float32)
    with rasterio.open(raster_path) as src:
        if band_index < 1 or band_index > src.count:
            raise ValueError(
                f"Invalid band_index={band_index}; raster has {src.count} band(s) for {raster_path}."
            )
        raster_nodata = src.nodata

        inv = ~src.transform
        x = xy[:, 0].astype(np.float64, copy=False)
        y = xy[:, 1].astype(np.float64, copy=False)
        cols = np.floor(inv.a * x + inv.b * y + inv.c).astype(np.int64, copy=False)
        rows = np.floor(inv.d * x + inv.e * y + inv.f).astype(np.int64, copy=False)
        inside_mask = (
            (rows >= 0)
            & (rows < src.height)
            & (cols >= 0)
            & (cols < src.width)
        )
        outside_count = int((~inside_mask).sum())

        if inside_mask.any():
            band = src.read(band_index).astype(np.float32, copy=False)
            sampled = band[rows[inside_mask], cols[inside_mask]]

            if raster_nodata is not None:
                nodata_mask = sampled == raster_nodata
                sampled = sampled.astype(np.float32, copy=True)
                sampled[nodata_mask] = fill_value
                nodata_points = int(nodata_mask.sum())
            else:
                nodata_points = 0

            values[inside_mask] = sampled.astype(np.float32, copy=False)
        else:
            nodata_points = 0

    stats = {
        "raster_nodata": raster_nodata,
        "num_points": int(xy.shape[0]),
        "num_points_outside_raster": outside_count,
        "num_points_nodata": nodata_points,
    }
    return values, stats


def build_lidar_ply_path(
    ply_root: str, dept_year: str, roi: str, scene_i_j: str
) -> str:
    """Expected labeled LiDAR PLY path for one tile."""
    stem = f"{dept_year}_LIDARHD_{roi}_{scene_i_j}"
    return os.path.join(
        ply_root, "LIDARHD", f"{dept_year}_LIDARHD", roi, f"{stem}.ply"
    )


def build_modality_patch_path(
    dataset_root: str,
    modality: str,
    dept_year: str,
    roi: str,
    lidar_patch_stem: str,
) -> str:
    """GeoTIFF path of one modality (NATURAL_HABITAT, DEM_ELEV, FOREST) for a tile."""
    modality_stem = lidar_patch_stem.replace("_LIDARHD_", f"_{modality}_")
    # DEM_ELEV archives may be extracted with or without the <dept_year>_<modality> level.
    if modality == "DEM_ELEV":
        default_path = os.path.join(
            dataset_root,
            modality,
            f"{dept_year}_{modality}",
            roi,
            f"{modality_stem}.tif",
        )
        if os.path.isfile(default_path):
            return default_path
        return os.path.join(
            dataset_root,
            modality,
            roi,
            f"{modality_stem}.tif",
        )
    return os.path.join(
        dataset_root,
        modality,
        f"{dept_year}_{modality}",
        roi,
        f"{modality_stem}.tif",
    )


def setup_file_logger(log_file_path: str) -> logging.Logger:
    """Create a logger writing both to console and a log file."""
    logger = logging.getLogger("preprocess_malibu3d")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def log_run_configuration(logger: logging.Logger, args: argparse.Namespace) -> None:
    """Log the resolved CLI configuration as a readable block."""
    logger.info("=" * 72)
    logger.info("Preprocess configuration")
    logger.info("=" * 72)
    for key in sorted(vars(args)):
        value = getattr(args, key)
        logger.info("  %s: %s", key, value)
    logger.info("=" * 72)


def write_missing_ply_preflight_report(
    output_path: str,
    missing_ply: List[Dict[str, str]],
) -> None:
    """Write the pre-flight report listing patches whose PLY is missing on disk.

    Called right after the PLY existence check so the list is available
    without waiting for the full preprocessing run to finish.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Missing PLY pre-flight report\n\n")
        f.write(f"missing_ply={len(missing_ply)}\n\n")
        f.write("## Missing PLY files (LIDARHD=True but file not found)\n")
        if not missing_ply:
            f.write("none\n")
        else:
            for item in missing_ply:
                f.write(
                    f"{item['split']},{item['patch_id']},{item['ply_path']}\n"
                )


def write_missing_scenes_report(
    output_path: str,
    missing_ply: List[Dict[str, str]],
    missing_modalities: List[Dict[str, str]],
    failed_tasks: List[Dict[str, str]],
) -> None:
    """Write a text report listing missing files and task failures."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Missing scenes report\n\n")
        f.write(f"missing_ply={len(missing_ply)}\n")
        f.write(f"missing_modalities={len(missing_modalities)}\n")
        f.write(f"failed_tasks={len(failed_tasks)}\n\n")

        f.write("## Missing PLY files (LIDARHD=True but file not found)\n")
        if not missing_ply:
            f.write("none\n")
        else:
            for item in missing_ply:
                f.write(
                    f"{item['split']},{item['patch_id']},{item['ply_path']}\n"
                )

        f.write("\n## Missing modality rasters\n")
        if not missing_modalities:
            f.write("none\n")
        else:
            for item in missing_modalities:
                f.write(
                    f"{item['split']},{item['patch_id']},{item['modality']},"
                    f"{item['path']}\n"
                )

        f.write("\n## Failed preprocessing tasks\n")
        if not failed_tasks:
            f.write("none\n")
        else:
            for item in failed_tasks:
                f.write(
                    f"{item['split']},{item['patch_id']},{item['error']}\n"
                )


def load_manifest_tasks(csv_path: str, splits: List[str]) -> List[PatchTask]:
    """
    Load ``tiles.csv`` (or an older ``scene_split_manifest.csv``) and return one
    PatchTask per LiDAR tile belonging to one of ``splits``.

    Other rows are filtered out silently. A tile_id that does not match its
    components, or duplicates with inconsistent flags, raise a ``ValueError``.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Split manifest CSV not found: {csv_path}")
    missing = REQUIRED_MANIFEST_COLUMNS - tile_csv_columns(csv_path)
    if missing:
        raise ValueError(
            f"Invalid manifest CSV {csv_path}: missing required columns: {sorted(missing)}"
        )

    tasks_by_patch: Dict[str, PatchTask] = {}
    for tile in iter_tiles(csv_path, splits):
        if not tile.scene_i_j:
            continue
        expected_pid = f"{tile.dept_year}_{tile.roi}_{tile.scene_i_j}"
        if tile.tile_id != expected_pid:
            raise ValueError(
                f"Inconsistent tile_id in {csv_path}: "
                f"row says '{tile.tile_id}' but components yield '{expected_pid}'."
            )
        task = PatchTask(
            split=tile.split,
            dept_year=tile.dept_year,
            roi=tile.roi,
            scene_i_j=tile.scene_i_j,
            patch_id=tile.tile_id,
            has_natural_habitat=bool(tile.has_natural_habitat),
            has_dem_elev=bool(tile.has_elevation),
            date_gap_days=tile.date_gap_days,
        )
        prev = tasks_by_patch.get(tile.tile_id)
        if prev is None:
            tasks_by_patch[tile.tile_id] = task
        elif prev != task:
            raise ValueError(
                f"Inconsistent duplicate tile_id '{tile.tile_id}' in {csv_path}: "
                f"first={prev}, duplicate={task}"
            )

    return list(tasks_by_patch.values())


def _build_scene_from_subtile(
    task: PatchTask,
    ply_root: str,
    dataset_root: str,
    label_definitions: PreprocessLabelDefinitions,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, str]]]:
    """
    Build a Pointcept scene dict from one PatchTask.

    Returns:
        scene: dict of numpy arrays to save (keys: coord, color, segment, and
            optionally strength, natural_habitat, elevation).
        missing_modalities: list of {split, patch_id, modality, path} entries
            for rasters expected (per manifest flags) but absent on disk.

    Raises:
        FileNotFoundError: if the LiDAR PLY itself is missing.
    """
    ply_path = build_lidar_ply_path(
        ply_root, task.dept_year, task.roi, task.scene_i_j
    )
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"LiDAR PLY not found: {ply_path}")

    attributes = read_ply_binary(ply_path)

    for axis in ("x", "y", "z"):
        if axis not in attributes:
            raise KeyError(f"Missing '{axis}' in {ply_path}")
    coord = np.stack(
        [attributes["x"], attributes["y"], attributes["z"]], axis=1
    ).astype(np.float32)

    if all(k in attributes for k in ("red", "green", "blue")):
        color = np.stack(
            [attributes["red"], attributes["green"], attributes["blue"]], axis=1
        ).astype(np.uint8)
    else:
        color = np.zeros((coord.shape[0], 3), dtype=np.uint8)

    segment = _segment_from_ply(
        attributes, ply_path, label_definitions.segment
    )

    out: Dict[str, np.ndarray] = {
        "coord": coord,
        "color": color,
        "segment": segment,
    }

    if "intensity" in attributes:
        strength = np.clip(attributes["intensity"].astype(np.float32), 0, 60000) / 60000
        out["strength"] = strength.astype(np.float32)

    missing_modalities: List[Dict[str, str]] = []
    lidar_patch_stem = f"{task.dept_year}_LIDARHD_{task.roi}_{task.scene_i_j}"

    if task.has_natural_habitat:
        natural_habitat_raster_path = build_modality_patch_path(
            dataset_root=dataset_root,
            modality="NATURAL_HABITAT",
            dept_year=task.dept_year,
            roi=task.roi,
            lidar_patch_stem=lidar_patch_stem,
        )
        if os.path.isfile(natural_habitat_raster_path):
            nh_def = label_definitions.natural_habitat
            natural_habitat_values, _ = sample_raster_to_points(
                raster_path=natural_habitat_raster_path,
                xy=coord[:, :2],
                fill_value=nh_def.missing_fill_raw_id,
            )
            out["natural_habitat"] = apply_remap(
                natural_habitat_values, nh_def
            ).astype(np.int16, copy=False)
        else:
            missing_modalities.append(
                {
                    "split": task.split,
                    "patch_id": task.patch_id,
                    "modality": "NATURAL_HABITAT",
                    "path": natural_habitat_raster_path,
                }
            )

    if task.has_dem_elev:
        dem_elev_raster_path = build_modality_patch_path(
            dataset_root=dataset_root,
            modality="DEM_ELEV",
            dept_year=task.dept_year,
            roi=task.roi,
            lidar_patch_stem=lidar_patch_stem,
        )
        if os.path.isfile(dem_elev_raster_path):
            dtm_values, _ = sample_raster_to_points_float(
                raster_path=dem_elev_raster_path,
                xy=coord[:, :2],
                fill_value=np.nan,
                band_index=2,  # DEM_ELEV bands: 1 = DSM, 2 = DTM
            )
            elevation = coord[:, 2].astype(np.float32, copy=False) - dtm_values
            out["elevation"] = elevation.astype(np.float32, copy=False)
        else:
            missing_modalities.append(
                {
                    "split": task.split,
                    "patch_id": task.patch_id,
                    "modality": "DEM_ELEV",
                    "path": dem_elev_raster_path,
                }
            )

    return out, missing_modalities


def _save_scene(
    output_scene_dir: str,
    scene: Dict[str, np.ndarray],
    task: PatchTask,
) -> None:
    """Persist scene arrays under the patch output directory.

    Files are written when present in ``scene``; for optional modalities, any
    stale file from a previous run is removed when the modality is disabled or
    its raster was missing.

    ``natural_habitat`` is expected as CarHab ids ``(N,)`` in ``scene``; it is
    converted to ``uint8 (N, 4)`` ecological axes before saving ``natural_habitat.npy``.

    ``coord.npy`` is intentionally written LAST so its presence on disk acts as
    a reliable completion marker for the resume / skip-existing logic in
    ``main_process``.
    """
    os.makedirs(output_scene_dir, exist_ok=True)
    np.save(os.path.join(output_scene_dir, "color.npy"), scene["color"].astype(np.uint8))
    np.save(os.path.join(output_scene_dir, "segment.npy"), scene["segment"].astype(np.uint8))
    if "strength" in scene:
        np.save(
            os.path.join(output_scene_dir, "strength.npy"),
            scene["strength"].astype(np.float32),
        )

    def _save_or_clean(filename: str, key: str, dtype, enabled: bool) -> None:
        path = os.path.join(output_scene_dir, filename)
        if enabled and key in scene:
            np.save(path, scene[key].astype(dtype, copy=False))
        elif os.path.isfile(path):
            os.remove(path)


    if task.has_natural_habitat and "natural_habitat" in scene:
        carhab = np.asarray(scene["natural_habitat"])
        if not is_nathab_carhab_array(carhab):
            raise ValueError(
                "preprocess expected in-memory natural_habitat as CarHab (N,) "
                f"before bake, got shape {carhab.shape}"
            )
        scene = dict(scene)
        scene["natural_habitat"] = carhab_to_nathab_axes(carhab)
        _save_or_clean(
            "natural_habitat.npy",
            "natural_habitat",
            np.uint8,
            enabled=True,
        )
    else:
        _save_or_clean(
            "natural_habitat.npy",
            "natural_habitat",
            np.uint8,
            enabled=False,
        )

    _save_or_clean("elevation.npy", "elevation", np.float32, enabled=task.has_dem_elev)

    # Write coord_translation.npy, then coord.npy LAST (completion marker for resume).
    # PLY coords = coord.astype(np.float64) + coord_translation.
    coord_f64 = scene["coord"].astype(np.float64, copy=False)
    if coord_f64.shape[0] > 0:
        translation = coord_f64[0].copy()
        coord = (coord_f64 - translation).astype(np.float32)
    else:
        translation = np.zeros(3, dtype=np.float64)
        coord = coord_f64.astype(np.float32)
    np.save(
        os.path.join(output_scene_dir, "coord_translation.npy"),
        translation.astype(np.float64, copy=False),
    )
    np.save(os.path.join(output_scene_dir, "coord.npy"), coord)


def _scene_meta_dict(
    task: PatchTask,
    label_definitions: PreprocessLabelDefinitions,
) -> Dict[str, Any]:
    """Build the meta.json payload (date gap, label definitions, nathab axes layout)."""
    label_meta = label_definitions.to_meta_dict()
    # The CarHab definition is only used for sampling; the stored array is the axes.
    label_meta = dict(label_meta)
    label_meta["natural_habitat"] = NATHAB_ONDISK_DEFINITION
    return {
        "date_gap_days": task.date_gap_days,
        "label_definitions": label_meta,
        "natural_habitat_layout": dict(NATHAB_LAYOUT_META),
    }


def _save_scene_meta(output_scene_dir: str, meta: Dict[str, Any]) -> None:
    meta_path = os.path.join(output_scene_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _process_subtile_task(
    task: PatchTask,
    ply_root: str,
    dataset_root: str,
    output_root: str,
    label_definitions: PreprocessLabelDefinitions,
) -> Tuple[str, List[Dict[str, str]]]:
    """Worker entry point: build and save one scene from a PatchTask.

    Returns the patch_id and a list of missing modality raster entries.
    Raises FileNotFoundError if the LiDAR PLY is missing.
    """
    scene, missing_modalities = _build_scene_from_subtile(
        task=task,
        ply_root=ply_root,
        dataset_root=dataset_root,
        label_definitions=label_definitions,
    )
    output_scene_dir = os.path.join(output_root, task.split, f"{task.dept_year}_LIDARHD", task.roi, task.patch_id)
    # meta.json first, then the arrays: ``_save_scene`` writes coord.npy last, so its
    # presence means the metadata and every other array were fully written.
    os.makedirs(output_scene_dir, exist_ok=True)
    _save_scene_meta(output_scene_dir, _scene_meta_dict(task, label_definitions))
    _save_scene(output_scene_dir, scene, task)
    return task.patch_id, missing_modalities


def _filter_tasks_for_processing(
    tasks: List[PatchTask], ply_root: str, output_root: str, force: bool
) -> Tuple[List[PatchTask], List[PatchTask], List[Dict[str, str]]]:
    """Pre-flight and resume check in a single pass.

    Returns:
        tasks_to_process: Tasks with a present PLY that need processing.
        already_done: Tasks skipped because coord.npy exists (empty if force=True).
        missing_ply: Tasks whose PLY file is missing on disk.
    """
    tasks_to_process: List[PatchTask] = []
    already_done: List[PatchTask] = []
    missing_ply: List[Dict[str, str]] = []

    for task in tasks:
        # 1. PLY existence
        ply_path = build_lidar_ply_path(
            ply_root, task.dept_year, task.roi, task.scene_i_j
        )
        if not os.path.isfile(ply_path):
            missing_ply.append(
                {
                    "split": task.split,
                    "patch_id": task.patch_id,
                    "ply_path": ply_path,
                }
            )
            continue

        # 2. Completion (resume)
        if not force:
            coord_path = os.path.join(output_root, task.split, f"{task.dept_year}_LIDARHD", task.roi, task.patch_id, "coord.npy")
            if os.path.isfile(coord_path):
                already_done.append(task)
                continue

        tasks_to_process.append(task)

    return tasks_to_process, already_done, missing_ply


def main_process():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ply_root",
        type=str,
        default="",
        help=(
            "Root directory of pre-labeled PLY files "
            "(layout: LIDARHD/<dept_year>_LIDARHD/<roi>/<stem>.ply)."
        ),
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Root directory containing the auxiliary GeoTIFFs (NATURAL_HABITAT, DEM_ELEV)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/malibu3d",
        help="Output root directory for Pointcept-formatted scenes",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        metavar="SPLIT",
        help="One or more output split folder names under output_root (e.g. train val test)",
    )
    parser.add_argument(
        "--split_manifest_csv",
        type=str,
        required=True,
        help=(
            "Required path to tiles.csv (or legacy scene_split_manifest.csv) with columns: "
            + ", ".join(sorted(REQUIRED_MANIFEST_COLUMNS))
            + ". See script docstring."
        ),
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Number of workers for preprocessing"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="",
        help="Optional path to write preprocessing logs. Defaults to <output_root>/preprocess_malibu3d.log",
    )
    parser.add_argument(
        "--missing_scenes_file",
        type=str,
        default="",
        help=(
            "Optional path to write missing-scenes report. "
            "Defaults to <output_root>/missing_scenes.txt"
        ),
    )
    parser.add_argument(
        "--missing_ply_preflight_file",
        type=str,
        default="",
        help=(
            "Optional path for the pre-flight report listing patches whose PLY "
            "is missing on disk. Written immediately after the existence check, "
            "before any scene is processed. "
            "Defaults to <output_root>/missing_ply_preflight.txt"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Reprocess every patch even if its output coord.npy already exists. "
            "By default the script resumes by skipping already-processed scenes."
        ),
    )
    for task_key in ("segment", "natural_habitat"):
        parser.add_argument(
            f"--{task_key}_definition",
            type=str,
            default=PREPROCESS_LABEL_DEFINITION_DEFAULTS[task_key],
            choices=supported_definitions(task_key),
            help=(
                f"Label definition for task '{task_key}' "
                f"(see malibu3d_label_remap.LABEL_DEFINITIONS)."
            ),
        )
    args = parser.parse_args()

    if not args.ply_root:
        parser.error("--ply_root is required")
    if not args.dataset_root:
        parser.error("--dataset_root is required")

    os.makedirs(args.output_root, exist_ok=True)
    log_file_path = args.log_file or os.path.join(args.output_root, "preprocess_malibu3d.log")
    logger = setup_file_logger(log_file_path)
    log_run_configuration(logger, args)

    splits = args.split if isinstance(args.split, list) else [args.split]
    logger.info("Using split manifest CSV: %s", args.split_manifest_csv)
    logger.info("Splits to process: %s", splits)

    tasks = load_manifest_tasks(args.split_manifest_csv, splits)
    logger.info("Loaded %d LiDAR tiles from manifest.", len(tasks))
    for split in splits:
        n = sum(1 for t in tasks if t.split == split)
        logger.info("  split '%s': %d tasks", split, n)

    missing_scenes_path = args.missing_scenes_file or os.path.join(
        args.output_root, "missing_scenes.txt"
    )
    missing_ply_preflight_path = args.missing_ply_preflight_file or os.path.join(
        args.output_root, "missing_ply_preflight.txt"
    )

    logger.info("PLY root: %s", args.ply_root)
    logger.info("Dataset root (rasters): %s", args.dataset_root)

    label_definitions = build_preprocess_label_definitions(
        segment=args.segment_definition,
        natural_habitat=args.natural_habitat_definition,
    )
    logger.info("Label definitions: %s", label_definitions.to_meta_dict())

    # Pre-create output split directories (only for splits that have tasks).
    for split in {t.split for t in tasks}:
        os.makedirs(os.path.join(args.output_root, split), exist_ok=True)

    # Pre-flight and resume checks in a single pass
    tasks_to_process, already_done, missing_ply = _filter_tasks_for_processing(
        tasks, args.ply_root, args.output_root, args.force
    )

    if missing_ply:
        logger.warning(
            "%d patch(es) have LIDARHD=True but their PLY file is missing on disk.",
            len(missing_ply),
        )

    # Persist the pre-flight report immediately so the missing-PLY list is
    # available without waiting for the whole run to finish.
    write_missing_ply_preflight_report(missing_ply_preflight_path, missing_ply)
    logger.info(
        "Wrote missing-PLY pre-flight report (%d entries) to: %s",
        len(missing_ply),
        missing_ply_preflight_path,
    )

    # Log resume status
    if args.force:
        logger.info(
            "--force enabled: reprocessing all %d patches with present PLY.",
            len(tasks_to_process),
        )
    else:
        if already_done:
            logger.info(
                "Resume mode: skipping %d already-processed patch(es); processing %d remaining.",
                len(already_done),
                len(tasks_to_process),
            )
            logger.warning(
                "Skip is based on coord.npy presence only. If upstream PLY labels, "
                "label definitions, or manifest modality flags "
                "(has_natural_habitat/has_elevation) changed since the previous run, "
                "re-run with --force to refresh outputs."
            )
        else:
            logger.info(
                "Resume mode: no existing scene found; processing all %d patches.",
                len(tasks_to_process),
            )

    missing_modalities: List[Dict[str, str]] = []
    failed_tasks: List[Dict[str, str]] = []
    scene_ids: List[str] = []

    if not tasks_to_process:
        logger.warning("No task to process after PLY existence and resume checks.")
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {
                pool.submit(
                    _process_subtile_task,
                    task,
                    args.ply_root,
                    args.dataset_root,
                    args.output_root,
                    label_definitions,
                ): task
                for task in tasks_to_process
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="patches",
                unit="patch",
            ):
                task = futures[future]
                try:
                    patch_id, task_missing = future.result()
                    scene_ids.append(patch_id)
                    for entry in task_missing:
                        missing_modalities.append(entry)
                        logger.warning(
                            "Missing %s raster for %s: %s",
                            entry["modality"],
                            entry["patch_id"],
                            entry["path"],
                        )
                except FileNotFoundError as exc:
                    # Rare (PLY existence was pre-checked): any file that disappeared
                    # or failed to open in the worker is reported as missing.
                    missing_ply.append(
                        {
                            "split": task.split,
                            "patch_id": task.patch_id,
                            "ply_path": str(exc),
                        }
                    )
                    logger.warning("Missing file for %s: %s", task.patch_id, exc)
                except Exception as exc:
                    failed_tasks.append(
                        {
                            "split": task.split,
                            "patch_id": task.patch_id,
                            "error": repr(exc),
                        }
                    )
                    logger.exception(
                        "Failed task for split=%s patch_id=%s",
                        task.split,
                        task.patch_id,
                    )

    logger.info("Done. Processed %d scenes into %s", len(scene_ids), args.output_root)

    write_missing_scenes_report(
        output_path=missing_scenes_path,
        missing_ply=missing_ply,
        missing_modalities=missing_modalities,
        failed_tasks=failed_tasks,
    )
    logger.info("Wrote missing-scenes report to: %s", missing_scenes_path)
    logger.info(
        "Summary: processed=%d skipped=%d missing_ply=%d missing_modalities=%d failed_tasks=%d",
        len(scene_ids),
        len(already_done),
        len(missing_ply),
        len(missing_modalities),
        len(failed_tasks),
    )

    log_run_configuration(logger, args)
    logger.info("Detailed logs saved to: %s", log_file_path)


if __name__ == "__main__":
    main_process()
