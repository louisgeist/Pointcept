r"""
Preprocessing script for Flair3D+ (LidarHD) — manifest-driven.

The split manifest CSV (e.g. ``data/flair3d_plus/raw/scene_split_manifest.csv``)
is the single source of truth: one row per patch, and only rows with
``LIDARHD=True`` produce a Pointcept scene folder containing:

- coord.npy
- color.npy
- segment.npy
- strength.npy   (LiDAR intensity)
- forest.npy     (FOREST GeoTIFF — always sampled; missing raster reported)
- natural_habitat.npy   (only when NATURAL_HABITAT=True in the manifest)
- land_use.npy          (only when LAND_USE=True in the manifest)
- elevation.npy         (only when DEM_ELEV=True in the manifest)
- meta.json      (date_gap_days)

Required manifest columns (extra columns are ignored):
    split, dept_year, roi, scene_i_j, patch_id,
    LIDARHD, NATURAL_HABITAT, LAND_USE, DEM_ELEV,
    date_gap_days

Conventions (must match scripts/build_csv_manifest.py):
    patch_id = f"{dept_year}_{roi}_{scene_i_j}"
    PLY path = LIDARHD/{dept_year}_LIDARHD/{roi}/{dept_year}_LIDARHD_{roi}_{scene_i_j}.ply

Examples:

python pointcept/datasets/preprocessing/flair3d_plus/preprocess_flair3d.py \
    --dataset_root data/flair3d_plus/raw \
    --output_root data/flair3d_plus \
    --label_definition inter_finerall6 \
    --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest.csv

python pointcept/datasets/preprocessing/flair3d_plus/preprocess_flair3d.py \
    --dataset_root data/flair3d_plus/raw \
    --output_root data/flair3d_plus \
    --label_definition inter_finerall8 \
    --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest_D067.csv
"""

import argparse
import csv
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

try:
    from pointcept.datasets.preprocessing.flair3d.flair3d_label_remap import (
        SUPPORTED_LABEL_REMAPS,
        build_segment,
    )
except ModuleNotFoundError:
    # Allow running this file directly without setting PYTHONPATH.
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    if THIS_DIR not in sys.path:
        sys.path.insert(0, THIS_DIR)
    from flair3d_label_remap import SUPPORTED_LABEL_REMAPS, build_segment

# Columns that must be present in --split_manifest_csv (others may be present and are ignored).
REQUIRED_MANIFEST_COLUMNS = frozenset(
    {
        "split",
        "dept_year",
        "roi",
        "scene_i_j",
        "patch_id",
        "LIDARHD",
        "NATURAL_HABITAT",
        "LAND_USE",
        "DEM_ELEV",
        "date_gap_days",
    }
)

_DATE_NA_TOKENS = frozenset({"", "<na>", "na", "none", "n/a", "nan", "null"})


@dataclass(frozen=True)
class PatchTask:
    """One patch to preprocess, fully specified by its manifest row."""

    split: str
    dept_year: str
    roi: str
    scene_i_j: str
    patch_id: str
    has_natural_habitat: bool
    has_land_use: bool
    has_dem_elev: bool
    date_gap_days: Optional[float]


def _parse_csv_bool(raw: Optional[str], field: str, patch_id: str, csv_path: str) -> bool:
    token = (raw or "").strip().lower()
    if token in ("true", "1", "yes"):
        return True
    if token in ("false", "0", "no"):
        return False
    raise ValueError(
        f"Invalid boolean for column '{field}' (patch_id={patch_id}) in {csv_path}: {raw!r}"
    )


def _parse_csv_float_optional(raw: Optional[str]) -> Optional[float]:
    """Parse an optional float column; '<NA>'/'nan'/empty → None."""
    token = (raw or "").strip().lower()
    if token in _DATE_NA_TOKENS:
        return None
    try:
        return float(token)
    except ValueError:
        return None


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


        # Fast vectorized XY -> pixel conversion using inverse affine transform
        # (rasterio.transform.rowcol is too slow.)
        # Use float64 for numerical stability on large projected coordinates.
        # (Outputs are the same as rasterio.transform.rowcol.)
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
    dataset_root: str, dept_year: str, roi: str, scene_i_j: str
) -> str:
    """Expected LiDAR PLY path for one patch (mirrors scripts/build_csv_manifest.py)."""
    stem = f"{dept_year}_LIDARHD_{roi}_{scene_i_j}"
    return os.path.join(
        dataset_root, "LIDARHD", f"{dept_year}_LIDARHD", roi, f"{stem}.ply"
    )


def build_modality_patch_path(
    dataset_root: str,
    modality: str,
    dept_year: str,
    roi: str,
    lidar_patch_stem: str,
) -> str:
    """Build one modality patch path from split metadata and LiDAR patch stem."""
    modality_stem = lidar_patch_stem.replace("_LIDARHD_", f"_{modality}_")
    # DEM_ELEV archives may be extracted with or without the <dept>_<modality> level.
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
    logger = logging.getLogger("preprocess_flair3d")
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
    Load ``scene_split_manifest.csv`` and return one PatchTask per row that:
    - belongs to one of ``splits``,
    - has ``LIDARHD=True``.

    Other rows are filtered out silently. Duplicates with inconsistent flags
    raise a ``ValueError``.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Split manifest CSV not found: {csv_path}")

    splits_set = set(splits)
    tasks_by_patch: Dict[str, PatchTask] = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV or no header row: {csv_path}")
        fields = set(reader.fieldnames)
        missing = REQUIRED_MANIFEST_COLUMNS - fields
        if missing:
            raise ValueError(
                f"Invalid manifest CSV {csv_path}: missing required columns: {sorted(missing)}"
            )

        for row in reader:
            split = (row.get("split") or "").strip().lower()
            dept_year = (row.get("dept_year") or "").strip()
            roi = (row.get("roi") or "").strip()
            scene_i_j = (row.get("scene_i_j") or "").strip()
            patch_id = (row.get("patch_id") or "").strip()
            if not split or not dept_year or not roi or not scene_i_j or not patch_id:
                continue
            if split not in splits_set:
                continue

            expected_pid = f"{dept_year}_{roi}_{scene_i_j}"
            if patch_id != expected_pid:
                raise ValueError(
                    f"Inconsistent patch_id in {csv_path}: "
                    f"row says '{patch_id}' but components yield '{expected_pid}'."
                )

            lidarhd = _parse_csv_bool(row.get("LIDARHD", ""), "LIDARHD", patch_id, csv_path)
            if not lidarhd:
                continue

            nh = _parse_csv_bool(
                row.get("NATURAL_HABITAT", ""), "NATURAL_HABITAT", patch_id, csv_path
            )
            lu = _parse_csv_bool(row.get("LAND_USE", ""), "LAND_USE", patch_id, csv_path)
            dem = _parse_csv_bool(row.get("DEM_ELEV", ""), "DEM_ELEV", patch_id, csv_path)
            date_gap = _parse_csv_float_optional(row.get("date_gap_days", ""))

            task = PatchTask(
                split=split,
                dept_year=dept_year,
                roi=roi,
                scene_i_j=scene_i_j,
                patch_id=patch_id,
                has_natural_habitat=nh,
                has_land_use=lu,
                has_dem_elev=dem,
                date_gap_days=date_gap,
            )

            if patch_id in tasks_by_patch:
                prev = tasks_by_patch[patch_id]
                if prev != task:
                    raise ValueError(
                        f"Inconsistent duplicate patch_id '{patch_id}' in {csv_path}: "
                        f"first={prev}, duplicate={task}"
                    )
            else:
                tasks_by_patch[patch_id] = task

    return list(tasks_by_patch.values())


def _build_scene_from_subtile(
    task: PatchTask,
    dataset_root: str,
    label_definition: str,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, str]]]:
    """
    Build a Pointcept scene dict from one PatchTask.

    Returns:
        scene: dict of numpy arrays to save (keys: coord, color, segment, and
            optionally strength, forest, natural_habitat, land_use, elevation).
        missing_modalities: list of {split, patch_id, modality, path} entries
            for rasters expected (per FOREST/manifest flags) but absent on disk.

    Raises:
        FileNotFoundError: if the LiDAR PLY itself is missing.
    """
    ply_path = build_lidar_ply_path(
        dataset_root, task.dept_year, task.roi, task.scene_i_j
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

    segment = build_segment(
        attributes=attributes,
        label_definition=label_definition,
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

    # FOREST: always sampled (assumed available everywhere by dataset convention).
    forest_raster_path = build_modality_patch_path(
        dataset_root=dataset_root,
        modality="FOREST",
        dept_year=task.dept_year,
        roi=task.roi,
        lidar_patch_stem=lidar_patch_stem,
    )
    if os.path.isfile(forest_raster_path):
        forest_values, _ = sample_raster_to_points(
            raster_path=forest_raster_path,
            xy=coord[:, :2],
            fill_value=2,  # Void idx
        )
        out["forest"] = forest_values.astype(np.int16, copy=False)
    else:
        missing_modalities.append(
            {
                "split": task.split,
                "patch_id": task.patch_id,
                "modality": "FOREST",
                "path": forest_raster_path,
            }
        )

    if task.has_natural_habitat:
        natural_habitat_raster_path = build_modality_patch_path(
            dataset_root=dataset_root,
            modality="NATURAL_HABITAT",
            dept_year=task.dept_year,
            roi=task.roi,
            lidar_patch_stem=lidar_patch_stem,
        )
        if os.path.isfile(natural_habitat_raster_path):
            natural_habitat_values, _ = sample_raster_to_points(
                raster_path=natural_habitat_raster_path,
                xy=coord[:, :2],
                fill_value=42,  # N/A index from natural_habitat_classes.txt
            )
            out["natural_habitat"] = natural_habitat_values.astype(np.int16, copy=False)
        else:
            missing_modalities.append(
                {
                    "split": task.split,
                    "patch_id": task.patch_id,
                    "modality": "NATURAL_HABITAT",
                    "path": natural_habitat_raster_path,
                }
            )

    if task.has_land_use:
        land_use_raster_path = build_modality_patch_path(
            dataset_root=dataset_root,
            modality="LAND_USE",
            dept_year=task.dept_year,
            roi=task.roi,
            lidar_patch_stem=lidar_patch_stem,
        )
        if os.path.isfile(land_use_raster_path):
            land_use_values, _ = sample_raster_to_points(
                raster_path=land_use_raster_path,
                xy=coord[:, :2],
                fill_value=19,  # "Usage inconnu" index from land_use_classes.txt
            )
            out["land_use"] = land_use_values.astype(np.int16, copy=False)
        else:
            missing_modalities.append(
                {
                    "split": task.split,
                    "patch_id": task.patch_id,
                    "modality": "LAND_USE",
                    "path": land_use_raster_path,
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
                band_index=2,  # 1: DSM, 2: DTM
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
    """
    os.makedirs(output_scene_dir, exist_ok=True)
    np.save(os.path.join(output_scene_dir, "coord.npy"), scene["coord"].astype(np.float32))
    np.save(os.path.join(output_scene_dir, "color.npy"), scene["color"].astype(np.uint8))
    np.save(os.path.join(output_scene_dir, "segment.npy"), scene["segment"].astype(np.int32))
    if "strength" in scene:
        np.save(
            os.path.join(output_scene_dir, "strength.npy"),
            scene["strength"].astype(np.float32),
        )

    def _save_or_clean(filename: str, key: str, dtype, enabled: bool) -> None:
        path = os.path.join(output_scene_dir, filename)
        if enabled and key in scene:
            np.save(path, scene[key].astype(dtype))
        elif os.path.isfile(path):
            os.remove(path)

    # FOREST is conceptually always enabled; any stale file is overwritten or kept.
    _save_or_clean("forest.npy", "forest", np.int16, enabled=True)
    _save_or_clean(
        "natural_habitat.npy", "natural_habitat", np.int16, enabled=task.has_natural_habitat
    )
    _save_or_clean("land_use.npy", "land_use", np.int16, enabled=task.has_land_use)
    _save_or_clean("elevation.npy", "elevation", np.float32, enabled=task.has_dem_elev)


def _save_scene_meta(output_scene_dir: str, meta: Dict[str, Any]) -> None:
    meta_path = os.path.join(output_scene_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _process_subtile_task(
    task: PatchTask,
    dataset_root: str,
    output_root: str,
    label_definition: str,
) -> Tuple[str, List[Dict[str, str]]]:
    """Worker entry point: build and save one scene from a PatchTask.

    Returns the patch_id and a list of missing modality raster entries.
    Raises FileNotFoundError if the LiDAR PLY is missing.
    """
    scene, missing_modalities = _build_scene_from_subtile(
        task=task,
        dataset_root=dataset_root,
        label_definition=label_definition,
    )
    output_scene_dir = os.path.join(output_root, task.split, task.patch_id)
    _save_scene(output_scene_dir, scene, task)
    _save_scene_meta(output_scene_dir, {"date_gap_days": task.date_gap_days})
    return task.patch_id, missing_modalities


def _split_tasks_by_ply_existence(
    tasks: List[PatchTask], dataset_root: str
) -> Tuple[List[PatchTask], List[Dict[str, str]]]:
    """Pre-flight check: separate tasks with a present PLY from missing ones."""
    present: List[PatchTask] = []
    missing: List[Dict[str, str]] = []
    for task in tasks:
        ply_path = build_lidar_ply_path(
            dataset_root, task.dept_year, task.roi, task.scene_i_j
        )
        if os.path.isfile(ply_path):
            present.append(task)
        else:
            missing.append(
                {
                    "split": task.split,
                    "patch_id": task.patch_id,
                    "ply_path": ply_path,
                }
            )
    return present, missing


def main_process():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/data/geist/datasets/sample_flairhub_3d",
        help="Root directory containing Flair3D/LidarHD tiles",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/data/geist/Pointcept/data/flair3d",
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
            "Required path to scene_split_manifest.csv with columns: "
            + ", ".join(sorted(REQUIRED_MANIFEST_COLUMNS))
            + ". See script docstring."
        ),
    )
    parser.add_argument(
        "--label_definition",
        type=str,
        required=True,
        choices=SUPPORTED_LABEL_REMAPS,
        help="Label definition (simple or fusion) from Flair3D mappings",
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Number of workers for preprocessing"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="",
        help="Optional path to write preprocessing logs. Defaults to <output_root>/preprocess_flair3d.log",
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
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    log_file_path = args.log_file or os.path.join(args.output_root, "preprocess_flair3d.log")
    missing_scenes_path = args.missing_scenes_file or os.path.join(
        args.output_root, "missing_scenes.txt"
    )
    logger = setup_file_logger(log_file_path)

    splits = args.split if isinstance(args.split, list) else [args.split]

    logger.info("Using label definition '%s'", args.label_definition)
    logger.info("Splits to process: %s", splits)
    logger.info("Using split manifest CSV: %s", args.split_manifest_csv)

    tasks = load_manifest_tasks(args.split_manifest_csv, splits)
    logger.info("Loaded %d tasks (LIDARHD=True) from manifest.", len(tasks))
    for split in splits:
        n = sum(1 for t in tasks if t.split == split)
        logger.info("  split '%s': %d tasks", split, n)

    # Pre-create output split directories (only for splits that have tasks).
    for split in {t.split for t in tasks}:
        os.makedirs(os.path.join(args.output_root, split), exist_ok=True)

    # Pre-flight: split into present-on-disk vs missing PLY.
    present_tasks, missing_ply = _split_tasks_by_ply_existence(tasks, args.dataset_root)
    if missing_ply:
        logger.warning(
            "%d patch(es) have LIDARHD=True but their PLY file is missing on disk.",
            len(missing_ply),
        )

    missing_modalities: List[Dict[str, str]] = []
    failed_tasks: List[Dict[str, str]] = []
    scene_ids: List[str] = []

    if not present_tasks:
        logger.warning("No task to process after PLY existence check.")
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {
                pool.submit(
                    _process_subtile_task,
                    task,
                    args.dataset_root,
                    args.output_root,
                    args.label_definition,
                ): task
                for task in present_tasks
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
                    # Should be rare here (we pre-checked PLY existence) but keep
                    # the safety net for any per-modality file missing in a way
                    # that escalates to an exception.
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
        "Summary: missing_ply=%d missing_modalities=%d failed_tasks=%d",
        len(missing_ply),
        len(missing_modalities),
        len(failed_tasks),
    )
    logger.info("Detailed logs saved to: %s", log_file_path)


if __name__ == "__main__":
    main_process()
