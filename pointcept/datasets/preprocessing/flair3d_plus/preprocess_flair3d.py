"""
Preprocessing script for Flair3D+ (LidarHD).

This script converts raw tiles/subtiles into Pointcept scene folders containing:
- coord.npy
- color.npy
- segment.npy
- strength.npy (LiDAR intensity)
- forest.npy (from GeoTIFF when present; no column in the manifest)
- natural_habitat.npy, land_use.npy, elevation.npy — only when enabled in the manifest

Required manifest CSV (one row per patch), e.g. ``data/flair3d_plus/raw/scene_split_manifest.csv``:
  split, dept_year, roi, patch_id, LIDARHD, NATURAL_HABITAT, LAND_USE, DEM_ELEV
(plus optional extra columns which are ignored)

Rules:
- Rows with LIDARHD=False are skipped (no scene output for that patch).
- When NATURAL_HABITAT / LAND_USE / DEM_ELEV is False, the matching .npy is not written.

python pointcept/datasets/preprocessing/flair3d_plus/preprocess_flair3d.py \\
    --dataset_root data/flair3d_plus/raw \\
    --output_root data/flair3d_plus \\
    --label_definition inter_finerall6 \\
    --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest.csv
    
python pointcept/datasets/preprocessing/flair3d_plus/preprocess_flair3d.py --dataset_root data/flair3d_plus/raw --output_root data/flair3d_plus --label_definition inter_finerall8 --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest.csv  

"""

import argparse
import csv
import glob
import json
import logging
import os
import re
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        "patch_id",
        "LIDARHD",
        "NATURAL_HABITAT",
        "LAND_USE",
        "DEM_ELEV",
    }
)


def _parse_csv_bool(raw: Optional[str], field: str, patch_id: str, csv_path: str) -> bool:
    token = (raw or "").strip().lower()
    if token in ("true", "1", "yes"):
        return True
    if token in ("false", "0", "no"):
        return False
    raise ValueError(
        f"Invalid boolean for column '{field}' (patch_id={patch_id}) in {csv_path}: {raw!r}"
    )


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


def build_lidar_roi_dir(dataset_root: str, dept_year: str, roi: str) -> str:
    """Build the expected LiDAR ROI directory from split metadata."""
    return os.path.join(dataset_root, "LIDARHD", f"{dept_year}_LIDARHD", roi)


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


def collect_split_groups(
    dataset_root: str, split: str, split_rois: Dict[str, List[Tuple[str, str]]]
) -> Tuple[List[Tuple[str, str, List[str]]], List[Dict[str, str]]]:
    """Collect valid (dept_year, roi, ply_files) groups and missing ROIs for one split."""
    rois = split_rois.get(split, [])
    groups: List[Tuple[str, str, List[str]]] = []
    missing_rois: List[Dict[str, str]] = []
    for dept_year, roi in rois:
        roi_dir = build_lidar_roi_dir(dataset_root, dept_year, roi)
        if not os.path.isdir(roi_dir):
            print(f"[WARN] Missing LiDAR ROI directory, skipped: {roi_dir}")
            missing_rois.append(
                {
                    "split": split,
                    "dept_year": dept_year,
                    "roi": roi,
                    "reason": "missing_roi_directory",
                    "path": roi_dir,
                }
            )
            continue
        ply_files = sorted(glob.glob(os.path.join(roi_dir, "*.ply")))
        if not ply_files:
            print(f"[WARN] No .ply files found in ROI directory: {roi_dir}")
            missing_rois.append(
                {
                    "split": split,
                    "dept_year": dept_year,
                    "roi": roi,
                    "reason": "missing_ply_files",
                    "path": roi_dir,
                }
            )
            continue
        groups.append((dept_year, roi, ply_files))
    return groups, missing_rois


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
    missing_rois: List[Dict[str, str]],
    failed_tasks: List[Dict[str, str]],
) -> None:
    """Write a text report listing missing scenes and task failures."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Missing scenes report\n\n")
        f.write(f"missing_rois={len(missing_rois)}\n")
        f.write(f"failed_tasks={len(failed_tasks)}\n\n")

        f.write("## Missing ROIs (from split manifest)\n")
        if not missing_rois:
            f.write("none\n")
        else:
            for item in missing_rois:
                f.write(
                    f"{item['split']},{item['dept_year']},{item['roi']},"
                    f"{item['reason']},{item['path']}\n"
                )

        f.write("\n## Failed preprocessing tasks\n")
        if not failed_tasks:
            f.write("none\n")
        else:
            for item in failed_tasks:
                f.write(
                    f"{item['split']},{item['dept_year']},{item['roi']},"
                    f"{item['ply_path']},{item['error']}\n"
                )


def load_scene_split_manifest(
    csv_path: str,
) -> Tuple[
    Dict[str, List[Tuple[str, str]]],
    Dict[str, bool],
    Dict[str, Tuple[bool, bool, bool]],
]:
    """
    Load ``scene_split_manifest.csv`` with required columns (see REQUIRED_MANIFEST_COLUMNS).

    Returns:
        split_rois: deduped (dept_year, roi) lists per split (preserving first-seen order).
        patch_lidarhd: patch_id -> whether to run LiDAR preprocessing for that patch.
        patch_modalities: patch_id -> (NATURAL_HABITAT, LAND_USE, DEM_ELEV) availability flags.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Split manifest CSV not found: {csv_path}")

    split_rois: Dict[str, List[Tuple[str, str]]] = {"train": [], "val": [], "test": []}
    roi_seen: set = set()
    patch_lidarhd: Dict[str, bool] = {}
    patch_modalities: Dict[str, Tuple[bool, bool, bool]] = {}

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
            patch_id = (row.get("patch_id") or "").strip()
            if not split or not dept_year or not roi or not patch_id:
                continue
            if split not in split_rois:
                raise ValueError(
                    f"Unsupported split '{split}' in {csv_path}. Expected one of: train, val, test."
                )

            lidarhd = _parse_csv_bool(row.get("LIDARHD", ""), "LIDARHD", patch_id, csv_path)
            nh = _parse_csv_bool(
                row.get("NATURAL_HABITAT", ""), "NATURAL_HABITAT", patch_id, csv_path
            )
            lu = _parse_csv_bool(row.get("LAND_USE", ""), "LAND_USE", patch_id, csv_path)
            dem = _parse_csv_bool(row.get("DEM_ELEV", ""), "DEM_ELEV", patch_id, csv_path)
            mods = (nh, lu, dem)

            if patch_id in patch_lidarhd:
                prev_l = patch_lidarhd[patch_id]
                prev_m = patch_modalities[patch_id]
                if prev_l != lidarhd or prev_m != mods:
                    raise ValueError(
                        f"Inconsistent duplicate patch_id '{patch_id}' in {csv_path}: "
                        f"first=({prev_l}, {prev_m}), duplicate=({lidarhd}, {mods})"
                    )
            else:
                patch_lidarhd[patch_id] = lidarhd
                patch_modalities[patch_id] = mods

            roi_key = (split, dept_year, roi)
            if roi_key not in roi_seen:
                roi_seen.add(roi_key)
                split_rois[split].append((dept_year, roi))

    return split_rois, patch_lidarhd, patch_modalities


def _build_scene_from_subtile(
    filepath: str,
    dataset_root: str,
    dept_year: str,
    roi: str,
    label_definition: str,
    include_natural_habitat: bool,
    include_land_use: bool,
    include_dem_elev: bool,
) -> Dict[str, np.ndarray]:
    """Build a Pointcept scene dict from one PLY subtile."""
    attributes = read_ply_binary(filepath)

    for axis in ("x", "y", "z"):
        if axis not in attributes:
            raise KeyError(f"Missing '{axis}' in {filepath}")
    coord = np.stack([attributes["x"], attributes["y"], attributes["z"]], axis=1).astype(
        np.float32
    )

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

    out = {
        "coord": coord,
        "color": color,
        "segment": segment,
    }

    if "intensity" in attributes:
        strength = np.clip(attributes["intensity"].astype(np.float32), 0, 60000) / 60000
        out["strength"] = strength.astype(np.float32)

    # FOREST point-wise labels from GeoTIFF.
    lidar_patch_stem = os.path.splitext(os.path.basename(filepath))[0]
    forest_raster_path = build_modality_patch_path(
        dataset_root=dataset_root,
        modality="FOREST",
        dept_year=dept_year,
        roi=roi,
        lidar_patch_stem=lidar_patch_stem,
    )
    if os.path.isfile(forest_raster_path):
        forest_values, _ = sample_raster_to_points(
            raster_path=forest_raster_path,
            xy=coord[:, :2],
            fill_value=2, # Void idx
        )
        out["forest"] = forest_values.astype(np.int16, copy=False)
    else:
        print(f"[WARN] Missing FOREST raster for {filepath}: {forest_raster_path}")

    if include_natural_habitat:
        natural_habitat_raster_path = build_modality_patch_path(
            dataset_root=dataset_root,
            modality="NATURAL_HABITAT",
            dept_year=dept_year,
            roi=roi,
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
            print(
                f"[WARN] Missing NATURAL_HABITAT raster for {filepath}: "
                f"{natural_habitat_raster_path}"
            )

    if include_land_use:
        land_use_raster_path = build_modality_patch_path(
            dataset_root=dataset_root,
            modality="LAND_USE",
            dept_year=dept_year,
            roi=roi,
            lidar_patch_stem=lidar_patch_stem,
        )
        if os.path.isfile(land_use_raster_path):
            land_use_values, _ = sample_raster_to_points(
                raster_path=land_use_raster_path,
                xy=coord[:, :2],
                fill_value=19,  # "Usage inconnu" index from land_use_classes.txt
            )

            # Mapping from 1-20 to 0-19 (we want "dense" indices).
            # land_use_values = land_use_values - 1  # Already dense in raster.

            out["land_use"] = land_use_values.astype(np.int16, copy=False)
        else:
            print(f"[WARN] Missing LAND_USE raster for {filepath}: {land_use_raster_path}")

    if include_dem_elev:
        dem_elev_raster_path = build_modality_patch_path(
            dataset_root=dataset_root,
            modality="DEM_ELEV",
            dept_year=dept_year,
            roi=roi,
            lidar_patch_stem=lidar_patch_stem,
        )
        if os.path.isfile(dem_elev_raster_path):
            dtm_values, _ = sample_raster_to_points_float(
                raster_path=dem_elev_raster_path,
                xy=coord[:, :2],
                fill_value=np.nan,
                band_index=2,  # 1 : DSM, 2 : DTM
            )
            elevation = coord[:, 2].astype(np.float32, copy=False) - dtm_values
            out["elevation"] = elevation.astype(np.float32, copy=False)
        else:
            print(f"[WARN] Missing DEM_ELEV raster for {filepath}: {dem_elev_raster_path}")

    return out


def _save_scene(
    output_scene_dir: str,
    scene: Dict[str, np.ndarray],
    include_natural_habitat: bool,
    include_land_use: bool,
    include_dem_elev: bool,
) -> None:
    os.makedirs(output_scene_dir, exist_ok=True)
    np.save(os.path.join(output_scene_dir, "coord.npy"), scene["coord"].astype(np.float32))
    np.save(os.path.join(output_scene_dir, "color.npy"), scene["color"].astype(np.uint8))
    np.save(os.path.join(output_scene_dir, "segment.npy"), scene["segment"].astype(np.int32))
    if "strength" in scene:
        np.save(
            os.path.join(output_scene_dir, "strength.npy"),
            scene["strength"].astype(np.float32),
        )
    if "forest" in scene:
        np.save(
            os.path.join(output_scene_dir, "forest.npy"),
            scene["forest"].astype(np.int16),
        )
    if include_natural_habitat and "natural_habitat" in scene:
        np.save(
            os.path.join(output_scene_dir, "natural_habitat.npy"),
            scene["natural_habitat"].astype(np.int16),
        )
    elif not include_natural_habitat:
        nh_path = os.path.join(output_scene_dir, "natural_habitat.npy")
        if os.path.isfile(nh_path):
            os.remove(nh_path)
    if include_land_use and "land_use" in scene:
        np.save(
            os.path.join(output_scene_dir, "land_use.npy"),
            scene["land_use"].astype(np.int16),
        )
    elif not include_land_use:
        lu_path = os.path.join(output_scene_dir, "land_use.npy")
        if os.path.isfile(lu_path):
            os.remove(lu_path)
    if include_dem_elev and "elevation" in scene:
        np.save(
            os.path.join(output_scene_dir, "elevation.npy"),
            scene["elevation"].astype(np.float32),
        )
    elif not include_dem_elev:
        el_path = os.path.join(output_scene_dir, "elevation.npy")
        if os.path.isfile(el_path):
            os.remove(el_path)


def _save_scene_meta(output_scene_dir: str, meta: Dict[str, Any]) -> None:
    meta_path = os.path.join(output_scene_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _build_scene_id(dept_year: str, roi: str, ply_path: str) -> str:
    stem = os.path.splitext(os.path.basename(ply_path))[0]
    suffix_match = re.search(r"(\d+-\d+)$", stem)
    if suffix_match:
        subtile_id = suffix_match.group(1)
    else:
        # Keep previous behavior when pattern is not present.
        print(f"[WARN] Could not extract '<num>-<num>' suffix from '{stem}', using full stem.")
        subtile_id = stem
    return f"{dept_year}_{roi}_{subtile_id}"


def load_date_gap_days_by_patch(gpkg_path: str) -> Dict[str, Optional[float]]:
    """Load patch_id -> date_gap_days from lidarhd_aerial_date_gap.gpkg."""
    if not os.path.isfile(gpkg_path):
        print(f"[WARN] Missing date gap geopackage: {gpkg_path}")
        return {}

    conn = sqlite3.connect(gpkg_path)
    try:
        rows = conn.execute(
            "SELECT patch_id, date_gap_days FROM date_per_patch_with_lidar"
        ).fetchall()
    finally:
        conn.close()

    out: Dict[str, Optional[float]] = {}
    for patch_id, date_gap_days in rows:
        if patch_id is None:
            continue
        out[str(patch_id)] = None if date_gap_days is None else float(date_gap_days)
    return out


def _process_subtile_task(
    ply_path: str,
    dept_year: str,
    roi: str,
    dataset_root: str,
    output_root: str,
    split: str,
    label_definition: str,
    date_gap_days: Optional[float],
    include_natural_habitat: bool,
    include_land_use: bool,
    include_dem_elev: bool,
) -> str:
    part = _build_scene_from_subtile(
        filepath=ply_path,
        dataset_root=dataset_root,
        dept_year=dept_year,
        roi=roi,
        label_definition=label_definition,
        include_natural_habitat=include_natural_habitat,
        include_land_use=include_land_use,
        include_dem_elev=include_dem_elev,
    )
    scene_id = _build_scene_id(dept_year=dept_year, roi=roi, ply_path=ply_path)
    output_scene_dir = os.path.join(output_root, split, scene_id)
    _save_scene(
        output_scene_dir,
        part,
        include_natural_habitat=include_natural_habitat,
        include_land_use=include_land_use,
        include_dem_elev=include_dem_elev,
    )
    _save_scene_meta(output_scene_dir, {"date_gap_days": date_gap_days})
    return scene_id


def _run_task(args_tuple):
    """Pool helper to avoid lambda pickling issues."""
    process_fn, fn_args = args_tuple
    return process_fn(*fn_args)


def main_process():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset_root",
        type=str,
        # required=True,
        default="/data/geist/datasets/sample_flairhub_3d",
        help="Root directory containing Flair3D/LidarHD tiles",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        # required=True,
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
        "--date_gap_gpkg",
        type=str,
        default="",
        help=(
            "Optional path to lidarhd_aerial_date_gap.gpkg. "
            "If empty, defaults to <dirname(dataset_root)>/flair3d_plus_010/lidarhd_aerial_date_gap.gpkg"
        ),
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

    # Normalize to list (argparse with nargs='+' returns list; default was single element)
    splits = args.split if isinstance(args.split, list) else [args.split]

    logger.info("Using label definition '%s'", args.label_definition)
    logger.info("Splits to process: %s", splits)

    split_rois, patch_lidarhd, patch_modalities = load_scene_split_manifest(
        args.split_manifest_csv
    )
    logger.info("Using split manifest CSV: %s", args.split_manifest_csv)

    if args.date_gap_gpkg:
        date_gap_gpkg_path = args.date_gap_gpkg
    else:
        date_gap_gpkg_path = os.path.join(
            os.path.dirname(os.path.abspath(args.dataset_root)),
            "flair3d_plus_010",
            "lidarhd_aerial_date_gap.gpkg",
        )
    date_gap_by_patch = load_date_gap_days_by_patch(date_gap_gpkg_path)
    logger.info("Date gap source: %s", date_gap_gpkg_path)

    all_missing_rois: List[Dict[str, str]] = []
    all_failed_tasks: List[Dict[str, str]] = []

    for split in splits:
        os.makedirs(os.path.join(args.output_root, split), exist_ok=True)

        logger.info("--- Split '%s' ---", split)
        split_groups, missing_rois = collect_split_groups(
            args.dataset_root, split, split_rois
        )
        all_missing_rois.extend(missing_rois)
        logger.info("Configured split ROIs: %d", len(split_rois.get(split, [])))
        if missing_rois:
            logger.warning("Found %d missing ROIs for split '%s'", len(missing_rois), split)

        process_fn = _process_subtile_task

        if len(split_groups) == 0:
            logger.warning("No task found for split '%s', skipping.", split)
            continue

        task_count = 0
        for dept_year, roi, ply_files in split_groups:
            for ply_path in ply_files:
                pid = _build_scene_id(dept_year=dept_year, roi=roi, ply_path=ply_path)
                if pid in patch_lidarhd and patch_lidarhd[pid]:
                    task_count += 1
        logger.info(
            "Found %d LiDAR patches to process for split '%s' (after LIDARHD manifest filter).",
            task_count,
            split,
        )

        scene_ids: List[str] = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            for dept_year, roi, ply_files in tqdm(
                split_groups,
                desc=f"{split}_rois",
                unit="roi",
                total=len(split_groups),
            ):
                process_args: List[tuple] = []
                for ply_path in ply_files:
                    patch_id = _build_scene_id(
                        dept_year=dept_year, roi=roi, ply_path=ply_path
                    )
                    if patch_id not in patch_lidarhd:
                        logger.warning(
                            "PLY not listed in manifest (patch_id=%s), skipping: %s",
                            patch_id,
                            ply_path,
                        )
                        continue
                    if not patch_lidarhd[patch_id]:
                        continue
                    nh, lu, dem = patch_modalities[patch_id]
                    process_args.append(
                        (
                            ply_path,
                            dept_year,
                            roi,
                            args.dataset_root,
                            args.output_root,
                            split,
                            args.label_definition,
                            date_gap_by_patch.get(patch_id),
                            nh,
                            lu,
                            dem,
                        )
                    )
                if not process_args:
                    continue

                futures = {
                    pool.submit(_run_task, (process_fn, process_arg)): process_arg
                    for process_arg in process_args
                }
                for future in as_completed(futures):
                    process_arg = futures[future]
                    try:
                        scene_ids.append(future.result())
                    except Exception as exc:
                        failed_entry = {
                            "split": split,
                            "dept_year": process_arg[1],
                            "roi": process_arg[2],
                            "ply_path": process_arg[0],
                            "error": repr(exc),
                        }
                        all_failed_tasks.append(failed_entry)
                        logger.exception(
                            "Failed task for split=%s dept_year=%s roi=%s ply_path=%s",
                            split,
                            process_arg[1],
                            process_arg[2],
                            process_arg[0],
                        )
        out_split_dir = os.path.join(args.output_root, split)
        logger.info("Done. Processed %d scenes into %s", len(scene_ids), out_split_dir)

    write_missing_scenes_report(
        output_path=missing_scenes_path,
        missing_rois=all_missing_rois,
        failed_tasks=all_failed_tasks,
    )
    logger.info("Wrote missing-scenes report to: %s", missing_scenes_path)
    logger.info(
        "Summary: missing_rois=%d failed_tasks=%d",
        len(all_missing_rois),
        len(all_failed_tasks),
    )
    logger.info("Detailed logs saved to: %s", log_file_path)


if __name__ == "__main__":
    main_process()
