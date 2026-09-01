"""
List all patches from lidarhd_aerial_date_gap.gpkg, assign train/val/test from
SPLIT/ FLAIR-HUB CSVs (same as convert_malibu3d_splits_to_json.py), and write a
CSV manifest with LiDAR / modality availability flags (including LAND_USE
rasters on disk where present).

Uses the same patch_id convention as scripts/convert_malibu3d_splits_to_json.py:
  patch_id = "{dept_year}_{roi}_{scene_i_j}"

The GeoPackage table date_per_patch_with_lidar has no split column; splits come
only from the CSV files.

After preprocess, enrich the manifest with n_points / n_voxels via
scripts/analyze_malibu3d_test_point_voxel_counts.py --write_manifest (needed for
val/test VoxelBudgetBatchSampler). This builder does not compute those columns
(raw GPKG stage has no coord.npy).

Network availability columns ``ROADS`` / ``RAILROADS`` / ``TRANSMISSION_LINES``
are **not** written here: they are added by Malibu3D-build
``scripts/export_network_graphs.py`` (``split_manifest_csv=...``), with
``True`` only when usable segments remain after export filters.

Example:
python scripts/flai3d_build_csv_manifest.py \
--dataset_root data/malibu3d_plus/raw \
--require_ply

``SPLIT/`` (CSVs), ``scene_split_manifest.csv`` (output), and the GeoPackage are
resolved relative to ``dataset_root`` (see ``_REL_*`` constants in the script).
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple

RepoRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Keep in sync with scripts/convert_malibu3d_splits_to_json.py (INPUT_BY_SPLIT).
_INPUT_BY_SPLIT: Dict[str, str] = {
    "train": "FLAIR-HUB_TRAIN.csv",
    "val": "FLAIR-HUB_VALID.csv",
    "test": "FLAIR-HUB_TEST.csv",
}

_DATE_NA_TOKENS = frozenset(
    {"", "<na>", "na", "none", "n/a", "nan", "null"}
)

# Paths relative to dataset_root (Malibu3D Plus raw directory).
_REL_SPLIT_DIR = "SPLIT"
_REL_MANIFEST_CSV = "scene_split_manifest.csv"
_REL_GPKG = os.path.join("lidarhd_aerial_date_gap.gpkg")


def resolve_dataset_root(path: str) -> str:
    """If path is relative, treat it as relative to the repository root."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(RepoRoot, path))


def parse_patch_id(patch_id: str) -> Tuple[str, str, str]:
    """
    Parse a patch_id like:
      D004-2021_AU-S1-33_1-1
    into:
      (dept_year, roi, scene_i_j)
    """
    parts = patch_id.strip().split("_", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid patch_id format (expected 3 underscore-separated fields): {patch_id}"
        )
    return parts[0], parts[1], parts[2]


def load_patch_id_to_split(split_dir: str) -> Dict[str, str]:
    """Load patch_id -> train|val|test from FLAIR-HUB split CSVs."""
    out: Dict[str, str] = {}
    for split, filename in _INPUT_BY_SPLIT.items():
        input_path = os.path.join(split_dir, filename)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(
                f"Missing input CSV for split '{split}': {input_path}"
            )
        with open(input_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            if reader.fieldnames is None or "patch_id" not in reader.fieldnames:
                raise KeyError(f"'patch_id' column not found in: {input_path}")
            for row in reader:
                pid = (row.get("patch_id") or "").strip()
                if not pid:
                    continue
                out[pid] = split
    return out


def load_patch_rows_from_gpkg(gpkg_path: str) -> Dict[str, Dict[str, Any]]:
    """Load date_per_patch_with_lidar keyed by patch_id."""
    if not os.path.isfile(gpkg_path):
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    conn = sqlite3.connect(gpkg_path)
    try:
        rows = conn.execute(
            "SELECT patch_id, date_aerial_rgb, date_lidarhd, date_gap_days "
            "FROM date_per_patch_with_lidar"
        ).fetchall()
    finally:
        conn.close()

    out: Dict[str, Dict[str, Any]] = {}
    for patch_id, date_aerial_rgb, date_lidarhd, date_gap_days in rows:
        if patch_id is None:
            continue
        pid = str(patch_id)
        out[pid] = {
            "date_aerial_rgb": date_aerial_rgb,
            "date_lidarhd": date_lidarhd,
            "date_gap_days": None if date_gap_days is None else float(date_gap_days),
        }
    return out


def lidarhd_present_from_gpkg_row(
    date_lidarhd: Any, date_gap_days: Optional[float]
) -> bool:
    """
    True when LiDAR HD acquisition is recorded for the patch.

    README: no point cloud => date_lidarhd '<NA>' and date_gap_days NULL.
    """
    if date_lidarhd is not None:
        token = str(date_lidarhd).strip().lower()
        if token and token not in _DATE_NA_TOKENS:
            return True
    if date_gap_days is not None:
        return True
    return False


def lidar_ply_path(dataset_root: str, dept_year: str, roi: str, scene_i_j: str) -> str:
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
    """Same logic as pointcept...preprocess_malibu3d.build_modality_patch_path (no package import)."""
    modality_stem = lidar_patch_stem.replace("_LIDARHD_", f"_{modality}_")
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


def modality_files_exist(
    dataset_root: str, dept_year: str, roi: str, scene_i_j: str
) -> Tuple[bool, bool, bool]:
    """Return (natural_habitat_exists, land_use_exists, dem_elev_exists)."""
    stem = f"{dept_year}_LIDARHD_{roi}_{scene_i_j}"
    nh_path = build_modality_patch_path(
        dataset_root, "NATURAL_HABITAT", dept_year, roi, stem
    )
    lu_path = build_modality_patch_path(
        dataset_root, "LAND_USE", dept_year, roi, stem
    )
    dem_path = build_modality_patch_path(
        dataset_root, "DEM_ELEV", dept_year, roi, stem
    )
    return (
        os.path.isfile(nh_path),
        os.path.isfile(lu_path),
        os.path.isfile(dem_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate GPKG patches with SPLIT/ CSV splits, LiDAR HD flags from "
            "metadata (and optional .ply), and write a manifest CSV. Only "
            "--dataset_root is configurable; SPLIT/, output CSV, and .gpkg paths "
            "are fixed relative to it."
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.path.join("data", "malibu3d_plus", "raw"),
        help=(
            "Malibu3D Plus raw directory (relative to repo root if not absolute). "
            f"Split CSVs: <dataset_root>/{_REL_SPLIT_DIR}/; "
            f"manifest: <dataset_root>/{_REL_MANIFEST_CSV}; "
            f"GeoPackage: <dataset_root>/{_REL_GPKG.replace(os.sep, '/')}. "
            "Also used for NATURAL_HABITAT / LAND_USE / DEM_ELEV / optional .ply checks."
        ),
    )
    parser.add_argument(
        "--require_ply",
        action="store_true",
        help="Require a LiDAR .ply file on disk in addition to GPKG metadata.",
    )
    parser.add_argument(
        "--only_with_lidar",
        action="store_true",
        help="CSV contains only rows where LIDARHD is True (default: all rows).",
    )
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.dataset_root)
    split_dir = os.path.normpath(os.path.join(dataset_root, _REL_SPLIT_DIR))
    out_csv = os.path.normpath(os.path.join(dataset_root, _REL_MANIFEST_CSV))
    gpkg = os.path.normpath(os.path.join(dataset_root, _REL_GPKG))

    gpkg_by_patch = load_patch_rows_from_gpkg(gpkg)
    split_by_patch = load_patch_id_to_split(split_dir)
    check_files = bool(dataset_root and os.path.isdir(dataset_root))
    if not check_files:
        print(
            "[WARN] dataset_root missing or not a directory — "
            "NATURAL_HABITAT, LAND_USE, and DEM_ELEV set to False; --require_ply cannot be satisfied.",
            file=sys.stderr,
        )
        if args.require_ply:
            raise SystemExit(
                "Cannot use --require_ply without a valid --dataset_root directory."
            )

    missing_split = 0
    bad_patch_id = 0
    csv_rows: List[Dict[str, Any]] = []

    for patch_id in sorted(gpkg_by_patch.keys()):
        row_meta = gpkg_by_patch[patch_id]
        d_aerial = row_meta["date_aerial_rgb"]
        d_lidar = row_meta["date_lidarhd"]
        gap = row_meta["date_gap_days"]
        present_gpkg = lidarhd_present_from_gpkg_row(d_lidar, gap)

        try:
            dept_year, roi, scene_i_j = parse_patch_id(patch_id)
        except ValueError:
            bad_patch_id += 1
            dept_year, roi, scene_i_j = "", "", ""

        split = split_by_patch.get(patch_id, "")
        if split == "" and dept_year:
            missing_split += 1

        ply_ok = True
        if args.require_ply:
            if dept_year and roi and scene_i_j:
                ply_path = lidar_ply_path(dataset_root, dept_year, roi, scene_i_j)
                ply_ok = os.path.isfile(ply_path)
            else:
                ply_ok = False

        lidarhd = bool(present_gpkg and ply_ok)

        if check_files and dept_year and roi and scene_i_j:
            has_nh, has_lu, has_dem = modality_files_exist(
                dataset_root, dept_year, roi, scene_i_j
            )
        else:
            has_nh, has_lu, has_dem = False, False, False

        rec = {
            "split": split,
            "dept_year": dept_year,
            "roi": roi,
            "scene_i_j": scene_i_j,
            "patch_id": patch_id,
            "LIDARHD": lidarhd,
            "NATURAL_HABITAT": has_nh,
            "LAND_USE": has_lu,
            "DEM_ELEV": has_dem,
            "date_aerial_rgb": d_aerial,
            "date_lidarhd": d_lidar,
            "date_gap_days": gap,
        }

        if not args.only_with_lidar or lidarhd:
            csv_rows.append(rec)

    if missing_split:
        print(
            f"[WARN] {missing_split} GPKG patches had no patch_id match in SPLIT CSVs.",
            file=sys.stderr,
        )
    if bad_patch_id:
        print(
            f"[WARN] {bad_patch_id} GPKG rows had patch_id strings that could not be parsed.",
            file=sys.stderr,
        )

    fieldnames = [
        "split",
        "dept_year",
        "roi",
        "scene_i_j",
        "patch_id",
        "LIDARHD",
        "NATURAL_HABITAT",
        "LAND_USE",
        "DEM_ELEV",
        "date_aerial_rgb",
        "date_lidarhd",
        "date_gap_days",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in csv_rows:
            row_out = {k: rec[k] for k in fieldnames}
            for key in ("date_aerial_rgb", "date_lidarhd", "date_gap_days"):
                if row_out[key] is None:
                    row_out[key] = ""
            writer.writerow(row_out)

    lidar_true = sum(1 for r in csv_rows if r["LIDARHD"])
    print(f"Read {len(gpkg_by_patch)} patches from GPKG {gpkg}")
    print(f"Loaded {len(split_by_patch)} patch_id -> split entries from {split_dir}")
    print(
        f"Wrote {len(csv_rows)} CSV rows to {out_csv} "
        f"({lidar_true} with LIDARHD=True)"
    )


if __name__ == "__main__":
    main()
