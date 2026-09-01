#!/usr/bin/env python3
"""
Audit per-subtile point and voxel counts for Malibu3D+ (test/val/train).

Uses all LIDARHD=True rows from the CSV manifest for the requested splits.
Dataset exclusions (missing / too_small / hardcoded) are NOT applied by default:
those tiles stay in the manifest and should still get n_points / n_voxels.
Pass --apply_dataset_exclusions only if you want Malibu3DDataset-matched audit stats.

Voxel count matches GridSample (grid floor + FNV unique keys) at the given grid_size.

Optionally enrich the scene_split_manifest CSV in place with n_points / n_voxels
columns (--write_manifest). Preferred source for VoxelBudgetBatchSampler.

Example (audit only):
python scripts/analyze_malibu3d_test_point_voxel_counts.py \
  --data_root data/malibu3d_plus \
  --csv_manifest data/malibu3d_plus/raw/scene_split_manifest.csv \
  --splits val,test \
  --grid_size 0.1 \
  --num_workers 16 \
  --output_dir stats/malibu3d/test_point_voxel_counts

Example (enrich scene_split_manifest after preprocess):
python scripts/analyze_malibu3d_test_point_voxel_counts.py \
  --data_root data/malibu3d_plus \
  --csv_manifest data/malibu3d_plus/raw/scene_split_manifest.csv \
  --splits val,test \
  --grid_size 0.1 \
  --write_manifest data/malibu3d_plus/raw/scene_split_manifest.csv \
  --num_workers 16 \
  --output_dir stats/malibu3d/test_point_voxel_counts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def build_scene_path(
    output_root: str,
    split: str,
    patch_id: str,
    dept_year: str,
    roi: str,
) -> str:
    return os.path.join(output_root, split, f"{dept_year}_LIDARHD", roi, patch_id)


@dataclass(frozen=True)
class SceneRecord:
    split: str
    patch_id: str
    scene_path: str


@dataclass
class SceneCountResult:
    split: str
    patch_id: str
    scene_path: str
    n_points: int
    n_voxels: int
    points_per_voxel: float
    error: str = ""


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def parse_splits(splits_arg: str) -> set[str]:
    return {token.strip() for token in splits_arg.split(",") if token.strip()}


def parse_manifest_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def load_hardcoded_excluded_tiles() -> set[tuple[str, str]]:
    excluded: set[tuple[str, str]] = set()
    details_csv = os.path.join(
        REPO_ROOT, "data", "malibu3d_plus", "missing_coord_tiles.details.csv"
    )
    if not os.path.isfile(details_csv):
        return excluded
    with open(details_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("reason") != "missing_coord_file":
                continue
            split = (row.get("split") or "").strip()
            patch_id = (row.get("patch_id") or "").strip()
            if split and patch_id:
                excluded.add((split, patch_id))
    return excluded


def load_missing_tiles_manifest(path: str | None) -> set[tuple[str, str]]:
    missing_tiles: set[tuple[str, str]] = set()
    if not path or not os.path.isfile(path):
        if path:
            print(f"Warning: missing tiles manifest not found: {path}")
        return missing_tiles
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [part.strip() for part in stripped.split(",", 2)]
            if len(parts) < 2:
                continue
            split, patch_id = parts[0], parts[1]
            if split and patch_id:
                missing_tiles.add((split, patch_id))
    return missing_tiles


def load_too_small_tiles_manifest(path: str | None) -> set[tuple[str, str]]:
    too_small_tiles: set[tuple[str, str]] = set()
    if not path or not os.path.isfile(path):
        if path:
            print(f"Warning: too-small tiles manifest not found: {path}")
        return too_small_tiles
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_split = (row.get("split") or "").strip()
            patch_id = (row.get("patch_id") or "").strip()
            if row_split and patch_id:
                too_small_tiles.add((row_split, patch_id))
    return too_small_tiles


def load_scene_records(
    data_root: str,
    csv_manifest: str,
    target_splits: set[str],
    excluded_tiles: set[tuple[str, str]],
) -> list[SceneRecord]:
    scene_records: list[SceneRecord] = []
    with open(csv_manifest, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "patch_id", "LIDARHD"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise KeyError(f"Missing required columns in manifest: {sorted(missing_cols)}")

        for row in reader:
            split = str(row["split"]).strip()
            patch_id = str(row["patch_id"]).strip()
            if not split or not patch_id:
                continue
            if split not in target_splits:
                continue
            if not parse_manifest_bool(row.get("LIDARHD")):
                continue
            if (split, patch_id) in excluded_tiles:
                continue

            dept_year = (row.get("dept_year") or "").strip() or patch_id.split("_", 2)[0]
            roi = (row.get("roi") or "").strip() or patch_id.split("_", 2)[1]
            scene_path = build_scene_path(data_root, split, patch_id, dept_year, roi)
            scene_records.append(
                SceneRecord(split=split, patch_id=patch_id, scene_path=scene_path)
            )
    return scene_records


def fnv_hash_vec(arr: np.ndarray) -> np.ndarray:
    """FNV64-1A, same as GridSample.fnv_hash_vec."""
    assert arr.ndim == 2
    arr = arr.astype(np.uint64, copy=False)
    hashed = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed *= np.uint64(1099511628211)
        hashed = np.bitwise_xor(hashed, arr[:, j])
    return hashed


def count_voxels(coord: np.ndarray, grid_size: float) -> int:
    """Unique voxels after floor(coord / grid_size), matching GridSample uniqueness."""
    if coord.shape[0] == 0:
        return 0
    grid_coord = np.floor(coord / np.asarray(grid_size, dtype=np.float64)).astype(
        np.int64
    )
    grid_coord = grid_coord - grid_coord.min(axis=0)
    keys = fnv_hash_vec(grid_coord)
    return int(np.unique(keys).shape[0])


def _empty_result(scene: SceneRecord, error: str = "") -> SceneCountResult:
    return SceneCountResult(
        split=scene.split,
        patch_id=scene.patch_id,
        scene_path=scene.scene_path,
        n_points=0,
        n_voxels=0,
        points_per_voxel=0.0,
        error=error,
    )


def analyze_scene(scene: SceneRecord, grid_size: float) -> SceneCountResult:
    coord_path = os.path.join(scene.scene_path, "coord.npy")
    if not os.path.isfile(coord_path):
        return _empty_result(scene, error="missing_coord")

    coord = np.load(coord_path, mmap_mode="r")
    if coord.ndim != 2 or coord.shape[1] < 3:
        return _empty_result(scene, error=f"bad_coord_shape:{tuple(coord.shape)}")

    n_points = int(coord.shape[0])
    if n_points == 0:
        return _empty_result(scene, error="empty_coord")

    # Materialize only what we need for hashing (mmap may be slow on random access).
    coord_xyz = np.asarray(coord[:, :3], dtype=np.float64)
    n_voxels = count_voxels(coord_xyz, grid_size)
    ppv = float(n_points) / float(n_voxels) if n_voxels > 0 else 0.0
    return SceneCountResult(
        split=scene.split,
        patch_id=scene.patch_id,
        scene_path=scene.scene_path,
        n_points=n_points,
        n_voxels=n_voxels,
        points_per_voxel=ppv,
    )


def _process_scene(args: Tuple[SceneRecord, float]) -> SceneCountResult:
    scene, grid_size = args
    try:
        return analyze_scene(scene, grid_size)
    except Exception as exc:  # noqa: BLE001 — keep worker alive, record error
        return _empty_result(scene, error=str(exc))


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def summarize_counts(
    results: Sequence[SceneCountResult], *, top_k: int = 20
) -> dict:
    ok = [r for r in results if not r.error]
    errors = [r for r in results if r.error]
    n_points = np.asarray([r.n_points for r in ok], dtype=np.int64)
    n_voxels = np.asarray([r.n_voxels for r in ok], dtype=np.int64)

    def _side_stats(arr: np.ndarray) -> dict:
        if arr.size == 0:
            return {
                "count": 0,
                "min": 0,
                "median": 0.0,
                "mean": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p98": 0.0,
                "p99": 0.0,
                "max": 0,
            }
        return {
            "count": int(arr.size),
            "min": int(arr.min()),
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "p90": _percentile(arr, 90),
            "p95": _percentile(arr, 95),
            "p98": _percentile(arr, 98),
            "p99": _percentile(arr, 99),
            "max": int(arr.max()),
        }

    by_voxels = sorted(ok, key=lambda r: r.n_voxels, reverse=True)
    top_outliers = [
        {
            "patch_id": r.patch_id,
            "split": r.split,
            "n_points": r.n_points,
            "n_voxels": r.n_voxels,
            "points_per_voxel": r.points_per_voxel,
            "scene_path": r.scene_path,
        }
        for r in by_voxels[:top_k]
    ]

    return {
        "n_scenes_ok": len(ok),
        "n_scenes_error": len(errors),
        "n_points": _side_stats(n_points),
        "n_voxels": _side_stats(n_voxels),
        "top_outliers_by_n_voxels": top_outliers,
        "error_examples": [
            {"patch_id": r.patch_id, "error": r.error, "scene_path": r.scene_path}
            for r in errors[:50]
        ],
    }


def _write_counts_csv(path: str, rows: List[SceneCountResult]) -> None:
    fieldnames = [
        "split",
        "patch_id",
        "scene_path",
        "n_points",
        "n_voxels",
        "points_per_voxel",
        "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_counts_into_manifest(
    manifest_path: str, results: Sequence[SceneCountResult]
) -> Tuple[int, int]:
    """Enrich scene_split_manifest with n_points / n_voxels (temp file then replace).

    Only successful results update cells. Other rows keep prior values (or empty
    for newly added columns). Returns (n_rows_updated, n_manifest_rows).
    """
    ok_by_key: Dict[Tuple[str, str], SceneCountResult] = {
        (r.split, r.patch_id): r for r in results if not r.error and r.n_voxels > 0
    }

    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty manifest CSV: {manifest_path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    for col in ("n_points", "n_voxels"):
        if col not in fieldnames:
            fieldnames.append(col)

    n_updated = 0
    for row in rows:
        split = (row.get("split") or "").strip()
        patch_id = (row.get("patch_id") or "").strip()
        key = (split, patch_id)
        if key in ok_by_key:
            result = ok_by_key[key]
            row["n_points"] = str(result.n_points)
            row["n_voxels"] = str(result.n_voxels)
            n_updated += 1
        else:
            row.setdefault("n_points", row.get("n_points") or "")
            row.setdefault("n_voxels", row.get("n_voxels") or "")

    manifest_dir = os.path.dirname(os.path.abspath(manifest_path)) or "."
    fd, tmp_path = tempfile.mkstemp(
        prefix=".scene_split_manifest_", suffix=".csv.tmp", dir=manifest_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp_path, manifest_path)
    except Exception:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise

    return n_updated, len(rows)


def _print_summary(summary: dict) -> None:
    print("\n=== Summary ===")
    print(f"ok={summary['n_scenes_ok']}  errors={summary['n_scenes_error']}")
    for key in ("n_points", "n_voxels"):
        stats = summary[key]
        print(
            f"{key}: min={stats['min']} median={stats['median']:.0f} "
            f"mean={stats['mean']:.0f} p90={stats['p90']:.0f} p95={stats['p95']:.0f} "
            f"p98={stats['p98']:.0f} p99={stats['p99']:.0f} max={stats['max']}"
        )
    print("\nTop outliers by n_voxels:")
    for row in summary["top_outliers_by_n_voxels"][:10]:
        print(
            f"  {row['patch_id']}: n_points={row['n_points']} "
            f"n_voxels={row['n_voxels']} ppv={row['points_per_voxel']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data/malibu3d_plus")
    parser.add_argument(
        "--csv_manifest",
        default="data/malibu3d_plus/raw/scene_split_manifest.csv",
    )
    parser.add_argument(
        "--splits",
        default="val,test",
        help="Comma-separated splits (default: val,test)",
    )
    parser.add_argument("--grid_size", type=float, default=0.1)
    parser.add_argument(
        "--missing_tiles_manifest",
        default="data/malibu3d_plus/missing_ply_preflight.txt",
    )
    parser.add_argument(
        "--too_small_tiles_manifest",
        default="data/malibu3d_plus/too_small_tiles.csv",
    )
    parser.add_argument(
        "--apply_dataset_exclusions",
        action="store_true",
        help=(
            "Exclude missing / too_small / hardcoded tiles like Malibu3DDataset. "
            "Default is off so all LIDARHD manifest rows can be enriched."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--max_scenes", type=int, default=0, help="Debug: limit scenes (0=all)")
    parser.add_argument(
        "--existing_only",
        action="store_true",
        help="Keep only scenes that already have coord.npy on disk",
    )
    parser.add_argument(
        "--write_manifest",
        default=None,
        help=(
            "Enrich this scene_split_manifest CSV with n_points/n_voxels "
            "(temp write then atomic replace). Typically the same path as --csv_manifest."
        ),
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--output_dir",
        default="stats/malibu3d/test_point_voxel_counts",
    )
    args = parser.parse_args()

    data_root = resolve_repo_path(args.data_root)
    csv_manifest = resolve_repo_path(args.csv_manifest)
    target_splits = parse_splits(args.splits)
    output_dir = resolve_repo_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    excluded: set[tuple[str, str]] = set()
    if args.apply_dataset_exclusions:
        excluded |= load_hardcoded_excluded_tiles()
        excluded |= load_missing_tiles_manifest(
            resolve_repo_path(args.missing_tiles_manifest)
        )
        excluded |= load_too_small_tiles_manifest(
            resolve_repo_path(args.too_small_tiles_manifest)
        )

    if not os.path.isfile(csv_manifest):
        raise FileNotFoundError(f"CSV manifest not found: {csv_manifest}")

    scenes = load_scene_records(data_root, csv_manifest, target_splits, excluded)
    if args.existing_only:
        before = len(scenes)
        scenes = [
            s
            for s in scenes
            if os.path.isfile(os.path.join(s.scene_path, "coord.npy"))
        ]
        print(f"existing_only: kept {len(scenes)}/{before} scenes with coord.npy")
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]

    tasks = [(scene, float(args.grid_size)) for scene in scenes]

    print(f"data_root={data_root}")
    print(f"csv_manifest={csv_manifest}")
    print(f"splits={sorted(target_splits)}")
    print(f"grid_size={args.grid_size}")
    print(f"apply_dataset_exclusions={args.apply_dataset_exclusions}")
    print(f"excluded tiles: {len(excluded)}")
    print(f"tiles to scan: {len(scenes)}")

    results: List[SceneCountResult] = []
    show_progress = not args.no_progress and len(tasks) > 0

    if args.num_workers <= 1:
        iterator = (_process_scene(task) for task in tasks)
        if show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc="Scenes", unit="scene")
        results = list(iterator)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            mapped = pool.map(_process_scene, tasks, chunksize=4)
            if show_progress:
                mapped = tqdm(mapped, total=len(tasks), desc="Scenes", unit="scene")
            results = list(mapped)

    split_tag = "_".join(sorted(target_splits))
    output_csv = os.path.join(output_dir, f"point_voxel_counts_{split_tag}.csv")
    summary_json = os.path.join(output_dir, f"point_voxel_summary_{split_tag}.json")

    _write_counts_csv(output_csv, results)
    summary = summarize_counts(results, top_k=args.top_k)
    summary["grid_size"] = float(args.grid_size)
    summary["splits"] = sorted(target_splits)
    summary["output_csv"] = output_csv

    if args.write_manifest:
        write_path = resolve_repo_path(args.write_manifest)
        n_updated, n_rows = write_counts_into_manifest(write_path, results)
        summary["write_manifest"] = write_path
        summary["manifest_rows_updated"] = n_updated
        summary["manifest_rows_total"] = n_rows
        print(f"\nEnriched manifest {write_path}: updated {n_updated}/{n_rows} rows")

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _print_summary(summary)
    print(f"\nWrote {output_csv} ({len(results)} rows)")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
