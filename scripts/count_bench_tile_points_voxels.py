#!/usr/bin/env python3
"""Raw-point vs GridSample-voxel counts for the inference-speed bench tiles.

The bench reports pts/s on ``coord.shape[0]`` *after* GridSample (one occupied
voxel at 0.1 m). Multiply those figures by ``sum(n_points) / sum(n_voxels)`` to
get raw LiDAR points/s.

Tile list (same 200 / 10-warmup as the JZ run):
  - ``--bench-dir`` with ``summary.json`` (exact names from the run), or
  - ``--resample``: LIDARHD=True test rows in CSV order, shuffle seed 42.

Voxel count applies CenterShift(z) + Z_MinShift then FNV unique keys, matching
the test voxelize input. If ``per_tile.csv`` is in ``--bench-dir``, the bench's
own ``num_points`` is also reported (ground truth for the published pts/s).

Examples (cluster, national test tiles)::

  python scripts/count_bench_tile_points_voxels.py \\
    --bench-dir stats/malibu3d/inference_speed_bench/<jobid>

  python scripts/count_bench_tile_points_voxels.py --resample --num_workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_malibu3d_test_point_voxel_counts import (  # noqa: E402
    build_scene_path,
    count_voxels,
    parse_manifest_bool,
    resolve_repo_path,
)


def load_manifest_index(
    csv_manifest: str, split: str, data_root: str
) -> tuple[dict[str, str], list[str]]:
    """patch_id -> scene_path, LIDARHD=True rows of ``split`` in CSV order."""
    index: dict[str, str] = {}
    order: list[str] = []
    with open(csv_manifest, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("split", "")).strip() != split:
                continue
            if not parse_manifest_bool(row.get("LIDARHD")):
                continue
            patch_id = str(row["patch_id"]).strip()
            dept_year = (row.get("dept_year") or "").strip() or patch_id.split("_", 2)[0]
            roi = (row.get("roi") or "").strip() or patch_id.split("_", 2)[1]
            index[patch_id] = build_scene_path(data_root, split, patch_id, dept_year, roi)
            order.append(patch_id)
    if not index:
        raise RuntimeError(f"No LIDARHD=True rows for split={split!r} in {csv_manifest}")
    return index, order


def resample_tile_names(order: list[str], num_tiles: int, seed: int) -> list[str]:
    n = len(order)
    if n < num_tiles:
        raise ValueError(f"Only {n} tiles in split, need --num-tiles={num_tiles}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [order[int(i)] for i in perm[:num_tiles]]


def load_bench_tile_names(bench_dir: Path) -> tuple[list[str], dict]:
    summary_path = bench_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing {summary_path}")
    with open(summary_path) as f:
        payload = json.load(f)
    names = payload.get("tile_names")
    if not names:
        raise ValueError(f"{summary_path} has no tile_names")
    return list(names), payload


def load_bench_voxel_counts(bench_dir: Path) -> dict[str, int]:
    """patch_id -> num_points from the sequential non-warmup pass."""
    csv_path = bench_dir / "per_tile.csv"
    if not csv_path.is_file():
        return {}
    out: dict[str, int] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("mode") != "sequential":
                continue
            warmup = str(row.get("warmup", "")).strip().lower() in {"1", "true"}
            if warmup:
                continue
            if str(row.get("oom", "")).strip().lower() in {"1", "true"}:
                continue
            n = row.get("num_points")
            if not n:
                continue
            out[row["patch_id"]] = int(float(n))
    return out


def apply_test_shift(coord: np.ndarray) -> np.ndarray:
    """CenterShift(apply_z=True) then Z_MinShift, as in data.test before voxelize."""
    x_min, y_min, z_min = coord.min(axis=0)
    x_max, y_max, _ = coord.max(axis=0)
    shift = np.array(
        [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, z_min], dtype=np.float64
    )
    shifted = coord - shift
    shifted[:, 2] -= shifted[:, 2].min()
    return shifted


def count_one(args: tuple[str, str, float]) -> dict:
    patch_id, scene_path, grid_size = args
    coord_path = os.path.join(scene_path, "coord.npy")
    rec = dict(
        patch_id=patch_id,
        scene_path=scene_path,
        n_points=0,
        n_voxels=0,
        points_per_voxel=0.0,
        error="",
    )
    if not os.path.isfile(coord_path):
        rec["error"] = "missing_coord"
        return rec
    coord = np.asarray(np.load(coord_path, mmap_mode="r")[:, :3], dtype=np.float64)
    if coord.ndim != 2 or coord.shape[1] < 3 or coord.shape[0] == 0:
        rec["error"] = f"bad_coord_shape:{tuple(coord.shape)}"
        return rec
    n_points = int(coord.shape[0])
    n_voxels = count_voxels(apply_test_shift(coord), grid_size)
    rec.update(
        n_points=n_points,
        n_voxels=n_voxels,
        points_per_voxel=float(n_points) / float(n_voxels) if n_voxels else 0.0,
    )
    return rec


def _fmt(n: float) -> str:
    return f"{n:,.0f}"


def convert_summaries(payload: dict, ratio: float) -> dict:
    converted = {}
    for name, modes in (payload.get("summaries") or {}).items():
        converted[name] = {}
        for mode, stats in modes.items():
            if not isinstance(stats, dict):
                continue
            row = {}
            for key, val in stats.items():
                if key.startswith("pts_per_sec_") and isinstance(val, (int, float)):
                    row[key] = val
                    row[f"{key}_raw_points"] = val * ratio
            converted[name][mode] = row
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data/malibu3d_plus")
    parser.add_argument(
        "--csv_manifest", default="data/malibu3d_plus/raw/scene_split_manifest.csv"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--grid_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tiles", type=int, default=200)
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument(
        "--bench-dir",
        default=None,
        help="Directory with summary.json (and optional per_tile.csv).",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Ignore --bench-dir tile list; shuffle CSV order with --seed.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--output_dir",
        default="stats/malibu3d/inference_speed_bench/point_voxel_ratio",
    )
    args = parser.parse_args()

    data_root = resolve_repo_path(args.data_root)
    csv_manifest = resolve_repo_path(args.csv_manifest)
    output_dir = Path(resolve_repo_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    index, order = load_manifest_index(csv_manifest, args.split, data_root)
    bench_payload = None
    bench_voxels: dict[str, int] = {}

    if args.bench_dir and not args.resample:
        bench_dir = Path(resolve_repo_path(args.bench_dir))
        all_names, bench_payload = load_bench_tile_names(bench_dir)
        bench_voxels = load_bench_voxel_counts(bench_dir)
        source = f"bench-dir {bench_dir}"
    else:
        all_names = resample_tile_names(order, args.num_tiles, args.seed)
        source = f"resample seed={args.seed} csv_order n={len(order)}"

    measured = all_names[args.num_warmup :]
    missing = [n for n in measured if n not in index]
    if missing:
        raise RuntimeError(
            f"{len(missing)} measured tile(s) not in manifest "
            f"(first: {missing[0]!r})"
        )

    tasks = [(name, index[name], float(args.grid_size)) for name in measured]
    print(
        f"[count] {len(measured)} measured tiles "
        f"(warmup skipped: {args.num_warmup}/{len(all_names)}) from {source}"
    )
    print(f"[count] grid_size={args.grid_size} workers={args.num_workers}")

    if args.num_workers <= 1:
        results = [count_one(t) for t in tqdm(tasks, desc="tiles")]
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            results = list(tqdm(pool.map(count_one, tasks), total=len(tasks), desc="tiles"))

    ok = [r for r in results if not r["error"]]
    errors = [r for r in results if r["error"]]
    if errors:
        print(f"[count] {len(errors)} tile(s) failed (first: {errors[0]})")
    if not ok:
        raise SystemExit("no successful tiles")

    sum_pts = int(sum(r["n_points"] for r in ok))
    sum_vox = int(sum(r["n_voxels"] for r in ok))
    ratio = float(sum_pts) / float(sum_vox)

    if bench_voxels:
        matched = [n for n in measured if n in bench_voxels]
        sum_vox_bench = int(sum(bench_voxels[n] for n in matched))
        ratio_bench = (
            float(sum(r["n_points"] for r in ok if r["patch_id"] in bench_voxels))
            / float(sum_vox_bench)
            if sum_vox_bench
            else None
        )
    else:
        matched, sum_vox_bench, ratio_bench = [], None, None

    csv_path = output_dir / "point_voxel_counts.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patch_id",
                "n_points",
                "n_voxels",
                "n_voxels_bench",
                "points_per_voxel",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                dict(
                    patch_id=r["patch_id"],
                    n_points=r["n_points"],
                    n_voxels=r["n_voxels"],
                    n_voxels_bench=bench_voxels.get(r["patch_id"], ""),
                    points_per_voxel=f"{r['points_per_voxel']:.6f}",
                    error=r["error"],
                )
            )

    converted = convert_summaries(bench_payload, ratio) if bench_payload else {}
    summary = dict(
        n_measured=len(ok),
        n_failed=len(errors),
        n_sampled=len(all_names),
        num_warmup=args.num_warmup,
        grid_size=args.grid_size,
        source=source,
        sum_n_points=sum_pts,
        sum_n_voxels=sum_vox,
        points_per_voxel_total=ratio,
        mean_points=float(np.mean([r["n_points"] for r in ok])),
        mean_voxels=float(np.mean([r["n_voxels"] for r in ok])),
        n_voxels_bench_matched=len(matched),
        sum_n_voxels_bench=sum_vox_bench,
        points_per_voxel_vs_bench=ratio_bench,
        converted_pts_per_sec=converted,
        note=(
            "Bench pts/s is occupied voxels/s. "
            "raw_pts/s = reported_pts/s * points_per_voxel_total."
        ),
    )
    json_path = output_dir / "point_voxel_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"tiles ok          {len(ok)}")
    print(f"sum n_points      {_fmt(sum_pts)}")
    print(f"sum n_voxels      {_fmt(sum_vox)}")
    print(f"points / voxel    {ratio:.4f}")
    if ratio_bench is not None:
        print(
            f"vs bench voxels   {ratio_bench:.4f} "
            f"(n={len(matched)}, sum_vox_bench={_fmt(sum_vox_bench)})"
        )
    print()
    print("Convert bench pts/s (voxels/s) -> raw LiDAR pts/s:")
    print(f"  multiply by {ratio:.4f}")
    if converted:
        print()
        print(f"{'backbone':<12} {'pipeline voxels/s':>18} {'pipeline pts/s':>16}")
        for name, modes in converted.items():
            pipe = modes.get("pipeline") or {}
            vox = pipe.get("pts_per_sec_pipeline")
            raw = pipe.get("pts_per_sec_pipeline_raw_points")
            if vox is None:
                continue
            print(f"{name:<12} {vox:18,.0f} {raw:16,.0f}")
    print(f"\n[count] wrote {csv_path}")
    print(f"[count] wrote {json_path}")


if __name__ == "__main__":
    main()
