#!/usr/bin/env python3
"""Rank D068/D075 ROIs by elevation MAE from already-dumped predictions.

Reads ``*_reg_elevation.npy`` under the job-873542 dump (already de-normalized
to metres) and the matching preprocessed ``elevation.npy`` GT. One-pass
accumulators, no concatenation of the full point cloud -- just MAE / RMSE /
R^2 / bias per ROI and per sub-tile, then rank hardest first.

Metrics use the finite mask ``isfinite(pred) & isfinite(gt)`` (same
convention as ``accumulate_regression_errors``). R^2 is
``1 - sum((pred-gt)^2) / sum((gt-mean(gt))^2)``.

Example::

    python scripts/rank_elevation_mae.py
    python scripts/rank_elevation_mae.py --top-subtiles 20
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

PRED_SUFFIX = "_reg_elevation.npy"
PRED_RE = re.compile(r"^(.*)_(\d+)-(\d+)_reg_elevation\.npy$")
SCENE_RE = re.compile(r"^(D\d{3}-\d{4})_(.+)$")

DEFAULT_PRED_ROOT = Path("/data/geist/superpixel_transformer_dev/local/temp/873542")
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "flair3d_plus"
DEFAULT_OUT_DIR = REPO_ROOT / "stats" / "flair3d" / "elevation_parity"
SPLITS = ("test", "val", "train")

ROI_FIELDS = [
    "rank", "scene", "split", "gt_roi", "n_subtiles",
    "n_total", "n_finite", "n_nonfinite",
    "mae", "rmse", "r2", "bias_pred_minus_gt",
]
SUBTILE_FIELDS = [
    "scene", "sub_tile", "n_total", "n_finite", "n_nonfinite",
    "mae", "rmse", "r2", "bias_pred_minus_gt",
]


def eprint(*args) -> None:
    print(*args, file=sys.stderr, flush=True)


def empty_acc() -> dict:
    return {
        "n_total": 0,
        "n_finite": 0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "sum_err": 0.0,
        "sum_gt": 0.0,
        "sum_gt2": 0.0,
    }


def add_file(acc: dict, pred: np.ndarray, gt: np.ndarray) -> None:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    acc["n_total"] += int(pred.size)
    m = np.isfinite(pred) & np.isfinite(gt)
    n = int(m.sum())
    if n == 0:
        return
    g, p = gt[m], pred[m]
    err = p - g
    acc["n_finite"] += n
    acc["sum_abs"] += float(np.abs(err).sum())
    acc["sum_sq"] += float(np.dot(err, err))
    acc["sum_err"] += float(err.sum())
    acc["sum_gt"] += float(g.sum())
    acc["sum_gt2"] += float(np.dot(g, g))


def merge_acc(dst: dict, src: dict) -> None:
    for k in dst:
        dst[k] += src[k]


def metrics_from_acc(acc: dict) -> dict:
    n = acc["n_finite"]
    n_total = acc["n_total"]
    nan = float("nan")
    out = {
        "n_total": n_total,
        "n_finite": n,
        "n_nonfinite": n_total - n,
        "mae": nan,
        "rmse": nan,
        "r2": nan,
        "bias_pred_minus_gt": nan,
    }
    if n == 0:
        return out
    out["mae"] = acc["sum_abs"] / n
    out["rmse"] = math.sqrt(acc["sum_sq"] / n)
    out["bias_pred_minus_gt"] = acc["sum_err"] / n
    mean_gt = acc["sum_gt"] / n
    ss_tot = acc["sum_gt2"] - n * mean_gt * mean_gt
    if ss_tot > 0.0:
        out["r2"] = 1.0 - acc["sum_sq"] / ss_tot
    return out


def parse_pred_stem(path: Path) -> tuple[str, str] | None:
    m = PRED_RE.match(path.name)
    if m is None:
        return None
    return m.group(1), path.name[: -len(PRED_SUFFIX)]


def parse_scene(scene: str) -> tuple[str, str]:
    m = SCENE_RE.match(scene)
    if m is None:
        raise ValueError(f"unexpected scene name (want Dxxx-yyyy_ROI): {scene}")
    return m.group(1), m.group(2)


def discover_preds(pred_root: Path, departments: list[str]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(pred_root.glob(f"**/*{PRED_SUFFIX}")):
        parsed = parse_pred_stem(path)
        if parsed is None:
            eprint(f"  [warn] skip unparseable pred file: {path.name}")
            continue
        scene, _ = parsed
        dept_year, _ = parse_scene(scene)
        dept = dept_year.split("-", 1)[0]
        if departments and dept not in departments:
            continue
        grouped[scene].append(path)
    return dict(grouped)


def resolve_gt_roi(
    data_root: Path, dept_year: str, roi_short: str, stem: str,
) -> tuple[Path, str]:
    lidar = f"{dept_year}_LIDARHD"
    for split in SPLITS:
        gt_path = data_root / split / lidar / roi_short / stem / "elevation.npy"
        if gt_path.is_file():
            return gt_path, split
    searched = ", ".join(str(data_root / s / lidar / roi_short / stem) for s in SPLITS)
    raise FileNotFoundError(
        f"missing GT elevation for {stem} (looked under {searched})"
    )


def fmt_m(x: float) -> str:
    return "   nan" if not math.isfinite(x) else f"{x:7.3f}"


def fmt_r2(x: float) -> str:
    return "    nan" if not math.isfinite(x) else f"{x:8.4f}"


def print_roi_table(rows: list[dict]) -> None:
    print()
    print(
        f"{'#':>4}  {'scene':<24}  {'split':<5}  {'n_sub':>5}  "
        f"{'n_finite':>12}  {'MAE m':>7}  {'RMSE m':>7}  {'R2':>8}  {'bias m':>8}"
    )
    print("-" * 108)
    for row in rows:
        print(
            f"{row['rank']:4d}  {row['scene']:<24}  {row['split']:<5}  "
            f"{row['n_subtiles']:5d}  {row['n_finite']:12,d}  "
            f"{fmt_m(row['mae'])}  {fmt_m(row['rmse'])}  "
            f"{fmt_r2(row['r2'])}  {row['bias_pred_minus_gt']:+8.3f}"
        )


def print_subtile_table(rows: list[dict], k: int) -> None:
    if k <= 0 or not rows:
        return
    top = rows[:k]
    print()
    print(f"Worst {len(top)} sub-tiles by MAE:")
    print(
        f"{'#':>4}  {'sub_tile':<32}  {'n_finite':>10}  "
        f"{'MAE m':>7}  {'RMSE m':>7}  {'R2':>8}  {'bias m':>8}"
    )
    print("-" * 92)
    for i, row in enumerate(top, start=1):
        print(
            f"{i:4d}  {row['sub_tile']:<32}  {row['n_finite']:10,d}  "
            f"{fmt_m(row['mae'])}  {fmt_m(row['rmse'])}  "
            f"{fmt_r2(row['r2'])}  {row['bias_pred_minus_gt']:+8.3f}"
        )


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    ap.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Flair3D+ preprocessed root (train/val/test/<dept>_LIDARHD/...)",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--departments", default="D068,D075",
        help="Comma-separated department prefixes to keep (empty = all).",
    )
    ap.add_argument("--top-subtiles", type=int, default=15)
    args = ap.parse_args()

    departments = [d.strip() for d in args.departments.split(",") if d.strip()]
    print(f"Predictions: {args.pred_root}")
    print(f"GT root:     {args.data_root}")
    print(f"Departments: {departments or 'all'}")

    grouped = discover_preds(args.pred_root, departments)
    if not grouped:
        raise FileNotFoundError(
            f"no {PRED_SUFFIX} files under {args.pred_root} "
            f"(departments={departments or 'all'})"
        )

    roi_rows: list[dict] = []
    subtile_rows: list[dict] = []

    for scene in sorted(grouped):
        pred_paths = grouped[scene]
        dept_year, roi_short = parse_scene(scene)
        print(f"[{scene}] {len(pred_paths)} sub-tiles ...", flush=True)
        roi_acc = empty_acc()
        split = None
        gt_roi = None
        for pp in pred_paths:
            parsed = parse_pred_stem(pp)
            assert parsed is not None
            _, stem = parsed
            gt_path, found_split = resolve_gt_roi(
                args.data_root, dept_year, roi_short, stem,
            )
            if split is None:
                split = found_split
                gt_roi = gt_path.parent.parent
            elif found_split != split:
                eprint(f"  [warn] {stem}: split {found_split} != {split}")
            pred = np.load(pp)
            gt = np.load(gt_path)
            if pred.shape != gt.shape:
                raise ValueError(
                    f"shape mismatch for {stem}: pred {pred.shape} vs gt {gt.shape}"
                )
            st_acc = empty_acc()
            add_file(st_acc, pred, gt)
            merge_acc(roi_acc, st_acc)
            subtile_rows.append({"scene": scene, "sub_tile": stem, **metrics_from_acc(st_acc)})
        metrics = metrics_from_acc(roi_acc)
        roi_rows.append({
            "scene": scene,
            "split": split or "",
            "gt_roi": str(gt_roi) if gt_roi is not None else "",
            "n_subtiles": len(pred_paths),
            **metrics,
        })
        print(
            f"  MAE={metrics['mae']:.4f} m  RMSE={metrics['rmse']:.4f} m  "
            f"R2={metrics['r2']:.4f}  n_finite={metrics['n_finite']:,}",
            flush=True,
        )

    roi_rows.sort(key=lambda r: (-(r["mae"] if math.isfinite(r["mae"]) else -1.0), r["scene"]))
    for i, row in enumerate(roi_rows, start=1):
        row["rank"] = i
    subtile_rows.sort(
        key=lambda r: (-(r["mae"] if math.isfinite(r["mae"]) else -1.0), r["sub_tile"])
    )

    print_roi_table(roi_rows)
    print_subtile_table(subtile_rows, int(args.top_subtiles))

    ranking_path = args.out_dir / "roi_mae_ranking.csv"
    subtile_path = args.out_dir / "per_subtile_all.csv"
    write_csv(ranking_path, ROI_FIELDS, roi_rows)
    write_csv(subtile_path, SUBTILE_FIELDS, subtile_rows)
    print(f"\nWrote {ranking_path}")
    print(f"Wrote {subtile_path}")


if __name__ == "__main__":
    main()
