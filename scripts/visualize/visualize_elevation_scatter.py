#!/usr/bin/env python3
"""Hexbin scatter: predicted vs ground-truth elevation for one or more ROIs.

Port of the Hydra `plot_elevation` script from the sibling repo. No Hydra, no
PLY: Pointcept already stores GT as `elevation.npy` (height above DTM, metres)
and the tester dumps predictions as `{tile}_reg_elevation.npy` (already
de-normalized to metres, row-aligned with GT).

Three input modes (first match wins):

1. ``--pairs path.npz`` -- cached finite (gt, pred) from
   ``scripts/export_elevation_parity.py`` (fastest).
2. ``--roi`` + ``--result-dir`` -- load every sub-tile under the ROI, matching
   each ``*_reg_elevation.npy`` to ``<roi>/<stem>/elevation.npy``.
3. no args -- both D075 test zones used in README_MALIBU3D.md,
   from the cached ``pairs.npz`` if present, else from the litept_b_multitask dumps.

Metrics (MAE / RMSE / R² / bias) are always computed on **all** finite pairs.
``--max-points`` only subsamples the hexbin rendering.

Example::

    python scripts/visualize/visualize_elevation_scatter.py

    python scripts/visualize/visualize_elevation_scatter.py \\
      --pairs stats/malibu3d/elevation_parity/D075-2021_AA-S2-2/pairs.npz

    python scripts/visualize/visualize_elevation_scatter.py \\
      --roi data/malibu3d_plus/test/D075-2021_LIDARHD/AA-S2-2 \\
      --result-dir exp/malibu3d/litept_b_multitask/result/D075_AA-S2-2 \\
      --output /tmp/AA-S2-2_elevation_hexbin.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

PRED_SUFFIX = "_reg_elevation.npy"

DEFAULT_PRED_ROOT = Path("exp/malibu3d/litept_b_multitask/result")
DEFAULT_GT_ROOT = REPO_ROOT / "data" / "malibu3d_plus" / "test" / "D075-2021_LIDARHD"
DEFAULT_PARITY_DIR = REPO_ROOT / "stats" / "malibu3d" / "elevation_parity"

# Same two D075 test zones as scripts/export_elevation_parity.py.
ZONES = [
    dict(scene="D075-2021_AA-S2-2", pred_subdir="D075_AA-S2-2", gt_subdir="AA-S2-2"),
    dict(scene="D075-2021_UU-S1-4", pred_subdir="D075_UU-S1-4", gt_subdir="UU-S1-4"),
]


def eprint(*args) -> None:
    print(*args, file=sys.stderr, flush=True)


def compute_regression_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Finite-masked MAE / RMSE / R² / bias, plus an OLS fit pred ~= a*gt + b."""
    gt_f = np.asarray(gt, dtype=np.float64).reshape(-1)
    pred_f = np.asarray(pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(gt_f) & np.isfinite(pred_f)
    n_valid = int(valid.sum())
    n_ignored = int((~valid).sum())
    nan = float("nan")
    if n_valid == 0:
        return {
            "mae": nan, "rmse": nan, "r2": nan, "bias_pred_minus_gt": nan,
            "ols_slope": nan, "ols_intercept": nan,
            "n_valid": 0, "n_ignored": n_ignored,
        }
    g, p = gt_f[valid], pred_f[valid]
    err = p - g
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((g - g.mean()) ** 2))
    a, b = np.polyfit(g, p, 1)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err * err))),
        "r2": nan if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot),
        "bias_pred_minus_gt": float(np.mean(err)),
        "ols_slope": float(a),
        "ols_intercept": float(b),
        "n_valid": n_valid,
        "n_ignored": n_ignored,
    }


def subsample_for_plot(
    gt: np.ndarray, pred: np.ndarray, max_points: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = gt.shape[0]
    if max_points <= 0 or n <= max_points:
        return gt, pred
    idx = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
    return gt[idx], pred[idx]


def load_pairs_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "gt" not in data or "pred" not in data:
        raise KeyError(f"{path} must contain arrays 'gt' and 'pred', got {list(data.keys())}")
    return np.asarray(data["gt"]).reshape(-1), np.asarray(data["pred"]).reshape(-1)


def load_from_result_dir(roi: Path, result_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate GT / pred elevation over every sub-tile dumped in result_dir."""
    pred_paths = sorted(result_dir.glob(f"*{PRED_SUFFIX}"))
    if not pred_paths:
        raise FileNotFoundError(f"no {PRED_SUFFIX} files under {result_dir}")

    gt_parts, pred_parts = [], []
    for pp in pred_paths:
        stem = pp.name[: -len(PRED_SUFFIX)]
        gt_path = roi / stem / "elevation.npy"
        if not gt_path.is_file():
            raise FileNotFoundError(f"missing GT elevation for {stem}: {gt_path}")
        pr = np.load(pp).astype(np.float64, copy=False).reshape(-1)
        gt = np.load(gt_path).astype(np.float64, copy=False).reshape(-1)
        if pr.shape != gt.shape:
            raise ValueError(f"shape mismatch for {stem}: pred {pr.shape} vs gt {gt.shape}")
        gt_parts.append(gt)
        pred_parts.append(pr)
        print(f"  {stem}: {gt.shape[0]:,} points")
    return np.concatenate(gt_parts), np.concatenate(pred_parts)


def load_zone_from_roots(zone: dict, pred_root: Path, gt_root: Path) -> tuple[np.ndarray, np.ndarray]:
    roi = gt_root / zone["gt_subdir"]
    result_dir = pred_root / zone["pred_subdir"]
    print(f"ROI: {roi}")
    print(f"Predictions: {result_dir}")
    return load_from_result_dir(roi, result_dir)


def plot_elevation_scatter(
    ax,
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    metrics: dict,
    n_plot: int,
    hexbin_gridsize: int,
    zone_name: str | None,
    cmap: str,
) -> object:
    hb = ax.hexbin(
        gt,
        pred,
        gridsize=int(hexbin_gridsize),
        bins="log",
        mincnt=1,
        cmap=cmap,
    )

    finite = np.isfinite(gt) & np.isfinite(pred)
    if finite.any():
        lo = float(min(gt[finite].min(), pred[finite].min()))
        hi = float(max(gt[finite].max(), pred[finite].max()))
        pad = 0.02 * (hi - lo) if hi > lo else 1.0
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, color="crimson", linewidth=1.5, label="y = x")
        a, b = metrics["ols_slope"], metrics["ols_intercept"]
        if math.isfinite(a) and math.isfinite(b):
            ax.plot(
                lims,
                [a * lims[0] + b, a * lims[1] + b],
                color="tab:blue",
                linewidth=1.3,
                linestyle="--",
                label=f"fit: {a:.3f}x {b:+.3f}",
            )
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Ground truth elevation (m)")
    ax.set_ylabel("Predicted elevation (m)")
    if zone_name:
        ax.set_title(f"Elevation regression: {zone_name}\nprediction vs ground truth")
    else:
        ax.set_title("Elevation regression: prediction vs ground truth")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    n_valid = metrics["n_valid"]
    stats_lines = [
        f"MAE  = {metrics['mae']:.3f} m",
        f"RMSE = {metrics['rmse']:.3f} m",
        f"R²   = {metrics['r2']:.4f}",
        f"bias = {metrics['bias_pred_minus_gt']:+.3f} m",
        f"N valid = {n_valid:,}",
    ]
    if n_plot != n_valid:
        stats_lines.append(f"N plotted = {n_plot:,}")
    ax.text(
        0.98,
        0.02,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    return hb


def save_figure(fig, output_path: Path, show: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path.resolve()}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    import matplotlib.pyplot as plt
    plt.close(fig)


def render_one(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    zone_name: str,
    output_path: Path,
    max_points: int,
    seed: int,
    hexbin_gridsize: int,
    cmap: str,
    show: bool,
) -> dict:
    metrics = compute_regression_metrics(gt, pred)
    print(
        f"Stats (all points): MAE={metrics['mae']:.4f} m  RMSE={metrics['rmse']:.4f} m  "
        f"R²={metrics['r2']:.4f}  bias={metrics['bias_pred_minus_gt']:+.4f} m  "
        f"N_valid={metrics['n_valid']:,}  N_ignored={metrics['n_ignored']:,}"
    )
    valid = np.isfinite(gt) & np.isfinite(pred)
    gt_plot, pred_plot = subsample_for_plot(gt[valid], pred[valid], max_points, seed)
    if gt_plot.shape[0] < int(valid.sum()):
        print(
            f"Plot subsample: {gt_plot.shape[0]:,} / {int(valid.sum()):,} "
            f"(max_points={max_points})"
        )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    hb = plot_elevation_scatter(
        ax, gt_plot, pred_plot,
        metrics=metrics, n_plot=int(gt_plot.shape[0]),
        hexbin_gridsize=hexbin_gridsize, zone_name=zone_name, cmap=cmap,
    )
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("log10(N)")
    fig.tight_layout()
    save_figure(fig, output_path, show)
    return metrics


def render_combined(
    panels: list[tuple[str, np.ndarray, np.ndarray, dict, int]],
    *,
    output_path: Path,
    hexbin_gridsize: int,
    cmap: str,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 7.0), squeeze=False)
    for ax, (zone_name, gt, pred, metrics, n_plot) in zip(axes[0], panels):
        hb = plot_elevation_scatter(
            ax, gt, pred,
            metrics=metrics, n_plot=n_plot,
            hexbin_gridsize=hexbin_gridsize, zone_name=zone_name, cmap=cmap,
        )
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("log10(N)")
    fig.tight_layout()
    save_figure(fig, output_path, show)


def default_jobs() -> list[dict]:
    jobs = []
    for zone in ZONES:
        pairs = DEFAULT_PARITY_DIR / zone["scene"] / "pairs.npz"
        if pairs.is_file():
            jobs.append(dict(kind="pairs", path=pairs, name=zone["scene"]))
        else:
            jobs.append(dict(kind="zone", zone=zone, name=zone["scene"]))
    return jobs


def resolve_output(args, name: str, n_jobs: int) -> Path:
    if args.output is None:
        return DEFAULT_PARITY_DIR / name / "hexbin.png"
    out = Path(args.output)
    if n_jobs > 1 or out.is_dir() or out.suffix == "":
        return out / f"{name}_hexbin.png"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--pairs", type=Path, nargs="*", default=None,
        help="Cached pairs.npz from export_elevation_parity.py (repeatable).",
    )
    ap.add_argument("--roi", type=Path, default=None, help="Preprocessed ROI directory (contains <scene>_<r>-<c>/elevation.npy).")
    ap.add_argument("--result-dir", type=Path, default=None, help="Tester dump directory holding *_reg_elevation.npy.")
    ap.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    ap.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    ap.add_argument(
        "--output", type=Path, default=None,
        help="PNG path (single plot) or directory (several). Default: stats/malibu3d/elevation_parity/<zone>/hexbin.png",
    )
    ap.add_argument("--combine", action="store_true", help="Also write a side-by-side figure when several zones are plotted.")
    ap.add_argument("--max-points", type=int, default=0, help="Random subsample for hexbin only (0 = all finite points).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hexbin-gridsize", type=int, default=80)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    jobs: list[dict] = []
    if args.pairs:
        for p in args.pairs:
            jobs.append(dict(kind="pairs", path=p, name=p.parent.name if p.name == "pairs.npz" else p.stem))
    elif args.roi is not None or args.result_dir is not None:
        if args.roi is None or args.result_dir is None:
            ap.error("--roi and --result-dir must be set together")
        name = f"{args.roi.parent.name}/{args.roi.name}"
        jobs.append(dict(kind="roi", roi=args.roi, result_dir=args.result_dir, name=name))
    else:
        jobs = default_jobs()

    combined_panels: list[tuple[str, np.ndarray, np.ndarray, dict, int]] = []
    for job in jobs:
        name = job["name"]
        print(f"[{name}] loading ...", flush=True)
        if job["kind"] == "pairs":
            gt, pred = load_pairs_npz(job["path"])
        elif job["kind"] == "roi":
            print(f"ROI: {job['roi']}")
            print(f"Predictions: {job['result_dir']}")
            gt, pred = load_from_result_dir(job["roi"], job["result_dir"])
        else:
            gt, pred = load_zone_from_roots(job["zone"], args.pred_root, args.gt_root)

        out = resolve_output(args, name.replace("/", "_"), len(jobs))
        metrics = render_one(
            gt, pred,
            zone_name=name, output_path=out,
            max_points=int(args.max_points), seed=int(args.seed),
            hexbin_gridsize=int(args.hexbin_gridsize), cmap=args.cmap,
            show=bool(args.show) and len(jobs) == 1,
        )
        if args.combine:
            valid = np.isfinite(gt) & np.isfinite(pred)
            gt_plot, pred_plot = subsample_for_plot(
                gt[valid], pred[valid], int(args.max_points), int(args.seed),
            )
            combined_panels.append((name, gt_plot, pred_plot, metrics, int(gt_plot.shape[0])))

    if args.combine and len(combined_panels) > 1:
        if args.output is not None and Path(args.output).suffix:
            combined_path = Path(args.output).with_name("hexbin_combined.png")
        else:
            dest = Path(args.output) if args.output is not None else DEFAULT_PARITY_DIR
            if dest.suffix:
                dest = dest.parent
            combined_path = dest / "hexbin_combined.png"
        render_combined(
            combined_panels,
            output_path=combined_path,
            hexbin_gridsize=int(args.hexbin_gridsize),
            cmap=args.cmap,
            show=bool(args.show),
        )


if __name__ == "__main__":
    main()
