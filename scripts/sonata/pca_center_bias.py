#!/usr/bin/env python3
"""PCA visualisation of frozen Sonata features on Flair3D+ test tiles + a
quantitative "center bias" check (features depending on distance to the tile
centre rather than on content).

Features = the same 1232ch multiscale encoder hypercolumn used by the Sonata
GridProbe configs / UMAP scripts (configs/flair3d_default/probe/
sonata-v1m2-flair3d-lin-grid.py, enc_mode=True), computed on the *whole*
voxelised tile (no SphereCrop), so "centre" = centre of the tile bbox.

Per tile, figure with 4 panels (top-down): input RGB, GT segment (v20), PCA
(3 comps -> RGB, robust-percentile scaled, fitted on that tile), and the
normalised radial distance r in [0, 1] (Chebyshev, 0 = centre, 1 = border).

Center-bias metrics (aggregated over --n-stat-tiles tiles):
  * mean feature L2 norm vs r
  * mean cosine similarity to the point's own class prototype (mean feature of
    the same class in the same tile) vs r -- content-controlled: a drop towards
    the border / the centre means same-class points look different depending on
    position
  * |corr(PC1 / PC2 / PC3, r)| per tile
  * ridge R^2 predicting r from PCA-64 features, trained on some tiles and
    evaluated on held-out tiles (chance = 0)

Usage::

    python scripts/sonata/pca_center_bias.py \\
      --weight ckpt/862680/epoch_120.pth \\
      --tiles D068-2021_AF-S1-17_1-1 --output-dir stats/sonata_pca
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _grid_probe_extract_common import build_model_for_extraction  # noqa: E402
from extract_flair3d_grid_probe_features import (  # noqa: E402
    _SEGMENT_V20_NAMES,
    build_extraction_transforms,
)

CONFIG = "configs/flair3d_default/probe/sonata-v1m2-flair3d-lin-grid.py"
N_RBINS = 10


def segment_palette(n=15):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    pal = np.array([cmap(i % 20)[:3] for i in range(n)])
    return np.vstack([pal, [[0.1, 0.1, 0.1]]])  # last = void


def pca_rgb(feat, n_fit=200_000, seed=0):
    """Top-3 PCA -> RGB, each comp clipped to its [2, 98] percentile."""
    feat = feat - feat.mean(0, keepdims=True)
    rng = np.random.default_rng(seed)
    sub = feat[rng.choice(len(feat), min(n_fit, len(feat)), replace=False)]
    _, _, vt = np.linalg.svd(sub, full_matrices=False)
    proj = feat @ vt[:3].T
    lo, hi = np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0)
    rgb = np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb, proj, vt


def radial_r(coord):
    xy = coord[:, :2]
    c = (xy.min(0) + xy.max(0)) / 2
    half = (xy.max(0) - xy.min(0)).max() / 2
    return np.abs(xy - c).max(1) / half


@__import__("torch").no_grad()
def forward_tile(model, dataset, idx, pre, post, device):
    import torch
    from pointcept.datasets import collate_fn

    vox = pre(dataset.get_data(idx))
    out = post(copy.deepcopy(vox))
    inp = collate_fn([out])
    for k in inp:
        if isinstance(inp[k], torch.Tensor):
            inp[k] = inp[k].to(device)
    # fp32 on purpose: under fp16 autocast ~4% of the 1232ch feature values come out
    # non-finite on a full ~250k-voxel tile (see PR notes / README_umap_geist.md).
    feat_by_norm, point, _ = model.prepare_batch(inp)
    feat = next(iter(feat_by_norm.values())).float().cpu().numpy()
    assert np.isfinite(feat).all(), "non-finite backbone features"
    return dict(
        coord=inp["coord"].float().cpu().numpy(),
        feat=feat,
        segment=inp["segment"].cpu().numpy(),
        color=out["feat"][:, 3:6].numpy() if out["feat"].shape[1] >= 6 else None,
    )


def plot_tile(name, d, rgb, r, path):
    import matplotlib.pyplot as plt

    xy = d["coord"][:, :2]
    seg = d["segment"]
    pal = segment_palette()
    seg_rgb = pal[np.where((seg >= 0) & (seg < 15), seg, 15)]
    panels = [("Input RGB", np.clip(d["color"], 0, 1) if d["color"] is not None else seg_rgb),
              ("GT segment v20", seg_rgb),
              ("Sonata PCA (3 comps)", rgb),
              ("radial distance r (0=centre, 1=border)", r)]
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.6))
    for ax, (t, c) in zip(axs, panels):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, s=0.25, rasterized=True,
                        **({"cmap": "viridis"} if c.ndim == 1 else {}))
        ax.set_aspect("equal"); ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
        if c.ndim == 1:
            fig.colorbar(sc, ax=ax, fraction=0.046)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weight", default="ckpt/862680/epoch_120.pth")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--rand-init", action="store_true", help="Random-init backbone control.")
    ap.add_argument("--csv-manifest", default="data/flair3d_plus/raw/scene_split_manifest_D068_D075.csv")
    ap.add_argument("--tiles", nargs="*", default=None, help="Tile names to plot (default: first of the stat tiles).")
    ap.add_argument("--n-stat-tiles", type=int, default=16)
    ap.add_argument("--n-plot-tiles", type=int, default=3)
    ap.add_argument("--output-dir", default="stats/sonata_pca")
    ap.add_argument("--tag", default="sonata_862680")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset

    np.random.seed(args.seed)
    cfg = Config.fromfile(args.config)
    val_cfg = copy.deepcopy(cfg.data.val)
    val_cfg.update(csv_manifest=args.csv_manifest, split="test", target_keys=["segment"],
                   primary_target_key="segment")
    val_cfg.pop("stratified_subset_manifest", None)
    val_cfg.pop("max_sample", None)
    dataset = build_dataset(val_cfg)
    names = [dataset.get_data_name(i) for i in range(len(dataset.data_list))]

    model = build_model_for_extraction(cfg, args.weight, args.device, rand_init=args.rand_init)
    pre, post = build_extraction_transforms(cfg, point_max=10**8)  # no crop: whole tile

    rng = np.random.default_rng(args.seed)
    if args.tiles:
        plot_idx = [names.index(t) for t in args.tiles]
    else:
        plot_idx = list(rng.choice(len(names), args.n_plot_tiles, replace=False))
    stat_idx = list(dict.fromkeys(plot_idx + list(rng.permutation(len(names)))))[: args.n_stat_tiles]

    out = Path(args.output_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    bins = np.linspace(0, 1, N_RBINS + 1)
    norm_bins, cos_bins, cnt_bins = [], [], []
    corr_rows, tile_pcs = [], []
    for idx in stat_idx:
        d = forward_tile(model, dataset, idx, pre, post, args.device)
        feat, seg = d["feat"], d["segment"]
        r = radial_r(d["coord"])
        rgb, proj, vt = pca_rgb(feat, seed=args.seed)
        print(f"[{names[idx]}] {len(feat):,} voxels, {feat.shape[1]}ch")
        if idx in plot_idx:
            plot_tile(names[idx], d, rgb, r, out / f"{names[idx]}.png")

        # norm vs r
        b = np.clip(np.digitize(r, bins) - 1, 0, N_RBINS - 1)
        nrm = np.linalg.norm(feat, axis=1)
        # cosine to own-class prototype (classes with >= 500 pts, void excluded)
        fn = feat / (nrm[:, None] + 1e-8)
        cos = np.full(len(feat), np.nan)
        for c in range(15):
            m = seg == c
            if m.sum() >= 500:
                proto = fn[m].mean(0); proto /= np.linalg.norm(proto) + 1e-8
                cos[m] = fn[m] @ proto
        nb, cb, kb = np.zeros(N_RBINS), np.zeros(N_RBINS), np.zeros(N_RBINS)
        for k in range(N_RBINS):
            m = b == k
            kb[k] = m.sum()
            nb[k] = nrm[m].mean() if m.any() else np.nan
            mc = m & ~np.isnan(cos)
            cb[k] = cos[mc].mean() if mc.any() else np.nan
        norm_bins.append(nb); cos_bins.append(cb); cnt_bins.append(kb)
        corr_rows.append([np.corrcoef(proj[:, j], r)[0, 1] for j in range(3)])
        sub = np.random.default_rng(idx).choice(len(feat), min(20000, len(feat)), replace=False)
        tile_pcs.append((fn[sub], r[sub]))

    # Ridge R^2 for predicting r from L2-normalised features, held-out tiles.
    n = len(tile_pcs); split = n // 2
    Xtr = np.vstack([t[0] for t in tile_pcs[:split]]); ytr = np.concatenate([t[1] for t in tile_pcs[:split]])
    Xte = np.vstack([t[0] for t in tile_pcs[split:]]); yte = np.concatenate([t[1] for t in tile_pcs[split:]])
    mu = Xtr.mean(0); Xtr -= mu; Xte -= mu
    A = Xtr.T @ Xtr + 1.0 * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()))
    pred = Xte @ w + ytr.mean()
    r2 = 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()

    corr = np.abs(np.array(corr_rows))
    norm_m, cos_m = np.nanmean(norm_bins, 0), np.nanmean(cos_bins, 0)
    print("\n=== center-bias summary ===")
    print("r-bin centre :", " ".join(f"{x:6.2f}" for x in (bins[:-1] + bins[1:]) / 2))
    print("mean ||f||   :", " ".join(f"{x:6.2f}" for x in norm_m))
    print("cos to proto :", " ".join(f"{x:6.3f}" for x in cos_m))
    print(f"|corr(PCk, r)| mean over {n} tiles: PC1={corr[:,0].mean():.3f} PC2={corr[:,1].mean():.3f} PC3={corr[:,2].mean():.3f}")
    print(f"ridge R^2 (r from features, held-out tiles) = {r2:.3f}")

    np.savez(out / "center_bias_stats.npz", bins=bins, norm=np.array(norm_bins), cos=np.array(cos_bins),
             counts=np.array(cnt_bins), pc_corr=np.array(corr_rows), ridge_r2=r2, tiles=np.array([names[i] for i in stat_idx]))

    import matplotlib.pyplot as plt
    ctr = (bins[:-1] + bins[1:]) / 2
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, arr, t in zip(axs, (norm_bins, cos_bins), ("feature L2 norm", "cosine to own-class prototype")):
        arr = np.array(arr)
        for row in arr:
            ax.plot(ctr, row, color="gray", alpha=0.3, lw=0.8)
        ax.plot(ctr, np.nanmean(arr, 0), "r-o", lw=2)
        ax.set_xlabel("r (0 = tile centre, 1 = border)"); ax.set_title(t)
    fig.suptitle(f"{args.tag} -- R^2(r | feat) = {r2:.3f}")
    fig.tight_layout(); fig.savefig(out / "center_bias_radial.png", dpi=130)
    print(f"\nwrote figures to {out}/")


if __name__ == "__main__":
    main()
