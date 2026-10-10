#!/usr/bin/env python3
"""Interactive 3D viser view of frozen Sonata features (PCA -> RGB) on one Flair3D+ tile.

Reuses the extraction path of ``pca_center_bias.py`` (same 1232ch multiscale encoder
hypercolumn as the Sonata GridProbe configs, whole voxelised tile, no SphereCrop, fp32).
The tile is shown as a point cloud in viser (works over SSH, forward the port) with a
dropdown to switch between colourings:

  * ``PCA 1-3`` / ``PCA 4-6``  -- top PCA components of the full 1232ch concat -> RGB (fitted on this tile, [2, 98] percentile scaling)
  * ``Stage k (...)``          -- PCA 1-3 fitted on the channels of encoder stage k only (voxel 0.1/0.3/0.9/2.7/8.1 m)
  * ``Input RGB``              -- the colours fed to the model
  * ``GT segment (v20)``       -- ground-truth classes
  * ``Elevation``              -- z, as a sanity reference

Features/PCA are cached in ``--output-dir`` (npz) so re-serving a tile is instant.

Usage::

    python scripts/sonata/visualize_pca_viser.py \\
      --weight ckpt/862680/epoch_120.pth --tile D075-2021_UU-S1-22_1-1

    # list available tiles of the manifest
    python scripts/sonata/visualize_pca_viser.py --list-tiles
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sonata"))

from pca_center_bias import (  # noqa: E402
    CONFIG,
    segment_palette,
)
from _grid_probe_extract_common import build_model_for_extraction  # noqa: E402
from extract_flair3d_grid_probe_features import (  # noqa: E402
    _SEGMENT_V20_NAMES,
    build_extraction_transforms,
)


def pca_components_rgb(feat, start, n_fit=200_000, seed=0):
    """PCA fitted once; return RGB for components [start, start+3) with [2, 98] clipping."""
    feat = feat - feat.mean(0, keepdims=True)
    rng = np.random.default_rng(seed)
    sub = feat[rng.choice(len(feat), min(n_fit, len(feat)), replace=False)]
    _, _, vt = np.linalg.svd(sub, full_matrices=False)
    proj = feat @ vt[: start + 3].T
    out = proj[:, start : start + 3]
    lo, hi = np.percentile(out, 2, axis=0), np.percentile(out, 98, axis=0)
    return np.clip((out - lo) / (hi - lo + 1e-8), 0, 1)


@__import__("torch").no_grad()
def forward_tile_raw(model, dataset, idx, pre, post, device):
    """Like pca_center_bias.forward_tile but returns the *raw* (un-normalised) 1232ch concat."""
    import torch
    from pointcept.datasets import collate_fn

    inp = collate_fn([post(copy.deepcopy(pre(dataset.get_data(idx))))])
    for k in inp:
        if isinstance(inp[k], torch.Tensor):
            inp[k] = inp[k].to(device)
    _, point, _ = model.prepare_batch(inp)  # fp32 on purpose, see pca_center_bias.forward_tile
    feat = point.feat.float().cpu().numpy()
    assert np.isfinite(feat).all(), "non-finite backbone features"
    colr = inp["feat"][:, 3:6].float().cpu().numpy() if inp["feat"].shape[1] >= 6 else None
    return dict(coord=inp["coord"].float().cpu().numpy(), feat=feat,
                segment=inp["segment"].cpu().numpy(), color=colr)


def stage_slices(cfg):
    """Channel slices per encoder stage; concat order is finest -> coarsest."""
    bb = cfg.model.backbone
    ch = list(bb["enc_channels"])
    size = cfg["grid_size"]
    out, a = [], 0
    for s, c in enumerate(ch):
        out.append((f"Stage {s + 1} ({size:g} m, {c}ch)", slice(a, a + c)))
        a += c
        if s < len(bb["stride"]):
            size *= bb["stride"][s]
    assert a == bb["enc_channels"][0] + sum(ch[1:]) == cfg.model.backbone_out_channels
    return out


def load_or_compute(args, names, dataset, idx):
    cache = Path(args.output_dir) / args.tag / f"{names[idx]}_pca.npz"
    if cache.exists() and not args.recompute:
        z = np.load(cache)
        if "pca_stage_0" in z.files:
            print(f"[cache] {cache}")
            return {k: z[k] for k in z.files}
        print("[cache] no per-stage PCA in cache -> recomputing")

    cfg = args.cfg
    model = build_model_for_extraction(cfg, args.weight, args.device, rand_init=args.rand_init)
    pre, post = build_extraction_transforms(cfg, point_max=10**8)  # whole tile
    d = forward_tile_raw(model, dataset, idx, pre, post, args.device)
    print(f"[{names[idx]}] {len(d['feat']):,} voxels, {d['feat'].shape[1]}ch")
    d["pca_1_3"] = pca_components_rgb(d["feat"], 0, seed=args.seed)
    d["pca_4_6"] = pca_components_rgb(d["feat"], 3, seed=args.seed)
    for k, (_, sl) in enumerate(stage_slices(cfg)):
        d[f"pca_stage_{k}"] = pca_components_rgb(d["feat"][:, sl], 0, seed=args.seed)
    del d["feat"]  # 1232ch x N is large; only keep the colourings
    if d["color"] is None:
        d["color"] = np.zeros((len(d["coord"]), 3), dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **d)
    print(f"[cache] wrote {cache}")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weight", default="ckpt/862680/epoch_120.pth")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--rand-init", action="store_true", help="Random-init backbone control.")
    ap.add_argument("--csv-manifest", default="data/flair3d_plus/raw/scene_split_manifest_D068_D075.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tile", default=None, help="Tile name (default: first of the split).")
    ap.add_argument("--list-tiles", action="store_true")
    ap.add_argument("--output-dir", default="stats/sonata_pca")
    ap.add_argument("--tag", default="sonata_862680")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-points", type=int, default=1_000_000)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--point-size", type=float, default=0.4)
    args = ap.parse_args()

    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset

    np.random.seed(args.seed)
    cfg = Config.fromfile(args.config)
    args.cfg = cfg
    val_cfg = copy.deepcopy(cfg.data.val)
    val_cfg.update(csv_manifest=args.csv_manifest, split=args.split, target_keys=["segment"],
                   primary_target_key="segment")
    val_cfg.pop("stratified_subset_manifest", None)
    val_cfg.pop("max_sample", None)
    dataset = build_dataset(val_cfg)
    names = [dataset.get_data_name(i) for i in range(len(dataset.data_list))]
    if args.list_tiles:
        print("\n".join(names))
        return
    idx = names.index(args.tile) if args.tile else 0

    d = load_or_compute(args, names, dataset, idx)
    coord = d["coord"].astype(np.float32)
    seg = d["segment"]
    pal = segment_palette()
    seg_rgb = pal[np.where((seg >= 0) & (seg < 15), seg, 15)]
    z = coord[:, 2]
    zn = np.clip((z - np.percentile(z, 1)) / (np.percentile(z, 99) - np.percentile(z, 1) + 1e-8), 0, 1)
    import matplotlib.pyplot as plt

    stage_cols = {name: d[f"pca_stage_{k}"] for k, (name, _) in enumerate(stage_slices(cfg))}
    colourings = {
        "PCA 1-3": d["pca_1_3"],
        **stage_cols,
        "PCA 4-6": d["pca_4_6"],
        "Input RGB": np.clip(d["color"], 0, 1),
        "GT segment (v20)": seg_rgb,
        "Elevation": plt.get_cmap("viridis")(zn)[:, :3],
    }
    colourings = {k: (v * 255).astype(np.uint8) for k, v in colourings.items()}

    n = len(coord)
    keep = np.arange(n)
    if n > args.max_points:
        keep = np.random.default_rng(args.seed).choice(n, args.max_points, replace=False)
    coord_c = coord - coord.mean(0)  # centred for viser's float32 camera

    import viser

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    present = np.unique(seg[(seg >= 0) & (seg < 15)])
    legend = "  ".join(
        f'<span style="color:rgb{tuple((pal[c] * 255).astype(int))}">■</span> {_SEGMENT_V20_NAMES[c]}'
        for c in present
    )
    server.gui.add_markdown(f"**Sonata features** — `{names[idx]}`\n\n`{args.weight}`\n\n"
                            f"voxels: `{n:,}` (shown `{len(keep):,}`)")
    mode = server.gui.add_dropdown("Colouring", tuple(colourings), initial_value="PCA 1-3")
    psize = server.gui.add_slider("point size", min=0.05, max=2.0, step=0.05, initial_value=args.point_size)
    server.gui.add_markdown(f"GT legend:\n\n{legend}")
    handle = server.scene.add_point_cloud("/tile", coord_c[keep], colourings[mode.value][keep],
                                          point_size=psize.value)

    @mode.on_update
    def _(_):
        handle.colors = colourings[mode.value][keep]

    @psize.on_update
    def _(_):
        handle.point_size = psize.value

    radius = max(float(np.linalg.norm(np.ptp(coord_c, axis=0))) / 2.0, 1.0)

    @server.on_client_connect
    def _(client):
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = (0.0, -radius * 1.3, radius * 1.1)
        client.camera.look_at = (0.0, 0.0, 0.0)

    print(f"[visualize_pca_viser] serving at http://localhost:{args.port}  (Ctrl-C to stop)")
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
