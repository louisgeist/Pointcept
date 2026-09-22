#!/usr/bin/env python3
"""Extract frozen-backbone multiscale features on DALES test tiles for UMAP viz.

Reuses one of the existing DALES GridProbe configs verbatim (backbone type,
enc/dec multiscale mode, feat_scales, CheckpointLoader rename/exclude rules)
so the extracted features are exactly what that config's linear probes were
trained on -- only the probe heads themselves are unused here:

  configs/dales/sonata-v1m2-dales-lin-grid.py        Sonata PT-v3m2 encoder MS (1232ch)
  configs/dales/litept-b-v1m0-dales-lin-grid.py       LitePT-B decoder hypercolumn MS (1404ch)
  configs/dales/litept-b-v1m0-dales-lin-grid-enc.py   LitePT-B encoder MS (1386ch)

Runs the *val* transform pipeline (deterministic, no augmentation -- same one
GridProbeEvaluator uses) on the DALES `test` split, which is already mirrored
locally (data/dales/test/, 11 tiles). Each tile is forwarded whole (matching
production val behaviour) under fp16 autocast -- the 1404ch hypercolumn on an
11M-point tile is ~31GB in fp32, autocast is not optional here (see OOM note
in litept-b-v1m0-dales-lin-grid.py's docstring for the same tensor under a
harder, multi-probe-backward workload).

No flash-attn install / spconv fixes needed here -- see
scripts/_grid_probe_extract_common.py (shared with
extract_flair3d_grid_probe_features.py) for the shim/workaround rationale.

Point selection: per-class reservoir over shuffled test tiles, capped at
--points-per-class per tile-contribution and again at the end -- keeps a
handful of very common classes (Ground/Vegetation) from drowning out rare
ones (Trucks/Power lines/Poles) in the eventual UMAP plot, and avoids ever
materializing more than a few tiles' full feature tensor at once. Tiles are
visited in a seeded-shuffled order and extraction stops as soon as every
class has reached its cap.

Usage::

    python scripts/extract_dales_grid_probe_features.py \\
      --config configs/dales/sonata-v1m2-dales-lin-grid.py \\
      --weight ckpt/862680/epoch_120.pth \\
      --output stats/umap/dales/data/sonata_enc.npz

    # From-scratch baseline (no checkpoint, fresh random init):
    python scripts/extract_dales_grid_probe_features.py \\
      --config configs/dales/litept-b-v1m0-dales-lin-grid-enc.py \\
      --rand-init \\
      --output stats/umap/dales/data/litept_enc_randinit.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _grid_probe_extract_common import (  # noqa: E402
    build_model_for_extraction,
    run_extraction,
    save_features_npz,
)


def build_extraction_transforms(cfg, point_max):
    """(pre, post) transform pair: `pre` voxelizes a whole raw tile once
    (cached per tile); `post` draws one SphereCrop from it -- cheap enough to
    call many times per tile for spatial diversity.

    Whole DALES test tiles are enormous post-GridSample (~11M points) and a
    single-shot backbone forward over all of them reliably segfaults spconv's
    SubMConv3d (illegal memory access, reproduced locally at grid_size=0.1 on
    the raw litept-b-v1m0-dales-lin-grid-enc.py backbone) -- capping at
    `point_max` sidesteps this by staying at the exact point budget every
    training/GridProbe config in this repo already forwards without issue
    (SphereCrop(point_max=102400) in the *train* pipeline of these same
    configs). We don't need whole-tile fidelity anyway, only a stratified
    per-class sample.

    grid_size/hash_type/feat_keys/feat_scales are read off cfg.data.val's own
    GridSample/Collect entries so this always matches whichever of the 3
    configs is passed, rather than hardcoding per-config assumptions.
    """
    from pointcept.datasets.transform import Compose

    val_list = cfg.data.val.transform
    grid_sample_cfg = next(t for t in val_list if t["type"] == "GridSample")
    collect_cfg = next(t for t in val_list if t["type"] == "Collect")

    pre = Compose([
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=grid_sample_cfg["grid_size"],
            hash_type=grid_sample_cfg.get("hash_type", "fnv"),
            mode="train",
            return_grid_coord=True,
        ),
    ])
    post = Compose([
        dict(type="SphereCrop", point_max=point_max, mode="random"),
        dict(type="CenterShift", apply_z=False),
        dict(type="FillMissingFeat", feat_key="color", feat_dim=3),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment"),
            feat_keys=collect_cfg["feat_keys"],
            feat_scales=collect_cfg.get("feat_scales", {}),
        ),
    ])
    return pre, post


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="One of the configs/dales/*-lin-grid*.py configs.")
    parser.add_argument("--weight", default=None, help="Local path to the checkpoint (download from JZ first -- cfg.weight is a lustre path). Required unless --rand-init.")
    parser.add_argument("--rand-init", action="store_true", help="Skip loading a checkpoint -- extract features from the backbone's fresh random init instead (from-scratch baseline, e.g. for a 'litept_enc_randinit' UMAP panel). --weight is ignored.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--points-per-class", type=int, default=3000)
    parser.add_argument("--point-max", type=int, default=102400, help="SphereCrop cap per draw -- keep at the codebase's proven-safe point_max (see build_extraction_transforms docstring).")
    parser.add_argument("--max-draws", type=int, default=60, help="Safety cap on (tile, random-crop) draws before giving up on under-filled classes.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.rand_init and args.weight is None:
        parser.error("--weight is required unless --rand-init is set.")

    np.random.seed(args.seed)  # SphereCrop(mode="random") draws its center from the global numpy RNG

    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset

    cfg = Config.fromfile(args.config)
    num_classes = cfg.model["num_classes"]
    class_names = cfg.names[:num_classes]

    model = build_model_for_extraction(cfg, args.weight, args.device, rand_init=args.rand_init)
    pre_transform, post_transform = build_extraction_transforms(cfg, args.point_max)
    dataset = build_dataset(cfg.data.val)

    collector, tile_names, used_tiles = run_extraction(
        model=model,
        dataset=dataset,
        pre_transform=pre_transform,
        post_transform=post_transform,
        num_classes=num_classes,
        target_key="segment",
        points_per_class=args.points_per_class,
        max_draws=args.max_draws,
        seed=args.seed,
        device=args.device,
    )

    for cls, name in enumerate(class_names):
        n = collector.count(cls)
        flag = "" if n >= args.points_per_class else "  (below cap -- exhausted --max-draws)"
        print(f"[extract] class {cls:2d} {name:12s}: {n:6d} pts{flag}")

    save_features_npz(
        args.output, collector, class_names, tile_names, used_tiles,
        config_path=str(args.config), weight_path=str(args.weight) if args.weight else "",
        rand_init=args.rand_init, points_per_class=args.points_per_class, seed=args.seed,
    )


if __name__ == "__main__":
    main()
