#!/usr/bin/env python3
"""Extract frozen-backbone multiscale features on ECLAIR val tiles for UMAP viz.

Reuses the ECLAIR GridProbe configs verbatim (backbone type, enc/dec
multiscale mode, feat_scales, CheckpointLoader rename/exclude rules) so the
extracted features are exactly what those configs' linear probes are trained
on -- only the probe heads themselves are unused here:

  configs/eclair/sonata-v1m2-eclair-lin-grid.py       Sonata PT-v3m2 encoder MS (1232ch)
  configs/eclair/litept-b-v1m0-eclair-lin-grid.py     LitePT-B decoder hypercolumn MS (1404ch, or --enc-mode for 1386ch encoder MS)

Runs the *val* transform pipeline (deterministic voxelization, no
augmentation -- same one GridProbeEvaluator uses) on the ECLAIR `val` split,
already mirrored locally (data/eclair/val/, 62 tiles, GT-only). Each tile is
voxelized once then repeatedly SphereCrop-ped and forwarded through the
frozen backbone under fp16 autocast, mirroring
extract_dales_grid_probe_features.py's approach.

ECLAIR has real RGB and on-disk strength (unlike DALES's zero-filled color),
so the post-transform uses NormalizeColor with no FillMissingFeat, matching
each config's own val pipeline.

Class balance on the val split is extreme (Ground/Vegetation in the
millions, Vehicle/Noise/Poles in the hundreds to low thousands *total across
all 62 tiles*) -- several classes can never reach a 3000/class cap no matter
how many draws, so --max-draws defaults higher than DALES/H3D to give rare
classes several passes over the full tile set before giving up.

No flash-attn install / spconv fixes needed here -- see
scripts/_grid_probe_extract_common.py (shared with the DALES/Flair3D+/H3D
extraction scripts) for the shim/workaround rationale.

Usage::

    python scripts/extract_eclair_grid_probe_features.py \\
      --config configs/eclair/sonata-v1m2-eclair-lin-grid.py \\
      --weight ckpt/862680/epoch_120.pth \\
      --output stats/umap/eclair/data/sonata_enc.npz

    python scripts/extract_eclair_grid_probe_features.py \\
      --config configs/eclair/litept-b-v1m0-eclair-lin-grid.py \\
      --weight ckpt/873542/model_best.pth \\
      --output stats/umap/eclair/data/litept_dec.npz

    python scripts/extract_eclair_grid_probe_features.py \\
      --config configs/eclair/litept-b-v1m0-eclair-lin-grid.py \\
      --weight ckpt/873542/model_best.pth \\
      --output stats/umap/eclair/data/litept_enc.npz --enc-mode

    # From-scratch baseline (no checkpoint, fresh random init):
    python scripts/extract_eclair_grid_probe_features.py \\
      --config configs/eclair/litept-b-v1m0-eclair-lin-grid.py \\
      --rand-init --enc-mode \\
      --output stats/umap/eclair/data/litept_enc_randinit.npz
"""

from __future__ import annotations

import argparse
import copy
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

    grid_size/hash_type/feat_keys/feat_scales are read off cfg.data.val's own
    GridSample/Collect entries so this always matches whichever of the 2
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
        dict(type="NormalizeColor"),
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
    parser.add_argument("--config", required=True, help="One of the configs/eclair/*-lin-grid.py configs.")
    parser.add_argument("--weight", default=None, help="Local path to the checkpoint (ckpt/862680/epoch_120.pth or ckpt/873542/model_best.pth). Required unless --rand-init.")
    parser.add_argument("--rand-init", action="store_true", help="Skip loading a checkpoint -- extract features from the backbone's fresh random init instead (from-scratch baseline, e.g. for a 'litept_enc_randinit' UMAP panel). --weight is ignored.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--enc-mode", action="store_true", help="LitePT-B only: read the encoder-multiscale hypercolumn (1386ch, enc_mode=True) instead of the native decoder hypercolumn (1404ch). No-op / unsupported for Sonata (already enc_mode=True in its config).")
    parser.add_argument("--points-per-class", type=int, default=3000)
    parser.add_argument("--point-max", type=int, default=102400, help="SphereCrop cap per draw -- keep at the codebase's proven-safe point_max (see build_extraction_transforms docstring).")
    parser.add_argument("--max-draws", type=int, default=300, help="Safety cap on (tile, random-crop) draws before giving up on under-filled classes (higher than DALES/H3D -- ECLAIR's rarest classes total only a few hundred to low-thousand points across the whole 62-tile val split, so several passes over all tiles are needed).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.rand_init and args.weight is None:
        parser.error("--weight is required unless --rand-init is set.")

    np.random.seed(args.seed)  # SphereCrop(mode="random") draws its center from the global numpy RNG

    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset

    cfg = Config.fromfile(args.config)

    if args.enc_mode:
        backbone_type = cfg.model["backbone"]["type"]
        if backbone_type != "LitePT-v1":
            raise ValueError(f"--enc-mode only applies to the LitePT-B backbone, got {backbone_type}.")
        cfg.model["backbone"] = copy.deepcopy(cfg.model["backbone"])
        cfg.model["backbone"]["enc_mode"] = True
        cfg.model["backbone_out_channels"] = sum(cfg.model["backbone"]["enc_channels"])

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
        print(f"[extract] class {cls:2d} {name:20s}: {n:6d} pts{flag}")

    save_features_npz(
        args.output, collector, class_names, tile_names, used_tiles,
        config_path=str(args.config), weight_path=str(args.weight) if args.weight else "",
        rand_init=args.rand_init, points_per_class=args.points_per_class, seed=args.seed,
    )


if __name__ == "__main__":
    main()
