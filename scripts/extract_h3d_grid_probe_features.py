#!/usr/bin/env python3
"""Extract frozen-backbone multiscale features on H3D val tiles for UMAP viz.

Reuses one of the existing H3D GridProbe experiment configs verbatim (backbone
type, enc/dec multiscale mode, feat_scales, CheckpointLoader rename/exclude
rules) so the extracted features are exactly what that config's linear probes
were trained on -- only the probe heads themselves are unused here:

  configs/experiment/w109/5/13h_adamw_h3d/sonata-v1m2-h3d-lin-grid_6.py   Sonata PT-v3m2 encoder MS (1232ch)
  configs/experiment/w109/5/13h_adamw_h3d/litept-b-v1m0-h3d-lin_4.py     LitePT-B decoder hypercolumn MS (1404ch)
  configs/experiment/w109/5/13h_adamw_h3d/litept-b-v1m0-h3d-lin_5.py     LitePT-B encoder MS (1386ch)

Runs the *val* transform pipeline (deterministic voxelization, no
augmentation -- same one GridProbeEvaluator uses) on the H3D `val` split,
which is already mirrored locally (data/h3d/val/, 4 tiles). Each tile is
voxelized once then repeatedly SphereCrop-ped and forwarded through the
frozen backbone under fp16 autocast, mirroring
extract_dales_grid_probe_features.py's approach (H3D val tiles are ~2-5M
points post-GridSample -- same whole-tile-forward spconv crash risk as DALES,
see that script's docstring and the shared point_max rationale in
scripts/_grid_probe_extract_common.py).

H3D-specific transform quirks (baked into build_extraction_transforms below,
matching the *_h3d configs' own val pipeline order): `Z_MinShift` (redundant
with `CenterShift(apply_z=True)` here since that already zeroes z_min, but
included for fidelity with the source configs), `NormalizeColor` (H3D color
is raw 0-255, unlike DALES), and `FillMissingFeat(feat_key="strength", ...)`
(H3D has no on-disk strength/intensity, zero-filled -- same idea as DALES'
FillMissingFeat("color"), just a different missing channel).

No flash-attn install / spconv fixes needed here -- see
scripts/_grid_probe_extract_common.py (shared with the DALES/Flair3D+
extraction scripts) for the shim/workaround rationale.

Point selection: per-class reservoir over shuffled val tiles, capped at
--points-per-class per tile-contribution and again at the end -- keeps common
classes (Low Vegetation/Roof/...) from drowning out rare ones (Chimney/
Vehicle/...) in the eventual UMAP plot. Tiles are visited in a
seeded-shuffled order and extraction stops as soon as every class has
reached its cap.

Usage::

    python scripts/extract_h3d_grid_probe_features.py \\
      --config configs/experiment/w109/5/13h_adamw_h3d/sonata-v1m2-h3d-lin-grid_6.py \\
      --weight ckpt/862680/epoch_120.pth \\
      --output stats/umap/h3d/data/sonata_enc.npz

    # From-scratch baseline (no checkpoint, fresh random init):
    python scripts/extract_h3d_grid_probe_features.py \\
      --config configs/experiment/w109/5/13h_adamw_h3d/litept-b-v1m0-h3d-lin_5.py \\
      --rand-init \\
      --output stats/umap/h3d/data/litept_enc_randinit.npz
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
    (cached per tile, includes the H3D-specific z-alignment steps); `post`
    draws one SphereCrop from it -- cheap enough to call many times per tile
    for spatial diversity.

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
        dict(type="Z_MinShift"),
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
        dict(type="FillMissingFeat", feat_key="strength", feat_dim=1, fill_value=0.0),
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
    parser.add_argument("--config", required=True, help="One of the *-h3d-lin*.py GridProbe experiment configs.")
    parser.add_argument("--weight", default=None, help="Local path to the checkpoint (download from JZ first -- cfg.weight is a lustre path). Required unless --rand-init.")
    parser.add_argument("--rand-init", action="store_true", help="Skip loading a checkpoint -- extract features from the backbone's fresh random init instead (from-scratch baseline, e.g. for a 'litept_enc_randinit' UMAP panel). --weight is ignored.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--points-per-class", type=int, default=3000)
    parser.add_argument("--point-max", type=int, default=102400, help="SphereCrop cap per draw -- keep at the codebase's proven-safe point_max (see build_extraction_transforms docstring).")
    parser.add_argument("--max-draws", type=int, default=100, help="Safety cap on (tile, random-crop) draws before giving up on under-filled classes.")
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
        print(f"[extract] class {cls:2d} {name:18s}: {n:6d} pts{flag}")

    save_features_npz(
        args.output, collector, class_names, tile_names, used_tiles,
        config_path=str(args.config), weight_path=str(args.weight) if args.weight else "",
        rand_init=args.rand_init, points_per_class=args.points_per_class, seed=args.seed,
    )


if __name__ == "__main__":
    main()
