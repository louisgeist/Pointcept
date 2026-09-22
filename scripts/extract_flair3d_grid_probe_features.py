#!/usr/bin/env python3
"""Extract frozen-backbone multiscale features on Flair3D+ test tiles (D068,
D075 by default) for UMAP viz, colored by segment (v20/finer12, 15 classes).

Companion to extract_dales_grid_probe_features.py -- same flash-attn/spconv
workarounds, same stratified per-class sampling (see
scripts/_grid_probe_extract_common.py for the shared machinery and full
rationale). The difference is entirely in the dataset/transform: Flair3D+ has
real color (FLAIR-HUB aerial imagery, NormalizeColor) instead of DALES's
zero-filled fake color, and its own local test-split mirroring.

Two backbones, both read straight from their *native* Flair3D+ configs (not
GridProbe configs -- there's no DALES-style downstream transfer step here,
these checkpoints were pretrained/finetuned directly on Flair3D+):

  configs/flair3d_default/probe/sonata-v1m2-flair3d-lin-grid.py   Sonata PT-v3m2 encoder MS (1232ch), frozen
  configs/flair3d_default/multi-litept-b-v1m0-flair3d.py          LitePT-B (dec hypercolumn 1404ch, or --enc-mode for 1386ch encoder MS)
  configs/flair3d_default/multi-ptv3-v1m0-flair3d.py              PT-v3-malibu (dec hypercolumn 1024ch via traceable=True, or --enc-mode for 992ch)

Same checkpoints already used for the DALES comparison (ckpt/862680, ckpt/873542,
ckpt/1095469) -- they *are* the Flair3D-native pretrain/finetune checkpoints,
no new download needed.

D068/D075 test tiles are already mirrored locally under
data/flair3d_plus/test/{D068,D075}-2021_LIDARHD/. Flair3DDataset only accepts
a single csv_manifest, so --csv-manifest defaults to a locally-built
concatenation of the two per-department manifests
(data/flair3d_plus/raw/scene_split_manifest_D068_D075.csv) -- rebuild it with
`csv.writer`/pandas if you want a different department combo.

Usage::

    python scripts/extract_flair3d_grid_probe_features.py \\
      --config configs/flair3d_default/probe/sonata-v1m2-flair3d-lin-grid.py \\
      --weight ckpt/862680/epoch_120.pth \\
      --output stats/umap/flair3d/data/sonata_enc.npz

    python scripts/extract_flair3d_grid_probe_features.py \\
      --config configs/flair3d_default/multi-litept-b-v1m0-flair3d.py \\
      --weight ckpt/873542/model_best.pth \\
      --output stats/umap/flair3d/data/litept_dec.npz

    python scripts/extract_flair3d_grid_probe_features.py \\
      --config configs/flair3d_default/multi-litept-b-v1m0-flair3d.py \\
      --weight ckpt/873542/model_best.pth \\
      --output stats/umap/flair3d/data/litept_enc.npz --enc-mode

    # From-scratch baseline (no checkpoint, fresh random init):
    python scripts/extract_flair3d_grid_probe_features.py \\
      --config configs/flair3d_default/multi-litept-b-v1m0-flair3d.py \\
      --rand-init --enc-mode \\
      --output stats/umap/flair3d/data/litept_enc_randinit.npz
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

_SEGMENT_V20_NAMES = (
    "Building", "Greenhouse", "Impervious surface", "Other soil", "Herbaceous",
    "Vineyard", "Brushwood", "Other infrastructures", "Swimming pool", "Water",
    "Deciduous", "Coniferous", "Bridge", "Agricultural soil", "Soil under vegetation",
)


def build_extraction_transforms(cfg, point_max):
    """Same SphereCrop-splice idea as the DALES script's
    build_extraction_transforms, but Flair3D+ has real color (unlike DALES's
    zero-filled fake color), so NormalizeColor replaces FillMissingFeat, read
    straight off whichever of cfg.data.val.transform's entries are present.
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
    parser.add_argument("--config", required=True, help="configs/flair3d_default/probe/sonata-v1m2-flair3d-lin-grid.py or configs/flair3d_default/multi-litept-b-v1m0-flair3d.py.")
    parser.add_argument("--weight", default=None, help="Local checkpoint path (ckpt/862680/epoch_120.pth or ckpt/873542/model_best.pth). Required unless --rand-init.")
    parser.add_argument("--rand-init", action="store_true", help="Skip loading a checkpoint -- extract features from the backbone's fresh random init instead (from-scratch baseline, e.g. for a 'litept_enc_randinit' UMAP panel). --weight is ignored.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--csv-manifest", default="data/flair3d_plus/raw/scene_split_manifest_D068_D075.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--enc-mode", action="store_true", help="LitePT-B only: read the encoder-multiscale hypercolumn (1386ch, enc_mode=True) instead of the native decoder hypercolumn (1404ch). No-op for Sonata (already enc_mode=True in its config).")
    parser.add_argument("--points-per-class", type=int, default=3000)
    parser.add_argument("--point-max", type=int, default=102400)
    parser.add_argument("--max-draws", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.rand_init and args.weight is None:
        parser.error("--weight is required unless --rand-init is set.")

    np.random.seed(args.seed)  # SphereCrop(mode="random") draws its center from the global numpy RNG

    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset

    cfg = Config.fromfile(args.config)

    # Both source configs are multitask/multi-probe; reduce to a plain
    # mono-task "segment" readout -- Flair3DDataset defaults to exactly this
    # (target_keys=("segment",)) and it's all we need for the UMAP colors.
    is_multitask = cfg.model.get("type") == "MultiTaskSegmentorV2"
    if is_multitask:
        backbone_cfg = copy.deepcopy(cfg.model["backbone"])
        if args.enc_mode:
            backbone_cfg["enc_mode"] = True
            backbone_out_channels = sum(backbone_cfg["enc_channels"])
        else:
            # Native multitask configs only expose the finest decoder stage
            # (LitePT 72ch / PT-v3-malibu 64ch). Flip on the backbone's
            # traceable flag so GridProbeSegmentorV2 builds the same
            # decoder-hypercolumn multiscale concat the GridProbe decoder
            # configs use. Purely additive bookkeeping -- doesn't change any
            # weight or computed value, just what gets stashed for
            # concatenation -- safe to enable post-hoc on a frozen checkpoint.
            # LitePT uses `dec_traceable`; PT-v3-malibu uses `traceable`.
            backbone_type = backbone_cfg.get("type", "")
            if backbone_type == "PT-v3-malibu":
                backbone_cfg["traceable"] = True
            else:
                backbone_cfg["dec_traceable"] = True
            dec_channels = cfg.model["backbone"]["dec_channels"]
            bottleneck = cfg.model["backbone"]["enc_channels"][-1]
            backbone_out_channels = sum(dec_channels) + bottleneck
        cfg.model = dict(
            type="GridProbeSegmentorV2",
            probes={"dummy": dict(criteria=[dict(type="CrossEntropyLoss")], input_norm=None, feat_norm=None, dropout=0.0)},
            num_classes=15,
            ignore_index=15,
            target_key="segment",
            backbone_out_channels=backbone_out_channels,
            backbone=backbone_cfg,
            freeze_backbone=True,
        )
    num_classes = cfg.model["num_classes"]
    class_names = list(_SEGMENT_V20_NAMES)
    assert len(class_names) == num_classes, (class_names, num_classes)

    # Point at the local D068/D075 test manifest -- cfg.data.val's own
    # csv_manifest is the national one (JZ-only, see CLAUDE.md), and its
    # `split` is "val" (dev-subset workflow) not "test".
    val_cfg = copy.deepcopy(cfg.data.val)
    val_cfg["csv_manifest"] = args.csv_manifest
    val_cfg["split"] = args.split
    val_cfg["target_keys"] = ["segment"]
    val_cfg["primary_target_key"] = "segment"
    val_cfg.pop("stratified_subset_manifest", None)
    val_cfg.pop("max_sample", None)

    model = build_model_for_extraction(cfg, args.weight, args.device, rand_init=args.rand_init)
    pre_transform, post_transform = build_extraction_transforms(cfg, args.point_max)
    dataset = build_dataset(val_cfg)

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
        print(f"[extract] class {cls:2d} {name:22s}: {n:6d} pts{flag}")

    save_features_npz(
        args.output, collector, class_names, tile_names, used_tiles,
        config_path=str(args.config), weight_path=str(args.weight) if args.weight else "",
        rand_init=args.rand_init, points_per_class=args.points_per_class, seed=args.seed,
        csv_manifest=args.csv_manifest,
    )


if __name__ == "__main__":
    main()
