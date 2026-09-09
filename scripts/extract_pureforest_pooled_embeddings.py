#!/usr/bin/env python3
"""Extract per-tile mean/max-pooled frozen-backbone embeddings on PureForest.

One GPU pass over train/val/test (or a subset). For each tile, runs the same
encoder-multiscale path as ``GridProbeClassifier``, then saves both scene-level
pools so a later sklearn probe can try mean / max / concat / sum without
re-forwarding.

Uses the *val* transform pipeline from a PureForest GridProbe config, but forces
``GridSample(mode="test", test_single_fragment=True)`` so voxel picks are
deterministic (configs still use ``mode="train"`` on val, which is stochastic).

Does **not** apply train-time SphereCrop / augs. Optional ``--point-max`` adds a
deterministic center SphereCrop after voxelization when you need a VRAM cap.

Usage::

    export PYTHONPATH="$PWD"
    python scripts/extract_pureforest_pooled_embeddings.py \\
      --config configs/pureforest/cls-sonata-v1m2-pureforest-lin-grid-enc.py \\
      --weight ckpt/malibu3d/sonata_outdoor/epoch_120.pth \\
      --output-dir stats/pureforest/embeddings/sonata_outdoor \\
      --splits train val test \\
      --batch-size 4

    # Toy smoke:
    python scripts/extract_pureforest_pooled_embeddings.py \\
      --config configs/pureforest/cls-sonata-v1m2-pureforest-lin-grid-enc.py \\
      --weight ckpt/malibu3d/sonata_outdoor/epoch_120.pth \\
      --data-root data/pureforest_toy \\
      --output-dir stats/pureforest/embeddings/sonata_outdoor_toy \\
      --splits train val test --batch-size 2
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch_scatter
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _grid_probe_extract_common import build_model_for_extraction  # noqa: E402


def _safe_segment_csr_mean(feat, indptr):
    pooled = torch_scatter.segment_csr(feat.float(), indptr, reduce="mean")
    return torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)


def _pool_mean_max(feat, offset):
    """Return (mean_pool, max_pool) each shaped [B, C]."""
    import torch.nn.functional as F

    indptr = F.pad(offset, (1, 0))
    mean_feat = _safe_segment_csr_mean(feat, indptr)
    max_feat = torch_scatter.segment_csr(feat, indptr, reduce="max")
    return mean_feat, max_feat


def build_extract_transforms(cfg, point_max=None):
    """Split val transforms around GridSample; force deterministic voxel pick.

    Returns ``(pre_compose, post_compose)`` where ``pre`` may return a
    single-element list from GridSample test mode (caller unwraps).
    """
    from pointcept.datasets.transform import Compose

    val_list = list(cfg.data.val.transform)
    grid_idx = next(i for i, t in enumerate(val_list) if t["type"] == "GridSample")
    pre_cfgs = copy.deepcopy(val_list[: grid_idx + 1])
    post_cfgs = copy.deepcopy(val_list[grid_idx + 1 :])

    grid_cfg = pre_cfgs[-1]
    grid_cfg["mode"] = "test"
    grid_cfg["test_single_fragment"] = True

    if point_max is not None and int(point_max) > 0:
        # Insert after GridSample unwrap, before the rest of the val pipeline.
        post_cfgs.insert(
            0,
            dict(type="SphereCrop", point_max=int(point_max), mode="center"),
        )

    return Compose(pre_cfgs), Compose(post_cfgs)


def _unwrap_grid_sample(sample):
    if isinstance(sample, list):
        if len(sample) != 1:
            raise RuntimeError(
                f"Expected a single GridSample fragment, got {len(sample)}."
            )
        return sample[0]
    return sample


def extract_dual_pools(model, input_dict):
    """Backbone multiscale feat -> mean and max scene pools [B, C]."""
    model._fill_masked_feat_with_learned_value(input_dict)
    with torch.no_grad():
        feat, point = model._forward_backbone(input_dict)
    if getattr(model, "drop_leading_channels", 0):
        feat = feat[..., model.drop_leading_channels :]

    offset = input_dict["offset"]
    from pointcept.models.utils.structure import Point

    if isinstance(point, Point) and "offset" in point.keys():
        offset = point.offset
    n_scenes = int(offset.numel())
    n_points = int(offset[-1].item()) if n_scenes > 0 else 0

    if feat.shape[0] == n_scenes:
        # Already scene-pooled inside the backbone (e.g. SpUNet enc_mode).
        return feat, feat.clone()
    if feat.shape[0] == n_points:
        return _pool_mean_max(feat, offset)
    raise ValueError(
        "Unexpected backbone feature shape "
        f"{tuple(feat.shape)} for offset with {n_scenes} scenes and "
        f"{n_points} points."
    )


def _move_batch_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


@torch.no_grad()
def run_split(
    *,
    model,
    dataset,
    pre_transform,
    post_transform,
    batch_size,
    device,
    use_amp,
):
    from pointcept.datasets import collate_fn

    n = len(dataset.data_list)
    names = []
    categories = []
    mean_chunks = []
    max_chunks = []
    already_pooled_warned = False

    indices = list(range(n))
    pbar = tqdm(range(0, n, batch_size), desc=f"extract[{dataset.split}]", leave=True)
    for start in pbar:
        batch_indices = indices[start : start + batch_size]
        samples = []
        batch_names = []
        batch_cats = []
        for idx in batch_indices:
            raw = dataset.get_data(idx)
            name = dataset.get_data_name(idx)
            category = int(raw["category"][0])
            voxelized = _unwrap_grid_sample(pre_transform(raw))
            sample = post_transform(voxelized)
            samples.append(sample)
            batch_names.append(name)
            batch_cats.append(category)

        input_dict = collate_fn(samples)
        input_dict = _move_batch_to_device(input_dict, device)

        amp_enabled = bool(use_amp and device.type == "cuda")
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            mean_feat, max_feat = extract_dual_pools(model, input_dict)

        if torch.equal(mean_feat, max_feat) and not already_pooled_warned:
            print(
                "[extract] warning: mean and max pools are identical "
                "(backbone likely already scene-pooled)."
            )
            already_pooled_warned = True

        mean_chunks.append(mean_feat.float().cpu().numpy().astype(np.float16))
        max_chunks.append(max_feat.float().cpu().numpy().astype(np.float16))
        names.extend(batch_names)
        categories.extend(batch_cats)

        del input_dict, mean_feat, max_feat, samples
        if device.type == "cuda":
            torch.cuda.empty_cache()

        pbar.set_postfix(tiles=f"{min(start + batch_size, n)}/{n}")

    return {
        "names": np.asarray(names),
        "category": np.asarray(categories, dtype=np.int64),
        "mean_feat": np.concatenate(mean_chunks, axis=0),
        "max_feat": np.concatenate(max_chunks, axis=0),
    }


def save_split_npz(output_path, payload, meta):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        names=payload["names"],
        category=payload["category"],
        mean_feat=payload["mean_feat"],
        max_feat=payload["max_feat"],
        **{k: np.asarray(v) for k, v in meta.items()},
    )
    size_mb = output_path.stat().st_size / 2**20
    n, c = payload["mean_feat"].shape
    print(
        f"[extract] wrote {output_path}  ({n:,} tiles x {c}ch mean/max, {size_mb:.1f} MB)"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        required=True,
        help="PureForest GridProbeClassifier config (e.g. configs/pureforest/cls-sonata-...).",
    )
    parser.add_argument(
        "--weight",
        default=None,
        help="Checkpoint path. Defaults to cfg.weight. Required unless --rand-init.",
    )
    parser.add_argument(
        "--rand-init",
        action="store_true",
        help="Skip checkpoint load (random backbone init baseline).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for {split}.npz and meta.json.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override cfg.data.*.data_root (e.g. data/pureforest_toy).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Splits to extract (default: train val test).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--point-max",
        type=int,
        default=None,
        help="Optional center SphereCrop after voxelization (VRAM / speed cap).",
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto).")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable fp16 autocast on CUDA (off by default; matches Sonata probe configs).",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Optional cap on tiles per split (smoke / debug).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from pointcept.utils.config import Config
    from pointcept.datasets.builder import build_dataset

    cfg = Config.fromfile(args.config)
    weight = args.weight or cfg.get("weight")
    if not args.rand_init and not weight:
        raise SystemExit("Provide --weight or set cfg.weight (or pass --rand-init).")

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[extract] device={device}  config={args.config}")

    model = build_model_for_extraction(
        cfg, weight, device, rand_init=args.rand_init
    )
    # Classification extract only needs the frozen backbone path.
    model.eval()

    pre_transform, post_transform = build_extract_transforms(
        cfg, point_max=args.point_max
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = list(cfg.data.get("names", cfg.get("class_names", [])))
    backbone_out_channels = int(cfg.model.get("backbone_out_channels", -1))
    meta_common = {
        "config": str(args.config),
        "weight": str(weight) if weight else "",
        "rand_init": bool(args.rand_init),
        "data_root": str(args.data_root or cfg.data.val.data_root),
        "grid_size": float(cfg.get("grid_size", -1)),
        "point_max_cfg": int(cfg.get("point_max", -1)),
        "point_max_extract": -1 if args.point_max is None else int(args.point_max),
        "backbone_out_channels": backbone_out_channels,
        "backbone_type": str(cfg.model.backbone.type),
        "class_names": class_names,
        "gridsample_mode": "test_single_fragment",
    }

    for split in args.splits:
        split_cfg = copy.deepcopy(cfg.data[split])
        # Dataset applies its own transform; we drive transforms manually.
        split_cfg["transform"] = []
        split_cfg["test_mode"] = False
        if args.data_root is not None:
            split_cfg["data_root"] = args.data_root
        dataset = build_dataset(split_cfg)
        if args.max_tiles is not None:
            dataset.data_list = dataset.data_list[: int(args.max_tiles)]
            print(f"[extract] capped {split} at {len(dataset.data_list)} tiles")

        payload = run_split(
            model=model,
            dataset=dataset,
            pre_transform=pre_transform,
            post_transform=post_transform,
            batch_size=max(1, int(args.batch_size)),
            device=device,
            use_amp=args.amp,
        )
        if backbone_out_channels > 0 and payload["mean_feat"].shape[1] != backbone_out_channels:
            print(
                "[extract] warning: feat dim "
                f"{payload['mean_feat'].shape[1]} != cfg.backbone_out_channels "
                f"{backbone_out_channels}"
            )

        split_meta = {
            **meta_common,
            "split": split,
            "num_tiles": int(payload["category"].shape[0]),
            "feat_dim": int(payload["mean_feat"].shape[1]),
        }
        save_split_npz(output_dir / f"{split}.npz", payload, {
            "split": split,
            "feat_dim": split_meta["feat_dim"],
            "backbone_out_channels": backbone_out_channels,
            "class_names": np.asarray(class_names),
            "config": args.config,
            "weight": weight or "",
            "data_root": split_meta["data_root"],
            "grid_size": split_meta["grid_size"],
            "point_max_extract": split_meta["point_max_extract"],
            "gridsample_mode": split_meta["gridsample_mode"],
        })

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({**meta_common, "splits": list(args.splits)}, f, indent=2)
    print(f"[extract] wrote {meta_path}")


if __name__ == "__main__":
    main()
