"""Shared, dataset-agnostic pieces for the extract_<dataset>_grid_probe_features.py
scripts (currently DALES and Flair3D+). See extract_dales_grid_probe_features.py
for the full design rationale (flash-attn shim, spconv ConvAlgo.Native
workaround, per-class stratified sampling). Only the dataset/transform-pipeline
construction is dataset-specific and stays in each script.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def install_sdpa_flash_shim(module) -> bool:
    """Monkeypatch `module.flash_attn` with an SDPA-based drop-in for
    `flash_attn_varlen_qkvpacked_func` (no real flash-attn install needed --
    same numerical op, non-fused kernel, irrelevant for a one-off extraction).
    No-op if flash_attn is already importable there.
    """
    if getattr(module, "flash_attn", None) is not None:
        return False

    class _SDPAFlashShim:
        @staticmethod
        def flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None
        ):
            n_pad, _, h, ch = qkv.shape
            q_all, k_all, v_all = qkv.unbind(dim=1)
            out = torch.empty(n_pad, h, ch, device=qkv.device, dtype=qkv.dtype)
            dp = dropout_p if isinstance(dropout_p, (int, float)) else 0.0
            starts = cu_seqlens[:-1].tolist()
            ends = cu_seqlens[1:].tolist()
            for s, e in zip(starts, ends):
                if e <= s:
                    continue
                q = q_all[s:e].permute(1, 0, 2).unsqueeze(0)
                k = k_all[s:e].permute(1, 0, 2).unsqueeze(0)
                v = v_all[s:e].permute(1, 0, 2).unsqueeze(0)
                o = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, scale=softmax_scale)
                out[s:e] = o.squeeze(0).permute(1, 0, 2)
            return out

    module.flash_attn = _SDPAFlashShim()
    return True


def find_checkpoint_loader_cfg(cfg):
    for h in cfg.hooks:
        if h["type"] == "CheckpointLoader":
            return h
    raise ValueError("No CheckpointLoader hook found in cfg.hooks.")


def load_checkpoint_into_model(model, weight_path, keywords="", replacement=None, exclude_keys=()):
    """Mirrors pointcept.engines.hooks.misc.CheckpointLoader.before_train,
    minus the trainer/DDP/resume machinery this standalone script has no use
    for -- same key rename / exclude / "module." stripping (single process).
    """
    replacement = keywords if replacement is None else replacement
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    weight = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        if not key.startswith("module."):
            key = "module." + key
        if exclude_keys and any(k in key for k in exclude_keys):
            continue
        if keywords and keywords in key:
            key = key.replace(keywords, replacement, 1)
        key = key[7:]  # module.xxx -> xxx (single process)
        weight[key] = value
    missing, unexpected = model.load_state_dict(weight, strict=False)
    print(f"[checkpoint] loaded {weight_path}")
    print(f"[checkpoint] missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    print(f"[checkpoint] unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")


def force_native_conv_algo(model):
    """Force every spconv layer to ConvAlgo.Native (spconv's plain
    gather-scatter implementation) instead of the default auto-tuned
    MaskImplicitGemm path.

    On this machine (spconv-cu118 2.3.8 / cumm-cu118 0.7.11, RTX A6000
    sm_86), the auto-tuner's kernel search comes up empty as soon as a
    *second* distinct (in_channels, out_channels, kernel_size) SubMConv3d
    shape is used in the same process -- reproduced down to a 2-call
    standalone repro (stem k=5 7->54 then any other shape both fail
    identically on real data and on synthetic data alike), so it's an
    autotuner/kernel-coverage gap in this spconv+cumm+driver combo, not a
    shape or data-density issue. ConvAlgo.Native sidesteps the tuner
    entirely (verified: full LitePT-B forward, 102400 pts, 0.6s). Slower
    than the fused kernel would be, irrelevant for a few dozen tiles' worth
    of one-off feature extraction.
    """
    import spconv.pytorch as spconv
    from spconv.pytorch.core import ConvAlgo

    n = 0
    for m in model.modules():
        if isinstance(m, spconv.conv.SparseConvolution):
            m.algo = ConvAlgo.Native
            n += 1
    return n


def build_model_for_extraction(cfg, weight_path, device, rand_init=False):
    """`rand_init=True` skips the CheckpointLoader entirely -- the model
    keeps its fresh nn.Module init (whatever the backbone's own reset_parameters
    does), used as a from-scratch baseline against the pretrained/finetuned
    features (e.g. a "litept_enc_randinit" UMAP panel). `weight_path` is
    ignored in that case.
    """
    from pointcept.models.builder import build_model

    model_cfg = copy.deepcopy(cfg.model)
    backbone_type = model_cfg["backbone"]["type"]
    if backbone_type == "PT-v3m2":
        model_cfg["backbone"]["enable_flash"] = False
    elif backbone_type == "LitePT-v1":
        import pointcept.models.litept.litept_v1 as litept_mod

        if install_sdpa_flash_shim(litept_mod):
            print("[flash-attn] not installed -- LitePT-v1 attention routed through SDPA instead.")
    elif backbone_type == "PT-v3-malibu":
        import pointcept.models.point_transformer_v3.point_transformer_v3_malibu as ptv3_malibu_mod

        if install_sdpa_flash_shim(ptv3_malibu_mod):
            print("[flash-attn] not installed -- PT-v3-malibu attention routed through SDPA instead.")
    elif backbone_type in (
        "SpUNet-v1m1",
        "SpUNet-v1m2",
        "SpUNet-v1m3",
        "kpconvx_base",
        "KPConvX",
    ):
        # No flash-attn path in these backbones.
        pass
    else:
        raise ValueError(f"Unhandled backbone type for flash-attn handling: {backbone_type}")

    model = build_model(model_cfg)
    n_native = force_native_conv_algo(model)
    print(f"[spconv] forced ConvAlgo.Native on {n_native} layer(s) (works around a local autotuner gap).")
    model = model.to(device)
    model.eval()

    if rand_init:
        print("[checkpoint] --rand-init set -- skipping CheckpointLoader, using the model's fresh random init.")
        return model

    ckpt_cfg = find_checkpoint_loader_cfg(cfg)
    load_checkpoint_into_model(
        model,
        weight_path,
        keywords=ckpt_cfg.get("keywords", ""),
        replacement=ckpt_cfg.get("replacement"),
        exclude_keys=ckpt_cfg.get("exclude_keys", ()),
    )
    return model


class PerClassCollector:
    """Caps each tile's contribution and the final pooled total at
    `points_per_class`, per class. Not a statistically exact reservoir
    (later tiles aren't down-weighted against earlier ones) -- fine for a
    UMAP sanity-check sample, and keeps memory bounded throughout.
    """

    def __init__(self, num_classes, points_per_class, rng):
        self.points_per_class = points_per_class
        self.rng = rng
        self.coord = [[] for _ in range(num_classes)]
        self.feat = [[] for _ in range(num_classes)]
        self.tile_idx = [[] for _ in range(num_classes)]

    def offer_tile(self, cls, coord_np, feat_np, tile_index):
        n = coord_np.shape[0]
        if n == 0:
            return
        if n > self.points_per_class:
            sel = self.rng.choice(n, size=self.points_per_class, replace=False)
            coord_np, feat_np = coord_np[sel], feat_np[sel]
            n = self.points_per_class
        self.coord[cls].append(coord_np)
        self.feat[cls].append(feat_np)
        self.tile_idx[cls].append(np.full(n, tile_index, dtype=np.int64))

    def count(self, cls):
        return sum(a.shape[0] for a in self.coord[cls])

    def finalize(self):
        all_coord, all_feat, all_segment, all_tile_idx = [], [], [], []
        for cls in range(len(self.coord)):
            if not self.coord[cls]:
                continue
            coord = np.concatenate(self.coord[cls], axis=0)
            feat = np.concatenate(self.feat[cls], axis=0)
            tile_idx = np.concatenate(self.tile_idx[cls], axis=0)
            n = coord.shape[0]
            if n > self.points_per_class:
                sel = self.rng.choice(n, size=self.points_per_class, replace=False)
                coord, feat, tile_idx = coord[sel], feat[sel], tile_idx[sel]
                n = self.points_per_class
            all_coord.append(coord)
            all_feat.append(feat)
            all_segment.append(np.full(n, cls, dtype=np.int64))
            all_tile_idx.append(tile_idx)
        return (
            np.concatenate(all_coord, axis=0),
            np.concatenate(all_feat, axis=0),
            np.concatenate(all_segment, axis=0),
            np.concatenate(all_tile_idx, axis=0),
        )


def run_extraction(
    *,
    model,
    dataset,
    pre_transform,
    post_transform,
    num_classes,
    target_key,
    points_per_class,
    max_draws,
    seed,
    device,
):
    """Shared (tile, random-crop) draw loop: voxelize each tile once (cached),
    draw repeated SphereCrops from it, forward through the frozen backbone,
    and feed a stratified PerClassCollector until every class is capped or
    max_draws is exhausted. Returns (collector, tile_names, used_tiles).
    """
    from pointcept.datasets import collate_fn

    n_tiles = len(dataset.data_list)
    rng = np.random.default_rng(seed)
    tile_names = [dataset.get_data_name(i) for i in range(n_tiles)]

    collector = PerClassCollector(num_classes, points_per_class, rng)
    voxelized_cache = {}
    used_tiles = set()
    tile_cycle = []

    for draw in range(max_draws):
        if all(collector.count(c) >= points_per_class for c in range(num_classes)):
            print(f"[extract] all {num_classes} classes at cap -- stopping early "
                  f"({draw} draws, {len(used_tiles)}/{n_tiles} tiles touched)")
            break

        if not tile_cycle:
            tile_cycle = list(rng.permutation(n_tiles))
        tile_index = int(tile_cycle.pop())

        if tile_index not in voxelized_cache:
            raw = dataset.get_data(tile_index)
            voxelized_cache[tile_index] = pre_transform(raw)
        voxelized = copy.deepcopy(voxelized_cache[tile_index])
        cropped = post_transform(voxelized)

        input_dict = collate_fn([cropped])
        for key in input_dict:
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].to(device, non_blocking=True)

        n_points = input_dict["coord"].shape[0]
        print(f"[extract] draw {draw}: tile {tile_names[tile_index]} ({n_points:,} pts) ...")
        used_tiles.add(tile_names[tile_index])

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            feat_by_norm, point, _active = model.prepare_batch(input_dict)
        del point
        feat = next(iter(feat_by_norm.values()))  # input_norm=None -> raw backbone feat
        target = input_dict[target_key]
        coord = input_dict["coord"]

        for cls in range(num_classes):
            if collector.count(cls) >= points_per_class:
                continue
            mask = target == cls
            if not bool(mask.any()):
                continue
            coord_np = coord[mask].float().cpu().numpy()
            feat_np = feat[mask].float().cpu().numpy().astype(np.float16)
            collector.offer_tile(cls, coord_np, feat_np, tile_index)

        del feat, feat_by_norm, target, coord, input_dict
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"[extract] hit --max-draws={max_draws} with some classes still under cap.")

    return collector, tile_names, used_tiles


def save_features_npz(output, collector, class_names, tile_names, used_tiles, **extra_meta):
    coord, feat, segment, tile_idx = collector.finalize()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        coord=coord.astype(np.float32),
        feat=feat.astype(np.float16),
        segment=segment.astype(np.int64),
        tile_idx=tile_idx.astype(np.int64),
        tile_names=np.array(tile_names),
        used_tiles=np.array(sorted(used_tiles)),
        class_names=np.array(class_names),
        backbone_out_channels=feat.shape[1],
        **extra_meta,
    )
    print(f"[extract] wrote {output}  ({feat.shape[0]:,} pts x {feat.shape[1]}ch, "
          f"{Path(output).stat().st_size / 2**20:.1f} MB)")
