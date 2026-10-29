"""LitePT-B and Sonata panels for one DALES tile.

Writes, next to ``coord.npy``:

- ``pred.npy`` — LitePT-B encoder linear probe (the Tab. 3 DALES head)
- ``pred_sonata.npy`` — MALiBU3D Sonata (v1m2) linear probe
- ``pca_litept.npy`` / ``pca_sonata.npy`` — RGB from a 3-component PCA of the
  frozen encoder features, one colour per point

The full tile is ~10M voxels at the LitePT grid, which does not fit a forward
on one GPU. The cloud is cut into XY blocks with a halo; only the core of each
block is kept. Each block is normalized the way the test pipeline normalizes a
scene (center, then one point per voxel). ``shuffle_orders`` is turned off so
the panels are reproducible.

PCA is fit on a subsample of voxel features, after per-channel standardization,
so the five encoder levels contribute on the same scale. Each component is then
stretched from its 2nd to its 98th percentile into 0–255. The sign of a
component is arbitrary.
"""
import argparse
import os.path as osp

import numpy as np
import torch

import pointcept.models  # noqa: F401  (registers backbones and GridProbeSegmentorV2)
import pointcept.models.losses  # noqa: F401
from pointcept.datasets.transform import (
    CenterShift,
    Collect,
    FillMissingFeat,
    GridSample,
    ToTensor,
    Z_MinShift,
)
from pointcept.models.builder import build_model

SCENE = "data/dales/test/5100_54440_new"
LITEPT_BACKBONE = "ckpt/malibu3d/litept_b_multitask/model_best.pth"
LITEPT_PROBE = "ckpt/lin_probe/dales/litept/probe_best.pth"
# MALiBU3D Sonata (v1m2), the backbone the DALES linear probe was trained on.
# The indoor release is a different stem (9 channels, stride 2, coordinates / 10).
SONATA_BACKBONE = "ckpt/malibu3d/sonata_outdoor/epoch_120.pth"
SONATA_PROBE = "ckpt/lin_probe/dales/sonata/probe_best.pth"

IGNORE = 8
NUM_CLASSES = 8
CORE_M = 100.0
HALO_M = 40.0
PCA_CAP = 200_000
SAMPLE_PER_BLOCK = 8_000


def load_filtered(path, replace=None, prefixes=()):
    """Checkpoint keys, optionally rewritten, restricted to ``prefixes``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cleaned = {}
    for key, value in state.items():
        if replace is not None and replace[0] in key:
            key = key.replace(replace[0], replace[1], 1)
        elif replace is not None:
            continue
        if key.startswith("module."):
            key = key[len("module."):]
        if prefixes and not any(key == prefix or key.startswith(prefix) for prefix in prefixes):
            continue
        cleaned[key] = value
    return cleaned


def load_into(model, path, replace=None, prefixes=()):
    state = load_filtered(path, replace=replace, prefixes=prefixes)
    info = model.load_state_dict(state, strict=False)
    missing = [key for key in info.missing_keys if key.startswith("backbone.")]
    if missing:
        raise RuntimeError(
            f"{path}: {len(missing)} backbone keys missing, e.g. {missing[:6]}")
    print(f"loaded {path}: {len(state)} tensors, "
          f"{len(info.unexpected_keys)} unexpected")


def probe_model(backbone, in_channels):
    """One linear head. The archived probe is ``linear.weight`` / ``linear.bias`` only."""
    return build_model(dict(
        type="GridProbeSegmentorV2",
        probes=dict(main=dict(
            criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=IGNORE)],
            input_norm=None,
            feat_norm=None,
            dropout=0.0,
        )),
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE,
        backbone_out_channels=in_channels,
        backbone=backbone,
        freeze_backbone=True,
        feature_mask_values=dict(enable=True, masked_feat_keys=["color", "strength"])
        if backbone["type"] == "LitePT-v1" else None,
    ))


def litept_model():
    channels = (54, 108, 216, 432, 576)
    model = probe_model(dict(
        type="LitePT-v1",
        in_channels=7,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(3, 3, 3, 3),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=channels,
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=False,
        pre_norm=True,
        enc_mode=True,
    ), sum(channels))
    load_into(
        model, LITEPT_BACKBONE,
        prefixes=("backbone.", "color_mask_value", "strength_mask_value"))
    head = torch.load(LITEPT_PROBE, map_location="cpu", weights_only=False)
    model.heads["main"].load_state_dict(head)
    return model.cuda().eval()


def sonata_model():
    """PT-v3m2 as in configs/dales/sonata-v1m2-dales-lin-grid.py.

    Seven input channels (coord, zero color, strength). Stride 3 and a 0.1 m
    grid, the MALiBU3D pretraining setup. No coordinate rescaling.
    """
    channels = (48, 96, 192, 384, 512)
    model = probe_model(dict(
        type="PT-v3m2",
        in_channels=7,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(3, 3, 3, 3),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=channels,
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=False,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=True,
        freeze_encoder=False,
    ), sum(channels))
    load_into(
        model, SONATA_BACKBONE,
        replace=("module.student.backbone", "backbone"),
        prefixes=("backbone.",))
    head = torch.load(SONATA_PROBE, map_location="cpu", weights_only=False)
    model.heads["main"].load_state_dict(head)
    return model.cuda().eval()


def prepare_litept(coord, strength):
    data = {"coord": coord.copy(), "strength": strength.copy()}
    data = CenterShift(apply_z=True)(data)
    data = Z_MinShift()(data)
    data = FillMissingFeat("color", 3)(data)
    return data


def prepare_sonata(coord, strength):
    """Same normalization as the v1m2 DALES test pipeline."""
    data = {"coord": coord.copy(), "strength": strength.copy()}
    data = CenterShift(apply_z=True)(data)
    data = Z_MinShift()(data)
    data = FillMissingFeat("color", 3)(data)
    return data


def subset(data, index):
    n = len(data["coord"])
    out = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and len(value) == n:
            out[key] = value[index]
        else:
            out[key] = value
    return out


def forward_block(model, batch):
    """Voxel features (V, C) and class ids (V,)."""
    with torch.no_grad():
        feat_by_norm, point, _active = model.prepare_batch(batch)
        head = model.heads["main"]
        feat = feat_by_norm[(head.input_norm, head.eps)].float()
        pred = head.linear(feat).argmax(dim=-1)
    del point, feat_by_norm, batch
    return feat, pred


def fit_pca(samples):
    """Per-channel standardization, then the top 3 right-singular vectors."""
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    centered = (samples - mean) / std
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3].astype(np.float32)
    projected = centered @ components.T
    low, high = np.percentile(projected, [2, 98], axis=0)
    high = np.where(high - low < 1e-6, low + 1.0, high)
    return mean.astype(np.float32), std.astype(np.float32), components, low, high


def project_rgb(feat, pca):
    """Map voxel features to uint8 RGB with the PCA fit on the reservoir."""
    mean, std, components, low, high = (
        torch.as_tensor(value, device=feat.device, dtype=torch.float32) for value in pca)
    projected = (feat - mean) / std @ components.T
    scaled = ((projected - low) / (high - low)).clamp(0, 1)
    return scaled.mul(255).round().to(torch.uint8).cpu().numpy()


def miou(pred, gt):
    scores = []
    valid = gt != IGNORE
    for cls in range(NUM_CLASSES):
        union = np.sum(((pred == cls) | (gt == cls)) & valid)
        if union == 0:
            continue
        scores.append(np.sum((pred == cls) & (gt == cls) & valid) / union)
    return float(np.mean(scores)) if scores else float("nan")


def rectangles(xy, core):
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0) + 1e-3
    nx = int(np.ceil((x1 - x0) / core))
    ny = int(np.ceil((y1 - y0) / core))
    xs = x0 + np.arange(nx + 1) * (x1 - x0) / nx
    ys = y0 + np.arange(ny + 1) * (y1 - y0) / ny
    for i in range(nx):
        for j in range(ny):
            yield float(xs[i]), float(xs[i + 1]), float(ys[j]), float(ys[j + 1])


def _run_blocks(name, model, data, xy, grid_size, feat_keys, feat_scales, segment,
                core_m, halo_m):
    """``core_m`` and ``halo_m`` are in the same units as ``xy`` (metres, or the
    backbone's scaled metres)."""
    n = len(xy)
    pred = np.full(n, -1, dtype=np.int32)
    samples = []

    def infer_rect(xa, xb, ya, yb, depth, store):
        halo = (
            (xy[:, 0] >= xa - halo_m) & (xy[:, 0] < xb + halo_m)
            & (xy[:, 1] >= ya - halo_m) & (xy[:, 1] < yb + halo_m))
        halo_idx = np.flatnonzero(halo)
        if halo_idx.size == 0:
            return
        block_xy = xy[halo_idx]
        core_local = np.flatnonzero(
            (block_xy[:, 0] >= xa) & (block_xy[:, 0] < xb)
            & (block_xy[:, 1] >= ya) & (block_xy[:, 1] < yb))
        if core_local.size == 0:
            return
        block = subset(data, halo_idx)
        fragment_inverse = batch = feat = voxel_pred = None
        try:
            fragment_inverse, batch = voxel_batch_and_inverse(
                block, grid_size, feat_keys, feat_scales)
            feat, voxel_pred = forward_block(model, batch)
        except torch.cuda.OutOfMemoryError:
            del fragment_inverse, batch, feat, voxel_pred
            torch.cuda.empty_cache()
            if depth >= 4:
                raise
            print(f"  {name}: OOM on {halo_idx.size} points, splitting")
            mx, my = (xa + xb) / 2, (ya + yb) / 2
            for rect in ((xa, mx, ya, my), (mx, xb, ya, my),
                         (xa, mx, my, yb), (mx, xb, my, yb)):
                infer_rect(*rect, depth + 1, store)
            return
        point_pred = voxel_pred[fragment_inverse].detach().cpu().numpy()
        global_core = halo_idx[core_local]
        store(global_core, point_pred[core_local], feat, fragment_inverse, core_local)
        print(f"  {name}: core {global_core.size} pts, voxels {feat.shape[0]}")
        del feat, voxel_pred, fragment_inverse, batch, point_pred
        torch.cuda.empty_cache()

    def store_pred(global_core, point_pred, feat, inverse, core_local):
        pred[global_core] = point_pred
        voxel_ids = inverse[torch.as_tensor(core_local, device=inverse.device)]
        # One row per voxel that the core actually uses, capped per block.
        used = np.unique(voxel_ids.detach().cpu().numpy())
        take = used if used.size <= SAMPLE_PER_BLOCK else np.random.default_rng(0).choice(
            used, SAMPLE_PER_BLOCK, replace=False)
        samples.append(feat[torch.as_tensor(take, device=feat.device)].cpu().numpy())

    for rect in rectangles(xy, core_m):
        infer_rect(*rect, 0, store_pred)
    missing = int(np.sum(pred < 0))
    if missing:
        raise RuntimeError(f"{name}: {missing} points received no prediction")
    print(f"{name} mIoU on this tile: {miou(pred, segment):.3f}")

    bank = np.concatenate(samples, axis=0)
    if bank.shape[0] > PCA_CAP:
        bank = bank[np.random.default_rng(0).choice(bank.shape[0], PCA_CAP, replace=False)]
    pca = fit_pca(bank)
    print(f"{name} PCA fit on {bank.shape[0]} voxels, feat dim {bank.shape[1]}")
    del samples, bank

    rgb = np.zeros((n, 3), dtype=np.uint8)

    def store_rgb(global_core, _point_pred, feat, inverse, core_local):
        colours = project_rgb(feat, pca)
        picked = inverse[torch.as_tensor(core_local, device=inverse.device)]
        rgb[global_core] = colours[picked.detach().cpu().numpy()]

    for rect in rectangles(xy, core_m):
        infer_rect(*rect, 0, store_rgb)
    return pred, rgb


def voxel_batch_and_inverse(data, grid_size, feat_keys, feat_scales):
    sampler = GridSample(
        grid_size=grid_size, hash_type="fnv", mode="test",
        return_grid_coord=True, test_single_fragment=True)
    fragment = sampler(data)[0]
    inverse = fragment["inverse"]
    fragment = CenterShift(apply_z=False)(fragment)
    fragment = ToTensor()(fragment)
    batch = Collect(
        keys=("coord", "grid_coord", "index"),
        optional_keys=("inverse",),
        feat_keys=feat_keys,
        feat_scales=feat_scales,
    )(fragment)
    batch = {
        key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    if not torch.is_tensor(inverse):
        inverse = torch.from_numpy(np.asarray(inverse)).long()
    return inverse.cuda(), batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=SCENE)
    parser.add_argument("--models", nargs="+", default=["litept", "sonata"],
                        choices=["litept", "sonata"])
    args = parser.parse_args()

    coord = np.load(osp.join(args.scene, "coord.npy"))
    strength = np.load(osp.join(args.scene, "strength.npy"))
    segment = np.load(osp.join(args.scene, "segment.npy")).astype(np.int64)
    print(f"{args.scene}: {len(coord)} points")

    if "litept" in args.models:
        model = litept_model()
        data = prepare_litept(coord, strength)
        pred, rgb = _run_blocks(
            "litept", model, data, data["coord"][:, :2], 0.1,
            ["coord", "color", "strength"],
            dict(coord=0.01, strength=1 / 60000),
            segment, core_m=CORE_M, halo_m=HALO_M)
        np.save(osp.join(args.scene, "pred.npy"), pred.astype(np.int32))
        np.save(osp.join(args.scene, "pca_litept.npy"), rgb)
        del model, data, pred, rgb
        torch.cuda.empty_cache()

    if "sonata" in args.models:
        model = sonata_model()
        data = prepare_sonata(coord, strength)
        pred, rgb = _run_blocks(
            "sonata", model, data, data["coord"][:, :2], 0.1,
            ["coord", "color", "strength"],
            dict(strength=1 / 60000),
            segment, core_m=CORE_M, halo_m=HALO_M)
        np.save(osp.join(args.scene, "pred_sonata.npy"), pred.astype(np.int32))
        np.save(osp.join(args.scene, "pca_sonata.npy"), rgb)


if __name__ == "__main__":
    main()
