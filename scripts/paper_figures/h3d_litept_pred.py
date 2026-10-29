"""Write LitePT-B linear-probe predictions for the merged H3D test block.

Same frozen backbone and normalization as configs/h3d/litept-b-v1m0-h3d-lin_enc.py
(encoder features, 11 classes, ignore Void). H3D has RGB and no intensity, so
strength is a learned fill. pred.npy is one class id per original point,
aligned with coord.npy.
"""
import gc
import os.path as osp
import sys

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import dales_feature_panels as panels
from pointcept.datasets.transform import CenterShift, FillMissingFeat, NormalizeColor, Z_MinShift

SCENE = "/home/lgeist/blendify/data/h3d/Mar18_test_main"
PROBE = "ckpt/lin_probe/h3d/litept/probe_best.pth"
NAMES = (
    "Low Vegetation",
    "Impervious Surface",
    "Vehicle",
    "Urban Furniture",
    "Roof",
    "Facade",
    "Shrub",
    "Tree",
    "Soil or Gravel",
    "Vertical Surface",
    "Chimney",
)
IGNORE = 11


def prepare(coord, color):
    data = {"coord": coord.copy(), "color": color.astype(np.float32, copy=False)}
    data = CenterShift(apply_z=True)(data)
    data = Z_MinShift()(data)
    data = NormalizeColor()(data)
    return FillMissingFeat("strength", 1, fill_value=0.0)(data)


# The merged block is ~49M points. A 100 m core with a 40 m halo OOMs on the
# dense part, and splitting the core does not shrink that halo. These windows
# stay near the size of one official ~95 m test cell.
CORE_M = 35.0
HALO_M = 15.0


def predict(model, data, xy, segment):
    """One class id per point. Skips the PCA second pass of the DALES panels."""
    n = len(xy)
    pred = np.full(n, -1, dtype=np.int32)

    def infer_rect(xa, xb, ya, yb, depth):
        halo = (
            (xy[:, 0] >= xa - HALO_M) & (xy[:, 0] < xb + HALO_M)
            & (xy[:, 1] >= ya - HALO_M) & (xy[:, 1] < yb + HALO_M))
        halo_idx = np.flatnonzero(halo)
        if halo_idx.size == 0:
            return
        block_xy = xy[halo_idx]
        core_local = np.flatnonzero(
            (block_xy[:, 0] >= xa) & (block_xy[:, 0] < xb)
            & (block_xy[:, 1] >= ya) & (block_xy[:, 1] < yb))
        if core_local.size == 0:
            return
        block = panels.subset(data, halo_idx)
        fragment_inverse = batch = feat = voxel_pred = None
        try:
            fragment_inverse, batch = panels.voxel_batch_and_inverse(
                block, 0.1, ["coord", "color", "strength"], dict(coord=0.01))
            feat, voxel_pred = panels.forward_block(model, batch)
        except torch.cuda.OutOfMemoryError:
            del fragment_inverse, batch, feat, voxel_pred
            gc.collect()
            torch.cuda.empty_cache()
            if depth >= 5:
                raise
            print(f"  litept-h3d: OOM on {halo_idx.size} points, splitting")
            mx, my = (xa + xb) / 2, (ya + yb) / 2
            for rect in ((xa, mx, ya, my), (mx, xb, ya, my),
                         (xa, mx, my, yb), (mx, xb, my, yb)):
                infer_rect(*rect, depth + 1)
            return
        point_pred = voxel_pred[fragment_inverse].detach().cpu().numpy()
        global_core = halo_idx[core_local]
        pred[global_core] = point_pred[core_local]
        print(f"  litept-h3d: core {global_core.size} pts, voxels {feat.shape[0]}")
        del feat, voxel_pred, fragment_inverse, batch, point_pred
        torch.cuda.empty_cache()

    for rect in panels.rectangles(xy, CORE_M):
        infer_rect(*rect, 0)
    missing = int(np.sum(pred < 0))
    if missing:
        raise RuntimeError(f"litept-h3d: {missing} points received no prediction")
    print(f"litept-h3d mIoU on this block: {panels.miou(pred, segment):.3f}")
    return pred


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else SCENE
    coord = np.load(osp.join(scene, "coord.npy"))
    color = np.load(osp.join(scene, "color.npy"))
    segment = np.load(osp.join(scene, "segment.npy")).astype(np.int64)
    print(f"{scene}: {len(coord)} points, color [{color.min():.1f}, {color.max():.1f}]")

    panels.NUM_CLASSES = len(NAMES)
    panels.IGNORE = IGNORE
    _load = panels.torch.load

    def _load_h3d(path, *args, **kwargs):
        if path == panels.LITEPT_PROBE:
            path = PROBE
        return _load(path, *args, **kwargs)

    panels.torch.load = _load_h3d
    model = panels.litept_model()
    print(f"LitePT-B head {PROBE} {tuple(model.heads['main'].linear.weight.shape)}")

    data = prepare(coord, color)
    pred = predict(model, data, data["coord"][:, :2], segment)
    out = osp.join(scene, "pred.npy")
    np.save(out, pred.astype(np.int32))
    valid = segment != IGNORE
    counts = np.bincount(pred[valid], minlength=len(NAMES))
    for name, count in zip(NAMES, counts):
        print(f"  {count:8d}  {name}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
