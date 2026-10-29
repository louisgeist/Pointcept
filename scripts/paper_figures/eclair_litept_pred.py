"""Write LitePT-B linear-probe predictions for one ECLAIR tile.

Same frozen backbone and normalization as configs/eclair/litept-b-v1m0-eclair-lin_enc.py
(encoder features, 11 classes). The head is the archived Tab. 3 probe
ckpt/lin_probe/eclair/litept/probe_best.pth. pred.npy is one class id per
original point, aligned with coord.npy, so a figure can paint it as `pred`.
"""
import os.path as osp
import sys

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import dales_feature_panels as panels
from pointcept.datasets.transform import CenterShift, NormalizeColor, Z_MinShift

SCENE = "data/eclair/test/pointcloud_943"
PROBE = "ckpt/lin_probe/eclair/litept/probe_best.pth"
NAMES = (
    "Unassigned",
    "Ground",
    "Vegetation",
    "Buildings",
    "Noise",
    "Transmission Wires",
    "Distribution Wires",
    "Poles",
    "Transmission Towers",
    "Fence",
    "Vehicle",
)


def prepare(coord, color, strength):
    data = {
        "coord": coord.copy(),
        "color": color.astype(np.float32, copy=False),
        "strength": strength.copy(),
    }
    data = CenterShift(apply_z=True)(data)
    data = Z_MinShift()(data)
    return NormalizeColor()(data)


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else SCENE
    coord = np.load(osp.join(scene, "coord.npy"))
    color = np.load(osp.join(scene, "color.npy"))
    strength = np.load(osp.join(scene, "strength.npy"))
    segment = np.load(osp.join(scene, "segment.npy")).astype(np.int64)
    print(f"{scene}: {len(coord)} points, color [{color.min():.1f}, {color.max():.1f}]")

    panels.NUM_CLASSES = len(NAMES)
    panels.IGNORE = -1
    _load = panels.torch.load

    def _load_eclair(path, *args, **kwargs):
        if path == panels.LITEPT_PROBE:
            path = PROBE
        return _load(path, *args, **kwargs)

    panels.torch.load = _load_eclair
    model = panels.litept_model()
    print(f"LitePT-B head {PROBE} {tuple(model.heads['main'].linear.weight.shape)}")

    data = prepare(coord, color, strength)
    pred, _rgb = panels._run_blocks(
        "litept-eclair", model, data, data["coord"][:, :2], 0.1,
        ["coord", "color", "strength"],
        dict(coord=0.01, strength=1 / 60000),
        segment, core_m=100.0, halo_m=40.0)
    out = osp.join(scene, "pred.npy")
    np.save(out, pred.astype(np.int32))
    counts = np.bincount(pred[pred >= 0], minlength=len(NAMES))
    for name, count in zip(NAMES, counts):
        print(f"  {count:8d}  {name}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
