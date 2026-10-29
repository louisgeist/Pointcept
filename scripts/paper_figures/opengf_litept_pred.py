"""Write LitePT-B linear-probe predictions for one OpenGF cell.

Same frozen backbone and normalization as
configs/opengf/litept-b-v1m0-opengf-lin-grid-enc.py. The head is the best
validation seed of Jean Zay job 26448 (2 classes: Ground, Non-ground).
OpenGF has no RGB, so color is a learned fill. Outliers stay label 2 on disk
and are ignored by the probe; the saved ids are only 0 or 1.
"""
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import dales_feature_panels as panels
from pointcept.datasets.transform import CenterShift, FillMissingFeat, Z_MinShift

SCENE = "data/opengf/test/T2_5-2"
PROBE = "ckpt/lin_probe/opengf/litept/probe_best.pth"
NAMES = ("Ground", "Non-ground")
IGNORE = 2


def prepare(coord, strength):
    data = {"coord": coord.copy(), "strength": strength.copy()}
    data = CenterShift(apply_z=True)(data)
    data = Z_MinShift()(data)
    return FillMissingFeat("color", 3)(data)


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else SCENE
    coord = np.load(osp.join(scene, "coord.npy"))
    strength = np.load(osp.join(scene, "strength.npy"))
    segment = np.load(osp.join(scene, "segment.npy")).astype(np.int64)
    print(f"{scene}: {len(coord)} points, strength [{strength.min():.1f}, {strength.max():.1f}]")

    panels.NUM_CLASSES = len(NAMES)
    panels.IGNORE = IGNORE
    _load = panels.torch.load

    def _load_opengf(path, *args, **kwargs):
        if path == panels.LITEPT_PROBE:
            path = PROBE
        return _load(path, *args, **kwargs)

    panels.torch.load = _load_opengf
    model = panels.litept_model()
    print(f"LitePT-B head {PROBE} {tuple(model.heads['main'].linear.weight.shape)}")

    data = prepare(coord, strength)
    pred, _rgb = panels._run_blocks(
        "litept-opengf", model, data, data["coord"][:, :2], 0.1,
        ["coord", "color", "strength"],
        dict(coord=0.01, strength=1 / 60000),
        segment, core_m=100.0, halo_m=40.0)
    out = osp.join(scene, "pred.npy")
    np.save(out, pred.astype(np.int32))
    counts = np.bincount(pred, minlength=len(NAMES))
    for name, count in zip(NAMES, counts):
        print(f"  {count:8d}  {name}")
    outliers = segment == IGNORE
    if outliers.any():
        outlier_pred = np.bincount(pred[outliers], minlength=len(NAMES))
        print(
            f"  outliers {int(outliers.sum())}: "
            + ", ".join(f"{name} {int(n)}" for name, n in zip(NAMES, outlier_pred))
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
