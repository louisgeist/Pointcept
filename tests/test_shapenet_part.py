"""ShapeNetPart port: per-shape part mIoU, category-masked argmax, and the dataset on the real data
(skipped when data/shapenetcore_partanno_segmentation_benchmark_v0_normal is absent).

Run with: PYTHONPATH=. pytest tests/test_shapenet_part.py -v
"""

import os

import numpy as np
import pytest
import torch

from pointcept.utils.misc import masked_part_argmax, shape_part_miou

DATA_ROOT = "data/shapenetcore_partanno_segmentation_benchmark_v0_normal"


def test_shape_part_miou():
    target = np.array([0, 0, 1, 1])
    # parts 2 and 3 are absent from both: they count as IoU 1
    assert shape_part_miou(target, target, [0, 1, 2, 3]) == 1.0
    pred = np.array([0, 1, 1, 1])
    # part 0: 1/2, part 1: 2/3, parts 2 and 3 absent from both: 1
    assert shape_part_miou(pred, target, [0, 1, 2, 3]) == pytest.approx(
        (0.5 + 2 / 3 + 1 + 1) / 4
    )
    pred = np.array([0, 0, 1, 2])  # part 2 predicted but absent: IoU 0
    assert shape_part_miou(pred, target, [0, 1, 2]) == pytest.approx((1 + 0.5 + 0) / 3)


def test_masked_part_argmax():
    logits = torch.tensor([[5.0, 0.0, 1.0, 2.0], [0.0, 9.0, 3.0, 1.0]])
    assert masked_part_argmax(logits, [2, 3]).tolist() == [3, 2]
    assert (
        masked_part_argmax(logits, [0, 1, 2, 3]).tolist() == logits.argmax(1).tolist()
    )


@pytest.mark.skipif(not os.path.isdir(DATA_ROOT), reason="ShapeNetPart not installed")
def test_dataset_splits_and_sample():
    from pointcept.datasets.shapenet_part import ShapeNetPartDataset

    sizes = {"train": 12137, "val": 1870, "test": 2874}
    for split, n in sizes.items():
        assert len(ShapeNetPartDataset(split=split, data_root=DATA_ROOT)) == n
    ds = ShapeNetPartDataset(split=["train", "val"], data_root=DATA_ROOT)
    assert len(ds) == sizes["train"] + sizes["val"]
    assert len(ds.categories) == 16
    sample = ds.get_data(0)
    parts = ds.category2part[ds.categories[sample["cls_token"]]]
    assert sample["coord"].shape == sample["normal"].shape
    assert sample["segment"].shape[0] == sample["coord"].shape[0]
    assert set(np.unique(sample["segment"])) <= set(parts)
    # the cache must not be modified by in-place transforms
    sample["coord"] += 1.0
    assert not np.allclose(ds.get_data(0)["coord"], sample["coord"])


def _toy_meter():
    from pointcept.utils.partseg_metrics import PartSegMetrics

    categories = ["A", "B"]
    category2part = {"A": [0, 1], "B": [2, 3]}
    return PartSegMetrics(categories, category2part, num_classes=4)


def _one_hot(pred, k=4):
    return torch.nn.functional.one_hot(torch.tensor(pred), k).float()


def test_partseg_metrics_confusion_over_points():
    meter = _toy_meter()
    # shape of category A: 4 points, one part-1 point predicted as part 2 (a B part),
    # with part 1 as its second choice
    logits = _one_hot([0, 0, 1, 2])
    logits[3, 1] = 0.5
    meter.update(logits, torch.tensor([0, 0, 1, 1]), 0)
    # shape of category B: 2 points, all right
    meter.update(_one_hot([2, 3]), torch.tensor([2, 3]), 1)
    m = meter.compute()
    # IoU per class over all 6 points: 0 -> 1, 1 -> 1/2, 2 -> 1/2, 3 -> 1
    assert m["mIoU"] == pytest.approx((1 + 0.5 + 0.5 + 1) / 4)
    assert m["allAcc"] == pytest.approx(5 / 6)
    assert m["mAcc"] == pytest.approx((1 + 0.5 + 1 + 1) / 4)
    # per-shape protocol: shape A plain argmax mIoU = (1 + 1/2) / 2, masked = 1
    assert m["ins_mIoU"] == pytest.approx((0.75 + 1) / 2)
    assert m["ins_mIoU_catmask"] == pytest.approx(1.0)


def test_update_partseg_metrics_batch_with_inverse():
    from pointcept.engines.hooks.evaluator import update_partseg_metrics

    meter = _toy_meter()
    # two shapes collated: 2 + 1 voxels, 4 + 2 original points
    logits = _one_hot([0, 1, 3])
    input_dict = dict(
        offset=torch.tensor([2, 3]),
        origin_offset=torch.tensor([4, 6]),
        inverse=torch.tensor([0, 0, 1, 1, 0, 0]),
        origin_segment=torch.tensor([0, 0, 1, 1, 3, 2]),
        segment=torch.tensor([0, 1, 3]),
        cls_token=torch.tensor([0, 1]),
    )
    update_partseg_metrics(meter, logits, input_dict)
    m = meter.compute()
    assert meter.target.sum() == 6
    # class 2: one target point, predicted 3 -> IoU 0; class 3: IoU 1/2
    assert m["mIoU"] == pytest.approx((1 + 1 + 0 + 0.5) / 4)
    assert m["allAcc"] == pytest.approx(5 / 6)
