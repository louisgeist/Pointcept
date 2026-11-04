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
