"""MultiTaskSegmentorV2._pixel_pool_and_gather pooling mode: "max" (default, used by network) or "mean" (used
by forest_2d). The model is built with object.__new__ since the method reads no instance state.

Run with: PYTHONPATH=. pytest tests/test_pixel_semantic_pooling.py
"""

import unittest

import torch

from pointcept.models.default import MultiTaskSegmentorV2


def _bare_model():
    return object.__new__(MultiTaskSegmentorV2)


class TestPixelPoolAndGather(unittest.TestCase):
    def setUp(self):
        self.model = _bare_model()
        # One scene, 3 points: points 0 and 1 share cell (0, 0); point 2 is alone
        # in cell (1, 1).
        self.feat = torch.tensor([[1.0], [3.0], [10.0]])
        self.cell = torch.tensor([[0, 0], [0, 0], [1, 1]])
        self.pix = torch.tensor([[0, 0], [0, 0], [1, 1]])
        self.offset = torch.tensor([3])
        self.point_labels = torch.tensor([[1], [1], [0]])

    def test_default_pooling_is_max_unchanged(self):
        pooled, targets, _ = self.model._pixel_pool_and_gather(
            self.feat, self.cell, self.pix, self.offset, self.point_labels,
            num_networks=1,
        )
        # Cell (0,0) max(1,3)=3; cell (1,1) is 10. Order follows torch.unique.
        pooled_sorted = sorted(pooled.flatten().tolist())
        self.assertEqual(pooled_sorted, [3.0, 10.0])

    def test_mean_pooling_averages_within_cell(self):
        pooled, targets, _ = self.model._pixel_pool_and_gather(
            self.feat, self.cell, self.pix, self.offset, self.point_labels,
            num_networks=1, pooling="mean",
        )
        pooled_sorted = sorted(pooled.flatten().tolist())
        # Cell (0,0) mean(1,3)=2; cell (1,1) is 10.
        self.assertEqual(pooled_sorted, [2.0, 10.0])

    def test_explicit_max_matches_default(self):
        pooled_default, _, _ = self.model._pixel_pool_and_gather(
            self.feat, self.cell, self.pix, self.offset, self.point_labels,
            num_networks=1,
        )
        pooled_explicit, _, _ = self.model._pixel_pool_and_gather(
            self.feat, self.cell, self.pix, self.offset, self.point_labels,
            num_networks=1, pooling="max",
        )
        torch.testing.assert_close(pooled_default, pooled_explicit)

    def test_invalid_pooling_raises(self):
        with self.assertRaises(ValueError):
            self.model._pixel_pool_and_gather(
                self.feat, self.cell, self.pix, self.offset, self.point_labels,
                num_networks=1, pooling="sum",
            )


if __name__ == "__main__":
    unittest.main()
