"""
Tests for binary_prf_counts (pointcept.utils.misc), the pure tp/fp/fn helper used
by MultiTaskTester to compute test-set precision/recall/F1 for pixel_semantic
tasks (e.g. forest_2d) from a scene's dense prediction/target grids.

Run with: PYTHONPATH=./ pytest tests/test_pixel_semantic_test_prf.py
"""

import unittest

import numpy as np

from pointcept.utils.misc import binary_prf_counts


class TestBinaryPrfCounts(unittest.TestCase):
    def test_simple_confusion_counts(self):
        # 2x2 grid. fg_idx=1, ignore_index=2.
        # (0,0): pred fg, gt fg -> TP
        # (0,1): pred fg, gt not-fg -> FP
        # (1,0): pred not-fg, gt fg -> FN
        # (1,1): pred not-fg, gt not-fg -> TN (not counted)
        prob = np.array([[0.9, 0.8], [0.1, 0.2]])
        target = np.array([[1, 0], [1, 0]])
        tp, fp, fn = binary_prf_counts(prob, target, ignore_index=2, fg_idx=1)
        self.assertEqual((tp, fp, fn), (1, 1, 1))

    def test_unobserved_pixels_excluded(self):
        prob = np.array([[0.9, np.nan]])
        target = np.array([[1, 1]])
        tp, fp, fn = binary_prf_counts(prob, target, ignore_index=2, fg_idx=1)
        self.assertEqual((tp, fp, fn), (1, 0, 0))

    def test_void_pixels_excluded(self):
        prob = np.array([[0.9, 0.9]])
        target = np.array([[1, 2]])  # second pixel is Void (ignore_index=2)
        tp, fp, fn = binary_prf_counts(prob, target, ignore_index=2, fg_idx=1)
        self.assertEqual((tp, fp, fn), (1, 0, 0))

    def test_all_excluded_gives_zero_counts(self):
        prob = np.full((2, 2), np.nan)
        target = np.full((2, 2), 2)
        tp, fp, fn = binary_prf_counts(prob, target, ignore_index=2, fg_idx=1)
        self.assertEqual((tp, fp, fn), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
