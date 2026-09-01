"""
Tests for binary_prf_counts (pointcept.utils.misc) and the dilated P/R/F1 path
used by MultiTaskTester for pixel_semantic tasks (e.g. network) from a scene's
dense prediction/target grids.

Run with: PYTHONPATH=./ pytest tests/test_pixel_semantic_test_prf.py
"""

import unittest

import numpy as np

from pointcept.datasets.malibu3d_config_utils import get_pixel_semantic_config
from pointcept.utils.dilated_metrics import (
    dilated_prf_enabled,
    dilated_precision_recall_counts,
    precision_recall_f1,
)
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


def _dilated_counts_from_dense(prob, target, ignore_index, fg_idx, radius_px):
    """Same 0.5-threshold masks as MultiTaskTester's dilated pixel_semantic path."""
    valid = np.isfinite(prob) & (target != ignore_index)
    pred_fg = (prob > 0.5) & valid
    gt_fg = (target == fg_idx) & valid
    return dilated_precision_recall_counts(
        pred_fg, gt_fg, valid, radius_px=radius_px
    )


class TestDilatedPrfFromDense(unittest.TestCase):
    def test_one_pixel_offset_kills_exact_f1_but_dilated_hits(self):
        # 3x3 grid: GT foreground at (1, 0), prediction shifted 1 px right to (1, 1).
        target = np.zeros((3, 3), dtype=np.int64)
        target[1, 0] = 1
        prob = np.full((3, 3), 0.1)
        prob[1, 1] = 0.9

        tp, fp, fn = binary_prf_counts(prob, target, ignore_index=2, fg_idx=1)
        exact_precision, exact_recall, exact_f1 = precision_recall_f1(
            tp, tp + fp, tp, tp + fn
        )
        self.assertEqual((tp, fp, fn), (0, 1, 1))
        self.assertAlmostEqual(exact_f1, 0.0)

        p_num, p_denom, r_num, r_denom = _dilated_counts_from_dense(
            prob, target, ignore_index=2, fg_idx=1, radius_px=1
        )
        dilated_precision, dilated_recall, dilated_f1 = precision_recall_f1(
            p_num, p_denom, r_num, r_denom
        )
        self.assertEqual((p_num, p_denom, r_num, r_denom), (1.0, 1.0, 1.0, 1.0))
        self.assertAlmostEqual(dilated_precision, 1.0)
        self.assertAlmostEqual(dilated_recall, 1.0)
        self.assertAlmostEqual(dilated_f1, 1.0)

    def test_unobserved_and_void_excluded_from_dilated_counts(self):
        # (0,0) observed hit; (0,1) unobserved; (1,0) void; (1,1) observed miss.
        prob = np.array([[0.9, np.nan], [0.9, 0.9]])
        target = np.array([[1, 1], [2, 0]])
        p_num, p_denom, r_num, r_denom = _dilated_counts_from_dense(
            prob, target, ignore_index=2, fg_idx=1, radius_px=1
        )
        # Only (0,0) is a valid GT fg pixel; (1,1) is a valid pred fg pixel but
        # lies next to the GT hit so dilated precision still counts it.
        self.assertEqual(r_denom, 1.0)
        self.assertEqual(p_denom, 2.0)

    def test_network_enables_dilated_prf_forest_2d_opts_out(self):
        self.assertTrue(dilated_prf_enabled(get_pixel_semantic_config("network")))
        self.assertFalse(dilated_prf_enabled(get_pixel_semantic_config("forest_2d")))


if __name__ == "__main__":
    unittest.main()
