"""
Tests for pointcept.utils.misc.f1_scores_from_hist — the per-class F1 / macro-F1
estimator shared by the test-time `test/f1_macro` logging and (since the H3D
GridProbe macro-F1 selection change) validation-time per-probe macro-F1.

Key property under test: the macro average is an UNMASKED mean over all class
bins (a class that is absent from the targets or never predicted contributes 0),
so validation macro-F1 and test macro-F1 are the exact same number for the same
confusion histogram.

Run with: PYTHONPATH=./ pytest tests/test_f1_scores_from_hist.py
"""

import unittest

import numpy as np

from pointcept.utils.misc import f1_scores_from_hist


def _hist(conf):
    """intersection/union/target counts from a KxK confusion matrix
    conf[t, p] = #points with true label t predicted as p."""
    conf = np.asarray(conf, dtype=np.float64)
    tp = np.diag(conf)
    target = conf.sum(axis=1)  # per true class
    pred = conf.sum(axis=0)  # per predicted class
    union = target + pred - tp
    return tp, union, target


class TestF1ScoresFromHist(unittest.TestCase):
    def test_matches_hand_computed_three_class(self):
        # conf[t][p] = #(true t, pred p)
        conf = np.array(
            [
                [8, 1, 1],
                [2, 5, 3],
                [1, 0, 9],
            ],
            dtype=np.float64,
        )
        tp, union, target = _hist(conf)
        f1, macro = f1_scores_from_hist(tp, union, target)

        # precision = TP / col-sum, recall = TP / row-sum, F1 = 2PR/(P+R)
        diag = np.diag(conf)
        precision = diag / conf.sum(axis=0)
        recall = diag / conf.sum(axis=1)
        expected = 2 * precision * recall / (precision + recall)

        np.testing.assert_allclose(f1, expected, rtol=1e-9)
        self.assertAlmostEqual(macro, float(expected.mean()), places=12)

    def test_absent_and_never_predicted_class_contributes_zero_to_mean(self):
        # 3 classes but class 2 has no targets and is never predicted.
        conf = [
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 0],
        ]
        tp, union, target = _hist(conf)
        f1, macro = f1_scores_from_hist(tp, union, target)
        # class 0 and 1 perfect, class 2 -> 0 (division guarded to 0), included.
        np.testing.assert_allclose(f1, [1.0, 1.0, 0.0])
        self.assertAlmostEqual(macro, 2.0 / 3.0, places=12)

    def test_perfect_prediction(self):
        conf = np.diag([7, 3, 11, 5])
        tp, union, target = _hist(conf)
        f1, macro = f1_scores_from_hist(tp, union, target)
        np.testing.assert_allclose(f1, np.ones(4))
        self.assertEqual(macro, 1.0)

    def test_all_wrong_prediction(self):
        # every point of class i predicted as class (i+1) % 3
        conf = [
            [0, 6, 0],
            [0, 0, 6],
            [6, 0, 0],
        ]
        tp, union, target = _hist(conf)
        f1, macro = f1_scores_from_hist(tp, union, target)
        np.testing.assert_allclose(f1, np.zeros(3))
        self.assertEqual(macro, 0.0)


if __name__ == "__main__":
    unittest.main()
