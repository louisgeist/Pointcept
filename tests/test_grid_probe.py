"""
Tests for shared unitsphere caching in GridProbeSegmentorV2 / ProbeHead, and for
GridProbeEvaluator's per-probe best-checkpoint selection metric (select_metric,
"mIoU" default or "macro_f1").

Run with: PYTHONPATH=./ pytest tests/test_grid_probe.py
"""

import unittest

import numpy as np
import torch

from pointcept.engines.hooks.grid_probe import (
    GridProbeCheckpointSaver,
    GridProbeEvaluator,
)
from pointcept.models.grid_probe import (
    ProbeHead,
    _input_norm_cache,
    _prepare_shared_feat,
    _unitsphere,
)


def _clone_head(src):
    dst = ProbeHead(
        in_channels=src.linear.in_features,
        num_classes=src.linear.out_features,
        input_norm=src.input_norm,
        eps=src.eps,
    )
    dst.load_state_dict(src.state_dict())
    return dst


class TestUnitsphere(unittest.TestCase):
    def test_none_returns_same_tensor(self):
        feat = torch.randn(4, 8)
        self.assertIs(_unitsphere(feat, None), feat)

    def test_l2_rows_are_unit_norm(self):
        feat = torch.randn(5, 7)
        out = _unitsphere(feat, "l2")
        norms = out.norm(p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


class TestInputNormCache(unittest.TestCase):
    def test_one_unitsphere_per_kind(self):
        feat = torch.randn(8, 16)
        heads = torch.nn.ModuleDict(
            {
                "a": ProbeHead(16, 3, input_norm="l2"),
                "b": ProbeHead(16, 3, input_norm="l2"),
                "c": ProbeHead(16, 3, input_norm="l2"),
                "d": ProbeHead(16, 3, input_norm=None),
            }
        )
        cache = _input_norm_cache(feat, heads, heads.keys())
        self.assertEqual(len(cache), 2)
        self.assertIs(cache[(None, heads["d"].eps)], feat)
        self.assertIs(cache[("l2", heads["a"].eps)], cache[("l2", heads["b"].eps)])
        self.assertIsNot(cache[("l2", heads["a"].eps)], feat)

    def test_prepare_shared_feat_contiguous(self):
        feat = torch.randn(4, 16)[:, ::2]  # [4, 8], not contiguous
        self.assertFalse(feat.is_contiguous())
        prepared = _prepare_shared_feat(feat)
        self.assertTrue(prepared.is_contiguous())
        self.assertEqual(prepared.shape, feat.shape)


class TestProbeHeadCacheEquivalence(unittest.TestCase):
    def test_cached_l2_matches_naive_heads(self):
        torch.manual_seed(0)
        feat = torch.randn(6, 12)
        naive_a = ProbeHead(12, 5, input_norm="l2")
        naive_b = _clone_head(naive_a)
        cached_a = _clone_head(naive_a)
        cached_b = _clone_head(naive_a)

        naive_out_a = naive_a(feat)
        naive_out_b = naive_b(feat)

        prepared = _prepare_shared_feat(feat)
        cache = _input_norm_cache(
            prepared,
            {"a": cached_a, "b": cached_b},
            ("a", "b"),
        )
        cached_out_a = cached_a(cache[("l2", cached_a.eps)], apply_input_norm=False)
        cached_out_b = cached_b(cache[("l2", cached_b.eps)], apply_input_norm=False)

        self.assertTrue(torch.allclose(cached_out_a, naive_out_a, atol=1e-6))
        self.assertTrue(torch.allclose(cached_out_b, naive_out_b, atol=1e-6))

    def test_l2_differs_from_none(self):
        torch.manual_seed(1)
        feat = torch.randn(6, 12)
        head_l2 = ProbeHead(12, 5, input_norm="l2")
        head_none = _clone_head(head_l2)
        head_none.input_norm = None
        out_l2 = head_l2(feat)
        out_none = head_none(feat)
        self.assertFalse(torch.allclose(out_l2, out_none, atol=1e-5))


def _cls_hist(iou_pair):
    """(intersection, union, target) arrays whose intersection/union == iou_pair."""
    union = np.array([10.0, 10.0])
    inter = np.array(iou_pair) * union
    return inter, union, union.copy()


class TestGridProbeSelectMetric(unittest.TestCase):
    """GridProbeEvaluator._update_bests + the selection helpers that
    GridProbeCheckpointSaver / GridProbeWinnerSelector read."""

    # per-epoch synthetic per-probe metrics: probe A peaks in mIoU at epoch 1
    # but in macro-F1 at (a later) epoch; probe B the other way around, and B's
    # macro-F1 ties its running best on epoch 3.
    EPOCHS = [
        # (m_iou, macro_f1, cls_iou_snapshot)
        {"A": (0.60, 0.50, [0.1, 0.1]), "B": (0.30, 0.90, [0.2, 0.2])},
        {"A": (0.40, 0.55, [0.9, 0.9]), "B": (0.50, 0.60, [0.3, 0.3])},
        {"A": (0.35, 0.52, [0.7, 0.7]), "B": (0.45, 0.90, [0.4, 0.4])},
    ]

    def _run(self, select_metric):
        hook = GridProbeEvaluator(select_metric=select_metric)
        for ep in self.EPOCHS:
            m_iou = {k: v[0] for k, v in ep.items()}
            f1 = {k: v[1] for k, v in ep.items()}
            f1_cls = {k: np.array([v[1], v[1]]) for k, v in ep.items()}
            cls_hist = {k: _cls_hist(v[2]) for k, v in ep.items()}
            hook._last_miou_by_probe = dict(m_iou)
            hook._last_f1_by_probe = dict(f1)
            hook._update_bests(m_iou, f1, f1_cls, cls_hist)
        return hook

    def test_running_bests_are_metric_maxes_regardless_of_select_metric(self):
        for sm in ("mIoU", "macro_f1"):
            hook = self._run(sm)
            self.assertAlmostEqual(hook._best_miou_by_probe["A"], 0.60)
            self.assertAlmostEqual(hook._best_miou_by_probe["B"], 0.50)
            self.assertAlmostEqual(hook._best_f1_by_probe["A"], 0.55)
            self.assertAlmostEqual(hook._best_f1_by_probe["B"], 0.90)

    def test_winner_depends_on_select_metric(self):
        iou_hook = self._run("mIoU")
        sel = iou_hook._selected_best_by_probe()
        self.assertEqual(max(sel, key=sel.get), "A")  # 0.60 > 0.50

        f1_hook = self._run("macro_f1")
        sel = f1_hook._selected_best_by_probe()
        self.assertEqual(max(sel, key=sel.get), "B")  # 0.90 > 0.55

    def test_per_class_snapshot_taken_at_selected_metric_best_epoch(self):
        # mIoU selection: A's best mIoU is epoch 1 -> snapshot [0.1, 0.1]
        iou_hook = self._run("mIoU")
        np.testing.assert_allclose(iou_hook._best_cls_iou_by_probe["A"], [0.1, 0.1])
        # macro-F1 selection: A's best F1 is epoch 2 -> snapshot [0.9, 0.9]
        f1_hook = self._run("macro_f1")
        np.testing.assert_allclose(f1_hook._best_cls_iou_by_probe["A"], [0.9, 0.9])

    def test_ge_tie_break_latest_epoch_wins(self):
        # B's macro-F1 is 0.90 at epoch 1 and again (tie) at epoch 3; the
        # >= rule must move the snapshot to epoch 3's hist ([0.4, 0.4]).
        f1_hook = self._run("macro_f1")
        np.testing.assert_allclose(f1_hook._best_cls_iou_by_probe["B"], [0.4, 0.4])

    def test_checkpointsaver_save_set_matches_snapshot_set(self):
        # The probes GridProbeCheckpointSaver would save this epoch (value
        # >= running best of the selected metric) must be exactly the probes
        # whose per-class snapshot _update_bests just (re)took.
        for sm in ("mIoU", "macro_f1"):
            hook = GridProbeEvaluator(select_metric=sm)
            # epoch 1
            ep = self.EPOCHS[0]
            m_iou = {k: v[0] for k, v in ep.items()}
            f1 = {k: v[1] for k, v in ep.items()}
            hook._last_miou_by_probe, hook._last_f1_by_probe = dict(m_iou), dict(f1)
            hook._update_bests(
                m_iou, f1,
                {k: np.array([v[1], v[1]]) for k, v in ep.items()},
                {k: _cls_hist(v[2]) for k, v in ep.items()},
            )
            cls_snapshot_keys = set(hook._best_cls_iou_by_probe)

            last = hook._selected_last_by_probe()
            best = hook._selected_best_by_probe()
            saver_keys = {
                name for name, val in last.items()
                if best.get(name) is not None and not (val < best[name])
            }
            self.assertEqual(saver_keys, cls_snapshot_keys)
        # smoke: the class is importable/usable
        self.assertTrue(hasattr(GridProbeCheckpointSaver, "after_epoch"))


class TestGridProbeHistoryCsvFieldnames(unittest.TestCase):
    def test_fieldnames_order_is_append_only(self):
        from pointcept.engines.hooks.grid_probe import _HISTORY_CSV_FIELDNAMES

        self.assertEqual(
            _HISTORY_CSV_FIELDNAMES,
            ("epoch", "probe_name", "mIoU", "mIoU_best", "f1_macro", "f1_macro_best"),
        )

    def test_short_row_without_f1_keys_does_not_raise(self):
        import tempfile

        from pointcept.engines.hooks.grid_probe import _atomic_write_history_csv

        with tempfile.NamedTemporaryFile(suffix=".csv") as f:
            _atomic_write_history_csv(
                f.name,
                [
                    {"epoch": 1, "probe_name": "p0", "mIoU": 0.5, "mIoU_best": 0.5},
                    {
                        "epoch": 2,
                        "probe_name": "p0",
                        "mIoU": 0.6,
                        "mIoU_best": 0.6,
                        "f1_macro": 0.55,
                        "f1_macro_best": 0.55,
                    },
                ],
            )
            with open(f.name) as fh:
                header = fh.readline().strip()
        self.assertEqual(
            header, "epoch,probe_name,mIoU,mIoU_best,f1_macro,f1_macro_best"
        )


if __name__ == "__main__":
    unittest.main()
