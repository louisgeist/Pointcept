"""
Tests for the HookBase.state_dict()/load_state_dict() checkpoint-resume contract.

Several evaluator hooks (ClsEvaluator, RegressionEvaluator, MultiTaskEvaluator) track
a "best metric so far" in a plain instance attribute purely for W&B/TensorBoard
display (checkpoint *selection* itself only ever depends on trainer.best_metric_value,
which CheckpointSaver/CheckpointLoader already persist). Before this state_dict hook
was added, those instance attributes were not persisted across a resume and silently
reset to their initial sentinel (-inf / {}), so a "best so far" curve — which should be
a monotonic running max by construction — could visibly dip right after a resume.

Run with: PYTHONPATH=./ pytest tests/test_hook_resume_state.py
"""

import unittest

from pointcept.engines.hooks.default import HookBase
from pointcept.engines.hooks.evaluator import (
    ClsEvaluator,
    MultiTaskEvaluator,
    RegressionEvaluator,
)
from pointcept.engines.hooks.grid_probe import GridProbeEvaluator


class TestHookBaseDefaults(unittest.TestCase):
    def test_default_state_dict_is_empty(self):
        self.assertEqual(HookBase().state_dict(), {})

    def test_default_load_state_dict_is_noop(self):
        hook = HookBase()
        hook.load_state_dict({"anything": 1})  # must not raise


class TestClsEvaluatorResumeState(unittest.TestCase):
    def test_round_trips_best_m_iou(self):
        hook = ClsEvaluator(metric="mIoU")
        hook._best_m_iou = 0.42

        state = hook.state_dict()
        self.assertEqual(state, {"best_m_iou": 0.42})

        fresh = ClsEvaluator(metric="mIoU")
        fresh.load_state_dict(state)
        self.assertEqual(fresh._best_m_iou, 0.42)

    def test_before_train_resets_on_fresh_run_not_on_resume(self):
        class _FakeCfg:
            resume = False

        class _FakeTrainer:
            cfg = _FakeCfg()

        hook = ClsEvaluator(metric="mIoU")
        hook._best_m_iou = 0.9
        hook.trainer = _FakeTrainer()

        # Fresh run: before_train wipes any leftover value back to -inf.
        hook.trainer.cfg.resume = False
        hook.before_train()
        self.assertEqual(hook._best_m_iou, float("-inf"))

        # Resume: CheckpointLoader.before_train (registered before evaluator hooks)
        # already restored the value; before_train must not clobber it.
        hook._best_m_iou = 0.9
        hook.trainer.cfg.resume = True
        hook.before_train()
        self.assertEqual(hook._best_m_iou, 0.9)


class TestRegressionEvaluatorResumeState(unittest.TestCase):
    def test_round_trips_best_neg_rmse(self):
        hook = RegressionEvaluator()
        hook._best_neg_rmse = -0.15

        state = hook.state_dict()
        self.assertEqual(state, {"best_neg_rmse": -0.15})

        fresh = RegressionEvaluator()
        fresh.load_state_dict(state)
        self.assertEqual(fresh._best_neg_rmse, -0.15)

    def test_load_state_dict_ignores_missing_key(self):
        hook = RegressionEvaluator()
        hook._best_neg_rmse = -0.15
        hook.load_state_dict({})
        self.assertEqual(hook._best_neg_rmse, -0.15)


class TestMultiTaskEvaluatorResumeState(unittest.TestCase):
    def test_round_trips_per_task_trackers(self):
        hook = MultiTaskEvaluator()
        hook._best_neg_rmse = -0.3
        hook._best_miou_by_task = {"segment": 0.55, "network": 0.61}
        hook._best_neg_kl_by_task = {"nathab_moisture_regime": -0.02}

        state = hook.state_dict()
        self.assertEqual(
            state,
            {
                "best_neg_rmse": -0.3,
                "best_miou_by_task": {"segment": 0.55, "network": 0.61},
                "best_neg_kl_by_task": {"nathab_moisture_regime": -0.02},
            },
        )

        fresh = MultiTaskEvaluator()
        fresh.load_state_dict(state)
        self.assertEqual(fresh._best_neg_rmse, -0.3)
        self.assertEqual(fresh._best_miou_by_task, {"segment": 0.55, "network": 0.61})
        self.assertEqual(
            fresh._best_neg_kl_by_task, {"nathab_moisture_regime": -0.02}
        )
        # Mutating the restored dict must not alias the saved state.
        fresh._best_miou_by_task["segment"] = 0.99
        self.assertEqual(state["best_miou_by_task"]["segment"], 0.55)

    def test_load_state_dict_defaults_missing_tasks_to_empty(self):
        hook = MultiTaskEvaluator()
        hook.load_state_dict({"best_neg_rmse": -0.1})
        self.assertEqual(hook._best_neg_rmse, -0.1)
        self.assertEqual(hook._best_miou_by_task, {})
        self.assertEqual(hook._best_neg_kl_by_task, {})


class TestGridProbeEvaluatorResumeState(unittest.TestCase):
    """GridProbeEvaluator tracks per-probe running-best mIoU AND macro-F1, plus
    per-class snapshots, all of which must survive a resume. It additionally
    hard-fails if select_metric is inconsistent with the checkpoint's metric,
    because trainer.best_metric_value (restored one hook earlier) is
    metric-specific and cannot be reinterpreted."""

    def test_rejects_unknown_select_metric(self):
        with self.assertRaises(AssertionError):
            GridProbeEvaluator(select_metric="bogus")

    def test_round_trips_all_trackers(self):
        hook = GridProbeEvaluator(select_metric="macro_f1")
        hook._best_miou_by_probe = {"p0": 0.51, "p1": 0.60}
        hook._best_f1_by_probe = {"p0": 0.63, "p1": 0.71}
        hook._best_cls_iou_by_probe = {"p0": [0.4, 0.5]}
        hook._best_cls_acc_by_probe = {"p0": [0.6, 0.7]}
        hook._best_cls_f1_by_probe = {"p0": [0.55, 0.6]}
        hook._history = [{"epoch": 1, "probe_name": "p0", "mIoU": 0.5}]

        state = hook.state_dict()
        self.assertEqual(state["select_metric"], "macro_f1")

        fresh = GridProbeEvaluator(select_metric="macro_f1")
        fresh.load_state_dict(state)
        self.assertEqual(fresh._best_miou_by_probe, {"p0": 0.51, "p1": 0.60})
        self.assertEqual(fresh._best_f1_by_probe, {"p0": 0.63, "p1": 0.71})
        self.assertEqual(fresh._best_cls_f1_by_probe, {"p0": [0.55, 0.6]})
        self.assertEqual(fresh._history, state["history"])
        # restored containers must not alias the saved state
        fresh._best_f1_by_probe["p0"] = 0.99
        self.assertEqual(state["best_f1_by_probe"]["p0"], 0.63)

    def test_selected_dict_helpers_follow_select_metric(self):
        f1_hook = GridProbeEvaluator(select_metric="macro_f1")
        f1_hook._best_f1_by_probe = {"p0": 0.7}
        f1_hook._best_miou_by_probe = {"p0": 0.5}
        f1_hook._last_f1_by_probe = {"p0": 0.68}
        f1_hook._last_miou_by_probe = {"p0": 0.49}
        self.assertIs(f1_hook._selected_best_by_probe(), f1_hook._best_f1_by_probe)
        self.assertIs(f1_hook._selected_last_by_probe(), f1_hook._last_f1_by_probe)

        iou_hook = GridProbeEvaluator()  # default "mIoU"
        self.assertIs(iou_hook._selected_best_by_probe(), iou_hook._best_miou_by_probe)
        self.assertIs(iou_hook._selected_last_by_probe(), iou_hook._last_miou_by_probe)

    def test_resume_pre_feature_checkpoint_with_miou_selection_ok(self):
        # Old checkpoint: no "select_metric" key, no F1 dicts.
        old_state = {
            "best_miou_by_probe": {"p0": 0.5},
            "best_cls_iou_by_probe": {},
            "best_cls_acc_by_probe": {},
            "history": [],
        }
        hook = GridProbeEvaluator()  # select_metric="mIoU"
        hook.load_state_dict(old_state)
        self.assertEqual(hook._best_miou_by_probe, {"p0": 0.5})
        self.assertEqual(hook._best_f1_by_probe, {})

    def test_resume_pre_feature_checkpoint_with_f1_selection_raises(self):
        old_state = {"best_miou_by_probe": {"p0": 0.5}, "history": []}
        hook = GridProbeEvaluator(select_metric="macro_f1")
        with self.assertRaises(RuntimeError):
            hook.load_state_dict(old_state)

    def test_resume_with_changed_select_metric_raises(self):
        state = GridProbeEvaluator(select_metric="mIoU").state_dict()
        hook = GridProbeEvaluator(select_metric="macro_f1")
        with self.assertRaises(RuntimeError):
            hook.load_state_dict(state)


class TestCheckpointSaverHookStateCollection(unittest.TestCase):
    """Mirrors the small collection loop in CheckpointSaver.after_epoch /
    CheckpointLoader.before_train without needing a full Trainer."""

    def test_only_hooks_with_nonempty_state_are_collected(self):
        evaluator = MultiTaskEvaluator()
        evaluator._best_miou_by_task = {"segment": 0.5}
        hooks = [HookBase(), evaluator]

        hook_states = {}
        for h in hooks:
            s = h.state_dict()
            if s:
                hook_states[h.__class__.__name__] = s

        self.assertEqual(list(hook_states.keys()), ["MultiTaskEvaluator"])
        self.assertEqual(hook_states["MultiTaskEvaluator"]["best_miou_by_task"], {"segment": 0.5})

    def test_restore_loop_only_touches_matching_hook_classes(self):
        evaluator = MultiTaskEvaluator()
        other = HookBase()
        hooks = [other, evaluator]
        hook_states = {
            "MultiTaskEvaluator": {"best_miou_by_task": {"network": 0.7}},
            "SomeUnrelatedHook": {"stale": True},
        }

        for h in hooks:
            state = hook_states.get(h.__class__.__name__)
            if state:
                h.load_state_dict(state)

        self.assertEqual(evaluator._best_miou_by_task, {"network": 0.7})


if __name__ == "__main__":
    unittest.main()
