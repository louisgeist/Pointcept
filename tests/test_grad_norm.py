"""
Tests for real GradNorm (Chen et al. 2018) — pointcept/utils/gradient_norm.py.

Covers the GradNormState update math (sum-to-T renorm, restoring force toward a
common training rate), the checkpoint round-trip (weights + Adam state + L_g(0)
anchors), the group-name derivation in build_grad_norm_state, and the mutual
exclusion with grad_norm_lite in default_config_parser.

Run with: PYTHONPATH=./ pytest tests/test_grad_norm.py
"""

import math
import unittest

import torch

from pointcept.utils.gradient_norm import (
    GradNormState,
    build_grad_norm_state,
    group_task_losses,
)


class TestGradNormStateUpdate(unittest.TestCase):
    def _state(self, **kw):
        kw.setdefault("group_names", ["a", "b"])
        kw.setdefault("alpha", 1.5)
        kw.setdefault("weight_lr", 0.1)
        return GradNormState(**kw)

    def test_renorm_sums_to_num_groups(self):
        st = self._state(group_names=["a", "b", "c"])
        st.capture_initial({"a": 2.0, "b": 2.0, "c": 2.0})
        for _ in range(20):
            info = st.update(
                grad_norms={"a": 3.0, "b": 1.0, "c": 0.5},
                loss_by_group={"a": 1.9, "b": 1.0, "c": 0.4},
            )
        total = sum(info["weights"].values())
        self.assertAlmostEqual(total, 3.0, places=4)
        for w in info["weights"].values():
            self.assertGreaterEqual(w, 0.0)

    def test_slow_learner_gets_more_weight(self):
        # Group "b" barely improved (loss ratio ~1) while "a" trained fast
        # (loss ratio ~0.3). GradNorm should push weight toward the slow "b".
        st = self._state()
        st.capture_initial({"a": 1.0, "b": 1.0})
        w_b = []
        for _ in range(50):
            info = st.update(
                grad_norms={"a": 1.0, "b": 1.0},
                loss_by_group={"a": 0.3, "b": 0.95},
            )
            w_b.append(info["weights"]["b"])
        self.assertGreater(w_b[-1], 1.0)
        self.assertGreater(w_b[-1], w_b[0])
        self.assertLess(info["weights"]["a"], 1.0)

    def test_equal_rates_stay_uniform(self):
        st = self._state()
        st.capture_initial({"a": 1.0, "b": 1.0})
        for _ in range(30):
            info = st.update(
                grad_norms={"a": 1.0, "b": 1.0},
                loss_by_group={"a": 0.5, "b": 0.5},
            )
        self.assertAlmostEqual(info["weights"]["a"], 1.0, places=2)
        self.assertAlmostEqual(info["weights"]["b"], 1.0, places=2)

    def test_capture_initial_is_sticky_and_guards_nonfinite(self):
        st = self._state()
        st.capture_initial({"a": 2.0, "b": float("nan")})
        st.capture_initial({"a": 9.0, "b": 3.0})
        self.assertEqual(st.L0["a"], 2.0)  # first finite value wins
        self.assertEqual(st.L0["b"], 3.0)

    def test_skips_step_when_fewer_than_two_active_groups(self):
        st = self._state()
        st.capture_initial({"a": 1.0})  # only "a" anchored
        info = st.update(grad_norms={"a": 1.0}, loss_by_group={"a": 0.5})
        self.assertTrue(math.isnan(info["grad_norm_loss"]))
        self.assertEqual(info["weights"]["a"], 1.0)
        self.assertEqual(info["weights"]["b"], 1.0)

    def test_per_task_scales_broadcasts_group_weight(self):
        st = self._state(group_names=["nathab", "segment"])
        st.weights.data.copy_(torch.tensor([1.4, 0.6]))
        groups = {
            "nathab_habitat_type": "nathab",
            "nathab_moisture_regime": "nathab",
            "segment": "segment",
        }
        scales = st.per_task_scales(list(groups.keys()), groups)
        self.assertAlmostEqual(scales["nathab_habitat_type"], 1.4, places=5)
        self.assertAlmostEqual(scales["nathab_moisture_regime"], 1.4, places=5)
        self.assertAlmostEqual(scales["segment"], 0.6, places=5)


class TestGradNormStateCheckpoint(unittest.TestCase):
    def test_state_dict_round_trip(self):
        src = GradNormState(["a", "b", "c"], alpha=1.5, weight_lr=0.05)
        src.capture_initial({"a": 1.0, "b": 2.0, "c": 3.0})
        for _ in range(5):
            src.update(
                grad_norms={"a": 2.0, "b": 1.0, "c": 0.5},
                loss_by_group={"a": 0.8, "b": 1.9, "c": 2.5},
            )
        state = src.state_dict()

        dst = GradNormState(["a", "b", "c"], alpha=1.5, weight_lr=0.05)
        dst.load_state_dict(state)

        self.assertTrue(torch.allclose(src.weights.detach(), dst.weights.detach()))
        self.assertEqual(src.L0, dst.L0)
        # Adam step counter carried over -> next update matches bit-for-bit.
        gn = {"a": 2.0, "b": 1.0, "c": 0.5}
        lg = {"a": 0.7, "b": 1.8, "c": 2.4}
        i_src = src.update(gn, lg)
        i_dst = dst.update(gn, lg)
        for g in ["a", "b", "c"]:
            self.assertAlmostEqual(i_src["weights"][g], i_dst["weights"][g], places=6)

    def test_load_state_dict_rejects_group_mismatch(self):
        src = GradNormState(["a", "b"], weight_lr=0.05)
        with self.assertRaises(ValueError):
            GradNormState(["a", "c"], weight_lr=0.05).load_state_dict(src.state_dict())


class _FakeModel:
    def __init__(self, tasks):
        self.tasks = tuple(tasks)


class _FakeCfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestBuildGradNormState(unittest.TestCase):
    def test_derives_grouped_ordered_group_names(self):
        cfg = _FakeCfg(
            grad_norm_task_groups={
                "nathab_habitat_type": "nathab",
                "nathab_moisture_regime": "nathab",
            },
            grad_norm_alpha=1.5,
            grad_norm_weight_lr=1e-2,
        )
        model = _FakeModel(
            ["segment", "forest_2d", "nathab_habitat_type", "nathab_moisture_regime"]
        )
        st = build_grad_norm_state(cfg, model, device="cpu")
        self.assertEqual(st.group_names, ["forest_2d", "nathab", "segment"])

    def test_ungrouped_is_one_group_per_task(self):
        cfg = _FakeCfg()
        st = build_grad_norm_state(cfg, _FakeModel(["a", "b", "c"]), device="cpu")
        self.assertEqual(st.group_names, ["a", "b", "c"])

    def test_raises_without_tasks(self):
        with self.assertRaises(ValueError):
            build_grad_norm_state(_FakeCfg(), _FakeModel([]), device="cpu")


class TestGroupTaskLosses(unittest.TestCase):
    def test_sums_within_group_and_skips_detached(self):
        loss_by_task = {
            "a1": torch.tensor(1.0, requires_grad=True),
            "a2": torch.tensor(2.0, requires_grad=True),
            "b": torch.tensor(5.0, requires_grad=True),
            "c": torch.tensor(9.0),  # detached -> skipped
        }
        groups = {"a1": "g", "a2": "g"}
        out = group_task_losses(loss_by_task, groups)
        self.assertAlmostEqual(out["g"], 3.0)
        self.assertAlmostEqual(out["b"], 5.0)
        self.assertNotIn("c", out)


class TestMutualExclusion(unittest.TestCase):
    def test_config_parser_rejects_both_flags(self):
        from pointcept.engines.defaults import default_config_parser

        with self.assertRaises(ValueError):
            default_config_parser(
                "configs/flair3d_default/spunet_nh_multilabel_toy.py",
                options={"grad_norm": True, "grad_norm_lite": True},
            )


if __name__ == "__main__":
    unittest.main()
