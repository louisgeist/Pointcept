"""
Tests for the tile_distribution task's shared pooling/KL utilities
(pointcept.utils.misc.pool_axis_distribution_from_probs / kl_divergence_rows),
the WeightedKLDivLoss criterion, and MultiTaskSegmentorV2's tile_distribution
loss branch.

Run with: PYTHONPATH=./ pytest tests/test_tile_distribution_pooling.py
"""

import unittest

import torch

from pointcept.utils.misc import kl_divergence_rows, pool_axis_distribution_from_probs
from pointcept.models.losses import build_criteria
from pointcept.models.default import MultiTaskSegmentorV2


class TestPoolAxisDistributionFromProbs(unittest.TestCase):
    def test_basic_pooling_excludes_void_points(self):
        # 2 tiles, 5 points, 3 classes, ignore_index=-1.
        probs = torch.tensor(
            [
                [0.7, 0.2, 0.1],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.0, 0.0, 0.0],  # void point (target=-1)
                [0.2, 0.2, 0.6],
            ],
            dtype=torch.float32,
        )
        target = torch.tensor([0, 0, 1, -1, 2], dtype=torch.long)
        indptr = torch.tensor([0, 3, 5])  # tile0: pts 0-2, tile1: pts 3-4

        pi_hat, q_t, n_t = pool_axis_distribution_from_probs(
            probs, target, indptr, ignore_index=-1, num_classes=3
        )

        torch.testing.assert_close(n_t, torch.tensor([3.0, 1.0]))
        torch.testing.assert_close(
            q_t, torch.tensor([[2 / 3, 1 / 3, 0.0], [0.0, 0.0, 1.0]])
        )
        expected_pi_hat_0 = probs[0:3].mean(dim=0)
        torch.testing.assert_close(pi_hat[0], expected_pi_hat_0)
        torch.testing.assert_close(pi_hat[1], probs[4])

    def test_tile_with_zero_valid_points_has_zero_count(self):
        probs = torch.zeros(2, 3)
        target = torch.tensor([-1, -1], dtype=torch.long)
        indptr = torch.tensor([0, 2])
        _, _, n_t = pool_axis_distribution_from_probs(
            probs, target, indptr, ignore_index=-1, num_classes=3
        )
        self.assertEqual(float(n_t.item()), 0.0)


class TestKlDivergenceRows(unittest.TestCase):
    def test_zero_when_distributions_match(self):
        q = torch.tensor([[0.5, 0.5]])
        p = torch.tensor([[0.5, 0.5]])
        kl = kl_divergence_rows(q, p)
        self.assertAlmostEqual(float(kl.item()), 0.0, places=5)

    def test_positive_when_distributions_differ(self):
        q = torch.tensor([[1.0, 0.0]])
        p = torch.tensor([[0.5, 0.5]])
        kl = kl_divergence_rows(q, p)
        self.assertGreater(float(kl.item()), 0.0)


class TestWeightedKLDivLoss(unittest.TestCase):
    def test_weighted_average_matches_manual_computation(self):
        criteria = build_criteria([dict(type="WeightedKLDivLoss")])
        pred = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
        target = torch.tensor([[0.6, 0.3, 0.1], [0.0, 1.0, 0.0]])
        weight = torch.tensor([3.0, 1.0])
        loss = criteria(pred, target, weight=weight)
        kl = kl_divergence_rows(target, pred)
        expected = (weight * kl).sum() / weight.sum()
        torch.testing.assert_close(loss, expected)


class TestMultiTaskSegmentorV2TileDistributionLoss(unittest.TestCase):
    def _make_model_stub(self, num_classes=3):
        model = object.__new__(MultiTaskSegmentorV2)
        task_config = dict(task_type="tile_distribution", num_classes=num_classes, ignore_index=-1)
        model.task_configs = {"axis_a": task_config}
        model.tasks = ("axis_a",)
        model.criteria_by_task = {"axis_a": build_criteria([dict(type="WeightedKLDivLoss")])}
        model.task_weights = {"axis_a": 1.0}
        return model

    def test_loss_computed_and_differentiable(self):
        model = self._make_model_stub()
        logits = torch.tensor(
            [
                [2.0, 0.5, 0.1],
                [1.8, 0.6, 0.1],
                [0.2, 2.0, 0.1],
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 2.0],
            ],
            requires_grad=True,
        )
        target = torch.tensor([0, 0, 1, -1, 2], dtype=torch.long)
        offset = torch.tensor([3, 5])
        input_dict = {"axis_a": target, "offset": offset}
        total_loss, loss_by_task = model._compute_loss(
            {"axis_a": logits}, {}, {}, input_dict
        )
        self.assertIn("axis_a", loss_by_task)
        self.assertTrue(torch.isfinite(total_loss))
        total_loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_all_void_batch_yields_zero_loss_with_live_grad_graph(self):
        model = self._make_model_stub()
        logits = torch.zeros(5, 3, requires_grad=True)
        target = torch.full((5,), -1, dtype=torch.long)
        offset = torch.tensor([3, 5])
        input_dict = {"axis_a": target, "offset": offset}
        total_loss, _ = model._compute_loss({"axis_a": logits}, {}, {}, input_dict)
        self.assertEqual(float(total_loss.item()), 0.0)
        total_loss.backward()  # must not raise (grad graph kept alive for DDP)
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
