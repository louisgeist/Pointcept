"""Tests for MultiTaskTester nathab point/tile prediction dumps.

Run with: PYTHONPATH=./ pytest tests/test_nathab_pred_dump.py
"""

import os
import tempfile
import unittest

import numpy as np
import torch


class TestFinalizeAndSaveTileDistribution(unittest.TestCase):
    def setUp(self):
        from pointcept.engines.test import MultiTaskTester

        self.fn = MultiTaskTester._finalize_and_save_tile_distribution
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.paths = dict(
            pred=os.path.join(self._tmpdir.name, "pred.npy"),
            tile=os.path.join(self._tmpdir.name, "tile.npy"),
            dist=os.path.join(self._tmpdir.name, "dist.npy"),
        )

    def test_point_argmax_and_tile_broadcast(self):
        avg_probs = torch.tensor(
            [
                [0.7, 0.2, 0.1],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
            ],
            dtype=torch.float32,
        )
        target = np.array([0, 0, 1], dtype=np.int64)
        pred, tile, dist, metrics = self.fn(
            avg_probs, target, ignore_index=-1, num_classes=3, cache_paths=self.paths
        )
        np.testing.assert_array_equal(pred, np.array([0, 0, 1], dtype=np.int32))
        self.assertEqual(tile.shape, (3,))
        self.assertEqual(len(np.unique(tile)), 1)
        self.assertEqual(dist.shape, (3,))
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics["weight"], 0)
        loaded_pred = np.load(self.paths["pred"])
        loaded_tile = np.load(self.paths["tile"])
        loaded_dist = np.load(self.paths["dist"])
        np.testing.assert_array_equal(loaded_pred, pred)
        np.testing.assert_array_equal(loaded_tile, tile)
        np.testing.assert_allclose(loaded_dist, dist)

    def test_all_void_fills_ignore_index(self):
        avg_probs = torch.tensor([[0.2, 0.8], [0.9, 0.1]], dtype=torch.float32)
        target = np.array([-1, -1], dtype=np.int64)
        pred, tile, dist, metrics = self.fn(
            avg_probs, target, ignore_index=-1, num_classes=2, cache_paths=self.paths
        )
        self.assertIsNone(metrics)
        np.testing.assert_array_equal(tile, np.array([-1, -1], dtype=np.int32))
        self.assertEqual(dist.shape, (2,))
        np.testing.assert_array_equal(pred, np.array([1, 0], dtype=np.int32))
