"""
Regression tests for the fixes from docs/bug_review/ (2026-09-27 review):

- CrossEntropyLoss on an all-ignore batch must return an in-graph zero
  (backward() must not raise, head gets zero grads).
- RemapSegment must map from the ORIGINAL labels (no cascading through
  chained / swapped mappings).
- OpenGF / ECLAIR preprocessing must recenter absolute projected coordinates
  in float64 before the float32 cast (no 0.125 m / 0.5 m XY lattice).

Run with: PYTHONPATH=./ python -m unittest tests/test_bug_review_2026_09_27.py
"""

import importlib.util
import os
import tempfile
import unittest

import numpy as np
import torch

from pointcept.models.losses.misc import CrossEntropyLoss
from pointcept.datasets.transform import RemapSegment

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, rel_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCrossEntropyAllIgnore(unittest.TestCase):
    def test_all_ignore_is_in_graph_zero(self):
        head = torch.nn.Linear(4, 3)
        logits = head(torch.randn(5, 4))
        target = torch.full((5,), 2, dtype=torch.long)
        loss = CrossEntropyLoss(ignore_index=2)(logits, target)
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(loss.requires_grad)
        loss.backward()  # used to raise: element 0 of tensors does not require grad
        self.assertTrue(torch.all(head.weight.grad == 0))

    def test_regular_batch_unchanged(self):
        logits = torch.randn(5, 3, requires_grad=True)
        target = torch.tensor([0, 1, 2, 2, 0])
        expected = torch.nn.functional.cross_entropy(logits, target, ignore_index=2)
        loss = CrossEntropyLoss(ignore_index=2)(logits, target)
        self.assertAlmostEqual(float(loss), float(expected), places=6)


class TestRemapSegment(unittest.TestCase):
    def _apply(self, mapping, segment):
        out = RemapSegment(mapping)({"segment": np.asarray(segment, dtype=np.int32)})
        return out["segment"].tolist()

    def test_single_merge(self):
        self.assertEqual(self._apply({2: 1}, [0, 1, 2, 2]), [0, 1, 1, 1])

    def test_swap(self):
        self.assertEqual(self._apply({0: 1, 1: 0}, [0, 1, 2]), [1, 0, 2])

    def test_chain_does_not_cascade(self):
        self.assertEqual(self._apply({1: 2, 2: 3}, [1, 2, 3]), [2, 3, 3])


def _write_las(path, xyz, extra):
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.floor(xyz.min(axis=0))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for key, value in extra.items():
        setattr(las, key, value)
    las.write(path)


@unittest.skipIf(importlib.util.find_spec("laspy") is None, "laspy not installed")
class TestProjectedCoordPrecision(unittest.TestCase):
    # Lambert-93-like magnitudes: float32 spacing is 0.125 m (x) / 0.5 m (y).
    ORIGIN = np.array([1_270_000.0, 5_014_000.0, 400.0])

    def _xyz(self, n=2000):
        rng = np.random.default_rng(0)
        local = np.round(rng.uniform([0, 0, 0], [50, 50, 10], size=(n, 3)), 2)
        return self.ORIGIN + local, local

    def _assert_centimetric(self, coord, translation, local):
        # Sub-lattice fractional parts survive (0.5 m lattice would give only {0, .5}).
        self.assertGreater(len(np.unique(np.round(np.mod(coord[:, 1], 1.0), 2))), 10)
        recon = coord.astype(np.float64) + np.asarray(translation)
        np.testing.assert_allclose(recon, self.ORIGIN + local, atol=0.006)

    def test_opengf_build_scene(self):
        mod = _load_module(
            "preprocess_opengf_under_test",
            "pointcept/datasets/preprocessing/opengf/preprocess_opengf.py",
        )
        xyz, local = self._xyz()
        n = xyz.shape[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tile.las")
            _write_las(
                path,
                xyz,
                dict(
                    classification=np.full(n, 2, dtype=np.uint8),
                    intensity=np.ones(n, dtype=np.uint16),
                ),
            )
            scene, translation = mod.build_scene(path)
        self.assertEqual(scene["coord"].dtype, np.float32)
        self._assert_centimetric(scene["coord"], translation, local)

    def test_eclair_build_scene(self):
        mod = _load_module(
            "preprocess_eclair_under_test",
            "pointcept/datasets/preprocessing/eclair/preprocess_eclair.py",
        )
        xyz, local = self._xyz()
        n = xyz.shape[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tile.las")
            _write_las(
                path,
                xyz,
                dict(
                    classification=np.full(n, mod.RAW_MIN, dtype=np.uint8),
                    intensity=np.ones(n, dtype=np.uint16),
                    red=np.zeros(n, dtype=np.uint16),
                    green=np.zeros(n, dtype=np.uint16),
                    blue=np.zeros(n, dtype=np.uint16),
                    return_number=np.ones(n, dtype=np.uint8),
                    number_of_returns=np.ones(n, dtype=np.uint8),
                ),
            )
            scene, translation = mod.build_scene(path)
        self.assertEqual(scene["coord"].dtype, np.float32)
        self._assert_centimetric(scene["coord"], translation, local)


if __name__ == "__main__":
    unittest.main()
