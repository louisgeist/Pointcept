"""
Regression tests for the fixes from docs/bug_review/ (2026-09-27 review):

- CrossEntropyLoss on an all-ignore batch must return an in-graph zero
  (backward() must not raise, head gets zero grads).
- RemapSegment must map from the ORIGINAL labels (no cascading through
  chained / swapped mappings).
- OpenGF / ECLAIR preprocessing must recenter absolute projected coordinates
  in float64 before the float32 cast (no 0.125 m / 0.5 m XY lattice).
- GridProbe AMP: one GradScaler per probe so a diverged head cannot collapse
  the healthy probes' scale / schedulers (docs/bug_review/03_*.md).

Run with: PYTHONPATH=./ python -m unittest tests/test_bug_review_2026_09_27.py
"""

import importlib.util
import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GradScaler fp16")
class TestGridProbePerProbeGradScaler(unittest.TestCase):
    """Isolation of AMP GradScaler across disjoint probe heads (bug 03)."""

    def _make_probe(self, C, K, device, big=False):
        head = nn.Linear(C, K).to(device)
        if big:
            with torch.no_grad():
                head.weight.mul_(1e6)  # fp16 logits overflow
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=1e-2, total_steps=300
        )
        return head, opt, sched

    def _run(self, per_probe_scaler, bad_probe, steps=300):
        torch.manual_seed(0)
        device = "cuda:0"
        C, K, N = 256, 8, 4096
        feat = torch.randn(N, C, device=device)
        target = (feat @ torch.randn(C, K, device=device)).argmax(1)

        probes = {"healthy": self._make_probe(C, K, device)}
        if bad_probe:
            probes["diverged"] = self._make_probe(C, K, device, big=True)

        if per_probe_scaler:
            scalers = {n: torch.amp.GradScaler("cuda") for n in probes}
        else:
            scalers = {"_shared": torch.amp.GradScaler("cuda")}

        for _ in range(steps):
            for _, o, _ in probes.values():
                o.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                losses = {
                    n: F.cross_entropy(h(feat), target)
                    for n, (h, _, _) in probes.items()
                }
            if per_probe_scaler:
                sum(scalers[n].scale(l) for n, l in losses.items()).backward()
                for n, (h, o, s) in probes.items():
                    before = scalers[n].get_scale()
                    scalers[n].unscale_(o)
                    torch.nn.utils.clip_grad_norm_(h.parameters(), 3.0)
                    scalers[n].step(o)
                    scalers[n].update()
                    if before <= scalers[n].get_scale():
                        s.step()
            else:
                shared = scalers["_shared"]
                shared.scale(sum(losses.values())).backward()
                before = shared.get_scale()
                for n, (h, o, _) in probes.items():
                    shared.unscale_(o)
                    torch.nn.utils.clip_grad_norm_(h.parameters(), 3.0)
                    shared.step(o)
                shared.update()
                if before <= shared.get_scale():
                    for _, _, s in probes.values():
                        s.step()

        healthy = probes["healthy"][0]
        acc = (healthy(feat).argmax(1) == target).float().mean().item()
        if per_probe_scaler:
            healthy_scale = scalers["healthy"].get_scale()
        else:
            healthy_scale = scalers["_shared"].get_scale()
        return healthy_scale, acc

    def test_shared_scaler_collapses_with_diverged_probe(self):
        scale, acc = self._run(per_probe_scaler=False, bad_probe=True)
        self.assertLess(scale, 1.0)
        self.assertLess(acc, 0.3)

    def test_per_probe_scaler_isolates_healthy_probe(self):
        scale_ok, acc_ok = self._run(per_probe_scaler=True, bad_probe=False)
        scale_bad, acc_bad = self._run(per_probe_scaler=True, bad_probe=True)
        self.assertGreater(scale_ok, 1.0)
        self.assertGreater(scale_bad, 1.0)
        self.assertGreater(acc_ok, 0.9)
        self.assertGreater(acc_bad, 0.9)


if __name__ == "__main__":
    unittest.main()
