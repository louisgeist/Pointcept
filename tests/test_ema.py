"""EMA of the model weights (cfg.use_ema): Trainer.build_ema / update_ema / eval_model,
checkpoint save + resume (CheckpointSaver / CheckpointLoader._load_ema) and the
validation forward of the hooks (HookBase._eval_forward). CPU only."""

import logging
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from pointcept.engines.hooks.default import HookBase
from pointcept.engines.hooks.misc import CheckpointLoader
from pointcept.engines.train import Trainer


class _Teacher:  # stands for MultiTaskTeacher: not an nn.Module
    def __init__(self):
        self.model = nn.Linear(3, 3)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)
        self.bn = nn.BatchNorm1d(2)
        self.teacher = _Teacher()

    def forward(self, x):
        return self.bn(self.lin(x))


def _trainer(decay=0.9, resume=False):
    t = SimpleNamespace(
        model=_Model(),
        cfg=SimpleNamespace(use_ema=True, ema_decay=decay, resume=resume),
        logger=logging.getLogger("test_ema"),
    )
    t.ema = Trainer.build_ema(t)
    return t


class TestEMA(unittest.TestCase):
    def test_build_shares_teacher_and_freezes(self):
        t = _trainer()
        self.assertIs(t.ema.teacher, t.model.teacher)
        self.assertIsNot(t.ema.lin, t.model.lin)
        self.assertFalse(t.ema.training)
        self.assertTrue(all(not p.requires_grad for p in t.ema.parameters()))
        self.assertTrue(all(p.requires_grad for p in t.model.parameters()))

    def test_update_formula_and_buffers(self):
        t = _trainer(decay=0.9)
        ema0 = {k: v.clone() for k, v in t.ema.state_dict().items()}
        with torch.no_grad():
            for p in t.model.parameters():
                p.add_(1.0)
            t.model.bn.running_mean.fill_(3.0)
        Trainer.update_ema(t)
        for name, p in t.model.named_parameters():
            expected = 0.9 * ema0[name] + 0.1 * p.detach()
            torch.testing.assert_close(dict(t.ema.named_parameters())[name], expected)
        torch.testing.assert_close(t.ema.bn.running_mean, torch.full((2,), 3.0))

    def test_eval_forward_uses_ema(self):
        t = _trainer()
        t.eval_model = t.ema
        hook = HookBase()
        hook.trainer = t
        t.model.eval()
        x = torch.randn(5, 4)
        with torch.no_grad():
            t.ema.lin.weight.zero_()
            t.ema.lin.bias.fill_(1.0)
            out = hook._eval_forward(x)
            self.assertTrue(torch.equal(out, t.ema(x)))
            self.assertFalse(torch.equal(out, t.model(x)))
        # Without EMA, the trained model.
        t.eval_model = None
        t.model.eval()
        x = torch.randn(5, 4)
        with torch.no_grad():
            self.assertTrue(torch.equal(hook._eval_forward(x), t.model(x)))

    def test_checkpoint_roundtrip(self):
        t = _trainer(decay=0.5)
        with torch.no_grad():
            for p in t.model.parameters():
                p.add_(2.0)
        Trainer.update_ema(t)
        ckpt = {
            "state_dict": t.model.state_dict(),
            "ema_state_dict": t.ema.state_dict(),
        }
        self.assertFalse(any("teacher" in k for k in ckpt["ema_state_dict"]))
        # Resume: the EMA weights come back as saved, not as the model weights.
        t2 = _trainer(resume=True)
        t2.model.load_state_dict(ckpt["state_dict"])
        loader = CheckpointLoader()
        loader.trainer = t2
        loader._load_ema(ckpt)
        for k, v in ckpt["ema_state_dict"].items():
            torch.testing.assert_close(t2.ema.state_dict()[k], v)
        self.assertFalse(
            torch.equal(t2.ema.lin.weight, t2.model.lin.weight)
        )
        # Initial weight (not a resume): the EMA starts from the loaded model.
        t3 = _trainer(resume=False)
        t3.model.load_state_dict(ckpt["state_dict"])
        loader.trainer = t3
        loader._load_ema(ckpt)
        for k, v in t3.model.state_dict().items():
            torch.testing.assert_close(t3.ema.state_dict()[k], v)
        # Through torch.save / torch.load with DDP-style "module." keys.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pth")
            torch.save(
                {
                    "ema_state_dict": {
                        "module." + k: v for k, v in ckpt["ema_state_dict"].items()
                    }
                },
                path,
            )
            t4 = _trainer(resume=True)
            loader.trainer = t4
            loader._load_ema(torch.load(path, weights_only=False))
            torch.testing.assert_close(t4.ema.lin.weight, t.ema.lin.weight)


if __name__ == "__main__":
    unittest.main()
