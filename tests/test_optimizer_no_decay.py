import unittest

import torch
from torch import nn

from pointcept.utils.config import ConfigDict
from pointcept.utils.optimizer import build_optimizer


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8))  # no keyword
        self.block = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))  # keyword "block"
        self.mask_value = nn.Parameter(torch.zeros(1))


def _wd(opt):
    return [g["weight_decay"] for g in opt.param_groups]


class TestNoDecayBiasNorm(unittest.TestCase):
    def test_default_unchanged(self):
        cfg = ConfigDict(type="AdamW", lr=1e-3, weight_decay=0.5)
        opt = build_optimizer(cfg, Toy(), [ConfigDict(keyword="block", lr=1e-4)])
        self.assertEqual(len(opt.param_groups), 2)
        self.assertEqual(_wd(opt), [0.5, 0.5])

    def test_split_layout_and_membership(self):
        model = Toy()
        cfg = ConfigDict(type="AdamW", lr=1e-3, weight_decay=0.5, no_decay_bias_norm=True)
        opt = build_optimizer(cfg, model, [ConfigDict(keyword="block", lr=1e-4)])
        g = opt.param_groups
        self.assertEqual(len(g), 4)  # [rest, block, rest_nd, block_nd]
        self.assertEqual(_wd(opt), [0.5, 0.5, 0.0, 0.0])
        self.assertEqual([x["lr"] for x in g], [1e-3, 1e-4, 1e-3, 1e-4])
        n = lambda grp: sorted(id(p) for p in grp["params"])
        ids = lambda *ps: sorted(id(p) for p in ps)
        self.assertEqual(n(g[0]), ids(model.stem[0].weight))
        self.assertEqual(n(g[1]), ids(model.block[0].weight))
        self.assertEqual(n(g[2]), ids(model.stem[0].bias, model.stem[1].weight, model.stem[1].bias, model.mask_value))
        self.assertEqual(n(g[3]), ids(model.block[0].bias, model.block[1].weight, model.block[1].bias))
        total = sum(len(x["params"]) for x in g)
        self.assertEqual(total, len(list(model.parameters())))

    def test_no_param_dicts(self):
        cfg = ConfigDict(type="AdamW", lr=1e-3, weight_decay=0.5, no_decay_bias_norm=True)
        opt = build_optimizer(cfg, Toy(), None)
        self.assertEqual(_wd(opt), [0.5, 0.0])

    def test_onecycle_accepts_doubled_max_lr(self):
        cfg = ConfigDict(type="AdamW", lr=1e-3, weight_decay=0.5, no_decay_bias_norm=True)
        opt = build_optimizer(cfg, Toy(), [ConfigDict(keyword="block", lr=1e-4)])
        torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-3, 1e-4, 1e-3, 1e-4], total_steps=10)
        with self.assertRaises(ValueError):
            torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-3, 1e-4], total_steps=10)


if __name__ == "__main__":
    unittest.main()
