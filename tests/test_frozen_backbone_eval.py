"""
freeze_backbone=True must keep the whole backbone in eval() mode permanently
(BatchNorm running stats frozen, DropPath/Dropout inactive), even after the
trainer's model.train() calls — for every segmentor/regressor/classifier in
pointcept/models/default.py.

Run with: PYTHONPATH=./ pytest tests/test_frozen_backbone_eval.py
"""

import unittest

import torch
import torch.nn as nn
from timm.layers import DropPath

from pointcept.models.builder import MODELS, build_model
from pointcept.models.default import FrozenBackboneEvalMixin
from pointcept.models.kpconvx.utils.generic_blocks import DropPathPack


@MODELS.register_module("_FrozenEvalBackbone")
class _FrozenEvalBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.blocks = nn.ModuleList(
            [DropPath(0.3), DropPathPack(0.3), nn.Dropout(0.5)]
        )

    def forward(self, point):
        return point


def _cfg(model_type, **kwargs):
    base = dict(
        type=model_type,
        backbone=dict(type="_FrozenEvalBackbone"),
        criteria=[],
    )
    if model_type in ("DefaultSegmentorV2", "DINOEnhancedSegmentor"):
        base.update(num_classes=2, backbone_out_channels=4)
    elif model_type == "DefaultRegressorV2":
        base.update(backbone_out_channels=4)
    elif model_type == "DefaultClassifier":
        base.update(num_classes=2, backbone_embed_dim=4)
    elif model_type == "MultiTaskSegmentorV2":
        base.update(
            backbone_out_channels=4,
            task_configs=dict(seg=dict(task_type="semantic", num_classes=2)),
        )
    base.update(kwargs)
    return base


MODEL_TYPES = (
    "DefaultSegmentor",
    "DefaultSegmentorV2",
    "DefaultRegressorV2",
    "MultiTaskSegmentorV2",
    "DefaultClassifier",
    "DINOEnhancedSegmentor",
)


class TestFrozenBackboneStaysEval(unittest.TestCase):
    def test_backbone_eval_at_construction_and_after_train(self):
        for t in MODEL_TYPES:
            with self.subTest(model=t):
                model = build_model(_cfg(t, freeze_backbone=True))
                self.assertFalse(model.backbone.training, "not eval at construction")
                model.train()
                self.assertTrue(model.training)
                for name, m in model.backbone.named_modules():
                    self.assertFalse(m.training, f"{name} in train mode after .train()")
                # And after a full eval -> train round trip, as the trainer does.
                model.eval()
                model.train()
                self.assertFalse(model.backbone.bn.training)

    def test_batchnorm_running_stats_do_not_drift(self):
        for t in MODEL_TYPES:
            with self.subTest(model=t):
                model = build_model(_cfg(t, freeze_backbone=True)).train()
                bn = model.backbone.bn
                mean, var = bn.running_mean.clone(), bn.running_var.clone()
                bn(torch.randn(32, 4) * 5 + 3)
                self.assertTrue(torch.equal(bn.running_mean, mean))
                self.assertTrue(torch.equal(bn.running_var, var))

    def test_unfrozen_backbone_follows_train_mode(self):
        for t in MODEL_TYPES:
            with self.subTest(model=t):
                model = build_model(_cfg(t, freeze_backbone=False)).train()
                self.assertTrue(model.backbone.bn.training)
                self.assertTrue(model.backbone.blocks[1].training)

    def test_lora_backbone_is_not_pinned(self):
        class _Stub(FrozenBackboneEvalMixin, nn.Module):
            def __init__(self, use_lora):
                super().__init__()
                self.backbone = nn.Sequential(nn.Dropout(0.1))
                self.freeze_backbone = True
                self.use_lora = use_lora

        self.assertTrue(_Stub(use_lora=True).train().backbone.training)
        self.assertFalse(_Stub(use_lora=False).train().backbone.training)


if __name__ == "__main__":
    unittest.main()
