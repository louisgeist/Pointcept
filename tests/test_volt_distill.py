"""Volt backbone + SpUNet-teacher distillation in MultiTaskSegmentorV2.

Builds the model of configs/malibu3d/multi-volt-s-v1m0-malibu3d.py (tasks reduced to
segment + elevation so a synthetic batch suffices; the teacher is the real multitask
SpUNet checkpoint) and runs it on random voxels. Needs CUDA, flash_attn and
ckpt_cleaned/spunet/model.pth; skipped otherwise.
"""

import os
import unittest
from unittest import mock

import torch

CONFIG = "configs/malibu3d/multi-volt-s-v1m0-malibu3d.py"


def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA not available"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "flash_attn not installed"
    from pointcept.utils.config import Config

    cfg = Config.fromfile(CONFIG)
    if not os.path.exists(cfg.teacher_weights):
        return f"teacher checkpoint {cfg.teacher_weights} missing"
    return None


def _build(with_teacher=True):
    # hecate's spconv autotuner crashes on the 2nd conv shape (teacher SpUNet): force
    # ConvAlgo.Native, as the regression harness does. Harmless elsewhere.
    import tests.regression.spconv_native  # noqa: F401
    from pointcept.models import build_model
    from pointcept.utils.config import Config

    cfg = Config.fromfile(CONFIG)
    tasks = ("segment", "elevation")
    cfg.model.task_configs = {t: cfg.model.task_configs[t] for t in tasks}
    cfg.model.task_criteria = {t: cfg.model.task_criteria[t] for t in tasks}
    cfg.model.task_weights = {t: 1.0 for t in tasks}
    if not with_teacher:
        cfg.model.teacher = None
    return cfg, build_model(cfg.model).cuda()


def _batch(n_per_scene=(8000, 6000), seed=0):
    g = torch.Generator().manual_seed(seed)
    grid, offset = [], []
    for n in n_per_scene:
        # 2.5D-ish aerial scene: 1000 x 1000 voxels (100 m) in xy, a thin z range.
        xy = torch.randint(0, 1000, (n, 2), generator=g)
        z = torch.randint(0, 40, (n, 1), generator=g)
        grid.append(torch.cat([xy, z], 1))
        offset.append(n)
    grid_coord = torch.cat(grid).int()
    n = grid_coord.shape[0]
    coord = grid_coord.float() * 0.1
    feat = torch.cat(
        [coord * 0.01, torch.rand(n, 3, generator=g), torch.rand(n, 1, generator=g)], 1
    )
    return dict(
        coord=coord.cuda(),
        grid_coord=grid_coord.cuda(),
        feat=feat.cuda(),
        offset=torch.tensor(offset).cumsum(0).cuda(),
        segment=torch.randint(0, 16, (n,), generator=g).cuda(),
        elevation=(torch.rand(n, generator=g) * 20).cuda(),
    )


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class TestVoltDistill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.cfg, cls.model = _build()

    def test_teacher_outside_student_state(self):
        model = self.model
        self.assertFalse(isinstance(model.teacher, torch.nn.Module))
        self.assertEqual(set(model.distill_heads.keys()), {"segment"})
        keys = model.state_dict().keys()
        self.assertTrue(any(k.startswith("distill_heads.segment") for k in keys))
        # No teacher parameter in the student (state_dict, optimizer, DDP).
        n_teacher = sum(p.numel() for p in model.teacher.model.parameters())
        self.assertGreater(n_teacher, 40e6)
        student_ids = {id(p) for p in model.parameters()}
        self.assertFalse(
            any(id(p) in student_ids for p in model.teacher.model.parameters())
        )
        self.assertFalse(model.teacher.model.training)

    def test_train_step_and_distill_mix(self):
        model = self.model
        model.train()
        batch = _batch()
        feat_before = batch["feat"].clone()
        labels = model.teacher.predict_labels(batch, ("segment",))["segment"]
        # The teacher works on a copy: the batch features are untouched.
        self.assertTrue(torch.equal(batch["feat"], feat_before))
        self.assertEqual(labels.shape, batch["segment"].shape)
        self.assertLess(int(labels.max()), 15)  # 15 classes, void never predicted

        with mock.patch.object(
            model.teacher, "predict_labels", wraps=model.teacher.predict_labels
        ) as spy, torch.autocast("cuda", dtype=torch.float16):
            out = model(dict(batch))
        spy.assert_called_once()
        loss = out["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.distill_heads["segment"].weight.grad)
        self.assertIsNotNone(model.backbone.tokenizer.proj.weight.grad)
        self.assertTrue(
            all(p.grad is None for p in model.teacher.model.parameters())
        )
        model.zero_grad(set_to_none=True)

        # loss_by_task["segment"] = 0.5 * L(gt) + 0.5 * L(teacher labels).
        model.eval()  # deterministic (DropPath off) for an exact comparison
        model.training = True  # but take the training branch of forward
        with torch.no_grad():
            out = model(dict(batch))
            feat, _ = model._forward_backbone(dict(batch))
            crit = model.criteria_by_task["segment"]
            l_gt = crit(model.seg_heads["segment"](feat), batch["segment"])
            l_t = crit(model.distill_heads["segment"](feat), labels)
        model.training = False
        torch.testing.assert_close(
            out["loss_by_task"]["segment"], 0.5 * l_gt + 0.5 * l_t, rtol=1e-4, atol=1e-5
        )

    def test_eval_does_not_call_teacher(self):
        model = self.model
        model.eval()
        with mock.patch.object(model.teacher, "predict_labels") as spy, torch.no_grad():
            out = model(_batch(seed=1))
        spy.assert_not_called()
        self.assertEqual(out["seg_logits"].shape[1], 15)
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_gradnorm_last_layer_is_volt_decoder(self):
        params = self.model.last_backbone_layer_parameters()
        dec = {id(p) for p in self.model.backbone.decoder.parameters()}
        self.assertTrue(params and all(id(p) in dec for p in params))

    def test_optimizer_groups_match_scheduler(self):
        from pointcept.utils.optimizer import build_optimizer

        opt = build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)
        self.assertEqual(len(opt.param_groups), len(self.cfg.scheduler.max_lr))
        self.assertEqual(opt.param_groups[-1]["weight_decay"], 0.0)
        self.assertEqual(opt.param_groups[0]["weight_decay"], 0.05)


class TestVoltNoTeacher(unittest.TestCase):
    @unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
    def test_without_teacher(self):
        _, model = _build(with_teacher=False)
        self.assertIsNone(model.teacher)
        self.assertEqual(len(model.distill_heads), 0)
        model.train()
        out = model(_batch(seed=2))
        self.assertEqual(set(out["loss_by_task"]), {"segment", "elevation"})


if __name__ == "__main__":
    unittest.main()
