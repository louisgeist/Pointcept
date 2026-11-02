"""Sonata pretraining inputs: prefixed view masks in Collect, coord feature scaling, vertical
registration and learned fill values of the masked color / intensity."""

import unittest

import numpy as np
import torch

from pointcept.datasets.transform import Collect, Compose
from pointcept.datasets.utils import point_collate_fn
from pointcept.models.sonata.sonata_v1m2_uni_teacher_head import Sonata
from pointcept.utils.config import Config

PRETRAIN_CONFIG = "configs/malibu3d/pretrain-sonata-v1m2-malibu3d.py"


def synthetic_tile(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    coord = np.stack(
        [
            rng.uniform(651000, 651100, n),
            rng.uniform(6862000, 6862100, n),
            rng.uniform(120, 160, n),
        ],
        axis=1,
    )
    return dict(
        coord=coord.astype(np.float32),
        color=rng.uniform(0, 255, (n, 3)).astype(np.float32),
        strength=rng.uniform(0, 1, (n, 1)).astype(np.float32),
        name="synthetic",
    )


def fill_model():
    model = Sonata.__new__(Sonata)
    torch.nn.Module.__init__(model)
    model._init_learned_masked_feat(
        feature_mask_values=dict(enable=True, masked_feat_keys=["color", "strength"])
    )
    with torch.no_grad():
        model.color_mask_value.copy_(torch.tensor([[7.0, 8.0, 9.0]]))
        model.strength_mask_value.copy_(torch.tensor([[5.0]]))
    return model.train()


class TestCollectViewMasks(unittest.TestCase):
    def test_prefixed_masks_and_scales(self):
        n = 6
        mask = np.array([1, 0, 1, 0, 0, 0], dtype=bool)
        data = {
            "global_coord": torch.full((n, 3), 100.0),
            "global_color": torch.ones(n, 3),
            "global_strength": torch.ones(n, 1),
            "global_color_mask": mask,
            "feat_color": torch.ones(n, 3),
        }
        out = Collect(
            keys=(),
            offset_keys_dict=dict(),
            global_feat_keys=("global_coord", "global_color", "global_strength"),
            feat_scales=dict(global_coord=0.01),
        )(data)
        self.assertTrue(torch.allclose(out["global_feat"][:, :3], torch.ones(n, 3)))
        self.assertTrue(torch.equal(out["global_color_mask"], torch.from_numpy(mask)))
        self.assertEqual(
            (out["global_color_feat_start"], out["global_color_feat_end"]), (3, 6)
        )
        # No strength mask in the input: none collected.
        self.assertNotIn("global_strength_mask", out)

    def test_plain_feat_unchanged(self):
        data = {
            "coord": torch.zeros(4, 3),
            "color": torch.ones(4, 3),
            "color_mask": np.ones(4, dtype=bool),
        }
        out = Collect(keys=("coord",), feat_keys=("coord", "color"))(data)
        self.assertEqual((out["color_feat_start"], out["color_feat_end"]), (3, 6))
        self.assertTrue(out["color_mask"].all())


class TestSonataPretrainPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = Config.fromfile(PRETRAIN_CONFIG)
        cls.cfg = cfg
        cls.transform = Compose(cfg.data.train.transform)

    def test_config_inputs(self):
        self.assertEqual(self.cfg.coord_feat_scale, 0.01)
        self.assertTrue(self.cfg.model.feature_mask_values["enable"])

    def test_views_carry_masks_scaled_coords_and_registered_z(self):
        np.random.seed(0)
        out = self.transform(synthetic_tile())
        for view in ("global", "local"):
            n = out[f"{view}_coord"].shape[0]
            self.assertEqual(out[f"{view}_color_mask"].shape, (n,))
            self.assertEqual(out[f"{view}_strength_mask"].shape, (n,))
            # coord features are the view coords x 0.01
            self.assertTrue(
                torch.allclose(out[f"{view}_feat"][:, :3], out[f"{view}_coord"] * 0.01)
            )
            # z stays measured from the lowest point of the tile (no per-view recentering):
            # synthetic heights span 40 m, so z must stay within [-1, 45] (offset, scale, tilt)
            z = out[f"{view}_coord"][:, 2]
            self.assertGreater(z.min().item(), -1.0)
            self.assertLess(z.max().item(), 45.0)
        # local views are cut without modality dropout
        self.assertFalse(out["local_color_mask"].any())
        self.assertFalse(out["local_strength_mask"].any())

    def test_scene_dropout_sets_global_masks(self):
        dropped = 0
        for seed in range(40):
            np.random.seed(seed)
            import random

            random.seed(seed)
            out = self.transform(synthetic_tile(n=4000, seed=seed))
            m = out["global_color_mask"]
            # scene-level dropout: all or nothing
            self.assertIn(m.float().mean().item(), (0.0, 1.0))
            dropped += int(m.all())
        self.assertGreater(dropped, 0)


class TestSonataLearnedFill(unittest.TestCase):
    def test_fill_and_gradient(self):
        model = fill_model()
        n = 5
        feat = torch.zeros(n, 7)
        batch = {
            "global_feat": feat,
            "global_color_mask": torch.tensor([True, False, True, False, False]),
            "global_strength_mask": torch.tensor([False, True, False, False, False]),
            "global_color_feat_start": torch.tensor([3]),
            "global_color_feat_end": torch.tensor([6]),
            "global_strength_feat_start": torch.tensor([6]),
            "global_strength_feat_end": torch.tensor([7]),
        }
        model._fill_view_masked_feat(batch, "global")
        out = batch["global_feat"]
        self.assertTrue(torch.equal(out[0, 3:6], torch.tensor([7.0, 8.0, 9.0])))
        self.assertTrue(torch.equal(out[1, 3:6], torch.zeros(3)))
        self.assertEqual(out[1, 6].item(), 5.0)
        self.assertEqual(out[0, 6].item(), 0.0)
        out.sum().backward()
        self.assertEqual(model.color_mask_value.grad.tolist(), [[2.0, 2.0, 2.0]])
        self.assertEqual(model.strength_mask_value.grad.tolist(), [[1.0]])

    def test_pipeline_batch_through_fill(self):
        cfg = Config.fromfile(PRETRAIN_CONFIG)
        transform = Compose(cfg.data.train.transform)
        np.random.seed(1)
        batch = point_collate_fn([transform(synthetic_tile(n=4000, seed=s)) for s in (1, 2)])
        model = fill_model()
        mask = batch["global_color_mask"]
        model._fill_view_masked_feat(batch, "global")
        model._fill_view_masked_feat(batch, "local")
        start = int(batch["global_color_feat_start"][0])
        if mask.any():
            self.assertTrue(
                (batch["global_feat"][mask, start : start + 3] == torch.tensor([7.0, 8.0, 9.0])).all()
            )
        self.assertFalse(
            (batch["local_feat"][:, start : start + 3] == torch.tensor([7.0, 8.0, 9.0])).all(1).any()
        )


class TestSonataConsumersMatchPretrain(unittest.TestCase):
    """Probe and fine-tuning configs of the MALiBU3D Sonata backbone read the same inputs."""

    CONSUMERS = [
        "configs/malibu3d/multi-sonata-ft-v1m0-malibu3d.py",
        "configs/malibu3d/probe/sonata-v1m2-malibu3d-lin.py",
        "configs/malibu3d/probe/sonata-v1m2-malibu3d-lin-grid-enc.py",
        "configs/dales/sonata-v1m2-dales-lin-grid-enc.py",
        "configs/h3d/sonata-v1m2-h3d-lin-grid-enc.py",
        "configs/eclair/sonata-v1m2-eclair-lin-grid-enc.py",
        "configs/opengf/sonata-v1m2-opengf-lin-grid-enc.py",
        "configs/pureforest/probe-sonata-v1m2-pureforest.py",
    ]

    def test_coord_scale_and_learned_fill(self):
        pretrain = Config.fromfile(PRETRAIN_CONFIG)
        for path in self.CONSUMERS:
            with self.subTest(path=path):
                cfg = Config.fromfile(path)
                self.assertEqual(cfg.coord_feat_scale, pretrain.coord_feat_scale)
                self.assertTrue(cfg.model.feature_mask_values["enable"])
                self.assertEqual(
                    list(cfg.model.feature_mask_values["masked_feat_keys"]),
                    ["color", "strength"],
                )
                self.assertIn(
                    "coord=coord_feat_scale", open(path).read().replace(" ", "")
                    .replace("coord=coord_feat_scale,", "coord=coord_feat_scale")
                )


if __name__ == "__main__":
    unittest.main()
