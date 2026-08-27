"""
Tests for tools/grid_then_seeds.py's seed-ensemble config generation and
CSV aggregation.

Run with: PYTHONPATH=./ pytest tests/test_grid_then_seeds_config.py
"""

import json
import os
import tempfile
import unittest

from pointcept.utils.config import Config
from tools.grid_then_seeds import (
    SUMMARY_FIELDS,
    append_summary_csv,
    build_seed_ensemble_config,
    split_info_from_cfg,
)

# A representative grid-probe config: loop-built `probes`, a `_base_`, unicode
# class names, PT-v3m2 enc_mode backbone.
GRID_CONFIG = "configs/experiment/w110/1/sonata_h3d_Ztrans_warm5/sonata-v1m2-h3d-lin-grid.py"

# ...and a cartesian_probes one (import inside the config file).
GRID_CONFIG_CARTESIAN = "configs/flair3d_default/probe/sonata-v1m2-flair3d-lin-grid-wide.py"

# DALES: no held-out val, every config points data.val and data.test at split="test"
GRID_CONFIG_DALES = "configs/dales/spunet-v1m0-dales-lin-grid-enc.py"

WINNER_NAME = "ce_lovasz_lr2e-2_wd0_do0_none_fnnone_adamw_w05"
WINNER_PROBE_CONFIG = {
    "criteria": [
        {"type": "CrossEntropyLoss", "loss_weight": 1.0, "ignore_index": 11},
        {"type": "LovaszLoss", "mode": "multiclass", "loss_weight": 1.0, "ignore_index": 11},
    ],
    "input_norm": None,
    "feat_norm": None,
    "dropout": 0.0,
    "optimizer": {"type": "AdamW", "lr": 0.02, "weight_decay": 0.0},
    "scheduler": {
        "type": "OneCycleLR",
        "max_lr": 0.02,
        "pct_start": 0.05,
        "anneal_strategy": "cos",
        "div_factor": 10.0,
        "final_div_factor": 1000.0,
    },
    "grad_clip": 3.0,
}


class TestBuildSeedEnsembleConfig(unittest.TestCase):
    def _generate(self, grid_config, n_seeds=10, winner_pc=None):
        winner_pc = winner_pc or WINNER_PROBE_CONFIG
        tmp = tempfile.mkdtemp()
        out = os.path.join(tmp, "seed_ensemble_config.py")
        path, probe_names, info = build_seed_ensemble_config(
            grid_config, WINNER_NAME, winner_pc, n_seeds, out
        )
        self._last_info = info
        return Config.fromfile(str(path)), probe_names

    def test_h3d_loop_config_roundtrips(self):
        cfg, probe_names = self._generate(GRID_CONFIG, n_seeds=10)

        self.assertEqual(len(cfg.model.probes), 10)
        self.assertEqual(list(cfg.model.probes), [f"seed{i}" for i in range(10)])
        self.assertEqual(probe_names, [f"seed{i}" for i in range(10)])

        # task_configs must mirror the probe names exactly
        self.assertEqual(list(cfg.data.task_configs), list(cfg.model.probes))
        for tc in cfg.data.task_configs.values():
            self.assertEqual(tc["task_type"], "semantic")
            self.assertEqual(tc["num_classes"], cfg.num_classes)
            self.assertEqual(tc["ignore_index"], cfg.ignore_index)

        # every probe carries the full winner probe_config
        for probe in cfg.model.probes.values():
            self.assertEqual(probe["optimizer"]["lr"], 0.02)
            self.assertEqual(probe["optimizer"]["type"], "AdamW")
            self.assertEqual(probe["scheduler"]["type"], "OneCycleLR")
            self.assertEqual(len(probe["criteria"]), 2)
            self.assertEqual(probe["grad_clip"], 3.0)

        hook_types = [h["type"] for h in cfg.hooks]
        self.assertIn("GridProbeSeedEnsembleTester", hook_types)
        self.assertNotIn("GridProbeWinnerSelector", hook_types)
        # order: the tester stays last, after the checkpoint savers
        self.assertEqual(hook_types[-1], "GridProbeSeedEnsembleTester")
        self.assertLess(
            hook_types.index("GridProbeCheckpointSaver"),
            hook_types.index("GridProbeSeedEnsembleTester"),
        )

        self.assertEqual(cfg.test["type"], "GridProbeSemSegTester")
        self.assertIs(cfg.log_test_f1, True)
        self.assertIn("SeedEnsemble", cfg.wandb_run_name)
        self.assertIn(WINNER_NAME, cfg.wandb_run_name)

        # backbone untouched
        self.assertEqual(cfg.model.backbone["type"], "PT-v3m2")
        self.assertIs(cfg.model.freeze_backbone, True)

    def test_cartesian_config_roundtrips(self):
        focal_sgd = {
            "criteria": [{"type": "FocalLoss", "gamma": 2.0, "loss_weight": 1.0, "ignore_index": 15}],
            "input_norm": "l2",
            "feat_norm": "batchnorm",
            "dropout": 0.1,
            "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4},
            "scheduler": {"type": "CosineAnnealingLR", "eta_min": 0.0},
            "grad_clip": None,
        }
        cfg, _ = self._generate(GRID_CONFIG_CARTESIAN, n_seeds=6, winner_pc=focal_sgd)

        self.assertEqual(len(cfg.model.probes), 6)
        probe = cfg.model.probes["seed3"]
        self.assertEqual(probe["input_norm"], "l2")
        self.assertEqual(probe["feat_norm"], "batchnorm")
        self.assertIsNone(probe["grad_clip"])
        self.assertEqual(probe["optimizer"]["type"], "SGD")
        self.assertEqual(probe["optimizer"]["momentum"], 0.9)

        hook_types = [h["type"] for h in cfg.hooks]
        self.assertIn("GridProbeSeedEnsembleTester", hook_types)
        self.assertNotIn("GridProbeWinnerSelector", hook_types)

    def test_h3d_val_and_test_are_distinct_splits(self):
        cfg, _ = self._generate(GRID_CONFIG, n_seeds=4)
        info = split_info_from_cfg(cfg)
        self.assertEqual(info["val_split"], "val")
        self.assertEqual(info["test_split"], "test")
        self.assertFalse(info["val_eq_test_split"])
        self.assertFalse(self._last_info["val_eq_test_split"])

    def test_dales_val_equals_test_split(self):
        # DALES has no held-out val -- both point at split="test", so the
        # seed-ensemble test_* metrics land on the winner-selection tiles.
        cfg, _ = self._generate(GRID_CONFIG_DALES, n_seeds=4)
        info = split_info_from_cfg(cfg)
        self.assertEqual(info["val_split"], "test")
        self.assertEqual(info["test_split"], "test")
        self.assertTrue(info["val_eq_test_split"])
        self.assertTrue(self._last_info["val_eq_test_split"])

    def test_select_metric_propagates_into_seed_config(self):
        # build_seed_ensemble_config copies every hook dict through verbatim
        # (only WinnerSelector -> SeedEnsembleTester is swapped), so a grid
        # config that selects on macro-F1 yields a seed config that does too.
        src = Config.fromfile(GRID_CONFIG)
        for hook in src.hooks:
            if hook["type"] == "GridProbeEvaluator":
                hook["select_metric"] = "macro_f1"
        tmp = tempfile.mkdtemp()
        patched = os.path.join(tmp, "grid_macro_f1.py")
        src.dump(patched)

        cfg, _ = self._generate(patched, n_seeds=4)
        evaluators = [h for h in cfg.hooks if h["type"] == "GridProbeEvaluator"]
        self.assertEqual(len(evaluators), 1)
        self.assertEqual(evaluators[0]["select_metric"], "macro_f1")

    def test_missing_winner_selector_raises(self):
        tmp = tempfile.mkdtemp()
        # a config file with a hooks list that has no GridProbeWinnerSelector
        bogus = os.path.join(tmp, "bogus_grid.py")
        with open(bogus, "w") as f:
            f.write(
                "model = dict(type='GridProbeSegmentorV2', probes={}, backbone=dict(type='x'))\n"
                "hooks = [dict(type='CheckpointLoader')]\n"
                "data = dict(num_classes=3, ignore_index=3, names=['a', 'b', 'c'])\n"
            )
        with self.assertRaises(ValueError):
            build_seed_ensemble_config(
                bogus, WINNER_NAME, WINNER_PROBE_CONFIG, 4, os.path.join(tmp, "out.py")
            )


class TestAppendSummaryCsv(unittest.TestCase):
    def _row(self, seed_dir):
        row = {k: "" for k in SUMMARY_FIELDS}
        row.update(seed_dir=seed_dir, winner_probe_name="w", test_mIoU_mean="0.5")
        return row

    def test_append_and_dedup(self):
        tmp = tempfile.mkdtemp()
        csv_path = os.path.join(tmp, "summary.csv")
        import pathlib

        self.assertTrue(append_summary_csv(pathlib.Path(csv_path), self._row("/x/seeds")))
        # same seed_dir -> not written again
        self.assertFalse(append_summary_csv(pathlib.Path(csv_path), self._row("/x/seeds")))
        # different seed_dir -> written
        self.assertTrue(append_summary_csv(pathlib.Path(csv_path), self._row("/y/seeds")))

        with open(csv_path) as f:
            import csv as _csv

            rows = list(_csv.DictReader(f))
        self.assertEqual([r["seed_dir"] for r in rows], ["/x/seeds", "/y/seeds"])


if __name__ == "__main__":
    unittest.main()
