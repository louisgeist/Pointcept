"""Released checkpoints load into their configs and give sensible predictions on two toy tiles
(multitask models: segment accuracy / mIoU and elevation MAE; Sonata SSL backbone: feature shape and spread).

Needs spconv, a GPU, the weights under ckpt/<model>/model.pth (docs/checkpoints.md) and the toy region installed
by the README Quick start (data/malibu3d/tiles_toy.csv); each model is skipped otherwise.

Run with: PYTHONPATH=. pytest tests/test_checkpoint_forward.py -v
"""

import random

import numpy as np
import pytest
import torch

from pointcept.utils.config import Config
from tools.forward_example import CONFIGS, REPO, build_input, load_model, seg_scores

TOY_CSV = "data/malibu3d/tiles_toy.csv"
TOY_DIR = "data/malibu3d/test/D075-2021_LIDARHD/UU-S1-4"
TILES = ("D075-2021_UU-S1-4_3-3", "D075-2021_UU-S1-4_4-2")
MAX_POINTS = 30000
# Lowest values over the five multitask checkpoints on these tiles: acc 0.78, mIoU 0.68, elevation MAE at most
# 0.1x the MAE of a constant prediction. The thresholds only catch real breakage.
SEG_ACC_MIN = 0.70
SEG_MIOU_MIN = 0.55
ELEV_MAE_MAX_RATIO = 0.5
SONATA_FEAT_DIM = 1232


def _setup(name, monkeypatch):
    pytest.importorskip("spconv")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    weight = REPO / "ckpt" / name / "model.pth"
    if not weight.is_file():
        pytest.skip(f"checkpoint not downloaded: {weight}")
    for tile in TILES:
        if not (REPO / TOY_DIR / tile).is_dir():
            pytest.skip(f"toy tile not installed: {tile}")
    monkeypatch.chdir(REPO)  # config data roots are repo-relative
    cfg = Config.fromfile(str(REPO / CONFIGS[name]))
    return cfg, load_model(name, str(weight), cfg)


def _input(cfg, tile):
    # the val pipeline's GridSample(mode="train") draws one random point per voxel
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return build_input(cfg, tile, "test", TOY_CSV, MAX_POINTS)


@pytest.mark.parametrize("name", [m for m in CONFIGS if m != "sonata-pretrained"])
@torch.no_grad()
def test_multitask_checkpoint_forward(name, monkeypatch):
    cfg, model = _setup(name, monkeypatch)
    for tile in TILES:
        inp = _input(cfg, tile)
        out = model(inp)
        for task, logits in out["seg_logits_by_task"].items():
            assert torch.isfinite(logits).all(), f"{name}/{tile}: non-finite {task} logits"
        acc, miou = seg_scores(
            out["seg_logits"].argmax(1).cpu().numpy(),
            inp["segment"].cpu().numpy(),
            int(cfg.data.ignore_index),
        )
        assert acc >= SEG_ACC_MIN, f"{name}/{tile}: segment acc {acc:.3f}"
        assert miou >= SEG_MIOU_MIN, f"{name}/{tile}: segment mIoU {miou:.3f}"
        pred = out["reg_pred_by_task"]["elevation"].float().cpu().numpy().reshape(-1)
        gt = inp["elevation"].float().cpu().numpy().reshape(-1)
        ok = np.isfinite(gt) & (gt > -1e4)
        mae = np.abs(pred[ok] - gt[ok]).mean()
        const_mae = np.abs(gt[ok] - np.median(gt[ok])).mean()
        assert (
            mae < ELEV_MAE_MAX_RATIO * const_mae
        ), f"{name}/{tile}: elevation MAE {mae:.2f} m vs {const_mae:.2f} m for a constant"


@torch.no_grad()
def test_sonata_pretrained_checkpoint_forward(monkeypatch):
    cfg, model = _setup("sonata-pretrained", monkeypatch)
    for tile in TILES:
        feat, _ = model._forward_backbone(_input(cfg, tile))
        feat = feat.float()
        assert feat.shape[1] == SONATA_FEAT_DIM, f"{tile}: features {tuple(feat.shape)}"
        assert torch.isfinite(feat).all(), f"{tile}: non-finite features"
        assert feat.std() > 0.1, f"{tile}: collapsed features (std {feat.std():.3f})"
