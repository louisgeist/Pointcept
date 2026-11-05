"""Sonata-v1m2 with a LitePT-v2 backbone (configs/shapenet_part/pretrain-sonata-litept-s-v2m0-shapenet-part.py):
the config's data pipeline feeds a forward/backward with a finite loss, non-empty point matches, gradients in the
student only (mask token included), and layer-wise lr keywords that match LitePT's module names.

Needs spconv, flash_attn, pointops and a GPU; skipped otherwise.

Run with: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest tests/test_sonata_litept_v2.py -v
"""

import numpy as np
import pytest
import torch

pytest.importorskip("spconv")
pytest.importorskip("flash_attn")
pytest.importorskip("pointops")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from pointcept.datasets import collate_fn
from pointcept.datasets.transform import Compose
from pointcept.models.builder import build_model
from pointcept.utils.config import Config

CONFIG = "configs/shapenet_part/pretrain-sonata-litept-s-v2m0-shapenet-part.py"


def synthetic_shape(rng, n=2600):
    """Points on an ellipsoid surface, in the format of ShapeNetPartDataset.get_data."""
    xyz = rng.normal(size=(n, 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    coord = (xyz * [0.6, 0.3, 0.2]).astype(np.float32)
    return dict(
        coord=coord,
        normal=xyz.astype(np.float32),
        segment=np.zeros(n, dtype=np.int32),
        cls_token=0,
        name="synthetic",
    )


@pytest.fixture(scope="module")
def setup():
    cfg = Config.fromfile(CONFIG)
    # smaller heads keep the test light; shapes are what matters
    cfg.model.head_hidden_channels = 256
    cfg.model.head_num_prototypes = 256
    torch.manual_seed(0)
    model = build_model(cfg.model).cuda().train()
    transform = Compose(cfg.data.train.transform)
    rng = np.random.default_rng(0)
    np.random.seed(0)
    batch = collate_fn([transform(synthetic_shape(rng)) for _ in range(4)])
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    return cfg, model, batch


def test_head_in_channels(setup):
    cfg, model, _ = setup
    assert cfg.model.head_in_channels == 144 + 252 + 504
    assert model.student.mask_head.mlp[0].in_features == 900


def test_forward_backward(setup):
    cfg, model, batch = setup
    matches = {}
    match_neighbour = model.match_neighbour

    def counting_match(*args):
        index = match_neighbour(*args)
        matches.setdefault("n", []).append(index.shape[0])
        return index

    model.match_neighbour = counting_match
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(dict(batch))
    finally:
        del model.match_neighbour
    for key in ("loss", "mask_loss", "roll_mask_loss", "unmask_loss"):
        assert torch.isfinite(out[key]), key
    assert len(matches["n"]) == 3 and min(matches["n"]) > 0, matches

    model.zero_grad()
    out["loss"].backward()
    student = dict(model.student.named_parameters())
    assert student["backbone.embedding.mask_token"].grad.abs().sum() > 0
    for name, p in student.items():
        if p.requires_grad:
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name
    assert all(p.grad is None for p in model.teacher.parameters())


def test_layer_wise_lr_keywords(setup):
    cfg, model, _ = setup
    names = [n for n, p in model.student.named_parameters()]
    for group in cfg.param_dicts:
        assert any(group.keyword in n for n in names), group.keyword
    # keywords must not overlap (enc0.block1. vs enc0.block10. etc.)
    for group in cfg.param_dicts:
        hits = [g.keyword for g in cfg.param_dicts if g.keyword in group.keyword]
        assert hits == [group.keyword]


def test_return_point_up_cast(setup):
    _, model, batch = setup
    data = dict(
        feat=batch["global_feat"],
        coord=batch["global_coord"],
        offset=batch["global_offset"],
        grid_size=batch["grid_size"][0],
    )
    with torch.no_grad():
        point = model(data, return_point=True)["point"]
    assert point.feat.shape[1] == 900


def test_target_stats(setup):
    _, model, batch = setup
    model.log_target_stats = True
    try:
        with torch.no_grad():
            out = model(dict(batch))
    finally:
        model.log_target_stats = False
    k = model.student.mask_head.prototype.out_features
    for name in ("mask", "unmask"):
        assert 1 <= out[f"{name}_target_ppl"] <= k + 1e-3
        assert 1 <= out[f"{name}_pred_ppl"] <= k + 1e-3
        assert 0 < out[f"{name}_proto_usage"] <= 1
