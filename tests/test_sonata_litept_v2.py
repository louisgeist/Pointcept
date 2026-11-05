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


UNET_CONFIG = (
    "configs/shapenet_part/pretrain-sonata-litept-s-v2m0-unet-shapenet-part.py"
)


def test_unet_decoder_losses():
    """U-Net variant: losses on the 72-d decoder output, on a random subset of points."""
    cfg = Config.fromfile(UNET_CONFIG)
    assert cfg.model.head_in_channels == cfg.model.backbone.dec_channels[0] == 72
    assert cfg.model.up_cast_level == 0 and cfg.model.head_point_ratio == 0.2
    cfg.model.head_hidden_channels = 256
    cfg.model.head_num_prototypes = 256
    torch.manual_seed(0)
    model = build_model(cfg.model).cuda().train()
    transform = Compose(cfg.data.train.transform)
    rng = np.random.default_rng(0)
    np.random.seed(0)
    batch = collate_fn([transform(synthetic_shape(rng)) for _ in range(4)])
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}

    with torch.no_grad():
        full = model.student.backbone(
            dict(
                feat=batch["global_feat"],
                coord=batch["global_coord"],
                origin_coord=batch["global_origin_coord"],
                offset=batch["global_offset"],
                grid_size=batch["grid_size"][0],
            )
        )
    assert full.feat.shape == (batch["global_coord"].shape[0], 72)
    sub = model._subsample_points(full)
    assert 0.1 < sub.feat.shape[0] / full.feat.shape[0] < 0.3
    assert torch.equal(sub.offset, torch.cumsum(sub.batch.bincount(), 0))

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(dict(batch))
    for key in ("loss", "mask_loss", "roll_mask_loss", "unmask_loss"):
        assert torch.isfinite(out[key]), key
    out["loss"].backward()
    decoder = [p for n, p in model.student.backbone.named_parameters() if "dec." in n]
    assert decoder and all(p.grad is not None for p in decoder)


def test_geom_probe_ridge():
    from tools.geom_probe import geom_metrics, ridge_fit

    g = torch.Generator().manual_seed(0)
    x = torch.randn(2000, 16, generator=g, dtype=torch.float64)
    normal = torch.nn.functional.normalize(
        x[:, :3] @ torch.randn(3, 3, generator=g, dtype=torch.float64), dim=1
    )
    height = x[:, 5] * 2 + 1
    predict = ridge_fit(x, torch.cat([normal, height[:, None]], 1), 1e-6)
    metrics = geom_metrics(predict(x), normal, height)
    assert metrics["height_r2"] > 0.999 and metrics["height_mae"] < 1e-2
    assert metrics["normal_abs_cos"] > 0.8


FINO_CONFIG = (
    "configs/shapenet_part/pretrain-sonata-fino-litept-s-v2m0-shapenet-part.py"
)


def _small_model_and_batch(config, **model_overrides):
    cfg = Config.fromfile(config)
    cfg.model.head_hidden_channels = 256
    cfg.model.head_num_prototypes = 256
    cfg.model.update(model_overrides)
    torch.manual_seed(0)
    model = build_model(cfg.model).cuda().train()
    transform = Compose(cfg.data.train.transform)
    rng = np.random.default_rng(0)
    np.random.seed(0)
    shapes = []
    for i in range(4):
        shape = synthetic_shape(rng)
        shape["cls_token"] = i % 3
        shapes.append(transform(shape))
    batch = collate_fn(shapes)
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    return model, batch


def _seeded_loss(model, batch):
    torch.manual_seed(1)
    with torch.no_grad():
        return model(dict(batch))["loss"]


def test_empty_meta_guidance_is_sonata():
    plain, batch = _small_model_and_batch(CONFIG)
    empty, _ = _small_model_and_batch(CONFIG, meta_guidance=[], sigreg=None)
    assert plain.meta is None and empty.meta is None
    assert list(plain.state_dict()) == list(empty.state_dict())
    empty.load_state_dict(plain.state_dict())
    assert torch.equal(_seeded_loss(plain, batch), _seeded_loss(empty, batch))


@pytest.mark.parametrize("student_view", ["masked_global", "global", "local"])
def test_fino_forward_backward(student_view):
    model, batch = _small_model_and_batch(FINO_CONFIG)
    model.meta_student_view = student_view
    assert batch["cls_token"].tolist() == [0, 1, 2, 0]
    proto_before = model.meta.branches["category"].prototype.clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(dict(batch))
    assert torch.isfinite(out["loss"]) and torch.isfinite(out["meta_category_loss"])
    assert 0 <= out["meta_category_acc"] <= 1
    out["loss"].backward()
    assert all(p.grad is None for p in model.teacher.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.student.backbone.parameters()
    )
    bank = model.meta.branches["category"]
    assert bank.initialised.tolist()[:3] == [True, True, True]
    assert not torch.equal(bank.prototype, proto_before)
    assert torch.allclose(bank.prototype.norm(dim=1), torch.ones(16, device="cuda"))
    assert int(model.meta.step) == 1


def test_sigreg_option():
    model, batch = _small_model_and_batch(FINO_CONFIG, sigreg=dict(weight=0.05))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(dict(batch))
    assert torch.isfinite(out["sigreg_loss"])
    out["loss"].backward()
