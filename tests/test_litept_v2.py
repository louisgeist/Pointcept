"""LitePT-v2 (pointcept/models/litept/litept_v2.py): forward/backward, enc_mode up-cast shapes, mask token,
bit-exact equivalence with LitePT-v1 under the "v1" options, and the float PointROPE used by the RoPE augmentation.

Needs spconv, flash_attn and a GPU; skipped otherwise.

Run with: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest tests/test_litept_v2.py -v
"""

import pytest
import torch

spconv = pytest.importorskip("spconv")
pytest.importorskip("flash_attn")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from pointcept.models.builder import build_model
from pointcept.models.litept.litept_v1 import PointROPE
from pointcept.models.litept.litept_v2 import float_pointrope
from pointcept.models.utils.structure import Point

# LitePT-S
SMALL = dict(
    in_channels=3,
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(36, 72, 144, 252, 504),
    enc_num_head=(2, 4, 8, 14, 28),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    dec_depths=(0, 0, 0, 0),
    dec_channels=(72, 72, 144, 252),
    dec_num_head=(4, 4, 8, 14),
    dec_patch_size=(1024, 1024, 1024, 1024),
    drop_path=0.0,
)
V1_OPTIONS = dict(norm="bn", stem="conv", mask_token=False, init="default")


def make_input(n_points=(3000, 2000), grid=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    grid_coord, offset = [], []
    for n in n_points:
        codes = torch.randperm(grid**3, generator=g)[:n]
        grid_coord.append(
            torch.stack([codes // grid**2, (codes // grid) % grid, codes % grid], 1)
        )
        offset.append(n)
    grid_coord = torch.cat(grid_coord).int()
    coord = grid_coord.float() * 0.02
    return dict(
        coord=coord.cuda(),
        grid_coord=grid_coord.cuda(),
        feat=coord.clone().cuda(),
        offset=torch.cumsum(torch.tensor(offset), 0).cuda(),
        grid_size=0.02,
    )


def build(type_="LitePT-v2", seed=0, **kwargs):
    torch.manual_seed(seed)
    return build_model(dict(type=type_, **{**SMALL, **kwargs})).cuda()


def run(model, data, seed=0):
    torch.manual_seed(seed)  # shuffle_orders draws a random serialization order
    return model({k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()})


@pytest.mark.parametrize("enc_mode", [False, True])
def test_forward_backward(enc_mode):
    model = build(enc_mode=enc_mode).train()
    point = run(model, make_input())
    assert torch.isfinite(point.feat).all()
    point.feat.square().mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_enc_mode_up_cast_shapes():
    data = make_input()
    model = build(enc_mode=True).eval()
    with torch.no_grad():
        point = run(model, data)
    channels = SMALL["enc_channels"]
    assert point.feat.shape[1] == channels[-1]
    up_channels = channels[-1]
    for level in reversed(range(len(channels) - 1)):
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        assert inverse.shape[0] == parent.feat.shape[0]
        assert int(inverse.max()) == point.feat.shape[0] - 1
        assert parent.feat.shape[1] == channels[level]
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        up_channels += channels[level]
        assert parent.feat.shape[1] == up_channels
        point = parent
    # the input voxels are distinct, so the full up-cast is one row per input point
    assert point.feat.shape == (data["coord"].shape[0], sum(channels))


@pytest.mark.parametrize("stem", ["linear", "conv"])
def test_mask_token(stem):
    data = make_input()
    mask = torch.rand(data["coord"].shape[0], device="cuda") < 0.5
    model = build(stem=stem, mask_token=True).train()
    with torch.no_grad():
        model.embedding.mask_token.normal_()

    def stem_out(with_mask):
        point = Point({**data, "mask": mask} if with_mask else dict(data))
        point.sparsify()
        point = model.embedding(point)
        assert torch.equal(point.sparse_conv_feat.features, point.feat)
        return point.feat

    masked, plain = stem_out(True), stem_out(False)
    token = model.embedding.mask_token.expand(int(mask.sum()), -1)
    assert torch.equal(masked[mask], token.to(masked.dtype))
    assert torch.equal(masked[~mask], plain[~mask])

    point = run(model, {**data, "mask": mask})
    point.feat.square().mean().backward()
    assert model.embedding.mask_token.grad.abs().sum() > 0


@pytest.mark.parametrize("enc_mode", [False, True])
def test_v1_options_match_litept_v1(enc_mode):
    data = make_input()
    v1 = build("LitePT-v1", seed=1, enc_mode=enc_mode).eval()
    v2 = build(seed=2, enc_mode=enc_mode, **V1_OPTIONS).eval()
    assert list(v1.state_dict()) == list(v2.state_dict())
    v2.load_state_dict(v1.state_dict(), strict=True)
    with torch.no_grad():
        out1, out2 = run(v1, data).feat, run(v2, data).feat
    assert torch.equal(out1, out2)


def test_options_change_parameters():
    v2 = build()  # Sonata defaults: LN, linear stem, trunc_normal init
    assert isinstance(v2.embedding.stem.linear, torch.nn.Linear)
    assert isinstance(v2.embedding.stem.norm, torch.nn.LayerNorm)
    assert isinstance(v2.enc.enc1.down.norm[0], torch.nn.LayerNorm)
    assert v2.embedding.mask_token is None
    linear = v2.enc.enc3.block0.mlp[0].fc1
    assert linear.bias.abs().max() == 0
    assert abs(linear.weight.std().item() - 0.02) < 0.003

    ls = build(layer_scale=1e-5)
    gamma = ls.enc.enc4.block0.ls1[0].gamma
    assert gamma.shape == (504,) and torch.allclose(gamma, torch.full_like(gamma, 1e-5))


def test_freeze_encoder():
    model = build(freeze_encoder=True)
    for name, p in model.named_parameters():
        frozen = name.startswith(("embedding.", "enc."))
        assert p.requires_grad != frozen, name


def test_float_pointrope_matches_kernel():
    torch.manual_seed(0)
    tokens = torch.randn(1, 4, 500, 36, device="cuda")
    pos = torch.randint(0, 200, (1, 500, 3), device="cuda")
    ref = PointROPE(freq=100.0)(tokens.clone(), pos)
    out = float_pointrope(tokens, pos.float(), 100.0)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_rope_augmentation_train_only():
    data = make_input()
    aug = dict(rope_shift=2.0, rope_jitter=1.1, rope_rescale=1.2)
    plain = build(seed=0)
    augmented = build(seed=0, **aug)
    augmented.load_state_dict(plain.state_dict())
    with torch.no_grad():
        plain.eval(), augmented.eval()
        assert torch.equal(run(plain, data).feat, run(augmented, data).feat)
        plain.train(), augmented.train()
        diff = (run(plain, data).feat - run(augmented, data).feat).abs().max()
    assert diff > 1e-3
    out = run(augmented, data)
    out.feat.square().mean().backward()
    assert all(p.grad is not None for p in augmented.parameters())


def test_enc_upcast_output_matches_enc_mode():
    data = make_input()
    unet = build(enc_upcast_output=True, dec_traceable=True).eval()
    enc = build(enc_mode=True).eval()
    enc.load_state_dict(
        {k: v for k, v in unet.state_dict().items() if not k.startswith("dec.")}
    )
    with torch.no_grad():
        point = run(unet, data)
        coarse = run(enc, data)
    finest, expected = enc.encoder_upcast(coarse)
    assert torch.equal(point["enc_upcast_feat"], expected)
    assert expected.shape == (data["coord"].shape[0], sum(SMALL["enc_channels"]))
