"""FINO metadata guidance (pointcept/models/sonata/meta_guidance.py), on CPU: gradient scaling and reversal,
DANN ramp, per-view pooling, prototype bank updates, gradient-norm equalisation, SIGReg.

Run with: PYTHONPATH=. pytest tests/test_meta_guidance.py -v
"""

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_scatter")

from pointcept.models.sonata.meta_guidance import (
    DiscreteBranch,
    MetaGuidance,
    dann_ramp,
    pool_views,
    scale_grad,
    sigreg_loss,
)


def test_scale_grad_reverses_and_scales():
    x = torch.randn(5, 3, requires_grad=True)
    scale_grad(x, -0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.5))
    y = torch.randn(5, 3, requires_grad=True)
    assert torch.equal(scale_grad(y, 2.0), y)


def test_dann_ramp():
    assert dann_ramp(0, 1000) == 0
    assert dann_ramp(1000, 1000) == pytest.approx(2 / (1 + math.exp(-10)) - 1)
    assert dann_ramp(100, 1000) < dann_ramp(500, 1000) < 1
    assert dann_ramp(-5, 1000) == 0
    assert dann_ramp(10, None) == 1


def test_pool_views():
    feat = torch.tensor([[1.0], [3.0], [5.0], [2.0], [4.0]])
    offset = torch.tensor([2, 5])
    assert pool_views(feat, offset, "mean").view(-1).tolist() == pytest.approx([2.0, 11 / 3])
    assert pool_views(feat, offset, "max").view(-1).tolist() == [3.0, 5.0]


def test_prototypes_update_and_normalise():
    branch = DiscreteBranch(in_channels=4, num_classes=3, alpha=0.9)
    assert not any(p.requires_grad for p in branch.parameters())
    before = branch.prototype.clone()
    phi = torch.tensor([[2.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 3.0, 0]])
    branch.update(phi, torch.tensor([0, 0, 1]))
    # first time a class is seen: its prototype is the normalised class mean
    assert torch.allclose(
        branch.prototype[0], F.normalize(torch.tensor([1.0, 1, 0, 0]), dim=0)
    )
    assert torch.allclose(branch.prototype[1], torch.tensor([0, 0, 1.0, 0]))
    assert torch.equal(branch.prototype[2], before[2])  # unseen class untouched
    assert branch.initialised.tolist() == [True, True, False]
    # then EMA towards the new mean, re-normalised
    branch.update(torch.tensor([[0, 0, 0, 5.0]]), torch.tensor([1]))
    expected = F.normalize(
        0.9 * torch.tensor([0, 0, 1.0, 0]) + 0.1 * torch.tensor([0, 0, 0, 1.0]), dim=0
    )
    assert torch.allclose(branch.prototype[1], expected)
    assert torch.allclose(branch.prototype.norm(dim=1), torch.ones(3))


def _meta(sign, **kwargs):
    torch.manual_seed(0)
    return MetaGuidance(
        8,
        [dict(name="h", type="continuous", range=(0.0, 10.0), sign=sign)],
        **kwargs,
    )


def test_sign_reverses_backbone_gradient_only():
    grads = {}
    for sign in ("+", "-"):
        meta = _meta(sign).eval()  # eval: no dropout, no buffer update
        phi = torch.randn(
            6, 8, generator=torch.Generator().manual_seed(1), requires_grad=True
        )
        loss, _ = meta(
            phi, phi.detach(), {"h": torch.linspace(0, 10, 6)}, {"h": torch.zeros(6)}
        )
        loss.backward()
        grads[sign] = (phi.grad.clone(), meta.branches["h"].mlp[0].weight.grad.clone())
    assert torch.allclose(grads["+"][0], -grads["-"][0])  # reversed at phi
    assert torch.allclose(grads["+"][1], grads["-"][1])  # the branch always minimises
    assert grads["+"][0].abs().sum() > 0


def test_ramp_scales_the_backbone_gradient():
    meta = _meta("+", ramp_iters=100).eval()
    phi = torch.randn(4, 8, requires_grad=True)
    loss, logs = meta(phi, phi.detach(), {"h": torch.ones(4)}, {"h": torch.ones(4)})
    loss.backward()
    assert logs["meta_gamma"] == 0 and phi.grad.abs().sum() == 0
    meta.train()
    meta.step.fill_(100)
    _, logs = meta(phi, phi.detach(), {"h": torch.ones(4)}, {"h": torch.ones(4)})
    assert logs["meta_gamma"] == pytest.approx(dann_ramp(100, 100))
    assert int(meta.step) == 101


def test_discrete_and_continuous_with_grad_norm_equalisation():
    meta = MetaGuidance(
        8,
        [
            dict(name="cat", key="cls", type="discrete", num_classes=4, sign="+"),
            dict(name="year", type="continuous", range=(2020, 2024), sign="-"),
        ],
    ).train()
    student = {
        "cls": torch.tensor([0, 1, 2, 3, 0, 1]),
        "year": torch.tensor([2020, 2021, 2022, 2023, 2024, 2021]),
    }
    teacher = {"cls": student["cls"], "year": student["year"]}
    for step in range(3):
        phi = torch.randn(6, 8, requires_grad=True)
        loss, logs = meta(phi, torch.randn(6, 8), student, teacher)
        loss.backward()
        assert torch.isfinite(loss)
        if step == 0:
            assert logs["meta_cat_scale"] == 1 and logs["meta_year_scale"] == 1
    assert (meta.grad_norm_ema > 0).all()
    scales = torch.tensor(meta.branch_scales())
    assert scales.log().sum().abs() < 1e-5  # geometric mean of the scales is 1
    assert {"meta_cat_loss", "meta_cat_acc", "meta_year_loss", "meta_year_mae"} <= set(
        logs
    )
    assert meta.branches["cat"].initialised.all()


def test_sigreg_prefers_gaussian_embeddings():
    torch.manual_seed(0)
    gaussian = sigreg_loss(torch.randn(4096, 64))
    collapsed = sigreg_loss(torch.ones(4096, 64) + 0.01 * torch.randn(4096, 64))
    assert gaussian < 0.01 < collapsed
