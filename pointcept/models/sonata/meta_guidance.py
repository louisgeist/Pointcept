"""
Metadata guidance (FINO, arXiv 2606.05107, Sec. 3 and App. A.4) for Sonata pretraining.

One branch per metadata factor reads a global embedding phi of each view (Sonata has no CLS
token: phi is the per-view mean or max of the encoder features up-cast for the heads):

- discrete factor (``type="discrete"``): a bank of ``num_classes`` l2-normalised prototypes (a
  buffer, no gradient); InfoNCE in cosine between the normalised student phi and the prototypes,
  temperature ``tau`` (0.023); after each training forward, every class present in the batch moves
  its prototype towards the mean of the normalised teacher phi of that class (EMA ``alpha``, 0.99),
  then the prototype is re-normalised. Under DDP the per-class sums and counts are all-reduced. A
  prototype is set to its class mean the first time the class is seen.
- continuous factor (``type="continuous"``): an MLP g (hidden 512 / 512 / 256, GELU, dropout 0.5)
  with a sigmoid (target min-max normalised to [0, 1], ``range=(min, max)``) or tanh ([-1, 1])
  output, L2 loss.

Sign: ``sign="+"`` (informative factor, M+) lets the branch gradient reach the backbone;
``sign="-"`` (spurious factor, M-) reverses it (gradient reversal between phi and the branch). The
branch's own parameters always minimise its loss. The gradient sent to the backbone is scaled by
the DANN ramp gamma(s) = 2 / (1 + exp(-10 s / S)) - 1 (``ramp_iters`` = S, s counted from
``ramp_start``; ``ramp_iters=None`` gives gamma = 1). Each branch loss is weighted by ``weight``
(lambda_meta, 0.03). With two branches or more, the gradients at phi are equalised: s_t =
geometric mean over branches of the EMA (``grad_norm_momentum``, 0.99) of ||d L_t / d phi||,
divided by branch t's EMA, detached. s_t comes from the EMA of the previous steps (one-step lag,
so each branch runs once per step); it is 1 until every branch has a gradient norm.
"""

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


def _world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


class _ScaleGrad(torch.autograd.Function):
    """Identity forward; multiplies the gradient by ``coeff`` (negative = reversal)."""

    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.coeff, None


def scale_grad(x, coeff):
    return _ScaleGrad.apply(x, float(coeff))


def dann_ramp(step, ramp_iters):
    """gamma(s) = 2 / (1 + exp(-10 s / S)) - 1, from 0 to 1 (1 without a ramp)."""
    if not ramp_iters:
        return 1.0
    return 2.0 / (1.0 + math.exp(-10.0 * max(step, 0) / ramp_iters)) - 1.0


def pool_views(feat, offset, mode="mean"):
    """One embedding per view (point cloud) of a batch split by ``offset``."""
    indptr = F.pad(offset.long(), (1, 0))
    return torch_scatter.segment_csr(feat, indptr, reduce=mode)


class DiscreteBranch(nn.Module):
    def __init__(self, in_channels, num_classes, tau=0.023, alpha=0.99):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.alpha = alpha
        self.register_buffer(
            "prototype", F.normalize(torch.randn(num_classes, in_channels), dim=1)
        )
        self.register_buffer("initialised", torch.zeros(num_classes, dtype=torch.bool))

    def forward(self, phi, label):
        # a copy: update() modifies the bank in place before the backward
        logits = F.normalize(phi, dim=1) @ self.prototype.clone().t() / self.tau
        loss = F.cross_entropy(logits.float(), label)
        acc = (logits.argmax(1) == label).float().mean().detach()
        return loss, dict(acc=acc)

    @torch.no_grad()
    def update(self, phi_teacher, label):
        phi = F.normalize(phi_teacher.float(), dim=1)
        sums = torch.zeros_like(self.prototype).index_add_(0, label, phi)
        counts = torch.bincount(label, minlength=self.num_classes).to(phi.dtype)
        if _world_size() > 1:
            dist.all_reduce(sums)
            dist.all_reduce(counts)
        seen = counts > 0
        mean = sums[seen] / counts[seen, None]
        first = seen & ~self.initialised
        proto = self.prototype.clone()
        proto[seen] = self.alpha * proto[seen] + (1 - self.alpha) * mean
        proto[first] = mean[first[seen]]
        self.prototype.copy_(F.normalize(proto, dim=1))
        self.initialised |= seen


class ContinuousBranch(nn.Module):
    def __init__(
        self,
        in_channels,
        value_range,
        activation="sigmoid",
        hidden=(512, 512, 256),
        dropout=0.5,
    ):
        super().__init__()
        assert activation in ("sigmoid", "tanh"), activation
        self.activation = activation
        self.register_buffer(
            "value_range", torch.tensor(value_range, dtype=torch.float)
        )
        layers, c = [], in_channels
        for h in hidden:
            layers += [nn.Linear(c, h), nn.GELU(), nn.Dropout(dropout)]
            c = h
        layers.append(nn.Linear(c, 1))
        self.mlp = nn.Sequential(*layers)

    def normalise(self, value):
        lo, hi = self.value_range
        unit = (value.float() - lo) / (hi - lo)
        return unit if self.activation == "sigmoid" else 2 * unit - 1

    def forward(self, phi, value):
        out = self.mlp(phi).squeeze(-1).float()
        out = torch.sigmoid(out) if self.activation == "sigmoid" else torch.tanh(out)
        target = self.normalise(value)
        loss = F.mse_loss(out, target)
        return loss, dict(mae=(out - target).abs().mean().detach())


class MetaGuidance(nn.Module):
    """The metadata branches of one model (see the module docstring).

    ``branches``: list of dicts with ``name``, ``type`` ("discrete" or "continuous"),
    ``sign`` ("+" or "-"), ``key`` (data_dict key of the metadata, default ``name``),
    ``weight`` (default ``weight``), and ``num_classes`` (discrete) or ``range`` /
    ``activation`` (continuous).
    """

    def __init__(
        self,
        in_channels,
        branches,
        weight=0.03,
        tau=0.023,
        alpha=0.99,
        ramp_iters=None,
        ramp_start=0,
        grad_norm_momentum=0.99,
    ):
        super().__init__()
        self.specs = []
        modules = {}
        for b in branches:
            b = dict(b)
            assert b["sign"] in ("+", "-"), b
            name = b["name"]
            if b["type"] == "discrete":
                modules[name] = DiscreteBranch(
                    in_channels, b["num_classes"], tau=b.get("tau", tau), alpha=alpha
                )
            elif b["type"] == "continuous":
                modules[name] = ContinuousBranch(
                    in_channels, b["range"], activation=b.get("activation", "sigmoid")
                )
            else:
                raise ValueError(f"unknown metadata branch type {b['type']!r}")
            self.specs.append(
                dict(
                    name=name,
                    type=b["type"],
                    sign=1.0 if b["sign"] == "+" else -1.0,
                    key=b.get("key", name),
                    weight=b.get("weight", weight),
                )
            )
        self.branches = nn.ModuleDict(modules)
        self.ramp_iters = ramp_iters
        self.ramp_start = ramp_start
        self.grad_norm_momentum = grad_norm_momentum
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("grad_norm_ema", torch.zeros(len(self.specs)))

    def gamma(self):
        return dann_ramp(int(self.step) - self.ramp_start, self.ramp_iters)

    def branch_scales(self):
        """s_t (detached): geometric mean of the EMA gradient norms at phi over the
        EMA of branch t, from the EMA of the previous steps; all 1 with a single
        branch or before the first update."""
        ema = self.grad_norm_ema
        if len(self.specs) < 2 or not bool((ema > 0).all()):
            return [1.0] * len(self.specs)
        geo = ema.log().mean().exp()
        return (geo / ema).tolist()

    @torch.no_grad()
    def update_grad_norms(self, phis, losses):
        """EMA of ||d L_t / d phi|| (taken at the input of each branch, i.e. before
        its gradient scaling)."""
        norms = torch.stack(
            [
                torch.autograd.grad(loss, phi, retain_graph=True)[0].norm()
                for phi, loss in zip(phis, losses)
            ]
        ).float()
        if _world_size() > 1:
            dist.all_reduce(norms, op=dist.ReduceOp.AVG)
        m = self.grad_norm_momentum
        ema = self.grad_norm_ema
        ema.copy_(torch.where(ema > 0, m * ema + (1 - m) * norms, norms))

    def forward(self, phi_student, phi_teacher, student_meta, teacher_meta):
        """Weighted metadata loss and log scalars. phi_student [V, C] (with grad) and
        phi_teacher [W, C] (no grad) hold one row per view; ``student_meta[key]`` /
        ``teacher_meta[key]`` one metadata value per student / teacher view."""
        gamma = self.gamma()
        scales = self.branch_scales()
        total = 0.0
        logs = {"meta_gamma": torch.tensor(gamma, device=phi_student.device)}
        phis, losses = [], []
        for spec, s_t in zip(self.specs, scales):
            # each branch reads phi through its own gradient scaling: sign (gradient
            # reversal for M-), DANN ramp, gradient-norm equalisation
            phi = scale_grad(phi_student, spec["sign"] * gamma * s_t)
            loss, extra = self.branches[spec["name"]](phi, student_meta[spec["key"]])
            total = total + spec["weight"] * loss
            phis.append(phi)
            losses.append(loss)
            name = spec["name"]
            logs[f"meta_{name}_loss"] = loss.detach()
            logs[f"meta_{name}_scale"] = torch.tensor(s_t, device=phi.device)
            logs.update({f"meta_{name}_{k}": v for k, v in extra.items()})
        if self.training and len(self.specs) > 1 and phi_student.requires_grad:
            self.update_grad_norms(phis, losses)
        if self.training:
            for spec in self.specs:
                branch = self.branches[spec["name"]]
                if spec["type"] == "discrete":
                    branch.update(phi_teacher, teacher_meta[spec["key"]])
            self.step += 1
        return total, logs


def sigreg_loss(z, num_slices=256, num_t=17, t_max=5.0):
    """SIGReg (LeJEPA): pushes embeddings towards an isotropic standard Gaussian.

    Epps-Pulley statistic of the 1D projections of z [N, D] on ``num_slices`` random unit
    directions: the integral over t in [-t_max, t_max] of the squared distance between the
    empirical characteristic function and exp(-t^2 / 2), weighted by exp(-t^2 / 2)
    (trapezoid rule), averaged over the slices (without LeJEPA's factor N).
    """
    z = z.float()
    direction = F.normalize(torch.randn(z.shape[1], num_slices, device=z.device), dim=0)
    proj = z @ direction  # [N, S]
    t = torch.linspace(-t_max, t_max, num_t, device=z.device)
    tx = proj[..., None] * t  # [N, S, T]
    real = tx.cos().mean(0) - torch.exp(-(t**2) / 2)
    imag = tx.sin().mean(0)
    stat = (real**2 + imag**2) * torch.exp(-(t**2) / 2)
    return torch.trapezoid(stat, t, dim=-1).mean()
