"""Utilities for per-task gradient norm diagnostics and loss balancing."""

import math

import torch
import torch.distributed as dist
import torch.nn as nn

import pointcept.utils.comm as comm


def l2_grad_norm(grads):
    total_sq = None
    for g in grads:
        if g is None:
            continue
        # float32 reduces AMP overflow when summing last-layer grads
        sq = g.detach().float().pow(2).sum()
        total_sq = sq if total_sq is None else total_sq + sq
    if total_sq is None:
        return 0.0
    value = float(total_sq.sqrt().item())
    if not math.isfinite(value):
        return 0.0
    return value


def l2_model_grad_norm(model):
    """Global L2 norm of accumulated parameter gradients."""
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    return l2_grad_norm(grads)


def snapshot_trainable_params(model):
    """Shallow snapshot {id(param): tensor} before optimizer.step()."""
    return {id(p): p.detach().clone() for p in model.parameters() if p.requires_grad}


def l2_model_update_norm(model, snapshot):
    """Global L2 norm of parameter updates after optimizer.step()."""
    total_sq = None
    for p in model.parameters():
        if not p.requires_grad:
            continue
        old = snapshot.get(id(p))
        if old is None:
            continue
        sq = (p.detach() - old).pow(2).sum()
        total_sq = sq if total_sq is None else total_sq + sq
    if total_sq is None:
        return 0.0
    return float(total_sq.sqrt().item())


def _flatten_grads(grads):
    parts = []
    for g in grads:
        if g is None:
            continue
        parts.append(g.detach().reshape(-1))
    if not parts:
        return None
    return torch.cat(parts)


def cosine_similarity_flat(a, b):
    norm_a = a.norm()
    norm_b = b.norm()
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float((a @ b / (norm_a * norm_b)).item())


def _pairwise_backbone_cosine_similarities(backbone_grads_by_task):
    task_names = list(backbone_grads_by_task.keys())
    cos_pairs = {}
    for i, task_a in enumerate(task_names):
        grad_a = backbone_grads_by_task[task_a]
        if grad_a is None:
            continue
        for task_b in task_names[i + 1 :]:
            grad_b = backbone_grads_by_task[task_b]
            if grad_b is None:
                continue
            pair_key = f"{task_a}__{task_b}"
            cos_pairs[pair_key] = cosine_similarity_flat(grad_a, grad_b)
    return cos_pairs


def _grad_norm_with_amp_probe(loss, params, probe_scale=1.0):
    """L2 grad norm of ``loss`` w.r.t. ``params``, AMP underflow-safe.

    Scene-level losses (mean-pooled multilabel/classification) produce tiny
    per-point grads that underflow to 0 in fp16 when probed without scaling.
    Multiplying the loss by ``probe_scale`` (then dividing the norm) mirrors
    GradScaler and recovers a finite norm. If the scaled probe overflows,
    fall back to an unscaled probe.
    """
    if not params:
        return 0.0
    # Detached / constant losses (e.g. all-ignore batch → new_zeros) have no
    # grad_fn; skip so GradNormLiteEMA keeps the previous scale for this task.
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        return 0.0
    scale = float(probe_scale)
    if scale != 1.0 and math.isfinite(scale) and scale > 0.0:
        grads = torch.autograd.grad(
            loss.float() * scale,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        norm = l2_grad_norm(grads) / scale
        if math.isfinite(norm) and norm > 0.0:
            return norm
    grads = torch.autograd.grad(
        loss.float(),
        params,
        retain_graph=True,
        allow_unused=True,
    )
    return l2_grad_norm(grads)


def compute_task_gradient_norms(
    model, loss_by_task, task_weights, accum_steps, probe_scale=1.0
):
    """Per-task L2 grad norms and pairwise backbone cosine similarities."""
    backbone_params = model.backbone_parameters()
    norms = {}
    backbone_grads_by_task = {}
    scale = float(probe_scale)
    use_probe = scale != 1.0 and math.isfinite(scale) and scale > 0.0
    for task_name, task_loss in loss_by_task.items():
        if not isinstance(task_loss, torch.Tensor) or not task_loss.requires_grad:
            norms[task_name] = {"backbone": 0.0, "head": 0.0}
            backbone_grads_by_task[task_name] = None
            continue
        w = float(task_weights.get(task_name, 1.0))
        scaled_loss = (task_loss * w) / accum_steps
        head_params = model.task_head_parameters(task_name)
        all_params = backbone_params + head_params
        if not all_params:
            norms[task_name] = {"backbone": 0.0, "head": 0.0}
            backbone_grads_by_task[task_name] = None
            continue
        probe = scale if use_probe else 1.0
        grads = torch.autograd.grad(
            scaled_loss.float() * probe,
            all_params,
            retain_graph=True,
            allow_unused=True,
        )
        n_bb = len(backbone_params)
        backbone_grads = grads[:n_bb]
        bb_norm = l2_grad_norm(backbone_grads) / probe
        head_norm = l2_grad_norm(grads[n_bb:]) / probe
        if use_probe and (
            not math.isfinite(bb_norm)
            or not math.isfinite(head_norm)
            or (bb_norm <= 0.0 and head_norm <= 0.0)
        ):
            grads = torch.autograd.grad(
                scaled_loss.float(),
                all_params,
                retain_graph=True,
                allow_unused=True,
            )
            backbone_grads = grads[:n_bb]
            bb_norm = l2_grad_norm(backbone_grads)
            head_norm = l2_grad_norm(grads[n_bb:])
        norms[task_name] = {
            "backbone": bb_norm,
            "head": head_norm,
        }
        backbone_grads_by_task[task_name] = _flatten_grads(backbone_grads)
    return {
        "norms": norms,
        "backbone_cos": _pairwise_backbone_cosine_similarities(backbone_grads_by_task),
    }


def _trainable_params(module):
    return [p for p in module.parameters() if p.requires_grad]


def _last_named_block(module):
    """Return the last child whose name starts with 'block', else None."""
    blocks = [
        (name, child)
        for name, child in module._modules.items()
        if name.startswith("block")
    ]
    if not blocks:
        return None
    return blocks[-1][1]


def resolve_last_backbone_layer_params(backbone):
    """Parameters of the last backbone layer (forward sense), per architecture.

    SpUNet: last BasicBlock of ``dec[0]`` (finest stage; applied last in
    ``reversed(range(num_stages))``).
    PTv3 / LitePT: last Block of ``dec.dec0``, or ``dec.dec0.up`` if no blocks.
    KPConvX: ``head`` (feature neck before external task heads).
    """
    if backbone is None:
        raise ValueError("backbone is None")

    # SpUNet family: ModuleList decoder, index 0 is last in the forward pass.
    if hasattr(backbone, "dec") and isinstance(backbone.dec, nn.ModuleList):
        if len(backbone.dec) == 0:
            raise ValueError(
                f"{type(backbone).__name__}: empty decoder; cannot resolve last layer."
            )
        last_stage = backbone.dec[0]
        last_block = _last_named_block(last_stage)
        module = last_block if last_block is not None else last_stage
        params = _trainable_params(module)
        if not params:
            raise ValueError(
                f"{type(backbone).__name__}: last decoder stage has no trainable params."
            )
        return params

    # PTv3 / LitePT: PointSequential with named stage ``dec0``.
    if hasattr(backbone, "dec") and "dec0" in getattr(backbone.dec, "_modules", {}):
        dec0 = backbone.dec.dec0
        last_block = _last_named_block(dec0)
        if last_block is not None:
            module = last_block
        elif "up" in dec0._modules:
            module = dec0.up
        else:
            module = dec0
        params = _trainable_params(module)
        if not params:
            raise ValueError(
                f"{type(backbone).__name__}: dec.dec0 has no trainable params."
            )
        return params

    # KPConvX: feature neck before external multitask heads.
    if hasattr(backbone, "head") and backbone.head is not None:
        params = _trainable_params(backbone.head)
        if not params:
            raise ValueError(
                f"{type(backbone).__name__}: head has no trainable params."
            )
        return params

    raise ValueError(
        "Unsupported backbone for last-layer gradient norms: "
        f"{type(backbone).__name__}. Supported: SpUNet, PTv3, LitePT, KPConvX."
    )


def compute_task_last_layer_grad_norms(
    model, loss_by_task, probe_scale=1.0, task_groups=None
):
    """Per-task (or per-group) L2 norm of grads w.r.t. the backbone last layer.

    Uses raw (unweighted) task losses. Under AMP, pass ``probe_scale`` > 1
    (e.g. 1024) so scene-level losses do not underflow to a zero norm in
    fp16. Does not write into ``.grad``. Non-finite / zero norms are omitted.

    ``task_groups`` optionally maps task_name -> group_name; tasks sharing a
    group have their losses summed before probing, so the returned dict is
    keyed by group_name (defaults to task_name for ungrouped tasks) and holds
    a single norm per group instead of one per task.
    """
    params = model.last_backbone_layer_parameters()
    if not params:
        return {}
    grouped_losses = {}
    for task_name, task_loss in loss_by_task.items():
        if not isinstance(task_loss, torch.Tensor) or not task_loss.requires_grad:
            continue
        group = task_groups.get(task_name, task_name) if task_groups else task_name
        grouped_losses.setdefault(group, []).append(task_loss)
    norms = {}
    for group, losses in grouped_losses.items():
        group_loss = losses[0] if len(losses) == 1 else sum(losses)
        norm = _grad_norm_with_amp_probe(
            group_loss, params, probe_scale=probe_scale
        )
        if math.isfinite(norm) and norm > 0.0:
            norms[group] = norm
    return norms


def resolve_grad_norm_lite_scales(ema, task_names, task_groups=None):
    """Per-task loss scales from a ``GradNormLiteEMA``, honoring ``task_groups``.

    Without ``task_groups`` this is just ``ema.scales(task_names)``. With it,
    every task in the same group shares the group's EMA scale (the EMA itself
    is keyed by group_name, matching ``compute_task_last_layer_grad_norms``).
    """
    task_names = list(task_names)
    if not task_groups:
        return ema.scales(task_names)
    groups = sorted(set(task_groups.get(t, t) for t in task_names))
    group_scales = ema.scales(groups)
    return {t: group_scales[task_groups.get(t, t)] for t in task_names}


def all_reduce_mean_task_norms(norms, task_names=None):
    """Average per-task norms across DDP ranks (no-op if world_size == 1)."""
    if task_names is None:
        task_names = sorted(norms.keys())
    else:
        task_names = list(task_names)
    if not task_names:
        return {}

    world_size = comm.get_world_size()
    local = {name: float(norms.get(name, 0.0)) for name in task_names}
    if world_size == 1:
        return local

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buf = torch.tensor(
        [local[name] for name in task_names],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    buf /= world_size
    return {name: float(buf[i].item()) for i, name in enumerate(task_names)}


def combine_weighted_task_losses(loss_by_task, task_weights, scales=None):
    """Combine per-task losses as total = Σ L_t * w_t * s_t (s_t defaults to 1.0)."""
    task_weights = task_weights or {}
    scales = scales or {}
    total_loss = None
    applied_scales = {}
    for task_name, task_loss in loss_by_task.items():
        w = float(task_weights.get(task_name, 1.0))
        scale = float(scales.get(task_name, 1.0))
        applied_scales[task_name] = scale
        weighted = task_loss * w * scale
        total_loss = weighted if total_loss is None else total_loss + weighted
    return total_loss, applied_scales


class GradNormLiteEMA:
    """Per-task EMA of last-layer gradient norms for loss reweighting."""

    def __init__(self, alpha=0.1, eps=1e-3):
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.ema = {}

    def update(self, norms):
        for task_name, norm in norms.items():
            value = float(norm)
            if not math.isfinite(value) or value <= 0.0:
                continue
            if task_name not in self.ema:
                self.ema[task_name] = value
            else:
                self.ema[task_name] = (
                    (1.0 - self.alpha) * self.ema[task_name] + self.alpha * value
                )

    def scale(self, task_name):
        """Return 1 / max(ema, eps), or 1.0 before the first valid observation."""
        ema = self.ema.get(task_name)
        if ema is None or not math.isfinite(ema) or ema <= 0.0:
            return 1.0
        return 1.0 / max(ema, self.eps)

    def scales(self, task_names=None):
        names = task_names if task_names is not None else list(self.ema.keys())
        return {name: self.scale(name) for name in names}


def group_task_losses(loss_by_task, task_groups=None):
    """Sum raw (graph-attached) task losses within each group.

    Returns ``{group_name: float(L_group)}`` (detached). Mirrors the grouping
    that ``compute_task_last_layer_grad_norms`` applies to the gradient probe,
    so the per-group loss ratio L_g(t)/L_g(0) lines up with the per-group norm.
    Tasks whose loss is detached / constant (all-ignore batch) are skipped.
    """
    out = {}
    for task_name, task_loss in loss_by_task.items():
        if not isinstance(task_loss, torch.Tensor) or not task_loss.requires_grad:
            continue
        group = task_groups.get(task_name, task_name) if task_groups else task_name
        out[group] = out.get(group, 0.0) + float(task_loss.detach())
    return out


class GradNormState:
    """Real GradNorm (Chen et al. 2018) learnable loss weights.

    One learnable weight per task group, trained by gradient descent on the
    GradNorm L1 objective and renormalized after every step so the weights sum
    to the number of groups. Unlike :class:`GradNormLiteEMA` this keeps genuine
    learnable parameters plus their Adam optimizer state and the per-group
    initial-loss anchors ``L_g(0)`` -- all checkpointed via
    ``CheckpointSaver`` / ``CheckpointLoader`` so a Slurm requeue does not reset
    the learned balance.

    Weight update (per :meth:`update` call, driven every ``grad_norm_interval``
    optimizer steps from ``Trainer.run_step``):

    * ``G_W^{(g)}(t) = w_g(t) * ||dL_g/dW||_2`` -- with ``w_g >= 0`` after the
      renorm this equals ``||d(w_g L_g)/dW||_2`` without a second-order graph,
      so the grad norm is probed on the detached raw loss and multiplied by the
      (differentiable) weight.
    * target ``Gbar(t) * r_g(t) ** alpha`` with ``r_g = L~_g / mean_h L~_h`` and
      ``L~_g = L_g(t) / L_g(0)`` -- treated as a constant.
    * ``L_grad = sum_g | G_W^{(g)}(t) - target_g |`` ; one Adam step on the
      weights; then ``clamp(min=clamp_min)`` and rescale to ``sum = num_groups``.
    """

    def __init__(
        self,
        group_names,
        alpha=1.5,
        weight_lr=1e-2,
        device=None,
        clamp_min=0.0,
    ):
        self.group_names = list(group_names)
        if len(self.group_names) < 2:
            raise ValueError(
                f"GradNormState needs >= 2 task groups, got {self.group_names}."
            )
        self.alpha = float(alpha)
        self.clamp_min = float(clamp_min)
        device = torch.device(device) if device is not None else torch.device("cpu")
        self.weights = torch.ones(
            len(self.group_names),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        self.optimizer = torch.optim.Adam([self.weights], lr=float(weight_lr))
        self.L0 = {}

    @property
    def num_groups(self):
        return len(self.group_names)

    def _index(self, group):
        return self.group_names.index(group)

    def capture_initial(self, loss_by_group):
        """Record ``L_g(0)`` for groups not seen yet (finite, > 0)."""
        for group, value in loss_by_group.items():
            v = float(value)
            if group in self.L0 or not math.isfinite(v) or v <= 0.0:
                continue
            self.L0[group] = v

    def update(self, grad_norms, loss_by_group):
        """One GradNorm step on the learnable weights.

        ``grad_norms``: ``{group: ||dL_g/dW||_2}`` detached floats.
        ``loss_by_group``: ``{group: current raw L_g}`` detached floats.
        Groups missing a grad norm or an ``L0`` anchor are skipped for the
        gradient this step but still take part in the sum-to-T renorm.
        Returns a dict of scalars for logging.
        """
        active = [
            g
            for g in self.group_names
            if g in grad_norms
            and g in loss_by_group
            and g in self.L0
            and math.isfinite(float(grad_norms[g]))
            and float(grad_norms[g]) > 0.0
        ]
        info = {
            "weights": {},
            "gw": {},
            "targets": {},
            "loss_ratio": {},
            "grad_norm_loss": float("nan"),
        }
        if len(active) >= 2:
            device = self.weights.device
            idx = torch.tensor(
                [self._index(g) for g in active], device=device, dtype=torch.long
            )
            g_vec = torch.tensor(
                [float(grad_norms[g]) for g in active],
                device=device,
                dtype=torch.float32,
            )
            loss_ratio = torch.tensor(
                [float(loss_by_group[g]) / self.L0[g] for g in active],
                device=device,
                dtype=torch.float32,
            )
            r = loss_ratio / loss_ratio.mean().clamp_min(1e-12)
            w_active = self.weights[idx]
            gw = w_active * g_vec  # G_W^{(g)}(t), differentiable in the weights
            target = (gw.detach().mean() * r.pow(self.alpha)).detach()
            grad_loss = (gw - target).abs().sum()
            self.optimizer.zero_grad(set_to_none=True)
            grad_loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                self.weights.data.clamp_(min=self.clamp_min)
                total = self.weights.data.sum().clamp_min(1e-12)
                self.weights.data.mul_(self.num_groups / total)
            info["grad_norm_loss"] = float(grad_loss.detach())
            for j, g in enumerate(active):
                info["gw"][g] = float(gw[j].detach())
                info["targets"][g] = float(target[j])
                info["loss_ratio"][g] = float(loss_ratio[j])
        with torch.no_grad():
            for g in self.group_names:
                info["weights"][g] = float(self.weights[self._index(g)])
        return info

    def per_task_scales(self, task_names, task_groups=None):
        """Per-task loss multiplier = the learned weight of the task's group."""
        out = {}
        with torch.no_grad():
            for t in task_names:
                g = task_groups.get(t, t) if task_groups else t
                out[t] = (
                    float(self.weights[self._index(g)])
                    if g in self.group_names
                    else 1.0
                )
        return out

    def state_dict(self):
        return {
            "weights": self.weights.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
            "L0": dict(self.L0),
            "group_names": list(self.group_names),
            "alpha": self.alpha,
            "clamp_min": self.clamp_min,
        }

    def load_state_dict(self, state):
        saved = list(state.get("group_names", self.group_names))
        if saved != self.group_names:
            raise ValueError(
                "GradNormState group set/order changed since checkpoint: "
                f"{saved} -> {self.group_names}"
            )
        with torch.no_grad():
            self.weights.data.copy_(
                state["weights"].to(self.weights.device, dtype=self.weights.dtype)
            )
        self.optimizer.load_state_dict(state["optimizer"])
        self.L0 = {k: float(v) for k, v in dict(state.get("L0", {})).items()}
        self.alpha = float(state.get("alpha", self.alpha))
        self.clamp_min = float(state.get("clamp_min", self.clamp_min))


def build_grad_norm_state(cfg, model, device=None):
    """Construct a :class:`GradNormState` from a config + a multi-task model.

    Group names are derived from ``model.tasks`` and ``cfg.grad_norm_task_groups``
    -- no data batch needed, so this works from both ``Trainer.__init__`` and
    ``CheckpointLoader`` before the first step.
    """
    tasks = list(getattr(model, "tasks", []) or [])
    if not tasks:
        raise ValueError(
            "build_grad_norm_state: model has no `tasks` "
            "(grad_norm needs MultiTaskSegmentorV2)."
        )
    task_groups = getattr(cfg, "grad_norm_task_groups", None)
    groups = sorted(
        set(task_groups.get(t, t) if task_groups else t for t in tasks)
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return GradNormState(
        group_names=groups,
        alpha=getattr(cfg, "grad_norm_alpha", 1.5),
        weight_lr=getattr(cfg, "grad_norm_weight_lr", 1e-2),
        device=device,
        clamp_min=getattr(cfg, "grad_norm_clamp_min", 0.0),
    )
