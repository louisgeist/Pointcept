"""Utilities for per-task gradient norm diagnostics during training."""

import torch


def l2_grad_norm(grads):
    total_sq = None
    for g in grads:
        if g is None:
            continue
        sq = g.detach().pow(2).sum()
        total_sq = sq if total_sq is None else total_sq + sq
    if total_sq is None:
        return 0.0
    return float(total_sq.sqrt().item())


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


def compute_task_gradient_norms(model, loss_by_task, task_weights, accum_steps):
    """Per-task L2 grad norms and pairwise backbone cosine similarities."""
    backbone_params = model.backbone_parameters()
    norms = {}
    backbone_grads_by_task = {}
    for task_name, task_loss in loss_by_task.items():
        w = float(task_weights.get(task_name, 1.0))
        scaled_loss = (task_loss * w) / accum_steps
        head_params = model.task_head_parameters(task_name)
        all_params = backbone_params + head_params
        if not all_params:
            norms[task_name] = {"backbone": 0.0, "head": 0.0}
            backbone_grads_by_task[task_name] = None
            continue
        grads = torch.autograd.grad(
            scaled_loss,
            all_params,
            retain_graph=True,
            allow_unused=True,
        )
        n_bb = len(backbone_params)
        backbone_grads = grads[:n_bb]
        norms[task_name] = {
            "backbone": l2_grad_norm(backbone_grads),
            "head": l2_grad_norm(grads[n_bb:]),
        }
        backbone_grads_by_task[task_name] = _flatten_grads(backbone_grads)
    return {
        "norms": norms,
        "backbone_cos": _pairwise_backbone_cosine_similarities(backbone_grads_by_task),
    }
