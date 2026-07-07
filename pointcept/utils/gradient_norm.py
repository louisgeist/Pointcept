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


def compute_task_gradient_norms(model, loss_by_task, task_weights, accum_steps):
    """Per-task L2 grad norms on shared backbone and task-specific head params."""
    backbone_params = model.backbone_parameters()
    norms = {}
    for task_name, task_loss in loss_by_task.items():
        w = float(task_weights.get(task_name, 1.0))
        scaled_loss = (task_loss * w) / accum_steps
        head_params = model.task_head_parameters(task_name)
        all_params = backbone_params + head_params
        if not all_params:
            norms[task_name] = {"backbone": 0.0, "head": 0.0}
            continue
        grads = torch.autograd.grad(
            scaled_loss,
            all_params,
            retain_graph=True,
            allow_unused=True,
        )
        n_bb = len(backbone_params)
        norms[task_name] = {
            "backbone": l2_grad_norm(grads[:n_bb]),
            "head": l2_grad_norm(grads[n_bb:]),
        }
    return norms
