"""
Centralized W&B metric axis definitions and tag helpers.

Call define_wandb_metrics once after wandb.init() so train/val/test charts use
Epoch and per-step charts use Iter instead of the global _step counter.

Tag helpers keep TensorBoard and W&B keys identical across train/val/test.
"""

import wandb


def class_name_slug(class_name):
    """Sanitize a class/label name for use in metric tag paths."""
    return "".join(
        c if (c.isalnum() or c in "._-") else "_"
        for c in str(class_name).strip().replace(" ", "_")
    )


def metric_tag(split, name, task=None):
    """Build a metric tag, e.g. ``val/segment/mIoU`` or ``train/mIoU``."""
    if task is None:
        return f"{split}/{name}"
    return f"{split}/{task}/{name}"


def iou_class_tag(split, slug, task=None):
    """Build a per-class IoU tag, e.g. ``val/segment/iou/building``."""
    if task is None:
        return f"{split}/iou/{slug}"
    return f"{split}/{task}/iou/{slug}"


def iou_channel_class_tag(split, channel_slug, class_slug, task=None):
    """Per-channel per-class IoU, e.g. ``val/network/iou/ROADS/Foreground``."""
    if task is None:
        return f"{split}/iou/{channel_slug}/{class_slug}"
    return f"{split}/{task}/iou/{channel_slug}/{class_slug}"


def best_miou_tag(split, task=None):
    """Build the best-mIoU tag, e.g. ``val/segment/mIoU_best``."""
    return metric_tag(split, "mIoU_best", task=task)


def define_wandb_metrics(task_names=None, semantic_task_names=None):
    """Register W&B metric axes for train/val/test and per-task namespaces.

    ``semantic_task_names`` is kept for backward compatibility and is merged
    into ``task_names`` when provided.
    """
    wandb.define_metric("Epoch")
    wandb.define_metric("Iter")

    wandb.define_metric("params/*", step_metric="Iter")
    wandb.define_metric("train_batch/*", step_metric="Iter")

    wandb.define_metric("train/*", step_metric="Epoch")
    wandb.define_metric("val/*", step_metric="Epoch")
    wandb.define_metric("val/reg/*", step_metric="Epoch")
    wandb.define_metric("test/*", step_metric="Epoch")
    wandb.define_metric("test/reg/*", step_metric="Epoch")
    wandb.define_metric("train/s_per_epoch", step_metric="Epoch")
    wandb.define_metric("val/s_per_epoch", step_metric="Epoch")
    wandb.define_metric("runtime/*", step_metric="Epoch")

    names = []
    for source in (task_names, semantic_task_names):
        if not source:
            continue
        for name in source:
            name = str(name)
            if name not in names:
                names.append(name)

    for task_name in names:
        wandb.define_metric(f"train/{task_name}/*", step_metric="Epoch")
        wandb.define_metric(f"val/{task_name}/*", step_metric="Epoch")
        wandb.define_metric(f"test/{task_name}/*", step_metric="Epoch")
