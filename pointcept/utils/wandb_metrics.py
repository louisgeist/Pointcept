"""
Centralized W&B metric axis definitions.

Call once after wandb.init() so train/val charts use Epoch and per-step
charts use Iter instead of the global _step counter.
"""

import wandb


def define_wandb_metrics(semantic_task_names=None):
    wandb.define_metric("Epoch")
    wandb.define_metric("Iter")

    wandb.define_metric("params/*", step_metric="Iter")
    wandb.define_metric("train_batch/*", step_metric="Iter")

    wandb.define_metric("train/*", step_metric="Epoch")
    wandb.define_metric("val/*", step_metric="Epoch")
    wandb.define_metric("val/reg/*", step_metric="Epoch")
    wandb.define_metric("train/s_per_epoch", step_metric="Epoch")
    wandb.define_metric("val/s_per_epoch", step_metric="Epoch")
    wandb.define_metric("runtime/*", step_metric="Epoch")

    for task_name in semantic_task_names or []:
        wandb.define_metric(f"train/{task_name}/*", step_metric="Epoch")
        wandb.define_metric(f"val/{task_name}/*", step_metric="Epoch")
