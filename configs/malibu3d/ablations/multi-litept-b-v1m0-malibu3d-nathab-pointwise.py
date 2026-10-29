"""
Ablation (Tab. A.3): LitePT-B multitask with point-wise CE + Lovasz supervision of the 4
natural-habitat axes instead of the tile-wise KL loss. Evaluation still pools predictions into tile
distributions. Otherwise identical to ../multi-litept-b-v1m0-malibu3d.py, which it inherits.
"""

_base_ = ["../multi-litept-b-v1m0-malibu3d.py"]

wandb_run_name = "LPT-B multi nathab-pointwise 1.1 iter=200000"

from pointcept.datasets.malibu3d_config_utils import (
    MALIBU3D_TARGET_KEYS,
    enable_nathab_pointwise_supervision,
    init_task_configs,
    init_task_criteria,
)

task_configs = enable_nathab_pointwise_supervision(
    init_task_configs(MALIBU3D_TARGET_KEYS)
)
task_criteria = init_task_criteria(task_configs)

# Delete the imported helpers: the config loader turns every non-dunder module
# attribute into a config entry, and Config.dump (yapf) fails on function objects.
del (
    MALIBU3D_TARGET_KEYS,
    enable_nathab_pointwise_supervision,
    init_task_configs,
    init_task_criteria,
)

model = dict(task_configs=task_configs, task_criteria=task_criteria)
data = dict(task_configs=task_configs)
