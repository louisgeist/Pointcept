"""
Ablation (Tab. A.2): LitePT-B multitask with GradNorm (Chen et al., 2018) instead of GradNormLite.
Same as ../multi-litept-b-v1m0-malibu3d.py, which it inherits, except the loss balancer (the 4
natural-habitat axes form one GradNorm group) and 4 GPUs with SyncBN.
"""

_base_ = ["../multi-litept-b-v1m0-malibu3d.py"]

# Loss balancing: real GradNorm, exclusive with GradNormLite.
grad_norm_lite = False
grad_norm = True
grad_norm_alpha = 1.5
grad_norm_weight_lr = 1e-2
grad_norm_interval = 1
# The 4 nathab axes share one learned loss weight (one "nathab" group).
grad_norm_task_groups = {
    "nathab_habitat_type": "nathab",
    "nathab_moisture_regime": "nathab",
    "nathab_soil_chemistry": "nathab",
    "nathab_bioclimatic_zone": "nathab",
}

# Hardware: 4 GPUs with SyncBN (batch sizes are totals across GPUs).
num_gpu = 4
sync_bn = True
num_worker = 32
batch_size_val = 32
batch_size_test = 32

wandb_run_name = "LPT-B multi GradNorm 1.1 iter=200000"
