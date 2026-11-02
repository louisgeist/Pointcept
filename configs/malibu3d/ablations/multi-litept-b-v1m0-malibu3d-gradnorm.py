"""
Ablation (Tab. A.2): LitePT-B multitask with GradNorm (Chen et al., 2018) instead of GradNormLite.
Same as ../multi-litept-b-v1m0-malibu3d.py, which it inherits, except the loss balancer (the 4
natural-habitat axes form one GradNorm group). Same hardware and schedule (1 GPU, batch 12, lr 1e-3).
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

wandb_run_name = "LPT-B multi GradNorm 1.1 iter=200000"
