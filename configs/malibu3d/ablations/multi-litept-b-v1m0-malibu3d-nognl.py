"""
Ablation (Tab. 4 / A.2): LitePT-B multitask without GradNormLite.
Same as ../multi-litept-b-v1m0-malibu3d.py, which it inherits (1 GPU, batch 12, lr 1e-3), except:
grad_norm_lite=False and static task weights (each of the 4 natural-habitat axes weighted 1/4).
"""

_base_ = ["../multi-litept-b-v1m0-malibu3d.py"]

grad_norm_lite = False
# Static task weights: the 4 nathab axes together weigh as much as one task.
task_weights = {
    "segment": 1.0,
    "forest_2d": 1.0,
    "elevation": 1.0,
    "nathab_habitat_type": 0.25,
    "nathab_moisture_regime": 0.25,
    "nathab_soil_chemistry": 0.25,
    "nathab_bioclimatic_zone": 0.25,
    "network": 1.0,
}
model = dict(task_weights=task_weights)

wandb_run_name = "LPT-B multi noGNL 1.1 iter=200000"
