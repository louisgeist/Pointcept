"""
Ablation (Tab. 4 / A.2): LitePT-B multitask without GradNormLite.
Same as ../multi-litept-b-v1m0-malibu3d.py, which it inherits, except: grad_norm_lite=False,
static task weights (each of the 4 natural-habitat axes weighted 1/4), lr 5e-4, 4 GPUs with SyncBN.
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

# Hardware: 4 GPUs with SyncBN (batch sizes are totals across GPUs).
num_gpu = 4
sync_bn = True
num_worker = 32
batch_size_val = 32
batch_size_test = 32

# Optimization: lr 5e-4 (the optimizer, scheduler and param_dicts of the base are
# built from its lr, so they are restated here).
lr = 5e-4
optimizer = dict(lr=lr)
scheduler = dict(max_lr=[lr, lr / 10])
param_dicts = [dict(keyword="block", lr=lr / 10)]

wandb_run_name = "LPT-B multi noGNL 1.1 iter=200000"
