"""
Ablation (Tab. 4): LitePT-B trained on land-cover segmentation only.
Same as ../segment/litept-b-v1m0-malibu3d.py, which it inherits, except lr 5e-4 and 4 GPUs with
SyncBN.
"""

_base_ = ["../segment/litept-b-v1m0-malibu3d.py"]

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
