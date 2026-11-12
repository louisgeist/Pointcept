"""
Volt-S on MALiBU3D with 0.2 m voxels and 5x5x5 patches (1 m; ~12.4k tokens per 102,400-voxel
crop, ~7.4 occupied voxels per token, i.e. 5.9 % of the K^3 slots).
Same as ../multi-volt-s-v1m0-malibu3d.py, which it inherits, except grid_pool=2: the data
pipeline (and the distillation teacher) stay at 0.1 m; Volt averages the features of the
0.1 m voxels into 0.2 m voxels, runs on those, and broadcasts its outputs back to the
0.1 m voxels, where the losses (GT and teacher labels) and metrics are computed.
"""

_base_ = ["../multi-volt-s-v1m0-malibu3d.py"]

model = dict(backbone=dict(grid_pool=2, sparse_patches=True))

wandb_run_name = "Volt-S distill pool2 K5 (0.2 m vox, 1 m patch) multi iter=200000"
