"""
Volt-S on MALiBU3D with 0.3 m voxels and 5x5x5 patches (1.5 m, as the K15 ablation).
Same as ../multi-volt-s-v1m0-malibu3d.py, which it inherits, except grid_pool=3: the data
pipeline (and the distillation teacher) stay at 0.1 m; Volt averages the features of the
0.1 m voxels into 0.3 m voxels, runs on those, and broadcasts its outputs back to the
0.1 m voxels, where the losses (GT and teacher labels) and metrics are computed.
"""

_base_ = ["../multi-volt-s-v1m0-malibu3d.py"]

model = dict(backbone=dict(grid_pool=3, sparse_patches=True))

wandb_run_name = "Volt-S distill pool3 K5 (0.3 m vox, 1.5 m patch) multi iter=200000"
