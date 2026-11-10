"""
Volt-S on MALiBU3D with 1.5 m patches: 0.1 m voxels, kernel_size = stride = 15.
Same as ../multi-volt-s-v1m0-malibu3d.py, which it inherits, except the patch size.
At 0.1 m the patches are nearly empty (~18 occupied voxels out of 15^3 on a 102,400-voxel
crop, ~5.7k tokens vs ~39k at K=5), hence sparse_patches (same math, memory scaling
with the voxels instead of tokens x K^3).
"""

_base_ = ["../multi-volt-s-v1m0-malibu3d.py"]

model = dict(backbone=dict(stride=15, kernel_size=15, sparse_patches=True))

wandb_run_name = "Volt-S distill K15 (0.1 m vox, 1.5 m patch) multi iter=200000"
