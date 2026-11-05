"""
Sonata pretraining of a LitePT-v2-S U-Net on ShapeNetPart: the Sonata losses are applied to the
decoder output (full resolution, 72 channels) instead of the encoder features up-cast over the
last three stages. Phase 3 of the hybrid-SSL project (decoder vs no decoder): everything else is
pretrain-sonata-litept-s-v2m0-shapenet-part.py.

Decoder: the one of the upstream LitePT-S semantic segmentation config
(configs/scannet/semseg-litept-v1m2-0-small.py): dec_depths 2-2-2-2, channels 72-72-144-252,
conv blocks in the three finest stages and attention in the coarsest.
The matching radius keeps the 4x margin over the voxel size, now at full resolution.
"""

_base_ = ["pretrain-sonata-litept-s-v2m0-shapenet-part.py"]

grid_size = 0.02
up_cast_level = 0
match_max_r = 4 * grid_size  # 0.08

wandb_run_name = "Sonata LitePT-v2-S U-Net pretrain ShapeNetPart"

model = dict(
    backbone=dict(
        enc_mode=False,
        dec_depths=(2, 2, 2, 2),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(True, True, True, False),
        dec_attn=(False, False, False, True),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
    ),
    head_in_channels=72,  # dec_channels[0]
    # Heads on a random 20 % of the decoder points: (a) supervises the ~18 % of the points
    # left at its matching level (stage 2, 0.02 grid), and all points do not fit at batch
    # 80 (43 GiB at batch 32 with every point).
    head_point_ratio=0.2,
    up_cast_level=up_cast_level,
    match_max_r=match_max_r,
)
