"""
Sonata pretraining of a LitePT-v2-S U-Net on ShapeNetPart, full decoder (down to D0), with the
losses on the stage-2 points (0.08 voxels, the points of the enc_mode setup): D2 (144) concatenated
with D1 (72) and D0 (72) averaged onto the stage-2 points through the pooling clusters
(feature_source="dec_pooled"), 288 channels. Same points, matching radius (0.32) and no point
subsampling, as pretrain-sonata-litept-s-v2m0-shapenet-part.py (enc_mode).
"""

_base_ = ["pretrain-sonata-litept-s-v2m0-unet-hyper-shapenet-part.py"]

grid_size = 0.02
match_max_r = 4 * grid_size * 2 * 2  # 0.32, as enc_mode

wandb_run_name = (
    "Sonata LitePT-v2-S U-Net (D0-D2 pooled to stage 2) pretrain ShapeNetPart"
)

model = dict(
    feature_source="dec_pooled",
    feature_stage=2,
    head_in_channels=288,  # 144 + 72 + 72
    head_point_ratio=None,
    match_max_r=match_max_r,
)
