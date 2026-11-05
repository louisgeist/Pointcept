"""
Sonata pretraining of a LitePT-v2-S U-Net on ShapeNetPart with the losses on the decoder
multi-scale features plus the encoder bottleneck (Feat. Dec, Scale multi + E4: D0-D3 + E4, 1044
channels at full resolution) instead of the decoder output (D0, 72 channels) of
pretrain-sonata-litept-s-v2m0-unet-shapenet-part.py, which this config otherwise keeps (heads on a
random 20 % of the points, head_point_ratio; match_max_r 0.08 at full resolution).
"""

_base_ = ["pretrain-sonata-litept-s-v2m0-unet-shapenet-part.py"]

batch_size = 80
total_iters = 5_000

wandb_run_name = "Sonata LitePT-v2-S U-Net (Dec multi + E4) pretrain ShapeNetPart"

model = dict(
    backbone=dict(dec_traceable=True),
    feature_source="dec_hypercolumn",
    head_in_channels=1044,  # 72 + 72 + 144 + 252 + 504
)
