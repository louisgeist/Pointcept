"""
Linear probe of the decoder output (72 channels, full resolution) of a frozen Sonata LitePT-v2-S
U-Net (pretrain-sonata-litept-s-v2m0-unet-shapenet-part.py) on ShapeNetPart part segmentation.
Everything else is probe-lin-litept-s-v2m0-shapenet-part.py; the encoder up-cast of the same
checkpoint is probed with that config.
"""

_base_ = ["probe-lin-litept-s-v2m0-shapenet-part.py"]

wandb_run_name = "Sonata LitePT-v2-S U-Net lin probe ShapeNetPart (decoder)"

model = dict(
    backbone_out_channels=72,
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
)
