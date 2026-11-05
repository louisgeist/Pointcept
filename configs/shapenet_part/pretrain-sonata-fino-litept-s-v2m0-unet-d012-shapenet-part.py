"""
FINO on the U-Net of pretrain-sonata-litept-s-v2m0-unet-d012-shapenet-part.py (Sonata losses on
D0-D2 pooled onto the stage-2 points, 288 channels): the shape category as an M+ discrete factor,
phi = per-view mean of the encoder bottleneck E4 (504 channels) of the student's masked global
views, no DANN ramp, lambda_meta 0.03.
"""

_base_ = ["pretrain-sonata-litept-s-v2m0-unet-d012-shapenet-part.py"]

wandb_run_name = "Sonata-FINO LitePT-v2-S U-Net (D0-D2 pooled to stage 2) pretrain ShapeNetPart | category M+"

model = dict(
    meta_guidance=[
        dict(
            name="category", key="cls_token", type="discrete", num_classes=16, sign="+"
        )
    ],
    meta_guidance_cfg=dict(
        weight=0.03,
        tau=0.023,
        alpha=0.99,
        ramp_iters=None,
        grad_norm_momentum=0.99,
        pooling="mean",
        pool_source="coarse",
        student_view="masked_global",
    ),
)
