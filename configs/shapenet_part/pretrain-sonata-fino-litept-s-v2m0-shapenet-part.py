"""
FINO (metadata-guided Sonata) pretraining of a LitePT-v2-S encoder on ShapeNetPart: phase 4 of the
hybrid-SSL project. Sonata as in pretrain-sonata-litept-s-v2m0-shapenet-part.py at batch 80, plus
one metadata branch (pointcept/models/sonata/meta_guidance.py): the shape category (16 classes,
``cls_token``) as an informative discrete factor (M+), prototype InfoNCE on phi.

phi: per-view mean of the encoder features of the student's masked global views (the views it
already encodes for the mask losses), taken from the last encoder stage (pool_source="coarse",
504 channels) or from the multi-scale features up-cast for the heads ("upcast", 900 channels).
No DANN ramp (gamma = 1): a single M+ branch has no gradient reversal to stabilise. SIGReg is
off (sigreg=dict(weight=0.05) turns it on). To revisit for MALiBU3D.
"""

_base_ = ["pretrain-sonata-litept-s-v2m0-shapenet-part.py"]

batch_size = 80
total_iters = 10_000

wandb_run_name = "Sonata-FINO LitePT-v2-S pretrain ShapeNetPart | category M+"

model = dict(
    meta_guidance=[
        dict(
            name="category",
            key="cls_token",
            type="discrete",
            num_classes=16,
            sign="+",
        )
    ],
    meta_guidance_cfg=dict(
        weight=0.03,  # lambda_meta per branch
        tau=0.023,
        alpha=0.99,  # prototype EMA
        ramp_iters=None,  # gamma = 1
        grad_norm_momentum=0.99,
        pooling="mean",
        pool_source="coarse",
        student_view="masked_global",
    ),
    sigreg=None,
)
