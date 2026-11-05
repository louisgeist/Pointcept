"""
FINO (metadata-guided Sonata) pretraining of a LitePT-v2-S encoder on ShapeNetPart: phase 4 of the
hybrid-SSL project. Sonata as in pretrain-sonata-litept-s-v2m0-shapenet-part.py at batch 80, plus
one metadata branch (pointcept/models/sonata/meta_guidance.py): the shape category (16 classes,
``cls_token``) as an informative discrete factor (M+), prototype InfoNCE on phi, the per-view mean
of the up-cast encoder features of the student's masked global views.

Choices to confirm before the first run: phi from the masked global views (the unmasked ones
cost one more student forward per step: student_view="global"), mean pooling, and the DANN ramp
over the whole run (ramp_iters = total_iters, from iteration 0, since there is no frozen phase
when training from scratch). SIGReg is off (sigreg=dict(weight=0.05) turns it on).
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
        ramp_iters=total_iters,
        ramp_start=0,
        grad_norm_momentum=0.99,
        pooling="mean",
        student_view="masked_global",
    ),
    sigreg=None,
)
