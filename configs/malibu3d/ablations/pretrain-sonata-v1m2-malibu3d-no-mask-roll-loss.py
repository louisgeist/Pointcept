"""
Ablation: Sonata MALiBU3D pretraining without the mask and roll mask losses (unmask loss only,
local -> global self-distillation, DINO-like).
Same as ../pretrain-sonata-v1m2-malibu3d.py, which it inherits, except mask_loss_weight=0 and
roll_mask_loss_weight=0 (unmask weight kept at 4/8, not renormalized). Without these losses the
student never sees a masked view: no mask head is built, the teacher uses the unmask head, and
the backbone mask token is dropped (it would be an unused parameter under DDP with
find_unused_parameters=False).
"""

_base_ = ["../pretrain-sonata-v1m2-malibu3d.py"]

model = dict(
    backbone=dict(mask_token=False),
    mask_loss_weight=0.0,
    roll_mask_loss_weight=0.0,
)

wandb_run_name = (
    "Sonata-v1m2 pretrain MALiBU3D | no mask+roll loss | bs=96 | grid=0.1 | iters=150000"
)
