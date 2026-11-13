"""
Ablation: Sonata MALiBU3D pretraining without the mask loss.
Same as ../pretrain-sonata-v1m2-malibu3d.py, which it inherits, except mask_loss_weight=0: the
student still sees the masked global views through the roll mask loss (cross-view prediction),
and the unmask (local -> global) loss is unchanged. The other weights are not renormalized.
"""

_base_ = ["../pretrain-sonata-v1m2-malibu3d.py"]

model = dict(mask_loss_weight=0.0)

wandb_run_name = "Sonata-v1m2 pretrain MALiBU3D | no mask loss | bs=96 | grid=0.1 | iters=150000"
