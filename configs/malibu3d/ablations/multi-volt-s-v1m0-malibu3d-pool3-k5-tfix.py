"""
Volt-S pool3-k5 (0.3 m voxels, 5x5x5 patches = 1.5 m) distilled from the SpUNet retrained on
the coordinate-fixed data (JZ 2135221, epoch 200, val segment mIoU 68.8, test LC 68.4).
Same as ./multi-volt-s-v1m0-malibu3d-pool3-k5.py except the teacher checkpoint: that run
uses ckpt_cleaned/spunet (paper SpUNet, trained on the float32-quantized coordinates, LC 65.4
when tested on the fixed data). Same teacher config; the two checkpoints have identical
keys and shapes.

On Jean Zay the checkpoint is a copy of
$WORK/Pointcept-malibu3d/runs/spunet/model/model_best.pth (runs/ is on $SCRATCH, purged).
"""

_base_ = ["./multi-volt-s-v1m0-malibu3d-pool3-k5.py"]

teacher_weights = "ckpt_coordfix/spunet/model_best.pth"
model = dict(teacher=dict(weights=teacher_weights))

wandb_run_name = (
    "Volt-S distill pool3 K5 (0.3 m vox, 1.5 m patch) teacher SpUNet coordfix multi iter=200000"
)
