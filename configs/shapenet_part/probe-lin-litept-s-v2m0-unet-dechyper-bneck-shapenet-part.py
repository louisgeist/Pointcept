"""
Linear probe of the multi-scale features of a frozen Sonata LitePT-v2-S U-Net on ShapeNetPart, with
the MALiBU3D *-lin-grid-dec convention: the 4 decoder stages and the encoder bottleneck gathered
onto the finest points, 72 + 72 + 144 + 252 + 504 = 1044 channels. Everything else is
probe-lin-litept-s-v2m0-unet-dechyper-shapenet-part.py (540 channels, without the bottleneck).
"""

_base_ = ["probe-lin-litept-s-v2m0-unet-dechyper-shapenet-part.py"]

wandb_run_name = (
    "Sonata LitePT-v2-S U-Net lin probe ShapeNetPart (decoder hypercolumn + bottleneck)"
)

model = dict(backbone_out_channels=1044, hypercolumn_bottleneck=True)
