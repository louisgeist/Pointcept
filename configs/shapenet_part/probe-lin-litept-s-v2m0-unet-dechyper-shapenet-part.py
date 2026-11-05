"""
Linear probe of the decoder hypercolumn of a frozen Sonata LitePT-v2-S U-Net on ShapeNetPart: the
outputs of the 4 decoder stages and the encoder bottleneck gathered onto the finest points
(dec_traceable=True, as the MALiBU3D *-lin-grid-dec probes): 72 + 72 + 144 + 252 + 504 = 1044
channels. Everything else is probe-lin-litept-s-v2m0-unet-dec-shapenet-part.py.
"""

_base_ = ["probe-lin-litept-s-v2m0-unet-dec-shapenet-part.py"]

wandb_run_name = "Sonata LitePT-v2-S U-Net lin probe ShapeNetPart (decoder hypercolumn)"

model = dict(backbone_out_channels=1044, backbone=dict(dec_traceable=True))
