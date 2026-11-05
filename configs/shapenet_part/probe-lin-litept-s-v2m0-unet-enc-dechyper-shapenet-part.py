"""
Linear probe of the encoder up-cast (1008 channels, as in enc_mode probes) concatenated with the
decoder hypercolumn (1044 channels) of a frozen Sonata LitePT-v2-S U-Net on ShapeNetPart: 2052
channels. Everything else is probe-lin-litept-s-v2m0-unet-dechyper-shapenet-part.py.
"""

_base_ = ["probe-lin-litept-s-v2m0-unet-dechyper-shapenet-part.py"]

wandb_run_name = (
    "Sonata LitePT-v2-S U-Net lin probe ShapeNetPart (encoder + decoder hypercolumn)"
)

model = dict(backbone_out_channels=2052, backbone=dict(enc_upcast_output=True))
