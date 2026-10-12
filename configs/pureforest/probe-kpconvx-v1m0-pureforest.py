"""
Frozen KPConvX (MALiBU3D multitask checkpoint, ckpt/kpconvx/model.pth) linear probe on PureForest
tile classification (13 tree species), Tab. 3 of the MALiBU3D paper.

Run with tools/probe_pureforest.py. Probe input: encoder multi-scale features (enc_mode=True, 5
stages, 928 channels), mean- and max-pooled per tile. A multinomial logistic regression is fitted
for every pooling in ``probe.aggs`` and every ``C`` in ``probe.Cs``; the best pair on validation
mIoU is tested.

PureForest has RGB but no intensity: strength is replaced by the checkpoint's learned mask value.
Test tiles overlapping MALiBU3D train/val forests are excluded
(exclude_malibu3d_leakage_tiles=True).
"""

seed = 0
grid_size = 0.1
coord_feat_scale = 0.01
kp_radius = 3.2
kp_sigma = kp_radius
radius_scaling = 3.0

data_root = "data/pureforest"
weight = "ckpt/kpconvx/model.pth"

feat_keys = ["coord", "color", "strength"]

class_names = [
    "deciduous_oak",
    "evergreen_oak",
    "beech",
    "chestnut",
    "black_locust",
    "maritime_pine",
    "scotch_pine",
    "black_pine",
    "aleppo_pine",
    "fir",
    "spruce",
    "larch",
    "douglas",
]

enc_channels = (96, 128, 192, 256, 256)

model = dict(
    type="PooledFeatureExtractor",
    backbone=dict(
        type="kpconvx_base",
        input_channels=7,
        num_classes=0,
        dim=3,
        task="cloud_segmentation",
        kp_mode="kpconvx",
        shell_sizes=(1, 14, 28),
        kp_radius=kp_radius,
        kp_aggregation="nearest",
        kp_influence="constant",
        kp_sigma=kp_sigma,
        share_kp=False,
        conv_groups=-1,
        inv_groups=8,
        inv_act="sigmoid",
        inv_grp_norm=True,
        kpx_upcut=False,
        subsample_size=grid_size,
        neighbor_limits=(12, 16, 20, 20, 20),
        layer_blocks=(3, 3, 9, 12, 3),
        init_channels=64,
        channel_scaling=1.414,
        radius_scaling=radius_scaling,
        decoder_layer=True,
        grid_pool=True,
        upsample_n=3,
        first_inv_layer=1,
        drop_path_rate=0,
        norm="batch",
        bn_momentum=0.1,
        smooth_labels=False,
        class_w=(),
        enc_mode=True,
    ),
    feature_mask_values=dict(
        enable=True,
        masked_feat_keys=["color", "strength"],
    ),
)

# Linear probe fitted by tools/probe_pureforest.py on the pooled features.
probe = dict(
    aggs=("mean", "max", "concat", "sum"),
    Cs=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3),
    max_iter=3000,
)

_fill_strength = dict(
    type="FillMissingFeat", feat_key="strength", feat_dim=1, fill_value=0.0
)

# Deterministic voxelization: one fragment, first point of each voxel.
transform = [
    dict(type="CenterShift", apply_z=True),
    dict(type="Z_MinShift"),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="test",
        test_single_fragment=True,
        return_min_coord=True,
    ),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    _fill_strength,
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "category"),
        feat_keys=feat_keys,
        feat_scales=dict(coord=coord_feat_scale),
        optional_keys=("name",),
    ),
]

data = dict(
    names=class_names,
    train=dict(
        type="PureForestDataset",
        split="train",
        data_root=data_root,
        class_names=class_names,
    ),
    val=dict(
        type="PureForestDataset",
        split="val",
        data_root=data_root,
        class_names=class_names,
    ),
    test=dict(
        type="PureForestDataset",
        split="test",
        data_root=data_root,
        class_names=class_names,
        exclude_malibu3d_leakage_tiles=True,
    ),
)
