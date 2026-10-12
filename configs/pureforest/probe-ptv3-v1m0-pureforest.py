"""
Frozen PTv3 (MALiBU3D multitask checkpoint, ckpt/ptv3/model.pth) linear probe on PureForest tile
classification (13 tree species), Tab. 3 of the MALiBU3D paper.

Run with tools/probe_pureforest.py. Probe input: encoder multi-scale features (enc_mode=True, 5
stages, 992 channels), mean- and max-pooled per tile. A multinomial logistic regression is fitted
for every pooling in ``probe.aggs`` and every ``C`` in ``probe.Cs``; the best pair on validation
mIoU is tested.

PureForest has RGB but no intensity: strength is replaced by the checkpoint's learned mask value.
Test tiles overlapping MALiBU3D train/val forests are excluded
(exclude_malibu3d_leakage_tiles=True).
"""

seed = 0
grid_size = 0.1
patch_size = 1024
coord_feat_scale = 0.01

data_root = "data/pureforest"
weight = "ckpt/ptv3/model.pth"

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

enc_channels = (32, 64, 128, 256, 512)

model = dict(
    type="PooledFeatureExtractor",
    backbone=dict(
        type="PT-v3-malibu",
        in_channels=7,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(3, 3, 3, 3),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=enc_channels,
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(patch_size, patch_size, patch_size, patch_size, patch_size),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
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
        return_grid_coord=True,
    ),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    _fill_strength,
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": grid_size}),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "grid_size", "category"),
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
