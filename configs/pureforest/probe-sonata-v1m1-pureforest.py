"""
Frozen official indoor Sonata release (ckpt/sonata/pretrain-sonata-v1m1-0-base.pth) linear probe on
PureForest tile classification (13 tree species), Tab. A.1 of the MALiBU3D paper.

Run with tools/probe_pureforest.py. Probe input: encoder multi-scale features (enc_mode=True, 1232
channels) of the indoor PT-v3m2 backbone (in_channels=9: coord, color, normal; stride 2), mean- and
max-pooled per tile. A multinomial logistic regression is fitted for every pooling in ``probe.aggs``
and every ``C`` in ``probe.Cs``; the best pair on validation mIoU is tested.

Coordinates are divided by 10 to approach indoor scene scale and the grid keeps the indoor
pretraining size (0.02). PureForest RGB is used; normals are zero-filled; intensity is not used.
Test tiles overlapping MALiBU3D train/val forests are excluded
(exclude_malibu3d_leakage_tiles=True).
"""

seed = 0
grid_size = 0.02  # native indoor Sonata pretrain
coord_scale = 1 / 10  # aerial coords scaled toward indoor scene size
patch_size = 1024

data_root = "data/pureforest"
weight = "ckpt/sonata/pretrain-sonata-v1m1-0-base.pth"
# Released Sonata checkpoints store the student backbone as student.backbone.*
weight_key_map = {"student.backbone.": "backbone."}

feat_keys = ["coord", "color", "normal"]

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

enc_channels = (48, 96, 192, 384, 512)

model = dict(
    type="PooledFeatureExtractor",
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,  # coord(3) + color(3, real) + normal(3, zero)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=enc_channels,
        enc_num_head=(3, 6, 12, 24, 32),
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
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=True,
        freeze_encoder=False,
    ),
)

# Linear probe fitted by tools/probe_pureforest.py on the pooled features.
probe = dict(
    aggs=("mean", "max", "concat", "sum"),
    Cs=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3),
    max_iter=3000,
)

_fill_normal = dict(type="FillMissingFeat", feat_key="normal", feat_dim=3)

# Deterministic voxelization: one fragment, first point of each voxel.
transform = [
    dict(type="CenterShift", apply_z=True),
    dict(type="FixedScaleCoord", scale=coord_scale),
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
    _fill_normal,
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": grid_size}),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "grid_size", "category"),
        feat_keys=feat_keys,
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
