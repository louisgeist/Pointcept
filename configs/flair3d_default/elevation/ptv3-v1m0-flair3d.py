"""
PT-v3m1 on Flair3D+ (coord + RGB + strength in feat_keys; grid_coord serialization).

Mono-task Flair3D+ config for point-wise elevation regression. Inherits only from
default_runtime.
"""

# -----------------------------------------------------------------------------
# Default
# -----------------------------------------------------------------------------
_base_ = ["../../_base_/default_runtime.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------

# Logging parameters
grp_exp = 1
num_exp = 1


# Hardware parameters
num_gpu = 1
num_worker = 8 * num_gpu
enable_amp = True

# Data parameters
batch_size = 4 * num_gpu  # total batch size across all gpus
batch_size_val = batch_size // 2
batch_size_test = batch_size // 2

grid_size = 0.1
point_max = 204800
mix_prob = 0.8

patch_size = 1024

# Optimization parameters
lr = 1e-3
total_iters = 10_000
warmup_iters = 5000

# Features
learned_masked_feat = True
feat_keys = ["coord", "color", "strength"]
coord_feat_scale = 0.01

# Wandb parameters
wandb_run_name = (
    f"Flair3D+ PTv3 mono elevation {grp_exp}.{num_exp}) lr={lr}"
)
wandb_project = "flair3d_elevation"

# -----------------------------------------------------------------------------
# Mono-task regression configuration
# -----------------------------------------------------------------------------
from pointcept.datasets.flair3d_config_utils import (
    FLAIR3D_COLLECT_PREFIX_GRID,
    init_multitask_collect_keys,
    ELEVATION_TARGET_SCALE,
    ELEVATION_SMOOTH_L1_BETA,
    get_regression_target_scales,
)

target_key = "elevation"
target_keys = (target_key,)
origin_target_key = f"origin_{target_key}"

elevation_target_scale = ELEVATION_TARGET_SCALE
elevation_key_scales = dict(elevation=elevation_target_scale)
target_scales = get_regression_target_scales(target_keys)

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=100),
    dict(type="RegressionEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

test_single_fragment = True
test = dict(type="RegressionTester", verbose=True)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = dict(
    type="DefaultRegressorV2",
    target_key=target_key,
    backbone_out_channels=64,
    criteria=[dict(type="SmoothL1Loss", beta=ELEVATION_SMOOTH_L1_BETA, loss_weight=1.0)],
    backbone=dict(
        type="PT-v3m1",
        in_channels=7,  # coord (3) + color (3) + strength (1)
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(patch_size, patch_size, patch_size, patch_size, patch_size),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(patch_size, patch_size, patch_size, patch_size),
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
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    feature_mask_values=dict(
        enable=learned_masked_feat,
        masked_feat_keys=["color", "strength"],
    ),
)

# -----------------------------------------------------------------------------
# Optimizer / scheduler
# -----------------------------------------------------------------------------
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.05)
scheduler = dict(
    type="LinearLR",
    start_factor=1 / 10,
    total_iters=warmup_iters,
)
param_dicts = [dict(keyword="block", lr=lr / 10)]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset_type = "Flair3DDataset"
data_root = "data/flair3d_plus"
csv_manifest = "data/flair3d_plus/raw/scene_split_manifest.csv"
missing_tiles_manifest = "data/flair3d_plus/missing_ply_preflight.txt"
too_small_tiles_manifest = "data/flair3d_plus/too_small_tiles.csv"

train_collect_keys, val_collect_keys, index_valid_keys = init_multitask_collect_keys(
    target_keys, collect_prefix_keys=FLAIR3D_COLLECT_PREFIX_GRID
)

del FLAIR3D_COLLECT_PREFIX_GRID, init_multitask_collect_keys, get_regression_target_scales

data = dict(
    target_scales=target_scales,
    target_key=target_key,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        csv_manifest=csv_manifest,
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
        target_keys=list(target_keys),
        primary_target_key=target_key,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(index_valid_keys)},
            ),
            dict(type="CenterShift", apply_z=True),
            dict(type="Z_MinShift"),
            dict(type="Z_RandomOffset"),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=point_max, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="RandomDropColor", drop_ratio=1.0, drop_application_ratio=0.2, keep_mask=True),
            dict(type="RandomDropColor", drop_ratio=0.1, drop_application_ratio=0.5, keep_mask=True),
            dict(type="RandomDropStrength", drop_ratio=1.0, drop_application_ratio=0.2, keep_mask=True),
            dict(type="RandomDropStrength", drop_ratio=0.1, drop_application_ratio=0.5, keep_mask=True),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=train_collect_keys,
                feat_keys=feat_keys,
                feat_scales=dict(coord=coord_feat_scale),
                key_scales=elevation_key_scales,
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        csv_manifest=csv_manifest,
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
        target_keys=list(target_keys),
        primary_target_key=target_key,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(index_valid_keys)},
            ),
            dict(type="CenterShift", apply_z=True),
            dict(type="Z_MinShift"),
            dict(
                type="Copy",
                keys_dict={target_key: origin_target_key},
            ),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=val_collect_keys,
                feat_keys=feat_keys,
                feat_scales=dict(coord=coord_feat_scale),
                key_scales=elevation_key_scales,
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        csv_manifest=csv_manifest,
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
        target_keys=list(target_keys),
        primary_target_key=target_key,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Z_MinShift"),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                test_single_fragment=test_single_fragment,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    optional_keys=("inverse",),
                    feat_keys=feat_keys,
                    feat_scales=dict(coord=coord_feat_scale),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ],
        ),
    ),
)
