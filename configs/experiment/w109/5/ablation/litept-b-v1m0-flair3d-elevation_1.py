"""
LitePT-Base on Flair3D+ (coord + RGB + strength in feat_keys).

Mono-task ablation for point-wise elevation regression (meters), 4x H100.
Standalone copy of flair3d_default/elevation/litept-b-v1m0-flair3d.py (Base
backbone + schedule from multi-litept-b). Inherits only from default_runtime.
"""

# -----------------------------------------------------------------------------
# Default
# -----------------------------------------------------------------------------
_base_ = ["../../../../_base_/default_runtime.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------

# Logging parameters
grp_exp = 1
num_exp = 1

# Hardware parameters
num_gpu = 4
num_worker = 16  # H100: fixed 16 DataLoader workers (total, not 8 * num_gpu)
sync_bn = True
enable_amp = True

# Data parameters
batch_size = 12  # total batch size across all gpus
batch_size_val = 8 * num_gpu
val_voxel_budget = 2_000_000
batch_size_test = 8 * num_gpu
test_voxel_budget = 2_000_000

grid_size = 0.1
point_max = 102400
mix_prob = 0.8

patch_size = 1024

# Optimization parameters
lr = 1e-3
total_iters = 200_000

# Features
learned_masked_feat = True
feat_keys = ["coord", "color", "strength"]
coord_feat_scale = 0.01

# Wandb parameters
wandb_run_name = (
    f"4xH100 LPT-B elevation {grp_exp}.{num_exp}) iter={total_iters}"
)
wandb_project = "flair3d_elevation"

# -----------------------------------------------------------------------------
# Mono-task regression configuration
# -----------------------------------------------------------------------------
from pointcept.datasets.flair3d_config_utils import (
    FLAIR3D_COLLECT_PREFIX_LITEPT,
    init_multitask_collect_keys,
)

target_key = "elevation"
target_keys = (target_key,)
origin_target_key = f"origin_{target_key}"

# Elevation in meters: no Collect key_scales, no denorm via target_scales
# (matches configs/experiment/w107/7/toward_bm/multi-litept-v1m0-flair3d_1.py).
target_scales = {}

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
# LitePT-Base official dims (deeper/wider than Small).
model = dict(
    type="DefaultRegressorV2",
    target_key=target_key,
    backbone_out_channels=72,
    # beta=1.0 (meters) -- elevation trained in raw meters, no Collect key_scales /
    # target_scales denorm. See multi-litept-v1m0-flair3d_1.py (w107/7/toward_bm).
    criteria=[dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0)],
    backbone=dict(
        type="LitePT-v1",
        in_channels=7,  # coord (3) + color (3) + strength (1)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(3, 3, 3, 3),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(patch_size, patch_size, patch_size, patch_size, patch_size),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 108, 216, 432),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(patch_size, patch_size, patch_size, patch_size),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=False,
    ),
    feature_mask_values=dict(
        enable=learned_masked_feat,
        masked_feat_keys=["color", "strength"],
    ),
)

# -----------------------------------------------------------------------------
# Optimizer / scheduler
# -----------------------------------------------------------------------------
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[lr, lr / 10],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=lr / 10)]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset_type = "Flair3DDataset"
data_root = "data/flair3d_plus"
csv_manifest = "data/flair3d_plus/raw/scene_split_manifest.csv"
min_points = {"train": 1000}
val_stratified_subset_manifest = "data/flair3d_plus/manifests/val_dev_subset_2000.csv"

train_collect_keys, val_collect_keys, index_valid_keys = init_multitask_collect_keys(
    target_keys, collect_prefix_keys=FLAIR3D_COLLECT_PREFIX_LITEPT
)

del FLAIR3D_COLLECT_PREFIX_LITEPT, init_multitask_collect_keys

data = dict(
    target_scales=target_scales,
    target_key=target_key,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        csv_manifest=csv_manifest,
        min_points=min_points,
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
            dict(type="Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="Collect",
                keys=train_collect_keys,
                feat_keys=feat_keys,
                feat_scales=dict(coord=coord_feat_scale),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        csv_manifest=csv_manifest,
        min_points=min_points,
        stratified_subset_manifest=val_stratified_subset_manifest,
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
            dict(type="Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="Collect",
                keys=val_collect_keys,
                feat_keys=feat_keys,
                feat_scales=dict(coord=coord_feat_scale),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        csv_manifest=csv_manifest,
        min_points=min_points,
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
