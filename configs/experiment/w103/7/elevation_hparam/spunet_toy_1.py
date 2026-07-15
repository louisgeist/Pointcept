"""
Toy SpUNet elevation hparam ablation (overfit_minimal).

aug off, no Z_RandomOffset, warmup=50, wd=0. Small subset (D067 manifest, max_sample caps), DefaultRegressorV2 on elevation only.
"""

# -----------------------------------------------------------------------------
# Default
# -----------------------------------------------------------------------------
_base_ = ["../../../../_base_/default_runtime.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------

# Logging parameters
grp_exp = 2
num_exp = 1

# Reproducibility
seed = 14028665

# Hardware parameters
num_gpu = 1
num_worker= 20 * num_gpu
enable_amp = True

# Data parameters
batch_size = 20 * num_gpu  # total batch size across all gpus
batch_size_val = batch_size // 2
batch_size_test = batch_size // 2

# One unique scene; train `loop` repeats it so DataLoader can form full batches
# (each index maps to the same tile via idx % len(data_list)).
overfit_unique_samples = 1
overfit_train_loop = batch_size

train_max_sample = overfit_unique_samples
val_max_sample = overfit_unique_samples
test_max_sample = overfit_unique_samples

grid_size = 0.1
point_max = 100000
mix_prob = 0.8

# Optimization parameters
lr = 1e-3
total_iters = 1000
iter_per_epoch = 10
eval_every = 1
warmup_iters = 50

# Features
learned_masked_feat = True
feat_keys = ["coord", "color", "strength"]
coord_feat_scale = 0.01
feat_scales = dict(coord=coord_feat_scale)

# Wandb parameters
wandb_run_name = (
    f"SpUNet elev overfit (2.1) | overfit_minimal"
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
    dict(type="InformationWriter", log_interval=1),
    dict(type="RegressionEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

test_single_fragment = True
test = dict(type="RegressionTester", verbose=True)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
backbone_channels = (32, 64, 128, 256, 256, 128, 96, 96)

model = dict(
    type="DefaultRegressorV2",
    target_key=target_key,
    backbone_out_channels=backbone_channels[-1],
    criteria=[dict(type="SmoothL1Loss", beta=ELEVATION_SMOOTH_L1_BETA, loss_weight=1.0)],
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=7,  # feat_keys channels
        num_classes=0,
        channels=backbone_channels,
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    feature_mask_values=dict(
        enable=True,
        masked_feat_keys=["color", "strength"],
    ),
)

# -----------------------------------------------------------------------------
# Optimizer / scheduler
# -----------------------------------------------------------------------------
optimizer = dict(type="AdamW", lr=lr, weight_decay=0)
scheduler = dict(
    type="LinearLR",
    start_factor=1 / 10,
    total_iters=warmup_iters,
)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset_type = "Flair3DDataset"
data_root = "data/flair3d_plus"
csv_manifest = "data/flair3d_plus/raw/scene_split_manifest_D067.csv"
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
        max_sample=train_max_sample,
        loop=overfit_train_loop,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(index_valid_keys)},
            ),
            dict(type="CenterShift", apply_z=True),
            dict(type="Z_MinShift"),
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
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=train_collect_keys,
                feat_keys=feat_keys,
                feat_scales=feat_scales,
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
        max_sample=val_max_sample,
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
        split="val",
        data_root=data_root,
        csv_manifest=csv_manifest,
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
        target_keys=list(target_keys),
        primary_target_key=target_key,
        max_sample=test_max_sample,
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
