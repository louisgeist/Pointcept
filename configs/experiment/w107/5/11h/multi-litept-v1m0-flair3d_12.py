"""
LitePT-Small on Flair3D+ multitask: same as _11 (bs=20 + GradNormLite pooled
nathab) with lr ×10.

_12: batch_size=20 + GradNormLite pooled nathab (from _11), lr=1e-2 (10× the
_11/_4-_10 default of 1e-3).

Tasks: segment (v20) + forest + elevation + 4 nathab tile_distribution axes
(Habitat Type / Moisture Regime / Soil Chemistry / Bioclimatic Zone), derived
on the fly from raw natural_habitat via Flair3DLabelRemap (storage definition
default / CarHab ids 0-43). Checkpoint selection uses main_task=segment.

stride=(2, 2, 2, 2), batch_size=20, lr=1e-2, num_gpu=1.
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
num_exp = 12

log_task_gradient_norms = False
grad_norm_lite = True
grad_norm_lite_interval = 100
grad_norm_lite_ema_alpha = 0.1
grad_norm_lite_eps = 1e-3

# Hardware parameters
num_gpu = 1
num_worker = 8 * num_gpu
enable_amp = True

# Data parameters
batch_size = 20 * num_gpu  # total batch size across all gpus
# Cap scenes/batch; actual packing uses *_voxel_budget (w105/6/19h: 2M worked).
batch_size_val = 8 * num_gpu
batch_size_test = 8 * num_gpu
test_voxel_budget = 2_000_000
val_voxel_budget = 2_000_000

grid_size = 0.1
point_max = 102400
mix_prob = 0.8

patch_size = 1024

# Optimization parameters
lr = 1e-2  # 10× vs. _11 (1e-3)
total_iters = 30_000

# Features
learned_masked_feat = True
feat_keys = ["coord", "color", "strength"]
coord_feat_scale = 0.01

# Backbone pooling stride (encoder stages)
stride = (2, 2, 2, 2)

# Wandb parameters
wandb_run_name = (
    f"Flair3D+ LitePT-S multi + nathab_distribution "
    f"{grp_exp}.{num_exp} stride={stride} batch_size={batch_size} lr={lr} "
    f"grad_norm_lite=pooled_nathab"
)
wandb_project = "flair3d_multi"

# -----------------------------------------------------------------------------
# Multitask configuration : targets configuration
# -----------------------------------------------------------------------------
from pointcept.datasets.flair3d_config_utils import (
    ELEVATION_TARGET_SCALE,
    FLAIR3D_TILE_DISTRIBUTION_TASKS,
    init_task_configs,
    init_task_criteria,
    FLAIR3D_COLLECT_PREFIX_LITEPT,
    init_multitask_collect_keys,
    get_regression_target_scales,
)

main_task = "segment"
nathab_keys = tuple(FLAIR3D_TILE_DISTRIBUTION_TASKS.keys())
target_keys = (main_task, "forest", "elevation") + nathab_keys
# natural_habitat is loader-only (remap source), not a supervised task.
dataset_target_keys = ("natural_habitat",) + target_keys

# GradNormLite: pool the 4 nathab axes into one "nathab" group instead of
# scaling each independently (see module docstring). segment/forest/elevation
# are left ungrouped (each keeps its own scale, keyed by its own task name).
grad_norm_lite_task_groups = {task_name: "nathab" for task_name in nathab_keys}

nathab_axis_remaps = dict(
    nathab_habitat_type=("natural_habitat", "by_habitat_type_ecological"),
    nathab_moisture_regime=("natural_habitat", "by_moisture_regime"),
    nathab_soil_chemistry=("natural_habitat", "by_soil_chemistry"),
    nathab_bioclimatic_zone=("natural_habitat", "by_climatic_domain"),
)
nathab_axis_storage_definitions = dict(natural_habitat="default")
nathab_axis_remap = dict(
    type="Flair3DLabelRemap",
    remaps=nathab_axis_remaps,
    storage_definitions=nathab_axis_storage_definitions,
)

elevation_target_scale = ELEVATION_TARGET_SCALE
elevation_key_scales = dict(elevation=elevation_target_scale)
target_scales = get_regression_target_scales(target_keys)

label_definitions = dict(
    segment="v20",
)

task_configs = init_task_configs(target_keys, definitions=label_definitions)
task_criteria = init_task_criteria(task_configs)
task_weights = {task_name: 1.0 for task_name in task_configs.keys()}

# Remove the imported helpers from this module's namespace so they do not leak
# into the Pointcept config dict. The config loader (pointcept/utils/config.py)
# treats every non-dunder module attribute as a config entry, and Config.dump
# pipes the resulting Python text through yapf. Yapf cannot reformat function
# objects rendered as "<function ... at 0x...>" and raises a SyntaxError.
del (
    init_task_configs,
    init_task_criteria,
    get_regression_target_scales,
    FLAIR3D_TILE_DISTRIBUTION_TASKS,
)

# main_task drives checkpoint selection / mIoU logging, so its num_classes,
# ignore_index and names are exposed at the data root for backward-compat hooks.
num_classes = task_configs[main_task]["num_classes"]
ignore_index = task_configs[main_task]["ignore_index"]
names = task_configs[main_task]["names"]

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=100),
    dict(type="MultiTaskEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

test_single_fragment = True
test = dict(type="MultiTaskTester", verbose=True, write_cls_iou=True)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# MultiTaskSegmentorV2 attaches per-task heads on top of backbone features
# (semantic: nn.Linear(backbone_out_channels, num_classes_task); elevation: 1;
# tile_distribution: WeightedKLDivLoss on pooled softmax).
model = dict(
    type="MultiTaskSegmentorV2",
    backbone_out_channels=72,
    backbone=dict(
        type="LitePT-v1",
        in_channels=7,  # coord (3) + color (3) + strength (1)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=stride,
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(patch_size, patch_size, patch_size, patch_size, patch_size),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
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
    task_configs=task_configs,
    main_task=main_task,
    task_criteria=task_criteria,
    task_weights=task_weights,
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
missing_tiles_manifest = "data/flair3d_plus/missing_ply_preflight.txt"
too_small_tiles_manifest = "data/flair3d_plus/too_small_tiles.csv"
val_stratified_subset_manifest = "data/flair3d_plus/manifests/val_dev_subset_2000.csv"

train_multitask_keys, val_multitask_keys, multitask_index_valid_keys = (
    init_multitask_collect_keys(
        target_keys, collect_prefix_keys=FLAIR3D_COLLECT_PREFIX_LITEPT
    )
)

del FLAIR3D_COLLECT_PREFIX_LITEPT, init_multitask_collect_keys

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=names,
    target_scales=target_scales,
    task_configs=task_configs,
    main_task=main_task,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        csv_manifest=csv_manifest,
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
        target_keys=list(dataset_target_keys),
        primary_target_key=main_task,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(multitask_index_valid_keys)},
            ),
            nathab_axis_remap,
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
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="Collect",
                keys=train_multitask_keys,
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
        stratified_subset_manifest=val_stratified_subset_manifest,
        target_keys=list(dataset_target_keys),
        primary_target_key=main_task,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(multitask_index_valid_keys)},
            ),
            nathab_axis_remap,
            dict(type="CenterShift", apply_z=True),
            dict(type="Z_MinShift"),
            dict(
                type="Copy",
                keys_dict={t: f"origin_{t}" for t in target_keys},
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
                keys=val_multitask_keys,
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
        target_keys=list(dataset_target_keys),
        primary_target_key=main_task,
        transform=[
            nathab_axis_remap,
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
