"""
LitePT on Flair3D+ with semantic multitask plus point-wise elevation regression.

Mirrors ``configs/flair3d_plus/spunet_multitask.py`` but uses LitePT-v1 as the
backbone (RGB + strength, ``feat_keys`` without XYZ concatenation). Inherits
only ``default_runtime``; dataset multitask blocks match SpUNet multitask.

LitePT expects ``grid_coord`` or ``coord`` + ``grid_size``; we inject scalar
``grid_size`` after ``ToTensor`` like ``configs/flair3d_plus/litept.py``.
"""

_base_ = ["../../../../_base_/default_runtime.py"]

test = dict(type="MultiTaskTester", verbose=True, write_cls_iou=True)

from pointcept.datasets.flair3d_plus_label_task_config import (
    get_multitask_regression_task_config_elevation,
    get_semantic_config,
)

# -----------------------------------------------------------------------------
# Run-level settings (LitePT training regime from configs/flair3d_plus/litept.py)
# -----------------------------------------------------------------------------
num_gpu = 1

grp_exp = 1
num_exp = 2

num_worker = 8 * num_gpu

test_single_fragment = True

batch_size_per_gpu = 20
batch_size = batch_size_per_gpu * num_gpu
mix_prob = 0.8
empty_cache = False
enable_amp = True

lr = 2e-3
warmup_steps = 500
grid_size = 0.1
point_max = 102400
patch_size = 1024

epoch = 3
eval_epoch = 3
feat_keys = ["color", "strength"]

learned_masked_feat = True

# -----------------------------------------------------------------------------
# Multitask targets and per-task specs (same tasks as spunet_multitask)
# -----------------------------------------------------------------------------
semantic_target_keys = ("segment", "forest", "land_use", "natural_habitat")
target_keys = semantic_target_keys + ("elevation",)
main_task = "segment"

task_configs = {task_name: get_semantic_config(task_name) for task_name in semantic_target_keys}
task_configs["elevation"] = get_multitask_regression_task_config_elevation()

del get_semantic_config, get_multitask_regression_task_config_elevation

num_classes = task_configs[main_task]["num_classes"]
ignore_index = task_configs[main_task]["ignore_index"]
names = task_configs[main_task]["names"]

task_criteria = {
    task_name: [
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=task_configs[task_name]["ignore_index"],
        ),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=task_configs[task_name]["ignore_index"],
        ),
    ]
    for task_name in semantic_target_keys
}
task_criteria["elevation"] = [
    dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
]

task_weights = {task_name: 1.0 for task_name in task_configs.keys()}

wandb_run_name = (
    f"LitePT {grp_exp}.{num_exp}) lr={lr}"
)
wandb_project = "flair3d+"

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=100),
    dict(type="MultiTaskEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=3),
    dict(type="PreciseEvaluator", test_last=False),
]

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = dict(
    type="MultiTaskSegmentorV2",
    backbone_out_channels=72,
    backbone=dict(
        type="LitePT-v1",
        in_channels=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
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
    type="LinearLR",
    start_factor=1 / 10,
    total_iters=warmup_steps,
)
param_dicts = [dict(keyword="block", lr=lr / 10)]

# -----------------------------------------------------------------------------
# Dataset (multitask pipeline aligned with spunet_multitask)
# -----------------------------------------------------------------------------
dataset_type = "Flair3DDataset"
data_root = "data/flair3d_plus"
csv_manifest = "data/flair3d_plus/raw/scene_split_manifest.csv"
missing_tiles_manifest = "data/flair3d_plus/missing_ply_preflight.txt"
too_small_tiles_manifest = "data/flair3d_plus/too_small_tiles.csv"

train_multitask_keys = (
    "coord",
    "grid_coord",
    "grid_size",
    "segment",
    "forest",
    "land_use",
    "natural_habitat",
    "elevation",
)
val_multitask_keys = (
    "coord",
    "grid_coord",
    "grid_size",
    "segment",
    "origin_segment",
    "forest",
    "origin_forest",
    "land_use",
    "origin_land_use",
    "natural_habitat",
    "origin_natural_habitat",
    "elevation",
    "origin_elevation",
    "inverse",
)
multitask_index_valid_keys = (
    "coord",
    "color",
    "normal",
    "color_mask",
    "normal_mask",
    "superpoint",
    "strength",
    "strength_mask",
    "segment",
    "instance",
    "forest",
    "land_use",
    "natural_habitat",
    "elevation",
)

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=names,
    task_configs=task_configs,
    main_task=main_task,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        csv_manifest=csv_manifest,
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
        target_keys=list(target_keys),
        primary_target_key=main_task,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(multitask_index_valid_keys)},
            ),
            dict(type="CenterShift", apply_z=True),
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
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="Collect",
                keys=train_multitask_keys,
                feat_keys=feat_keys,
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
        primary_target_key=main_task,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(multitask_index_valid_keys)},
            ),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "segment": "origin_segment",
                    "forest": "origin_forest",
                    "land_use": "origin_land_use",
                    "natural_habitat": "origin_natural_habitat",
                    "elevation": "origin_elevation",
                },
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
        primary_target_key=main_task,
        transform=[dict(type="CenterShift", apply_z=True), dict(type="NormalizeColor")],
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
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ],
        ),
    ),
)
