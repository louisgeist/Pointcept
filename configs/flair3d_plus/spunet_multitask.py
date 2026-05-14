"""
SpUNet on Flair3D+ with semantic multitask plus point-wise elevation regression.

Mirrors litept_multitask_semantic_elevation.py but uses SpUNet-v1m1 as the
backbone. Adds ``elevation`` to ``target_keys`` and a regression task in
``task_configs`` (``task_type: regression``).

This config is intentionally self-contained: it inherits only from
default_runtime and duplicates everything it needs from ``spunet.py`` so it can
be read top-to-bottom without cross-referencing other Flair3D+ configs.
"""

# -----------------------------------------------------------------------------
### Specificties for hecate debuging
# -----------------------------------------------------------------------------
# Test set is the val
# Log every train step
# lower batch size
# csv_manifest

# epoch; eval_epoch; max_sample;

_base_ = ["../_base_/default_runtime.py"]

test = dict(type="MultiTaskTester", verbose=True, write_cls_iou=True)

from pointcept.datasets.flair3d_plus_label_task_config import (
    get_multitask_regression_task_config_elevation,
    get_semantic_config,
)

# -----------------------------------------------------------------------------
# Run-level settings (copied from configs/flair3d_plus/spunet.py)
# -----------------------------------------------------------------------------
num_gpu = 1

grp_exp = 1
num_exp = 1

num_worker = 8 * num_gpu

test_single_fragment = True

batch_size = 24 * num_gpu  # total batch size across all gpus
gradient_accumulation_steps = 2
mix_prob = 0.8
empty_cache = False
enable_amp = True

lr = 5e-3
warmup_steps = 2500
grid_size = 0.1
point_max = 100000

epoch = 1
eval_epoch = 1
feat_keys = ["coord", "color", "strength"]

learned_masked_feat = True

# -----------------------------------------------------------------------------
# Multitask targets and per-task specs
# -----------------------------------------------------------------------------
semantic_target_keys = ("segment", "forest", "land_use", "natural_habitat")
target_keys = semantic_target_keys + ("elevation",)
main_task = "segment"

task_configs = {task_name: get_semantic_config(task_name) for task_name in semantic_target_keys}
task_configs["elevation"] = get_multitask_regression_task_config_elevation()

# Remove the imported helpers from this module's namespace so they do not leak
# into the Pointcept config dict. The config loader (pointcept/utils/config.py)
# treats every non-dunder module attribute as a config entry, and Config.dump
# pipes the resulting Python text through yapf. Yapf cannot reformat function
# objects rendered as "<function ... at 0x...>" and raises a SyntaxError.
del get_semantic_config, get_multitask_regression_task_config_elevation

# main_task drives checkpoint selection / mIoU logging, so its num_classes,
# ignore_index and names are exposed at the data root for backward-compat hooks.
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

task = "semseg_multitask_regression"
wandb_run_name = (
    f"Flair3D+ SpUNet multitask + elevation {grp_exp}.{num_exp}) lr={lr}"
)
wandb_project = "flair3dplus_hecate"

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=1),
    dict(type="MultiTaskEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Backbone produces per-point features (num_classes=0 disables its final 1x1
# conv). MultiTaskSegmentorV2 attaches per-task linear heads on top of these
# features (one nn.Linear(backbone_out_channels, num_classes_task) per
# semantic task, one nn.Linear(backbone_out_channels, 1) for elevation).
backbone_channels = (32, 64, 128, 256, 256, 128, 96, 96)

model = dict(
    type="MultiTaskSegmentorV2",
    backbone_out_channels=backbone_channels[-1],
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=7,  # coord (3) + color (3) + strength (1)
        num_classes=0,
        channels=backbone_channels,
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
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

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset_type = "Flair3DDataset"
data_root = "data/flair3d_plus"
# csv_manifest =  "data/flair3d_plus/raw/scene_split_manifest_D067.csv" 
csv_manifest =  "data/flair3d_plus/raw/scene_split_manifest.csv"
missing_tiles_manifest = "data/flair3d_plus/missing_ply_preflight.txt"
too_small_tiles_manifest = "data/flair3d_plus/too_small_tiles.csv"

train_multitask_keys = (
    "coord",
    "grid_coord",
    "segment",
    "forest",
    "land_use",
    "natural_habitat",
    "elevation",
)
val_multitask_keys = (
    "coord",
    "grid_coord",
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
        max_sample=30,
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
            dict(
                type="Collect",
                keys=train_multitask_keys,
                feat_keys=feat_keys,
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        max_sample=10,
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
            dict(
                type="Collect",
                keys=val_multitask_keys,
                feat_keys=feat_keys,
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        max_sample=10,
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
