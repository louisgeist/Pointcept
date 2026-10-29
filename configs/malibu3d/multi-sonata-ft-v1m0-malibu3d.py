"""
MALiBU3D multitask benchmark, Sonata (Tab. 2 of the MALiBU3D paper): PT-v3m2 backbone pretrained
self-supervised on MALiBU3D with pretrain-sonata-v1m2-malibu3d.py, then fully fine-tuned on the
five tasks jointly.

Same tasks, losses, GradNormLite balancing, schedule (200k iterations, batch 12 on 1 GPU, AMP)
and data pipeline as multi-litept-b-v1m0-malibu3d.py; the model block and the weight decay
differ. The encoder hyperparameters must match the pretraining config exactly (the checkpoint
is loaded by shape); the decoder, with the dimensions of upstream Sonata's ScanNet full
fine-tuning config, and the task heads are randomly initialized.
Released checkpoint: ckpt/sonata-ft/model.pth (see docs/checkpoints.md).
"""

# -----------------------------------------------------------------------------
# Default
# -----------------------------------------------------------------------------
_base_ = ["../_base_/default_runtime.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------

# Logging parameters
grp_exp = 1
num_exp = 1

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
wandb_run_name = f"Sonata-FT multi {grp_exp}.{num_exp} iter={total_iters}"
wandb_project = "malibu3d_multi"

# -----------------------------------------------------------------------------
# Pretrained weight
# -----------------------------------------------------------------------------
# Sonata pretraining checkpoint (epoch 120 of the 150-epoch run), downloaded with
#   hf download LouisGeist/MALiBU3D-backbones sonata-pretrained/model.pth --local-dir ckpt
weight = "ckpt/sonata-pretrained/model.pth"

# -----------------------------------------------------------------------------
# Multitask configuration: targets, losses, task weights
# -----------------------------------------------------------------------------
# The five tasks and their losses (see malibu3d_config_utils.init_task_criteria):
# segment (CE + Lovasz), forest_2d (CE + Lovasz), elevation in meters (SmoothL1),
# the 4 nathab_* axes (tile-wise weighted KL) and network (ROADS, CE with a 5x weight
# on road pixels).
from pointcept.datasets.malibu3d_config_utils import (
    MALIBU3D_TARGET_KEYS,
    MALIBU3D_TILE_DISTRIBUTION_TASKS,
    MALIBU3D_COLLECT_PREFIX_LITEPT,
    init_multitask_collect_keys,
    init_task_configs,
    init_task_criteria,
)

main_task = "segment"
target_keys = MALIBU3D_TARGET_KEYS
task_configs = init_task_configs(target_keys)
task_criteria = init_task_criteria(task_configs)
task_weights = {task_name: 1.0 for task_name in task_configs}
# GradNormLite: pool the 4 nathab axes into one "nathab" group.
grad_norm_lite_task_groups = {
    task_name: "nathab" for task_name in MALIBU3D_TILE_DISTRIBUTION_TASKS
}

# Delete the imported helpers: the config loader turns every non-dunder module
# attribute into a config entry, and Config.dump (yapf) fails on function objects.
del (
    MALIBU3D_TARGET_KEYS,
    MALIBU3D_TILE_DISTRIBUTION_TASKS,
    init_task_configs,
    init_task_criteria,
)

# main_task drives checkpoint selection; its num_classes / ignore_index / names are
# also exposed at the top level and in `data` for the hooks that read them there.
num_classes = task_configs[main_task]["num_classes"]
ignore_index = task_configs[main_task]["ignore_index"]
names = task_configs[main_task]["names"]

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
hooks = [
    # Loads the student backbone of a raw Sonata checkpoint (student.backbone.* ->
    # backbone.*) as well as the released backbone-only file (already backbone.*).
    # strict=False ignores teacher/head keys and leaves the decoder and task heads
    # randomly initialized.
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=100),
    dict(type="MultiTaskEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
    # Must follow PreciseEvaluator: scores its saved test logits with APLS (never on val).
    dict(type="NetworkAPLSEvaluator"),
]

test_single_fragment = True
test = dict(type="MultiTaskTester", verbose=True, write_cls_iou=True)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# MultiTaskSegmentorV2: one linear head per task on the backbone's per-point features.
# Encoder dims must match pretrain-sonata-v1m2-malibu3d.py; the decoder (enc_mode=False)
# uses the dims of upstream Sonata's ScanNet full fine-tuning config.
model = dict(
    type="MultiTaskSegmentorV2",
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m2",
        in_channels=7,  # coord (3) + color (3) + strength (1)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(3, 3, 3, 3),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
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
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
    ),
    feature_mask_values=dict(
        enable=learned_masked_feat,
        masked_feat_keys=["color", "strength"],
    ),
    task_configs=task_configs,
    main_task=main_task,
    task_criteria=task_criteria,
    task_weights=task_weights,
    # Full fine-tuning (the probe/ configs freeze the backbone instead).
    freeze_backbone=False,
)


# -----------------------------------------------------------------------------
# Optimizer / scheduler
# -----------------------------------------------------------------------------
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.05)
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
dataset_type = "MALiBU3DDataset"
data_root = "data/malibu3d"
csv_manifest = "data/malibu3d/tiles.csv"
min_points = {"train": 1000}
val_stratified_subset_manifest = "data/malibu3d/val_subset_2000tiles.csv"

# Opt-in APLS scoring of the PreciseEvaluator test logits (NetworkAPLSEvaluator /
# tools/test.py). The mask -> graph parameters keep the defaults of
# tools/eval_network_apls.py::run; ground-truth road graphs are read from each ROI
# directory (Hugging Face layout of the dataset).
network_apls_eval = dict(split="test")

(
    train_multitask_keys,
    val_multitask_keys,
    test_multitask_keys,
    multitask_index_valid_keys,
) = init_multitask_collect_keys(
    target_keys, collect_prefix_keys=MALIBU3D_COLLECT_PREFIX_LITEPT
)

del MALIBU3D_COLLECT_PREFIX_LITEPT, init_multitask_collect_keys

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
        min_points=min_points,
        target_keys=list(target_keys),
        primary_target_key=main_task,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(multitask_index_valid_keys)},
            ),
            # Keep absolute Lambert XY (before recentering and augmentation) for the
            # raster-target lookup in NetworkRasterToPointLabels.
            dict(type="ExtractAbsXY"),
            dict(type="CenterShift", apply_z=True),
            dict(type="Z_MinShift"),
            dict(type="Z_RandomOffset"),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
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
            dict(
                type="RandomDropColor",
                drop_ratio=1.0,
                drop_application_ratio=0.2,
                keep_mask=True,
            ),
            dict(
                type="RandomDropColor",
                drop_ratio=0.1,
                drop_application_ratio=0.5,
                keep_mask=True,
            ),
            dict(
                type="RandomDropStrength",
                drop_ratio=1.0,
                drop_application_ratio=0.2,
                keep_mask=True,
            ),
            dict(
                type="RandomDropStrength",
                drop_ratio=0.1,
                drop_application_ratio=0.5,
                keep_mask=True,
            ),
            dict(type="NetworkRasterToPointLabels"),
            dict(type="NetworkRasterToPointLabels", target_key="forest_2d"),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="Collect",
                keys=train_multitask_keys,
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
        primary_target_key=main_task,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(multitask_index_valid_keys)},
            ),
            dict(type="ExtractAbsXY"),
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
            dict(type="NetworkRasterToPointLabels"),
            dict(type="NetworkRasterToPointLabels", target_key="forest_2d"),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="Collect",
                keys=val_multitask_keys,
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
        primary_target_key=main_task,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(multitask_index_valid_keys)},
            ),
            dict(type="ExtractAbsXY"),
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
                dict(type="NetworkRasterToPointLabels"),
                dict(type="NetworkRasterToPointLabels", target_key="forest_2d"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=test_multitask_keys,
                    optional_keys=("inverse",),
                    feat_keys=feat_keys,
                    feat_scales=dict(coord=coord_feat_scale),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ]
            ],
        ),
    ),
)
