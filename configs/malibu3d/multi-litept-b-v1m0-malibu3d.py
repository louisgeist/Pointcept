"""
MALiBU3D multitask benchmark, LitePT-B trained from scratch (Tab. 2 of the MALiBU3D paper).

The five tasks are learned jointly on one backbone: land-cover segmentation (`segment`, v20
labels), road-network prediction (`network`, roads only, 1 m raster, scored with APLS at test
time), natural-habitat distribution on 4 ecological axes (`nathab_*`, tile-wise weighted KL),
forest cover (`forest_2d` raster) and elevation regression (meters). GradNormLite balances the
task losses. 200k iterations, batch 12 on 1 GPU, AMP, Mix3D.
Released checkpoint: ckpt/litept-b/model.pth (see README_checkpoints.md).
Self-contained: inherits only from _base_/default_runtime.py.
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
batch_size = 12   # total batch size across all gpus
batch_size_val = 8 * num_gpu
val_voxel_budget = 2_000_000
# Max scenes per val/test batch; batches are packed up to the voxel budgets.
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
    f"LPT-B multi {grp_exp}.{num_exp} iter={total_iters}"
)
wandb_project = "malibu3d_multi"

# -----------------------------------------------------------------------------
# Multitask configuration: targets, losses, task weights
# -----------------------------------------------------------------------------
from pointcept.datasets.malibu3d_config_utils import (
    MALIBU3D_TILE_DISTRIBUTION_TASKS,
    init_task_configs,
    init_task_criteria,
    MALIBU3D_COLLECT_PREFIX_LITEPT,
    init_multitask_collect_keys,
)

main_task = "segment"
nathab_keys = tuple(MALIBU3D_TILE_DISTRIBUTION_TASKS.keys())
target_keys = (main_task, "forest_2d", "elevation") + nathab_keys + ("network",)
# nathab_* axes are unpacked from on-disk natural_habitat.npy (N, 4) in get_data.
# GradNormLite: pool the 4 nathab axes into one "nathab" group.
grad_norm_lite_task_groups = {task_name: "nathab" for task_name in nathab_keys}


# Elevation is regressed in meters: no target rescaling, hence no denormalization
# in the metrics.
target_scales = {}

label_definitions = dict(
    segment="v20",
)

task_configs = init_task_configs(target_keys, definitions=label_definitions)
# Network task: ROADS channel only (the default also has RAILROADS), trained with
# cross-entropy alone (no Lovasz) and a 5x weight on road pixels (see below).
task_configs["network"]["num_networks"] = 1
task_configs["network"]["channel_names"] = ["ROADS"]
task_criteria = init_task_criteria(task_configs)
# The default beta (1e-2) assumes targets scaled by 0.01; with targets in meters,
# beta=1.0 keeps the L1/L2 transition at 1 m.
task_criteria["elevation"] = [
    dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
]
_network_ignore = int(task_configs["network"]["ignore_index"])
task_criteria["network"] = [
    dict(
        type="CrossEntropyLoss",
        loss_weight=1.0,
        ignore_index=_network_ignore,
        weight=[1.0, 5.0],  # Background, Foreground
    ),
]
del _network_ignore
task_weights = {task_name: 1.0 for task_name in task_configs.keys()}

# Delete the imported helpers: the config loader turns every non-dunder module
# attribute into a config entry, and Config.dump (yapf) fails on function objects.
del (
    init_task_configs,
    init_task_criteria,
    MALIBU3D_TILE_DISTRIBUTION_TASKS,
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
    dict(type="CheckpointLoader"),
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
# Backbone: official LitePT-B dimensions.
model = dict(
    type="MultiTaskSegmentorV2",
    backbone_out_channels=72,
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
dataset_type = "MALiBU3DDataset"
data_root = "data/malibu3d"
csv_manifest = "data/malibu3d/tiles.csv"
min_points = {"train": 1000}
val_stratified_subset_manifest = "data/malibu3d/val_subset_2000tiles.csv"

# Opt-in APLS scoring of PreciseEvaluator test logits (see NetworkAPLSEvaluator /
# tools/test.py). ``split`` must match ``data.test.split`` so stitched ROIs find
# ``{patch_id}_logits_network.npy``. ``network_graphs_root=None`` reads each ground-truth
# road graph from its ROI directory (Hugging Face layout of the dataset).
network_apls_eval = dict(
    network_graphs_root=None,
    split="test",
    threshold=0.2,
    overlap_combine="nanmean",
    connectivity=8,
    rdp_epsilon_m=2.0,
    endpoint_fix_enabled=False,
    endpoint_fix_stage="pre_rdp",
    merge_hop_threshold=2.5,
    max_rois=None,
    radius_fix_radius_m=5,
    # Mask -> graph (from-mask path): drop noise blobs, then skeletonize to 1px.
    remove_small_objects_enabled=False,
    remove_small_objects_min_size_px=8,
    skeletonize_enabled=True,
    open_iterations=0,
    close_iterations=5,
    morph_connectivity=8,
    min_component_nodes=5,
    # APLS metric parameters are fixed (apls_metric.APLS_PROTOCOL), not configurable;
    # everything in this dict builds the predicted graph. See tools/eval_network_apls.py.
)

train_multitask_keys, val_multitask_keys, multitask_index_valid_keys = (
    init_multitask_collect_keys(
        target_keys, collect_prefix_keys=MALIBU3D_COLLECT_PREFIX_LITEPT
    )
)

del MALIBU3D_COLLECT_PREFIX_LITEPT, init_multitask_collect_keys

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
        min_points=min_points,
        target_keys=list(target_keys),
        primary_target_key=main_task,
        task_configs=task_configs,
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
        task_configs=task_configs,
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
        task_configs=task_configs,
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
                    keys=(
                        "coord",
                        "grid_coord",
                        "index",
                        "network",
                        "network_cell",
                        "network_pix",
                        "network_origin_x",
                        "network_origin_y",
                        "network_pixel_m",
                        "network_height",
                        "network_width",
                        "forest_2d",
                        "forest_2d_cell",
                        "forest_2d_pix",
                        "forest_2d_origin_x",
                        "forest_2d_origin_y",
                        "forest_2d_pixel_m",
                        "forest_2d_height",
                        "forest_2d_width",
                    ),
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
