"""
KPConvX Malibu3D multitask (segment v20, forest_2d, elevation, 4 nathab axes,
roads-only network + APLS). Self-contained; inherits only default_runtime.
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
num_gpu = 6
num_worker = 8 * num_gpu  # A100: 8 workers per GPU
sync_bn = True  # required for DDP: syncs GradNormLite EMA and BatchNorm
enable_amp = True

# Data parameters
# Train batch_size is the global effective batch (4 per GPU when num_gpu=6).
# val/test caps are 8 * num_gpu so per-GPU occupancy stays 8 after
# default_setup // world_size. Voxel budgets are already per-rank.
batch_size = 24  # total across GPUs (4 per GPU when num_gpu=6)
batch_size_val = 8 * num_gpu
val_voxel_budget = 2_000_000
batch_size_test = 8 * num_gpu
test_voxel_budget = 2_000_000

grid_size = 0.1
point_max = 40000
mix_prob = 0.8
kp_radius = 3.2
kp_sigma = kp_radius
radius_scaling = 3.0

# Optimization parameters
lr = 5e-3
total_iters = 200_000

# Features (RGB + XYZ + strength concatenated into feat)
learned_masked_feat = True
feat_keys = ["coord", "color", "strength"]
coord_feat_scale = 0.01

# Wandb parameters
wandb_run_name = (
    f"6xA100  KPConvX multi {grp_exp}.{num_exp}) iter={total_iters}"
)
wandb_project = "malibu3d_multi"

# -----------------------------------------------------------------------------
# Multitask configuration : targets configuraiton
# -----------------------------------------------------------------------------
from pointcept.datasets.malibu3d_config_utils import (
    MALIBU3D_TILE_DISTRIBUTION_TASKS,
    init_task_configs,
    init_task_criteria,
    init_multitask_collect_keys,
)

main_task = "segment"
nathab_keys = tuple(MALIBU3D_TILE_DISTRIBUTION_TASKS.keys())
target_keys = (main_task, "forest_2d", "elevation") + nathab_keys + ("network",)
# natural_habitat is loader-only (remap source), not a supervised task.
dataset_target_keys = ("natural_habitat",) + target_keys

# GradNormLite: pool the 4 nathab axes into one "nathab" group.
grad_norm_lite_task_groups = {task_name: "nathab" for task_name in nathab_keys}

nathab_axis_remaps = dict(
    nathab_habitat_type=("natural_habitat", "by_habitat_type_ecological"),
    nathab_moisture_regime=("natural_habitat", "by_moisture_regime"),
    nathab_soil_chemistry=("natural_habitat", "by_soil_chemistry"),
    nathab_bioclimatic_zone=("natural_habitat", "by_climatic_domain"),
)
nathab_axis_storage_definitions = dict(natural_habitat="default")
nathab_axis_remap = dict(
    type="Malibu3DLabelRemap",
    remaps=nathab_axis_remaps,
    storage_definitions=nathab_axis_storage_definitions,
)

# Elevation in meters: no Collect key_scales, no denorm via target_scales
# (matches configs/experiment/w107/7/toward_bm/multi-litept-v1m0-malibu3d_1.py).
target_scales = {}

label_definitions = dict(
    segment="v20",
)

task_configs = init_task_configs(target_keys, definitions=label_definitions)
# Network head: ROADS only (RAILROADS channel dropped), supervised with CE only
# and weight=5 on the foreground pixel class. Mirrors
# configs/experiment/w107/5/18h/mono_network_ce_road_w5.py.
task_configs["network"]["num_networks"] = 1
task_configs["network"]["channel_names"] = ["ROADS"]
task_criteria = init_task_criteria(task_configs)
# Elevation in meters (no Collect key_scales / target_scales denorm above).
# beta=1.0 in meters ≡ former ELEVATION_SMOOTH_L1_BETA=1e-2 in the old ×0.01
# space (~1 m Huber threshold). See multi-litept-v1m0-malibu3d_1.py (w107/7/toward_bm).
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

# Remove the imported helpers from this module's namespace so they do not leak
# into the Pointcept config dict. The config loader (pointcept/utils/config.py)
# treats every non-dunder module attribute as a config entry, and Config.dump
# pipes the resulting Python text through yapf. Yapf cannot reformat function
# objects rendered as "<function ... at 0x...>" and raises a SyntaxError.
del (
    init_task_configs,
    init_task_criteria,
    MALIBU3D_TILE_DISTRIBUTION_TASKS,
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
    # After PreciseEvaluator only (end of training / tools/test.py) -- not on val.
    dict(type="NetworkAPLSEvaluator"),
]

test_single_fragment = True
test = dict(type="MultiTaskTester", verbose=True, write_cls_iou=True)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Backbone returns per-point features (num_classes=0 skips internal classifier).
backbone_feat_dim = 64

model = dict(
    type="MultiTaskSegmentorV2",
    backbone_out_channels=backbone_feat_dim,
    backbone=dict(
        type="kpconvx_base",
        input_channels=7,
        num_classes=0,
        dim=3,
        task="cloud_segmentation",
        kp_mode="kpconvx",
        shell_sizes=(1, 14, 28),
        kp_radius=kp_radius,
        kp_aggregation="nearest",
        kp_influence="constant",
        kp_sigma=kp_sigma,
        share_kp=False,
        conv_groups=-1,
        inv_groups=8,
        inv_act="sigmoid",
        inv_grp_norm=True,
        kpx_upcut=False,
        subsample_size=grid_size,
        neighbor_limits=(12, 16, 20, 20, 20),
        layer_blocks=(3, 3, 9, 12, 3),
        init_channels=64,
        channel_scaling=1.414,
        radius_scaling=radius_scaling,
        decoder_layer=True,
        grid_pool=True,
        upsample_n=3,
        first_inv_layer=1,
        drop_path_rate=0.3,
        norm="batch",
        bn_momentum=0.1,
        smooth_labels=False, # True only for classification
        class_w=(),
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
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=lr,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=10000.0,
)

# -----------------------------------------------------------------------------
# Dataset (KPConvX-style subsampling; multitask keys from Malibu3D)
# -----------------------------------------------------------------------------
dataset_type = "Malibu3DDataset"
data_root = "data/malibu3d"
csv_manifest = "data/malibu3d/raw/scene_split_manifest.csv"
min_points = {"train": 1000}
val_stratified_subset_manifest = "data/malibu3d/manifests/val_dev_subset_2000.csv"

# Opt-in APLS scoring of PreciseEvaluator test logits (see NetworkAPLSEvaluator /
# tools/test.py). ``split`` must match ``data.test.split`` so stitched ROIs find
# data/malibu3d_build/data/network_graphs).
network_apls_eval = dict(
    network_graphs_root="data/network_graphs",
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
    # APLS scoring itself (parameters that feed apls_symmetric_score directly);
    # everything above builds the predicted graph. See tools/eval_network_apls.py.
    apls_max_nodes_exact=None,  # None = no |V| cap after densify
    apls_densify=50.0,
    apls_snap_to_edge=4.0,
    apls_symmetric=True,
    apls_min_path_length_m=5,
)

train_multitask_keys, val_multitask_keys, multitask_index_valid_keys = (
    init_multitask_collect_keys(target_keys)
)

del init_multitask_collect_keys

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
        target_keys=list(dataset_target_keys),
        primary_target_key=main_task,
        task_configs=task_configs,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(multitask_index_valid_keys)},
            ),
            # Freeze Lambert XY before geometric augs / recentering.
            dict(type="ExtractAbsXY"),
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
                return_min_coord=True,
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
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
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
        target_keys=list(dataset_target_keys),
        primary_target_key=main_task,
        task_configs=task_configs,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(multitask_index_valid_keys)},
            ),
            dict(type="ExtractAbsXY"),
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
                return_min_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="NetworkRasterToPointLabels"),
            dict(type="NetworkRasterToPointLabels", target_key="forest_2d"),
            dict(type="ToTensor"),
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
        target_keys=list(dataset_target_keys),
        primary_target_key=main_task,
        task_configs=task_configs,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(multitask_index_valid_keys)},
            ),
            dict(type="ExtractAbsXY"),
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
                return_inverse=True,
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
