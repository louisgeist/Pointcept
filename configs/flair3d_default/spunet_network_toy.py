"""
Toy SpUNet config for debugging network pixel segmentation on hecate (D067).

Mono-task: ``network`` (binary 1 m Lambert masks for ROADS / RAILROADS).
Requires ``network.npy`` (or empty ``meta.network``) from ``rasterize_network.py``
and ``ExtractAbsXY`` before geometric shifts.

Hyper-short schedule + end-of-run APLS smoke: ``data.test`` is capped to the
first D067 val ROI (AN-S1-15, 100 subtiles) and ``network_apls_eval.max_rois=1``
scores only that ROI after PreciseEvaluator.

Example::

    export PYTHONPATH="$PWD"
    python tools/train.py --config-file configs/flair3d_default/spunet_network_toy.py \\
      --num-gpus 1
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

# Hardware parameters
num_gpu = 1
num_worker = 8 * num_gpu
enable_amp = True

# Data parameters
batch_size = 20 * num_gpu  # total batch size across all gpus
batch_size_val = batch_size // 4
batch_size_test = batch_size // 4
train_max_sample = 20
val_max_sample = 100
# First D067 val ROI in the manifest (AN-S1-15) has exactly 100 subtiles and
# sits at the head of the val list -- keep test_max_sample aligned so PreciseEvaluator
# writes a complete ROI for APLS stitching.
test_max_sample = 100  # == n_subtiles of first val ROI (AN-S1-15)

# Stratified fixed val/test subset (see README_geist.md "Fast dev val/test")
val_stratified_subset_manifest = None
test_stratified_subset_manifest = None

grid_size = 0.1
point_max = 100000
# Mix3D OK: NetworkRasterToPointLabels makes network point-wise before collate.
mix_prob = 0.8

# Optimization parameters
lr = 1e-3
total_iters = 15
iter_per_epoch = 5
warmup_iters = 5

# Features
learned_masked_feat = True
feat_keys = ["coord", "color", "strength"]
coord_feat_scale = 0.01

# Wandb parameters
wandb_run_name = f"SpUNet network toy {grp_exp}.{num_exp} lr={lr}"
wandb_project = "flair3d_network"

# -----------------------------------------------------------------------------
# Mono-task configuration : targets configuration
# -----------------------------------------------------------------------------
from pointcept.datasets.flair3d_config_utils import (
    init_task_configs,
    init_task_criteria,
    FLAIR3D_COLLECT_PREFIX_GRID,
    init_multitask_collect_keys,
)

main_task = "network"
target_keys = (main_task,)

task_configs = init_task_configs(target_keys)
task_criteria = init_task_criteria(task_configs)
task_weights = {main_task: 1.0}

# Remove the imported helpers from this module's namespace so they do not leak
# into the Pointcept config dict. The config loader (pointcept/utils/config.py)
# treats every non-dunder module attribute as a config entry, and Config.dump
# pipes the resulting Python text through yapf. Yapf cannot reformat function
# objects rendered as "<function ... at 0x...>" and raises a SyntaxError.
del init_task_configs, init_task_criteria

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
    dict(type="InformationWriter", log_interval=1),
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
# Backbone produces per-point features (num_classes=0 disables its final 1x1
# conv). MultiTaskSegmentorV2 max-pools into 1 m pixels then Linear(C -> 2*r).
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
min_points = {"train": 1000}

# Opt-in APLS on PreciseEvaluator logits. ``data.test`` uses the val split on D067
# (no local test rows) -- keep APLS ``split`` aligned so logits are found.
# max_rois=1: score only the first complete ROI covered by test_max_sample.
network_apls_eval = dict(
    network_graphs_root="/data/geist/Flair3D-build/data/network_graphs",
    split="val",
    threshold=0.5,
    overlap_combine="nanmean",
    connectivity=4,
    rdp_epsilon_m=2.0,
    endpoint_fix_stage="pre_rdp",
    merge_weight_threshold=2.5,
    max_nodes_exact=None,
    max_rois=1,
    densify=50.0,
    snap_to_edge=4.0,
    symmetric=True,
    radius_fix_radius_m=5,
    min_path_length_m=5,
)

train_multitask_keys, val_multitask_keys, multitask_index_valid_keys = (
    init_multitask_collect_keys(
        target_keys, collect_prefix_keys=FLAIR3D_COLLECT_PREFIX_GRID
    )
)

del FLAIR3D_COLLECT_PREFIX_GRID, init_multitask_collect_keys

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
        min_points=min_points,
        target_keys=list(target_keys),
        primary_target_key=main_task,
        max_sample=train_max_sample,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(multitask_index_valid_keys)},
            ),
            # Freeze Lambert XY before geometric augs / recentering.
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
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
        min_points=min_points,
        target_keys=list(target_keys),
        primary_target_key=main_task,
        max_sample=val_max_sample,
        stratified_subset_manifest=val_stratified_subset_manifest,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(multitask_index_valid_keys)},
            ),
            dict(type="ExtractAbsXY"),
            dict(type="CenterShift", apply_z=True),
            dict(type="Z_MinShift"),
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
        split="val",
        data_root=data_root,
        csv_manifest=csv_manifest,
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
        min_points=min_points,
        target_keys=list(target_keys),
        primary_target_key=main_task,
        max_sample=test_max_sample,
        stratified_subset_manifest=test_stratified_subset_manifest,
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
