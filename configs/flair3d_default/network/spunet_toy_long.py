"""
SpUNet longer toy overfit for network pixel segmentation on Hecate (D067).

Same recipe as ``spunet_toy.py`` (1 tile, Mix3D off, no crop), but a longer
schedule so pixel logits can move away from ~0.5 before evaluating
collapse-to-background.

Example::

    export PYTHONPATH="$PWD"
    python tools/train.py --config-file configs/flair3d_default/network/spunet_toy_long.py \\
      --num-gpus 1
"""

# -----------------------------------------------------------------------------
# Default
# -----------------------------------------------------------------------------
_base_ = ["../../_base_/default_runtime.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------

grp_exp = 1
num_exp = 2
seed = 14028665

num_gpu = 1
num_worker = 8 * num_gpu
enable_amp = True

# batch_size=1: SphereCrop is removed below (see train transform) so each replica
# is the full ~287k-point tile at grid_size=0.1 -- too big to batch at 4x.
batch_size = 1 * num_gpu
batch_size_val = batch_size
batch_size_test = 1

# One unique scene; train ``loop`` repeats it so DataLoader can form full batches
# (each index maps to the same tile via idx % len(data_list)).
overfit_unique_samples = 1
overfit_train_loop = batch_size

train_max_sample = overfit_unique_samples
val_max_sample = overfit_unique_samples
test_max_sample = overfit_unique_samples

grid_size = 0.1
# Single-tile overfit: no Mix3D (would only pair copies of the same scene).
mix_prob = 0.0

lr = 1e-3
total_iters = 10000
iter_per_epoch = 100
eval_every = 5
warmup_iters = 200

learned_masked_feat = True
feat_keys = ["coord", "color", "strength"]
coord_feat_scale = 0.01

wandb_run_name = (
    f"SpUNet network overfit-long D067 | lr={lr}, iters={total_iters}, "
    f"mix_prob={mix_prob}"
)
wandb_project = "flair3d_network_overfit"

# -----------------------------------------------------------------------------
# Mono-task: network
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

del init_task_configs, init_task_criteria

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
    dict(type="InformationWriter", log_interval=10),
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

optimizer = dict(type="AdamW", lr=lr, weight_decay=0.005)
scheduler = dict(
    type="LinearLR",
    start_factor=1 / 10,
    total_iters=warmup_iters,
)

# -----------------------------------------------------------------------------
# Dataset (D067 ROI on Hecate)
# -----------------------------------------------------------------------------
dataset_type = "Flair3DDataset"
data_root = "data/flair3d_plus"
csv_manifest = "data/flair3d_plus/raw/scene_split_manifest_D067.csv"
missing_tiles_manifest = "data/flair3d_plus/missing_ply_preflight.txt"
too_small_tiles_manifest = "data/flair3d_plus/too_small_tiles.csv"
min_points = {"train": 1000}

# Opt-in APLS on PreciseEvaluator logits. Overfit uses train for data.test, and
# D067 has no test split -- keep APLS ``split`` aligned with that.
network_apls_eval = dict(
    network_graphs_root="/data/geist/Flair3D-build/data/network_graphs",
    split="train",
    threshold=0.5,
    overlap_combine="nanmean",
    connectivity=4,
    rdp_epsilon_m=2.0,
    endpoint_fix_stage="pre_rdp",
    merge_weight_threshold=2.5,
    max_nodes_exact=None,
    max_rois=None,
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

# Same split for train/val/test so overfit metrics watch the same tile.
_overfit_split = "train"

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=names,
    task_configs=task_configs,
    main_task=main_task,
    train=dict(
        type=dataset_type,
        split=_overfit_split,
        data_root=data_root,
        csv_manifest=csv_manifest,
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
        min_points=min_points,
        target_keys=list(target_keys),
        primary_target_key=main_task,
        max_sample=train_max_sample,
        loop=overfit_train_loop,
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
            ),
            # No crop: single-tile overfit, train on the whole tile every step
            # (SphereCrop(mode="center") always anchored the same fixed point --
            # ~38% of the tile was never seen across any training iteration).
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="NetworkRasterToPointLabels"),
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
        split=_overfit_split,
        data_root=data_root,
        csv_manifest=csv_manifest,
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
        min_points=min_points,
        target_keys=list(target_keys),
        primary_target_key=main_task,
        max_sample=val_max_sample,
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
        split=_overfit_split,
        data_root=data_root,
        csv_manifest=csv_manifest,
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
        min_points=min_points,
        target_keys=list(target_keys),
        primary_target_key=main_task,
        max_sample=test_max_sample,
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
