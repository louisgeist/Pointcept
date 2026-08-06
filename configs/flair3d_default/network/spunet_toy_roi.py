"""
SpUNet ROI overfit for network pixel segmentation on Hecate (D067).

Same recipe as ``spunet_toy_long.py``, but overfits on the full
D067-2021_AF-S1-22 ROI (100 tiles, all locally mirrored on Hecate) instead of
a single subtile. With real per-tile diversity, the single-tile
loop-to-fill-a-batch trick is no longer needed: SphereCrop(mode="random")
gives a random center per sample, batch_size is raised accordingly, and
mix_prob>0 is safe again (NetworkRasterToPointLabels rasterizes labels
per-point after collate, same as ``spunet_network_toy.py``).

Example::

    export PYTHONPATH="$PWD"
    python tools/train.py --config-file configs/flair3d_default/network/spunet_toy_roi.py \\
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
num_exp = 3
seed = 14028665

num_gpu = 1
num_worker = 8 * num_gpu
enable_amp = True

batch_size = 8 * num_gpu
batch_size_val = batch_size
batch_size_test = 1

# ROI manifest is already filtered to the 100 D067-2021_AF-S1-22 tiles, so no
# max_sample truncation is needed -- every row is used.
train_max_sample = None
val_max_sample = None
test_max_sample = None

# No single-tile repeat trick needed anymore (real per-tile diversity).
overfit_train_loop = 1

grid_size = 0.1
point_max = 100000
# Real multi-tile diversity: Mix3D is safe again (see module docstring).
mix_prob = 0.8

lr = 1e-3
total_iters = 8_000
iter_per_epoch = 100
eval_every = 5
warmup_iters = 200

learned_masked_feat = True
feat_keys = ["coord", "color", "strength"]
coord_feat_scale = 0.01

wandb_run_name = (
    f"SpUNet network overfit-ROI D067-2021_AF-S1-22 | lr={lr}, iters={total_iters}, "
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
# Dataset (D067-2021_AF-S1-22 ROI on Hecate, 100 tiles)
# -----------------------------------------------------------------------------
dataset_type = "Flair3DDataset"
data_root = "data/flair3d_plus"
csv_manifest = "data/flair3d_plus/raw/scene_split_manifest_D067-2021_AF-S1-22.csv"
missing_tiles_manifest = "data/flair3d_plus/missing_ply_preflight.txt"
too_small_tiles_manifest = "data/flair3d_plus/too_small_tiles.csv"

# Runs tools/eval_network_apls.py at the end of tools/test.py and from
# NetworkAPLSEvaluator.after_train (after PreciseEvaluator only -- not on val).
# ``split`` must match ``data.test.split`` so stitched ROIs find logits on disk.
# This ROI manifest is train-only, and the overfit loop uses train for test too.
network_apls_eval = dict(
    network_graphs_root="/data/geist/Flair3D-build/data/network_graphs",  # Hecate path
    split="train",
    threshold=0.5,
    overlap_combine="nanmean",  # nanmean|max|first, combines overlapping subtile predictions
    connectivity=4,  # pixel-graph connectivity for the predicted mask: 4 or 8
    rdp_epsilon_m=2.0,  # Ramer-Douglas-Peucker simplification epsilon (meters)
    endpoint_fix_stage="pre_rdp",  # pre_rdp|post_rdp: when the diagonal endpoint-fix runs
    merge_weight_threshold=2.5,  # post-RDP node-merge edge-weight threshold
    # Hard cap on exact O(V^2) APLS after densification (raises rather than silently
    # subsampling). None disables the cap. The whole run_network_apls_eval_if_configured()
    # call is one try/except around the *entire* eval_network_apls.run() -- a single
    # oversized ROI with a finite cap would otherwise abort APLS for every other ROI.
    max_nodes_exact=None,
    max_rois=None,  # optional debug limit on number of ROIs scored
    densify=50.0,  # SpaceNet-aligned max edge length (meters) before matching; None to disable
    snap_to_edge=4.0,  # snap-to-edge control-point matching radius (meters); None = unrestricted NN
    symmetric=True,  # score both GT->pred and pred->GT, take the harmonic mean
    radius_fix_radius_m=5,  # predicted-graph endpoint/isolated-node radius reconnection (meters); None = disabled
    min_path_length_m=5,  # SpaceNet-style short-path filter (meters); None = disabled
)

train_multitask_keys, val_multitask_keys, multitask_index_valid_keys = (
    init_multitask_collect_keys(
        target_keys, collect_prefix_keys=FLAIR3D_COLLECT_PREFIX_GRID
    )
)

del FLAIR3D_COLLECT_PREFIX_GRID, init_multitask_collect_keys

# Same split for train/val/test so overfit metrics watch the same ROI tiles.
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
            dict(type="SphereCrop", point_max=point_max, mode="random"),
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
