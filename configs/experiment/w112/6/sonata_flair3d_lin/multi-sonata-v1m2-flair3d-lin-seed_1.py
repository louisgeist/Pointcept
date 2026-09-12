"""
Sonata-v1m2 multi-task linear probe on Flair3D+ — seed-ensemble template
(w112/6/sonata_flair3d_lin). Frozen PT-v3m2 encoder (enc_mode=True, 1232ch)
+ linear heads for the 5-task recipe:

  segment (v20) + forest_2d + elevation + 4 nathab tile_distribution axes
  (WeightedKL) + network (roads only, CE foreground weight=5).

Task wiring matches configs/flair3d_default/multi-litept-b-v1m0-flair3d.py;
backbone / freeze / CheckpointLoader match the sibling GridProbe
(sonata-v1m2-flair3d-lin-grid_1.py). No GradNormLite (trunk frozen), no
learned mask values, no RandomDropColor/Strength, no coord_feat_scale
(Sonata pretrain does not use one).

`lr` / `seed` / `num_exp` are patched by
scripts/sonata/gen_flair3d_multitask_lin_seeds.py from the GridProbe winner
(val mIoU). Placeholder lr below is 2e-3 until that sweep finishes.

Frozen checkpoint: pretrain job 862680 / epoch_120 (W_SONATA).
Launch after generation: scripts/sonata/sbatch_flair3d_multitask_lin_seeds_h100.sh
"""

_base_ = ["../../../../_base_/default_runtime.py"]

# Sonata pretrain ckpt (job 862680, epoch 120) — remap via CheckpointLoader.
weight = "/lustre/fsn1/projects/rech/unv/usi32yh/logs/pointcept_logs/slurm/862680/model/epoch_120.pth"

# -----------------------------------------------------------------------------
# Seed-ensemble knobs (patched by gen_flair3d_multitask_lin_seeds.py)
# -----------------------------------------------------------------------------
grp_exp = 2
num_exp = 1
seed = 0
lr = 2e-3  # placeholder; replace with GridProbe winner lr

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------
num_gpu = 1
num_worker = 16  # H100 Jean-Zay
num_worker_test = 2
enable_amp = True
empty_cache = False
evaluate = True
clip_grad = 3.0

batch_size = 24
batch_size_val = 8
batch_size_test = 8
val_voxel_budget = 2_000_000
test_voxel_budget = 2_000_000

grid_size = 0.1
point_max = 102400
mix_prob = 0.8
patch_size = 1024

total_iters = 10000
iter_per_epoch = 1000
eval_every = 2

feat_keys = ["coord", "color", "strength"]
backbone_out_channels = 1232  # enc_mode: 48+96+192+384+512

wandb_project = "flair3d_sonata"
wandb_run_name = (
    f"Sonata-v1m2 multi lin-probe {grp_exp}.{num_exp} H100 | "
    f"seed={seed} lr={lr} | iters={total_iters}"
)

# -----------------------------------------------------------------------------
# Multitask configuration
# -----------------------------------------------------------------------------
from pointcept.datasets.flair3d_config_utils import (
    FLAIR3D_TILE_DISTRIBUTION_TASKS,
    init_task_configs,
    init_task_criteria,
    FLAIR3D_COLLECT_PREFIX_LITEPT,
    init_multitask_collect_keys,
)

main_task = "segment"
nathab_keys = tuple(FLAIR3D_TILE_DISTRIBUTION_TASKS.keys())
target_keys = (main_task, "forest_2d", "elevation") + nathab_keys + ("network",)

# Elevation in meters: no Collect key_scales, no denorm via target_scales.
target_scales = {}

label_definitions = dict(
    segment="v20",
)

task_configs = init_task_configs(target_keys, definitions=label_definitions)
# Network head: ROADS only (RAILROADS channel dropped), CE + weight=5 on
# the foreground pixel class.
task_configs["network"]["num_networks"] = 1
task_configs["network"]["channel_names"] = ["ROADS"]
task_criteria = init_task_criteria(task_configs)
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

del (
    init_task_configs,
    init_task_criteria,
    FLAIR3D_TILE_DISTRIBUTION_TASKS,
)

num_classes = task_configs[main_task]["num_classes"]
ignore_index = task_configs[main_task]["ignore_index"]
names = task_configs[main_task]["names"]

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=100),
    dict(type="MultiTaskEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
    dict(type="NetworkAPLSEvaluator"),
]

test_single_fragment = True
test = dict(type="MultiTaskTester", verbose=True, write_cls_iou=True)

# -----------------------------------------------------------------------------
# Model — frozen Sonata encoder, linear heads only
# -----------------------------------------------------------------------------
model = dict(
    type="MultiTaskSegmentorV2",
    backbone_out_channels=backbone_out_channels,
    backbone=dict(
        type="PT-v3m2",
        in_channels=7,  # coord(3) + color(3) + strength(1)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(3, 3, 3, 3),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(patch_size, patch_size, patch_size, patch_size, patch_size),
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
        enc_mode=True,
        freeze_encoder=False,
    ),
    freeze_backbone=True,
    task_configs=task_configs,
    main_task=main_task,
    task_criteria=task_criteria,
    task_weights=task_weights,
)

# -----------------------------------------------------------------------------
# Optimizer / scheduler — heads only (backbone frozen)
# -----------------------------------------------------------------------------
# Empty param_dicts (not None): build_optimizer skips frozen params instead of
# stuffing the whole PT-v3m2 into AdamW state. OneCycleLR max_lr is a scalar
# (single param group).
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.0)
scheduler = dict(
    type="OneCycleLR",
    max_lr=lr,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = []

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset_type = "Flair3DDataset"
data_root = "data/flair3d_plus"
csv_manifest = "data/flair3d_plus/raw/scene_split_manifest.csv"
min_points = {"train": 1000}
val_stratified_subset_manifest = "data/flair3d_plus/manifests/val_dev_subset_2000.csv"

network_apls_eval = dict(
    network_graphs_root="/lustre/fsn1/projects/rech/unv/usi32yh/data_flair3d_build/network_graphs",
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
    remove_small_objects_enabled=False,
    remove_small_objects_min_size_px=8,
    skeletonize_enabled=True,
    open_iterations=0,
    close_iterations=5,
    morph_connectivity=8,
    min_component_nodes=5,
    apls_max_nodes_exact=None,
    apls_densify=50.0,
    apls_snap_to_edge=4.0,
    apls_symmetric=True,
    apls_min_path_length_m=5,
)

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
            dict(type="NetworkRasterToPointLabels"),
            dict(type="NetworkRasterToPointLabels", target_key="forest_2d"),
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
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ],
        ),
    ),
)
