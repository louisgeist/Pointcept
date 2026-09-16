"""
Sonata-v1m2 SSL-pretrained PT-v3m2 backbone (Flair3D+ fork, job 862680/epoch_120,
same checkpoint used as the frozen Sonata baseline in README_grid_then_seed.md),
fine-tuned in Flair3D+ multitask mode: segment (v20) + forest_2d + elevation + 4
nathab tile_distribution axes (WeightedKL; Habitat Type / Moisture Regime / Soil
Chemistry / Bioclimatic Zone, remapped on the fly from natural_habitat) + network
(roads only, CE + foreground weight=5; railroads/transmission lines dropped;
scored via APLS at test time).

Task set / criteria / hooks / Collect pipeline copied verbatim from the "classique"
multi-task recipe (configs/flair3d_default/multi-litept-b-v1m0-flair3d.py, the
LitePT-B checkpoint referenced as W_LPT/873542 in README_grid_then_seed.md) --
only the model block differs.

Backbone is PT-v3m2, not LitePT-B: the Sonata checkpoint's state_dict shapes
only match a PT-v3m2 backbone built with the exact pretrain encoder hyperparams
(configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py) -- enc_depths=
(3,3,3,12,3), enc_channels=(48,96,192,384,512), stride=(3,3,3,3). Loading into
LitePT-B would hard-fail on a shape mismatch inside matching keys (strict=False
only tolerates missing/extra *keys*, not mismatched *shapes*).

Full fine-tune with a freshly-initialized decoder on top of the loaded Sonata
encoder (enc_mode=False): dec_depths/dec_channels/dec_num_head/dec_patch_size
match the upstream official Sonata full-FT reference
(configs/sonata/semseg-sonata-v1m1-0c-scannet-ft.py), which uses the same
enc_channels=(48,96,192,384,512) pretrain backbone. The decoder has no
counterpart in the SSL checkpoint's state_dict, so CheckpointLoader's
strict=False load leaves it randomly initialized (same mechanism the upstream
config itself relies on).
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
# batch_size carried over unchanged from the malibu/LitePT multitask baselines;
# not recalibrated for this (decoder-less, lighter) backbone -- rerun
# scripts/find_max_batch_size.py --mix-prob 0.8 before trusting it at scale.
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
wandb_run_name = f"Sonata-FT multi {grp_exp}.{num_exp}) iter={total_iters}"
wandb_project = "flair3d_multi"

# -----------------------------------------------------------------------------
# Pretrained weight
# -----------------------------------------------------------------------------
# Sonata-v1m2 SSL pretrain on Flair3D+ (job 862680), epoch_120 -- the canonical
# checkpoint used as W_SONATA across cross-domain GridProbe evals
# (README_grid_then_seed.md), not the final epoch_150.
weight = "/lustre/fsn1/projects/rech/unv/usi32yh/logs/pointcept_logs/slurm/862680/model/epoch_120.pth"

# -----------------------------------------------------------------------------
# Multitask configuration : targets configuraiton
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
# nathab_* axes are unpacked from on-disk natural_habitat.npy (N, 4) in get_data.
# GradNormLite: pool the 4 nathab axes into one "nathab" group.
grad_norm_lite_task_groups = {task_name: "nathab" for task_name in nathab_keys}


# Elevation in meters: no Collect key_scales, no denorm via target_scales
# (matches configs/experiment/w107/7/toward_bm/multi-litept-v1m0-flair3d_1.py).
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
# space (~1 m Huber threshold). See multi-litept-v1m0-flair3d_1.py (w107/7/toward_bm).
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
    # Remap the SSL teacher/student checkpoint's backbone onto this model's
    # plain backbone; strict=False (default) tolerates the multi-task heads
    # and teacher/momentum-encoder keys having no counterpart on either side.
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
    # After PreciseEvaluator only (end of training / tools/test.py) -- not on val.
    dict(type="NetworkAPLSEvaluator"),
]

test_single_fragment = True
test = dict(type="MultiTaskTester", verbose=True, write_cls_iou=True)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# MultiTaskSegmentorV2 attaches per-task heads on top of backbone features
# (semantic: nn.Linear(backbone_out_channels, num_classes_task); elevation: 1).
# PT-v3m2, enc_mode=False: encoder dims match the Sonata pretrain checkpoint
# (loaded by CheckpointLoader below); dec_* is a freshly-initialized decoder,
# dims copied from the upstream official Sonata full-FT reference
# (configs/sonata/semseg-sonata-v1m1-0c-scannet-ft.py, same enc_channels).
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
    # Real fine-tune, not a frozen probe (contrast with
    # configs/flair3d_default/probe/sonata-v1m2-flair3d-lin.py's freeze_backbone=True).
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
dataset_type = "Flair3DDataset"
data_root = "data/flair3d_plus"
csv_manifest = "data/flair3d_plus/raw/scene_split_manifest.csv"
min_points = {"train": 1000}
val_stratified_subset_manifest = "data/flair3d_plus/manifests/val_dev_subset_2000.csv"

# Opt-in APLS scoring of PreciseEvaluator test logits (see NetworkAPLSEvaluator /
# tools/test.py). ``split`` must match ``data.test.split`` so stitched ROIs find
# ``{patch_id}_logits_network.npy``. Jean Zay graphs root (Hecate uses
# /data/geist/Flair3D-build/data/network_graphs).
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
