"""
LitePT-v1 on Flair3D+ (coord + RGB + strength in feat_keys).

Mono-task Flair3D+ config for target ``network`` (binary 1 m Lambert pixel
semantic segmentation of ROADS / RAILROADS only). TRANSMISSION_LINES is not
trained. Loss: FocalLoss + LovaszLoss (replaces CrossEntropy + Lovasz).

Train/val/test use the roads+railroads filtered manifest (build on Jean Zay)::

    python - <<'PY'
    import csv
    src = "data/flair3d_plus/raw/scene_split_manifest.csv"
    dst = "data/flair3d_plus/raw/scene_split_manifest_roads_railroads.csv"
    with open(src, newline="", encoding="utf-8") as f_in, open(
        dst, "w", newline="", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        n = 0
        for row in reader:
            if row.get("ROADS") == "True" or row.get("RAILROADS") == "True":
                writer.writerow(row)
                n += 1
    print(f"Wrote {n} rows -> {dst}")
    PY

Online val is further restricted to ``val_dev_subset_2000.csv`` (intersection
with the filtered CSV). Requires ``network.npy`` from ``rasterize_network.py``
and ``ExtractAbsXY`` before geometric shifts.
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
num_exp = 2


# Hardware parameters
num_gpu = 1
num_worker = 8 * num_gpu
enable_amp = True

# Data parameters
batch_size = 20 * num_gpu  # total batch size across all gpus
batch_size_val = 5
batch_size_test = batch_size // 4

grid_size = 0.1
point_max = 102400
# Mix3D OK: NetworkRasterToPointLabels makes network point-wise before collate.
mix_prob = 0.8

patch_size = 1024

# Optimization parameters
lr = 1e-3
total_iters = 30_000
warmup_iters = 500

# Features
learned_masked_feat = True
feat_keys = ["coord", "color", "strength"]
coord_feat_scale = 0.01

# Wandb parameters
wandb_run_name = (
    f"Flair3D+ LitePT mono network roads+railroads focal+lovasz "
    f"{grp_exp}.{num_exp} lr={lr} iters={total_iters} filtered+strat_val"
)
wandb_project = "flair3d_network"

# -----------------------------------------------------------------------------
# Mono-task configuration : targets configuration
# -----------------------------------------------------------------------------
from pointcept.datasets.flair3d_config_utils import (
    init_task_configs,
    init_task_criteria,
    FLAIR3D_COLLECT_PREFIX_LITEPT,
    init_multitask_collect_keys,
)

main_task = "network"
target_keys = (main_task,)

task_configs = init_task_configs(target_keys)
task_criteria = init_task_criteria(task_configs)
# Focal + Lovasz for network (replace default CE + Lovasz).
_ignore = int(task_configs[main_task]["ignore_index"])
task_criteria[main_task] = [
    dict(
        type="FocalLoss",
        gamma=2.0,
        alpha=0.5,
        loss_weight=1.0,
        ignore_index=_ignore,
    ),
    dict(
        type="LovaszLoss",
        mode="multiclass",
        loss_weight=1.0,
        ignore_index=_ignore,
    ),
]
del _ignore
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
# MultiTaskSegmentorV2: max-pool points into 1 m pixels, then Linear(C -> 2*r).
model = dict(
    type="MultiTaskSegmentorV2",
    backbone_out_channels=72,
    backbone=dict(
        type="LitePT-v1",
        in_channels=7,  # coord (3) + color (3) + strength (1)
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
    total_iters=warmup_iters,
)
param_dicts = [dict(keyword="block", lr=lr / 10)]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset_type = "Flair3DDataset"
data_root = "data/flair3d_plus"
csv_manifest = "data/flair3d_plus/raw/scene_split_manifest_roads_railroads.csv"
missing_tiles_manifest = "data/flair3d_plus/missing_ply_preflight.txt"
too_small_tiles_manifest = "data/flair3d_plus/too_small_tiles.csv"
val_stratified_subset_manifest = "data/flair3d_plus/manifests/val_dev_subset_2000.csv"

# Opt-in APLS scoring of PreciseEvaluator test logits (see NetworkAPLSEvaluator /
# tools/test.py). ``split`` must match ``data.test.split``. Jean Zay graphs root.
network_apls_eval = dict(
    network_graphs_root="/lustre/fsn1/projects/rech/unv/usi32yh/data_flair3d_build/network_graphs",
    split="test",
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
        target_keys, collect_prefix_keys=FLAIR3D_COLLECT_PREFIX_LITEPT
    )
)

del FLAIR3D_COLLECT_PREFIX_LITEPT, init_multitask_collect_keys

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
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
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
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
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
