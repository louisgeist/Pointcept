"""
KPConvX semantic segmentation on H3D (coord + RGB point features).

This config is intentionally self-contained: it inherits only from
default_runtime and can be read top-to-bottom without cross-referencing
other H3D configs.
"""

# -----------------------------------------------------------------------------
# Default
# -----------------------------------------------------------------------------
_base_ = ["../../../../_base_/default_runtime.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------

# Logging parameters
grp_exp = 2
num_exp = 1

# Hardware parameters
num_gpu = 1
num_worker = 8 * num_gpu
enable_amp = False

# Data parameters
batch_size = 2 * num_gpu  # total batch size across all gpus
batch_size_val = batch_size // 2
# batch_size_test = batch_size // 2

grid_size = 0.1
point_max = 40000
mix_prob = 0.8

# Optimization parameters
lr = 1e-2
epoch = 100
eval_epoch = epoch // 10

# Dataset / task
num_classes = 11
ignore_index = num_classes

# Features
feat_keys = ["coord", "color"]

# Test
test_single_fragment = True

# Wandb parameters
wandb_run_name = f"H3D KPConvX semseg ({grp_exp}.{num_exp}) lr={lr} epoch={epoch}"
wandb_project = "pointcept_h3d"

log_test_f1 = True

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=100),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

test = dict(type="SemSegTester", verbose=True)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="kpconvx_base",
        input_channels=6,  # coord (3) + RGB (3)
        num_classes=num_classes,
        dim=3,
        task="cloud_segmentation",
        kp_mode="kpconvx",
        shell_sizes=(1, 14, 28),
        kp_radius=2.3,
        kp_aggregation="nearest",
        kp_influence="constant",
        kp_sigma=2.3,
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
        radius_scaling=2.2,
        decoder_layer=True,
        grid_pool=True,
        upsample_n=3,
        first_inv_layer=1,
        drop_path_rate=0.3,
        norm="batch",
        bn_momentum=0.1,
        smooth_labels=False,
        class_w=(),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
    ],
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
    div_factor=100.0,
    final_div_factor=1000.0,
)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset_type = "H3DDataset"
data_root = "data/h3d"

class_names = [
    "Low Vegetation",
    "Impervious Surface",
    "Vehicle",
    "Urban Furniture",
    "Roof",
    "Façade",
    "Shrub",
    "Tree",
    "Soil or Gravel",
    "Vertical Surface",
    "Chimney",
    "Void",
]

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
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
                return_min_coord=True,
            ),
            dict(type="SphereCrop", point_max=point_max, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=feat_keys,
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
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
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "origin_segment", "inverse"),
                feat_keys=feat_keys,
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
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
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index"),
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
