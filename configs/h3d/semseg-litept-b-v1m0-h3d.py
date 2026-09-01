"""
LitePT-Base semantic segmentation on H3D (coord + RGB point features).

Backbone dims and optimization recipe mirror the LitePT-Base used in
configs/malibu3d_default/multi-litept-b-v1m0-malibu3d.py / configs/dales/semseg-litept-b-v1m0-dales.py,
adapted to H3D single-task semseg (DefaultSegmentorV2, no multitask wiring)
and trained from scratch (no pretrained weight). See
configs/h3d/semseg-litept-v1m0-h3d.py for the LitePT-Small counterpart.

H3D has real RGB but no native intensity (LAS intensity is all-zero), so
features stay native: coord + color only. No FillMissingFeat / strength
channel (those are reserved for GridProbes that must match a Malibu3D
pretrained 7-ch input). RandomDropColor + learned color masking follow the
ECLAIR / Malibu3D Lite-B RGB recipe.

This config is intentionally self-contained: it inherits only from
default_runtime and can be read top-to-bottom without cross-referencing
other H3D configs.
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
enable_amp = True  # LitePT-Base is heavier than Small; matches multi-litept-b-v1m0-malibu3d.py

# Data parameters
batch_size = 12  # total batch size across all gpus; LitePT-Base convention (vs 24 for Small)
batch_size_val = batch_size // 2

grid_size = 0.1
point_max = 102400
mix_prob = 0.8

patch_size = 1024

# Optimization parameters
lr = 1e-3
epoch = 200
eval_epoch = epoch // 10

# Dataset / task
num_classes = 11
ignore_index = num_classes

# Features (native H3D: RGB, no real intensity)
learned_masked_feat = True
feat_keys = ["coord", "color"]
coord_feat_scale = 0.01

# Test
test_single_fragment = True

# Wandb parameters
wandb_run_name = f"H3D LitePT-B semseg from-scratch ({grp_exp}.{num_exp}) lr={lr}"
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
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=72,
    backbone=dict(
        type="LitePT-v1",
        in_channels=6,  # coord (3) + color (3)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        # LitePT-Base dims (matches multi-litept-b-v1m0-malibu3d.py backbone).
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
        masked_feat_keys=["color"],
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
    ],
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
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "grid_size"),
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
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Z_MinShift"),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
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
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
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
        transform=[
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
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
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
