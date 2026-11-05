"""
LitePT-S part segmentation on ShapeNetPart (50 parts, 16 categories), trained from scratch.

Phase 0 sanity check of the hybrid-SSL project: the ShapeNetPart port (dataset, evaluator,
tester) trains and reports mIoU over the 50 parts (plus the per-shape instance and category
mIoU). Input features: coord only (in_channels=3). Trained on train + val, validated on test
(upstream Pointcept convention).
The backbone (LitePT-v1, or LitePT-v2 with its norm/stem options) is set by the variables
below, for the BN vs LN reference of phase 1.

Data: data/shapenetcore_partanno_segmentation_benchmark_v0_normal (see docs/data.md).
"""

# -----------------------------------------------------------------------------
# Default
# -----------------------------------------------------------------------------
_base_ = ["../_base_/default_runtime.py", "../_base_/dataset/shapenet_part.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------

# Hardware parameters
num_gpu = 1
num_worker = 8 * num_gpu
enable_amp = True

# Data parameters
batch_size = 32  # total batch size across all gpus
batch_size_val = 32
grid_size = 0.01

# Optimization parameters
lr = 2e-3
epoch = 10
eval_epoch = 10

# Dataset / task
num_classes = 50
ignore_index = -1  # dummy: every point has a part label
feat_keys = ["coord"]

# Backbone: "LitePT-v1", or "LitePT-v2" with v2_options (e.g. dict(norm="bn", stem="conv"))
backbone_type = "LitePT-v1"
v2_options = dict()

# Wandb parameters
enable_wandb = False
wandb_project = "hybrid_ssl"
wandb_run_name = f"ShapeNetPart {backbone_type}-S semseg lr={lr} ep={epoch}"

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=50),
    dict(type="ShapeNetPartSegEvaluator", metric="mIoU", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator"),
]

test = dict(type="ShapeNetPartSegTester", verbose=True)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=72,
    backbone=dict(
        type=backbone_type,
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        # LitePT-S
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
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
        **v2_options,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=ignore_index,
        ),
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
dataset_type = "ShapeNetPartDataset"
data_root = "data/shapenetcore_partanno_segmentation_benchmark_v0_normal"

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    train=dict(
        type=dataset_type,
        split=["train", "val"],
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "cls_token"),
                feat_keys=feat_keys,
            ),
        ],
        loop=1,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "cls_token",
                    "origin_segment",
                    "inverse",
                ),
                feat_keys=feat_keys,
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[dict(type="CenterShift", apply_z=True)],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "cls_token"),
                    feat_keys=feat_keys,
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[1, 1], anisotropic=True)],
            ],
        ),
    ),
)
