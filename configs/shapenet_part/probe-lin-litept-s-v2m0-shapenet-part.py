"""
Linear probe of a frozen Sonata LitePT-v2-S encoder (pretrain-sonata-litept-s-v2m0-shapenet-part.py)
on ShapeNetPart part segmentation: one CE + Lovasz linear head on the encoder features up-cast
to full resolution (36 + 72 + 144 + 252 + 504 = 1008 channels). Short schedule (2000
iterations), validation on the test split; reports instance / category mIoU with and without
category masking of the logits. The best ins_mIoU_catmask is written to save_path/metrics.json
(read by PeriodicProbeHook). Pass the checkpoint with --options weight=<path>.

The input pipeline matches the pretraining: NormalizeCoord, 0.02 grid, coord features.
probe_branch selects the student or the EMA teacher backbone of the Sonata checkpoint.
"""

_base_ = ["../_base_/default_runtime.py", "../_base_/dataset/shapenet_part.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------
num_gpu = 1
batch_size = 32
batch_size_val = 32
num_worker = 8 * num_gpu
mix_prob = 0
clip_grad = 3.0
enable_amp = True
evaluate = True

grid_size = 0.02
feat_keys = ["coord"]

# 2000 steps / 200 per epoch -> 10 trainer epochs, validation every 5
total_iters = 2000
iter_per_epoch = 200
eval_every = 5

probe_branch = "student"  # or "teacher" (EMA)

num_classes = 50
ignore_index = -1  # dummy: every point has a part label

enable_wandb = False
wandb_project = "hybrid_ssl"
wandb_run_name = f"Sonata LitePT-v2-S lin probe ShapeNetPart | {probe_branch}"

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=1008,
    backbone=dict(
        type="LitePT-v2",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=True,
        norm="ln",
        stem="linear",
        mask_token=True,  # loaded from the checkpoint, unused without a mask
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
    freeze_backbone=True,
)

# -----------------------------------------------------------------------------
# Optimizer / scheduler
# -----------------------------------------------------------------------------
lr = 0.002
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=lr,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

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
            dict(type="NormalizeCoord"),
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
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
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="NormalizeCoord"),
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
)

hooks = [
    dict(
        type="CheckpointLoader",
        keywords=f"module.{probe_branch}.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=50),
    dict(type="ShapeNetPartSegEvaluator", metric="ins_mIoU_catmask"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="MetricsJsonWriter"),
]
