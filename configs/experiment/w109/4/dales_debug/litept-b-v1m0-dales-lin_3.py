"""
LitePT-Base linear probing on DALES — BN-only ablation (isolates the
BatchNorm running-stat effect from DropPath, see litept-b-v1m0-dales-lin_2.py
in this same folder for the "both effects" classic-protocol baseline, and
w109/3/dales_lin/litept-b-v1m0-dales-lin_2.py for the original grid-probe
run this is meant to explain).

Context: grid-probe sweeps of this exact LitePT-B checkpoint (job 873542)
topped out around mIoU=0.25 on DALES val, while the classic single-probe
config with matching hyperparams (litept-b-v1m0-dales-lin_2.py, ce_lovasz,
lr=2e-2, wd=5e-3, no input_norm) reached 0.31 after just one epoch. Root
cause: GridProbeSegmentorV2's freeze_backbone forces the backbone into
eval() mode even during training (see pointcept/models/grid_probe.py
docstring), whereas DefaultSegmentorV2's freeze_backbone only sets
requires_grad=False and lets the backbone follow the trainer's normal
train()/eval() propagation. For LitePT-v1 this eval-pinning affects two
independent things: BatchNorm1d running_mean/var stop drifting toward the
DALES batch distribution (a free, gradient-free domain recalibration the
classic config was getting "by accident"), and DropPath(0.3) stochastic
depth is disabled.

This config isolates the BatchNorm piece alone: `bn_eval_mode=False`
(BN running stats adapt during training, exactly like the classic protocol)
+ `drop_path_eval_mode=True` (DropPath stays inactive, exactly like the
original grid-probe runs). Everything else — optimizer/scheduler/loss/
GridProbeTrainer machinery — is unchanged from the grid-probe path, with a
single probe replicating that sweep's `ce_lovasz_lr2e-2_wd5e-3_none` point,
so the only variable against that historical ~0.25 run is this one flag.
"""

_base_ = ["../../../../_base_/default_runtime.py"]

grp_exp = 1
num_exp = 3

num_classes = 8
ignore_index = 8
grid_size = 0.1
point_max = 102400
coord_feat_scale = 0.01  # must match Flair3D multitask pretrain

num_gpu = 1
epoch = 10
eval_epoch = 10
lr = 2e-2
patch_size = 1024

test_single_fragment = True

# misc custom setting
batch_size_per_gpu = 24
batch_size = batch_size_per_gpu * num_gpu
batch_size_val = 1
batch_size_test = 1  # DALES test tiles are unchunked full scenes (~12M raw pts each);
# batch_size // 2 packs multiple into one forward pass and crashes spconv (CUBLAS/illegal
# memory access in the stem indice_conv). 1 mirrors batch_size_val, which already works.
num_worker = 8 * num_gpu
num_worker_test = 2
mix_prob = 0.8
empty_cache = False
enable_amp = False

# dataset settings
dataset_type = "DALESDataset"
data_root = "data/dales"

weight = "/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/873542/model/model_best.pth"

wandb_project = f"pointcept_{dataset_type[:-7].lower()}"

# Hooks — same GridProbe hook chain as the original sweep (see ordering note
# in w109/3/dales_lin/litept-b-v1m0-dales-lin_2.py); only 1 probe here.
hooks = [
    dict(
        type="CheckpointLoader",
        exclude_keys=("seg_heads", "reg_heads", "cls_heads", "pixel_seg_heads", "cls_attn_pools"),
    ),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=10),
    dict(type="GridProbeEvaluator", write_cls_iou=True),
    dict(type="GridProbeCheckpointSaver"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="GridProbeWinnerSelector", skip_test=False),
]

feat_keys = ["coord", "color", "strength"]

names = [
    "Ground",
    "Vegetation",
    "Cars",
    "Trucks",
    "Power lines",
    "Fences",
    "Poles",
    "Buildings",
    "Unknown",
]

# Decoder levels (72, 108, 216, 432) + raw encoder bottleneck (576).
dec_channels = (72, 108, 216, 432)
bottleneck_channels = 576
backbone_out_channels = sum(dec_channels) + bottleneck_channels  # 1404

# Single probe replicating the grid sweep's ce_lovasz_lr2e-2_wd5e-3_none point.
probes = {
    "ce_lovasz_lr2e-2_wd5e-3_none": dict(
        criteria=[
            dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
            dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
        ],
        input_norm=None,
        feat_norm=None,
        dropout=0.0,
        optimizer=dict(type="AdamW", lr=lr, weight_decay=0.005),
        scheduler=dict(
            type="OneCycleLR",
            max_lr=lr,
            pct_start=0.05,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=1000.0,
        ),
        grad_clip=3.0,
    ),
}

wandb_run_name = (
    f"LitePT-B GridProbe DALES {grp_exp}.{num_exp}) BN-only ablation "
    f"(bn_eval_mode=False, drop_path_eval_mode=True), decoder hypercolumn 1404ch, epoch={epoch}"
)

# model settings
model = dict(
    type="GridProbeSegmentorV2",
    probes=probes,
    num_classes=num_classes,
    ignore_index=ignore_index,
    target_key="segment",
    backbone_out_channels=backbone_out_channels,
    backbone=dict(
        type="LitePT-v1",
        in_channels=7,  # coord(3) + color(3, fake) + strength(1)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(3, 3, 3, 3),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, bottleneck_channels),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(patch_size, patch_size, patch_size, patch_size, patch_size),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=dec_channels,
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
        dec_traceable=True,
    ),
    freeze_backbone=True,
    bn_eval_mode=False,  # let BatchNorm running stats adapt to DALES during training
    drop_path_eval_mode=True,  # keep DropPath inactive, like the original grid-probe runs
    feature_mask_values=dict(
        enable=True,
        masked_feat_keys=["color", "strength"],
    ),
)

# trainer settings — GridProbeTrainer builds one optimizer/scheduler per probe
# (see probes above); no top-level optimizer/scheduler/param_dicts here.
train = dict(type="GridProbeTrainer")

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=names,
    task_configs={
        name: dict(
            task_type="semantic",
            num_classes=num_classes,
            ignore_index=ignore_index,
            names=names,
        )
        for name in probes
    },
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
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=point_max, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="FillMissingFeat", feat_key="color", feat_dim=3),
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
            dict(type="CenterShift", apply_z=False),
            dict(type="FillMissingFeat", feat_key="color", feat_dim=3),
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
            dict(type="FillMissingFeat", feat_key="color", feat_dim=3),
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
                    optional_keys=("inverse",),  # for test_single_fragment broadcast
                    feat_keys=feat_keys,
                    feat_scales=dict(coord=coord_feat_scale),
                ),
            ],
            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]],
        ),
    ),
)
