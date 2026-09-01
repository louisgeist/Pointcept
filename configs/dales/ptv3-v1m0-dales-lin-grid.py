"""
PT-v3-malibu grid-search linear probing on DALES — decoder hypercolumn /
multi-scale variant (transfer from Malibu3D+ multitask supervised pretrain at
w109/2/ptv3_wd/multi-ptv3-v1m0-malibu3d_5.py, job 1052200).

Unlike w109/4/ptv3_lin/ptv3-v1m0-dales-lin_1.py (finest decoder stage only,
64ch), this config sets `traceable=True` so GridProbeSegmentorV2 concatenates
every decoder-stage feature + the encoder bottleneck into one 1024-dim vector
per point (64+64+128+256 + 512). Same DALES-has-no-RGB handling as the other
PTv3/LitePT DALES lin configs.

Grid (12 probes): ce_lovasz, AdamW/wd0/OneCycleLR warmup5%, lr sweep {1e-4 … 5e-1}.
epoch=400 / eval_epoch=10. AMP enabled (fp16).
`bn_eval_mode=True` is a no-op for PT-v3-malibu (LayerNorm only);
`drop_path_eval_mode=True` keeps DropPath(0.3) inactive during probe training.
"""

_base_ = ["../_base_/default_runtime.py"]

grp_exp = 1
num_exp = 1

num_classes = 8
ignore_index = 8
grid_size = 0.1
point_max = 102400
coord_feat_scale = 0.01  # must match Malibu3D multitask pretrain
strength_feat_scale = 1 / 60000  # DALES raw intensity → Malibu3D [0,1] convention

num_gpu = 1
epoch = 400
eval_epoch = 10
lr = 5e-2
patch_size = 1024

test_single_fragment = True

# misc custom setting
batch_size_per_gpu = 24
batch_size = batch_size_per_gpu * num_gpu
batch_size_val = 1
batch_size_test = 1
num_worker = 24 * num_gpu  num_worker_test = 2
mix_prob = 0.8
empty_cache = False
enable_amp = True

# dataset settings
dataset_type = "DALESDataset"
data_root = "data/dales"

weight = "ckpt/malibu3d/ptv3_multitask/model_best.pth"

wandb_project = f"pointcept_{dataset_type[:-7].lower()}"

# Hooks
# Order matters: GridProbeEvaluator before GridProbeCheckpointSaver/CheckpointSaver;
# GridProbeWinnerSelector last (frees the per-probe optimizers/schedulers, then runs
# its own SemSegTester pass on the winning probe — replaces PreciseEvaluator, which
# never sets raw_model.active_probe and would break under GridProbeSegmentorV2).
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
    dict(type="GridProbeWinnerSelector", skip_test=True),
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

# Decoder levels (64, 64, 128, 256) + raw encoder bottleneck (512).
dec_channels = (64, 64, 128, 256)
bottleneck_channels = 512
backbone_out_channels = sum(dec_channels) + bottleneck_channels  # 1024

# Grid-search probes — ce_lovasz, AdamW/wd0/OneCycleLR warmup5%, lr sweep only (12 probes).
_criteria = [
    dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
    dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
]
_lrs = {
    "1e-4": 1e-4,
    "2e-4": 2e-4,
    "5e-4": 5e-4,
    "1e-3": 1e-3,
    "2e-3": 2e-3,
    "5e-3": 5e-3,
    "1e-2": 1e-2,
    "2e-2": 2e-2,
    "5e-2": 5e-2,
    "1e-1": 1e-1,
    "2e-1": 2e-1,
    "5e-1": 5e-1,
}

probes = {
    f"ce_lovasz_lr{lr_name}": dict(
        criteria=_criteria,
        input_norm=None,
        feat_norm=None,
        dropout=0.0,
        optimizer=dict(type="AdamW", lr=lr, weight_decay=0.0),
        scheduler=dict(
            type="OneCycleLR",
            max_lr=lr,
            pct_start=0.05,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=1000.0,
        ),
        grad_clip=3.0,
    )
    for lr_name, lr in _lrs.items()
}
del _criteria, _lrs

wandb_run_name = (
    f"PT-v3-malibu GridProbe DALES {grp_exp}.{num_exp}) decoder hypercolumn 1024ch, "
    f"{len(probes)} probes, epoch={epoch}"
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
        type="PT-v3-malibu",
        in_channels=7,  # coord(3) + color(3, fake) + strength(1)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(3, 3, 3, 3),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, bottleneck_channels),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(patch_size, patch_size, patch_size, patch_size, patch_size),
        dec_depths=(2, 2, 2, 2),
        dec_channels=dec_channels,
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(patch_size, patch_size, patch_size, patch_size),
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
        enc_mode=False,
        traceable=True,
    ),
    freeze_backbone=True,
    bn_eval_mode=True,  # no-op for PT-v3-malibu (LayerNorm only, no BatchNorm) — set explicitly anyway
    drop_path_eval_mode=True,  # keep DropPath(0.3) inactive during probe training
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
            dict(type="Z_MinShift"),
            dict(type="Z_RandomOffset"),
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
                feat_scales=dict(coord=coord_feat_scale, strength=strength_feat_scale),
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
            dict(type="FillMissingFeat", feat_key="color", feat_dim=3),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=feat_keys,
                feat_scales=dict(coord=coord_feat_scale, strength=strength_feat_scale),
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
                    feat_scales=dict(coord=coord_feat_scale, strength=strength_feat_scale),
                ),
            ],
            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]],
        ),
    ),
)
