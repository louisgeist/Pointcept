"""
Sonata-ECLAIR lin probe (w116/4, Tab. 4 "eclairSonata"): frozen Sonata-v1m2 backbone
pretrained on ECLAIR only (Jean Zay job 48309, configs/eclair/pretrain-
sonata-v1m2-eclair.py), checkpoint epoch_60 = best ECLAIR val lin-probe mIoU (70.30) over
epochs 10..150.

Copy of `configs/experiment/w116/1/opengf_lin_nomix/sonata-v1m2-opengf-lin-grid-nomix_2.py`,
the OpenGF no-Mix3D Sonata reference (job 612602, test T2 mIoU 90.59; Mix3D hurts Sonata
probes on OpenGF), 12-LR grid, mix_prob=0, epoch=50.

Only `weight`, `grp_exp`/`num_exp`, `_base_` path, H100 `num_worker` and the wandb name
change otherwise. Same archi/features/grid_size as the Flair3D+ Sonata pretrain
(coord+color+strength, ECLAIR intensity scaled by 1/60000 -> [0,1]), so the downstream
feature handling (strength_feat_scale / FillMissingFeat) is kept as is. Local copy on
hecate: ckpt/48309/epoch_60.pth.
"""

_base_ = ["../../../../_base_/default_runtime.py"]

grp_exp = 1
num_exp = 3

num_classes = 2
ignore_index = 2  # unreachable after RemapSegment merges raw label 2 into 1
grid_size = 0.1
point_max = 102400
strength_feat_scale = 1 / 60000  # OpenGF raw intensity → Flair3D [0,1] convention (DALES-like range)

num_gpu = 1
epoch = 50  # ~3900 iters (1359 train tiles // 24) -- roughly DALES's ballpark, see decision context
eval_epoch = 10
lr = 5e-2
patch_size = 1024

test_single_fragment = True

# misc custom setting
batch_size_per_gpu = 24
batch_size = batch_size_per_gpu * num_gpu
batch_size_val = 1
batch_size_test = 1
num_worker = 16  # H100 Jean-Zay
num_worker_test = 2
mix_prob = 0  # Mix3D off (was 0.8) -- see docstring
empty_cache = False
enable_amp = True

# dataset settings
dataset_type = "OpenGFDataset"
data_root = "data/opengf"

weight = "/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/ckpt/48309/epoch_60.pth"

wandb_project = f"pointcept_{dataset_type[:-7].lower()}"

# Hooks
# Order matters: GridProbeEvaluator before GridProbeCheckpointSaver/CheckpointSaver;
# GridProbeWinnerSelector last (frees the per-probe optimizers/schedulers, then runs
# its own SemSegTester pass on the winning probe — replaces PreciseEvaluator, which
# never sets raw_model.active_probe and would break under GridProbeSegmentorV2).
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
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
    "Non-ground",
]

# Encoder levels (enc_mode): 48+96+192+384+512 = 1232
backbone_out_channels = 1232

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
    f"Sonata-v1m2 GridProbe OpenGF {grp_exp}.{num_exp}) H100 noMix3D, Sonata-ECLAIR 48309 epoch_60, enc multiscale "
    f"{backbone_out_channels}ch, {len(probes)} probes, epoch={epoch}"
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
        type="PT-v3m2",
        in_channels=7,  # coord(3) + color(3, fake/zero) + strength(1)
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
            dict(type="RemapSegment", mapping={2: 1}),  # Qin et al. CVPRW 2021 §4.4: merge outliers into NG (they participate in training)
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
                feat_scales=dict(strength=strength_feat_scale),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="RemapSegment", mapping={2: 1}),  # Qin et al. CVPRW 2021 §4.4: merge outliers into NG (they participate in training)
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
                feat_scales=dict(strength=strength_feat_scale),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        include_names="T2",  # Test II specifically (T1/T3 have no outlier points)
        transform=[
            dict(type="DropSegmentClass", labels=[2]),  # Qin et al. CVPRW 2021 §4.5: Test II w/o outliers (physically deleted, not merged)
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
                    feat_scales=dict(strength=strength_feat_scale),
                ),
            ],
            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]],
        ),
    ),
)
