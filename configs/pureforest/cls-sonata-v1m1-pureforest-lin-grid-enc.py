"""
Sonata-v1m1 (PT-v3m2) GridProbeClassifier on PureForest — official Meta/HuggingFace
indoor SSL release (pretrain-sonata-v1m1-0-base.pth), not the Flair3D-fork outdoor
checkpoint used by cls-sonata-v1m2-pureforest-lin-grid-enc.py.

Indoor backbone: in_channels=9 (coord+color+normal), stride=(2,2,2,2),
grid_size=0.02 (native indoor pretrain). PureForest has real color (unlike DALES) so
only "normal" is zero-filled via FillMissingFeat; color keeps NormalizeColor as in
the outdoor config. Scene coords are rescaled by coord_scale=1/10 before GridSample
(indoor Sonata was pretrained on room-scale scenes; see README_grid_then_seed.md's
"Sonata-v1m1 indoor" section — coord_scale is a real ablation axis there, N ∈
{1,5,10,25,50}; 1/10 picked here, do not reuse a probe lr winner from another scale).
Z_MinShift/Z_RandomOffset dropped (not used by the DALES/H3D/ECLAIR indoor configs
either).

Encoder multiscale (enc_mode=True -> 48+96+192+384+512 = 1232ch), scene mean-pool,
N linear heads. AdamW / wd=0 / OneCycleLR warmup 5%, lr sweep {1e-4 .. 5e-1}
(12 probes), CE only. select_metric=mIoU.
"""

_base_ = ["../_base_/default_runtime.py"]

grp_exp = 1
num_exp = 1

num_classes = 13
ignore_index = -1
grid_size = 0.02  # native indoor Sonata pretrain
coord_scale = 1 / 10  # aerial coord rescale to indoor-pretrain scale
point_max = 5000
patch_size = 1024

num_gpu = 1
epoch = 100
eval_epoch = 10
lr = 5e-2

log_test_f1 = True

batch_size = 24 * num_gpu
batch_size_val = batch_size // 2
batch_size_test = batch_size // 2
num_worker = 24 * num_gpu
enable_amp = False

dataset_type = "PureForestDataset"
data_root = "data/pureforest"

weight = "ckpt/sonata/pretrain-sonata-v1m1-0-base.pth"

wandb_project = "pointcept_pureforest"

feat_keys = ["coord", "color", "normal"]

class_names = [
    "deciduous_oak",
    "evergreen_oak",
    "beech",
    "chestnut",
    "black_locust",
    "maritime_pine",
    "scotch_pine",
    "black_pine",
    "aleppo_pine",
    "fir",
    "spruce",
    "larch",
    "douglas",
]

enc_channels = (48, 96, 192, 384, 512)
backbone_out_channels = sum(enc_channels)

_criteria = [
    dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
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

probes = {}
for _lr_name, _lr in _lrs.items():
    probes[f"ce_lr{_lr_name}_wd0_adamw_w05"] = dict(
        criteria=_criteria,
        input_norm=None,
        feat_norm=None,
        dropout=0.0,
        optimizer=dict(type="AdamW", lr=_lr, weight_decay=0.0),
        scheduler=dict(
            type="OneCycleLR",
            max_lr=_lr,
            pct_start=0.05,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=1000.0,
        ),
        grad_clip=3.0,
    )
del _criteria, _lrs, _lr_name, _lr

wandb_run_name = (
    f"Sonata-v1m1 indoor GridProbeClassifier PureForest ({grp_exp}.{num_exp}) "
    f"HF pretrain, enc multiscale {backbone_out_channels}ch, coord/10, "
    f"{len(probes)} probes, epoch={epoch}"
)

hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=100),
    dict(type="GridProbeEvaluator", write_cls_iou=True, select_metric="mIoU"),
    dict(type="GridProbeCheckpointSaver"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="GridProbeWinnerSelector", skip_test=False),
]

test = dict(type="ClsTester")

model = dict(
    type="GridProbeClassifier",
    probes=probes,
    num_classes=num_classes,
    ignore_index=ignore_index,
    target_key="category",
    pooling="mean",
    backbone_out_channels=backbone_out_channels,
    channel_blocks=enc_channels,
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,  # coord(3) + color(3, real) + normal(3, zero)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=enc_channels,
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
    bn_eval_mode=True,
    drop_path_eval_mode=True,
)

train = dict(type="GridProbeTrainer")

_fill_normal = dict(type="FillMissingFeat", feat_key="normal", feat_dim=3)

_val_test_transform = [
    dict(type="CenterShift", apply_z=True),
    dict(type="FixedScaleCoord", scale=coord_scale),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    _fill_normal,
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": grid_size}),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "grid_size", "category"),
        feat_keys=feat_keys,
        optional_keys=("name",),
    ),
]

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,
    task_configs={
        name: dict(
            task_type="classification",
            num_classes=num_classes,
            ignore_index=ignore_index,
            names=class_names,
        )
        for name in probes
    },
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        class_names=class_names,
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
            dict(type="FixedScaleCoord", scale=coord_scale),
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
            _fill_normal,
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "grid_size", "category"),
                feat_keys=feat_keys,
                optional_keys=("name",),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        class_names=class_names,
        transform=_val_test_transform,
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        exclude_flair3d_leakage_tiles=True,
        data_root=data_root,
        class_names=class_names,
        transform=_val_test_transform,
        test_mode=False,
    ),
)
