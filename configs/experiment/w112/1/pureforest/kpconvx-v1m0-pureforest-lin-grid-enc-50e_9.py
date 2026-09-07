"""
KPConvX GridProbeClassifier on PureForest — encoder multiscale
(enc_mode=True -> 96+128+192+256+256 = 928ch), scene mean-pool, N linear
heads. Frozen Malibu3D multitask ckpt. Strength zero-fill.

AdamW / wd=0 / OneCycleLR warmup 5%, lr sweep {1e-4 .. 5e-1} (12 probes),
CE only. select_metric=mIoU. 50-epoch schedule (eval every 5 trainer epochs).
"""

_base_ = ["../../../../_base_/default_runtime.py"]

grp_exp = 1
num_exp = 9

num_classes = 13
ignore_index = -1
grid_size = 0.1
point_max = 5000
kp_radius = 3.2
kp_sigma = kp_radius
radius_scaling = 3.0
coord_feat_scale = 0.01

num_gpu = 1
epoch = 50
eval_epoch = 5
lr = 5e-2

log_test_f1 = True

batch_size = 24 * num_gpu
batch_size_val = batch_size // 2
batch_size_test = batch_size // 2
num_worker = 24 * num_gpu
enable_amp = False

dataset_type = "PureForestDataset"
data_root = "data/pureforest"

weight = "ckpt/malibu3d/kpconvx_multitask/model_best.pth"

wandb_project = "pointcept_pureforest"

learned_masked_feat = True
feat_keys = ["coord", "color", "strength"]

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

enc_channels = (96, 128, 192, 256, 256)
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
    f"KPConvX GridProbeClassifier PureForest ({grp_exp}.{num_exp}) "
    f"enc multiscale {backbone_out_channels}ch, {len(probes)} probes, epoch={epoch}"
)

hooks = [
    dict(
        type="CheckpointLoader",
        exclude_keys=(
            "seg_heads",
            "reg_heads",
            "cls_heads",
            "pixel_seg_heads",
            "cls_attn_pools",
        ),
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
        type="kpconvx_base",
        input_channels=7,
        num_classes=0,
        dim=3,
        task="cloud_segmentation",
        kp_mode="kpconvx",
        shell_sizes=(1, 14, 28),
        kp_radius=kp_radius,
        kp_aggregation="nearest",
        kp_influence="constant",
        kp_sigma=kp_sigma,
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
        radius_scaling=radius_scaling,
        decoder_layer=True,
        grid_pool=True,
        upsample_n=3,
        first_inv_layer=1,
        drop_path_rate=0,
        norm="batch",
        bn_momentum=0.1,
        smooth_labels=False,
        class_w=(),
        enc_mode=True,
    ),
    freeze_backbone=True,
    bn_eval_mode=True,
    drop_path_eval_mode=True,
    feature_mask_values=dict(
        enable=learned_masked_feat,
        masked_feat_keys=["color", "strength"],
    ),
)

train = dict(type="GridProbeTrainer")

_fill_strength = dict(
    type="FillMissingFeat", feat_key="strength", feat_dim=1, fill_value=0.0
)

_val_test_transform = [
    dict(type="CenterShift", apply_z=True),
    dict(type="Z_MinShift"),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_min_coord=True,
    ),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    _fill_strength,
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "category"),
        feat_keys=feat_keys,
        feat_scales=dict(coord=coord_feat_scale),
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
                return_min_coord=True,
            ),
            dict(type="SphereCrop", point_max=point_max, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            _fill_strength,
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "category"),
                feat_keys=feat_keys,
                feat_scales=dict(coord=coord_feat_scale),
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
        data_root=data_root,
        class_names=class_names,
        transform=_val_test_transform,
        test_mode=False,
    ),
)
