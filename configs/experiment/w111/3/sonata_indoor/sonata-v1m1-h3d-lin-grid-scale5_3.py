"""
Sonata-v1m1 grid-search linear probing on H3D — w111/3/sonata_indoor (coord scale 1/5) — official indoor release baseline
for suppmat (cross-domain aerial).

Frozen PT-v3m2 encoder (enc_mode=True → multi-scale concat 1232ch) from the
Meta/HuggingFace Sonata pretrain (pretrain-sonata-v1m1-0-base.pth). Indoor
backbone: in_channels=9 (coord+color+normal), stride=(2,2,2,2). H3D provides
RGB (NormalizeColor); normals are zero-filled via FillMissingFeat. Scene coords
are rescaled by 1/5 before GridSample; grid_size=0.02 (native indoor Sonata pretrain).

AdamW / wd=0 / OneCycleLR warmup 5%, lr swept over 12 values. select_metric=
macro_f1. epoch=2000 / eval_epoch=10. Chain into seed ensemble via
tools/grid_then_seeds.py.
"""

_base_ = ["../../../../_base_/default_runtime.py"]

grp_exp = 3
num_exp = 3

num_classes = 11
ignore_index = 11
grid_size = 0.02
coord_scale = 1 / 5
point_max = 102400

num_gpu = 1
epoch = 2000
eval_epoch = 10
lr = 5e-2
patch_size = 1024

test_single_fragment = True
log_test_f1 = True

batch_size = 24
batch_size_val = 1
batch_size_test = 1
num_worker = 24 * num_gpu
num_worker_test = 2
mix_prob = 0.8
empty_cache = False
enable_amp = True

dataset_type = "H3DDataset"
data_root = "data/h3d"

weight = "ckpt/sonata/pretrain-sonata-v1m1-0-base.pth"

wandb_project = f"pointcept_{dataset_type[:-7].lower()}"

hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=1),
    dict(type="GridProbeEvaluator", write_cls_iou=True, select_metric="macro_f1"),
    dict(type="GridProbeCheckpointSaver"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="GridProbeWinnerSelector", skip_test=False),
]

feat_keys = ["coord", "color", "normal"]

names = [
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

backbone_out_channels = 1232

_losses = {
    "ce_lovasz": [
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
    ],
}
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
_wds = {"0": 0.0}
_dropouts = {"0": 0.0}
_norms = {"none": None}
_feat_norms = {"none": None}
_optimizers = {"adamw": "AdamW"}
_warmups = {"w05": 0.05}

probes = {}
for _loss_name, _criteria in _losses.items():
    for _lr_name, _lr in _lrs.items():
        for _wd_name, _wd in _wds.items():
            for _do_name, _dropout in _dropouts.items():
                for _norm_name, _input_norm in _norms.items():
                    for _fn_name, _feat_norm in _feat_norms.items():
                        for _opt_name, _opt_type in _optimizers.items():
                            for _wu_name, _pct_start in _warmups.items():
                                _name = (
                                    f"{_loss_name}_lr{_lr_name}_wd{_wd_name}_do{_do_name}_"
                                    f"{_norm_name}_fn{_fn_name}_{_opt_name}_{_wu_name}"
                                )
                                _optimizer = dict(type=_opt_type, lr=_lr, weight_decay=_wd)
                                if _opt_type == "SGD":
                                    _optimizer["momentum"] = 0.9
                                probes[_name] = dict(
                                    criteria=_criteria,
                                    input_norm=_input_norm,
                                    feat_norm=_feat_norm,
                                    dropout=_dropout,
                                    optimizer=_optimizer,
                                    scheduler=dict(
                                        type="OneCycleLR",
                                        max_lr=_lr,
                                        pct_start=_pct_start,
                                        anneal_strategy="cos",
                                        div_factor=10.0,
                                        final_div_factor=1000.0,
                                    ),
                                    grad_clip=3.0,
                                )

del _losses, _lrs, _wds, _dropouts, _norms, _feat_norms, _optimizers, _warmups
del _loss_name, _criteria, _lr_name, _lr, _wd_name, _wd, _do_name, _dropout
del _norm_name, _input_norm, _fn_name, _feat_norm, _opt_name, _opt_type
del _wu_name, _pct_start, _optimizer, _name

wandb_run_name = (
    f"Sonata-v1m1 indoor w111.3 GridProbe H3D {grp_exp}.{num_exp}) HF pretrain, "
    f"enc {backbone_out_channels}ch, coord/5, {len(probes)} probes, epoch={epoch}"
)

model = dict(
    type="GridProbeSegmentorV2",
    probes=probes,
    num_classes=num_classes,
    ignore_index=ignore_index,
    target_key="segment",
    backbone_out_channels=backbone_out_channels,
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,  # coord(3) + color(3) + normal(3, zero)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
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
            dict(type="FillMissingFeat", feat_key="normal", feat_dim=3),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "grid_size"),
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
            dict(type="Z_MinShift"),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="FixedScaleCoord", scale=coord_scale),
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
            dict(type="FillMissingFeat", feat_key="normal", feat_dim=3),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
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
            dict(type="Z_MinShift"),
            dict(type="FixedScaleCoord", scale=coord_scale),
            dict(type="NormalizeColor"),
            dict(type="FillMissingFeat", feat_key="normal", feat_dim=3),
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
                ),
            ],
            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]],
        ),
    ),
)
