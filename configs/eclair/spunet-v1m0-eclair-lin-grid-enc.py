"""
SpUNet-v1m1 grid-search linear probing on ECLAIR — encoder multiscale
variant (same frozen checkpoint as spunet-v1m0-eclair-lin-grid-dec.py, job
1052217, Flair3D+ multitask supervised pretrain: channels=(32,64,128,256,256,
128,96,96), layers=(2,3,4,6,2,2,2,2), stride=3).

SpUNetBase never returned a `Point` with `pooling_parent`/`unpooling_parent`
links, and its pre-existing `enc_mode=True` is a *classification* mode
(scatter-mean-pools every voxel to one feature per whole scene — see
modelnet40/cls-spunet-v1m1-0-base.py) — unusable for a per-point segmentation
probe (shape mismatch against `segment`).

Uses the new `point_mode=True` backbone flag (spconv_unet_v1m1_base.py) that
instead builds a `Point` chain (stem -> stage0 -> ... -> bottleneck) with
`pooling_parent`/`pooling_inverse` populated the same way PT-v3's GridPooling
does, so GridProbeSegmentorV2's existing generic encoder-multiscale walk
concatenates every stage's per-point feature broadcast up to the finest
(stem) resolution — no scatter, no new trainable parameters (see
tests/test_spunet_point_mode.py). Levels concatenated (finest first):
stem(32) + stage0(32) + stage1(64) + stage2(128) + stage3/bottleneck(256) =
512ch. Counterpart to the sibling decoder config (96ch, standard
full-resolution U-Net output).

ECLAIR provides real RGB: ChromaticAutoContrast/Translation/Jitter (train) +
NormalizeColor (like H3D / semseg-litept ECLAIR); strength uses 1/60000 like
DALES. Same probe grid as litept-b-v1m0-eclair-lin_enc.py (ce_lovasz x 12 LRs
x wd=0 x dropout=0 x input_norm=None x AdamW x warmup=5%) for cross-backbone
comparability. epoch=200 / eval_epoch=10.

`bn_eval_mode=True` freezes SpUNet's BatchNorm running stats (real
BatchNorm1d); `drop_path_eval_mode=True` is a no-op (SpUNet has no DropPath
modules).
"""

_base_ = ["../_base_/default_runtime.py"]

grp_exp = 1
num_exp = 2

num_classes = 11
ignore_index = -1
grid_size = 0.1
point_max = 102400
coord_feat_scale = 0.01  # must match Flair3D multitask pretrain
strength_feat_scale = 1 / 60000  # raw uint16 intensity → Flair3D [0,1] convention

num_gpu = 1
epoch = 200
eval_epoch = 10
lr = 5e-2

test_single_fragment = True

batch_size_per_gpu = 24
batch_size = batch_size_per_gpu * num_gpu
batch_size_val = 1
batch_size_test = 1
num_worker = 24  # H100 Jean-Zay
num_worker_test = 2
mix_prob = 0.8
empty_cache = False
enable_amp = True

dataset_type = "ECLAIRDataset"
data_root = "data/eclair"

weight = "/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/1052217/model/model_best.pth"

wandb_project = f"pointcept_{dataset_type[:-7].lower()}"

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
    "Unassigned",
    "Ground",
    "Vegetation",
    "Buildings",
    "Noise",
    "Transmission Wires",
    "Distribution Wires",
    "Poles",
    "Transmission Towers",
    "Fence",
    "Vehicle",
]

# point_mode multiscale concat: stem(32) + stage0(32) + stage1(64) + stage2(128) + stage3/bottleneck(256)
backbone_channels = (32, 64, 128, 256, 256, 128, 96, 96)
backbone_out_channels = 32 + sum(backbone_channels[:4])  # 512

# -----------------------------------------------------------------------------
# Grid-search probes — AdamW / OneCycleLR: ce_lovasz x lr x wd=0 x dropout=0 x
# input_norm=none x feat_norm=none x optimizer=AdamW, warmup=5%
# (1 x 12 x 1 x 1 x 1 x 1 x 1 = 12 probes).
# -----------------------------------------------------------------------------
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

probes = {}
for _loss_name, _criteria in _losses.items():
    for _lr_name, _lr in _lrs.items():
        for _wd_name, _wd in _wds.items():
            for _do_name, _dropout in _dropouts.items():
                for _norm_name, _input_norm in _norms.items():
                    for _fn_name, _feat_norm in _feat_norms.items():
                        for _opt_name, _opt_type in _optimizers.items():
                            _name = (
                                f"{_loss_name}_lr{_lr_name}_wd{_wd_name}_do{_do_name}_"
                                f"{_norm_name}_fn{_fn_name}_{_opt_name}"
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
                                    pct_start=0.05,
                                    anneal_strategy="cos",
                                    div_factor=10.0,
                                    final_div_factor=1000.0,
                                ),
                                grad_clip=3.0,
                            )

del _losses, _lrs, _wds, _dropouts, _norms, _feat_norms, _optimizers
del _loss_name, _criteria, _lr_name, _lr, _wd_name, _wd, _do_name, _dropout
del _norm_name, _input_norm, _fn_name, _feat_norm, _opt_name, _opt_type
del _optimizer, _name


wandb_run_name = (
    f"SpUNet GridProbe ECLAIR {grp_exp}.{num_exp}) enc multiscale {backbone_out_channels}ch, "
    f"AdamW/wd0/OneCycleLR warmup5%, {len(probes)} probes, epoch={epoch}"
)

model = dict(
    type="GridProbeSegmentorV2",
    probes=probes,
    num_classes=num_classes,
    ignore_index=ignore_index,
    target_key="segment",
    backbone_out_channels=backbone_out_channels,
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=7,  # coord(3) + color(3) + strength(1)
        num_classes=0,  # unused in point_mode (no final conv applied) — kept for checkpoint key compat
        channels=backbone_channels,
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        stride=3,
        point_mode=True,
    ),
    freeze_backbone=True,
    bn_eval_mode=True,  # freeze BatchNorm running stats during probe training (real BN here)
    drop_path_eval_mode=True,  # no-op — SpUNet has no DropPath modules
    feature_mask_values=dict(
        enable=True,
        masked_feat_keys=["color", "strength"],
    ),
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
        include_pseudo=True,
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
        split="val",
        data_root=data_root,
        include_pseudo=True,
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
                feat_scales=dict(coord=coord_feat_scale, strength=strength_feat_scale),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        include_pseudo=True,
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
                    feat_scales=dict(coord=coord_feat_scale, strength=strength_feat_scale),
                ),
            ],
            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]],
        ),
    ),
)
