"""
Sonata-v1m2 grid-search linear probe on Flair3D+ segment (v20) —
Jean Zay experiment w112/6/sonata_flair3d_lin (_1), 1x H100, no test.

12 probes share one frozen PT-v3m2 encoder forward per batch
(GridProbeSegmentorV2 + GridProbeTrainer). Same lr recipe as the H3D/DALES
"comme d'hab" GridProbe (ce_lovasz x 12 AdamW lrs, wd=0, OneCycleLR warmup
5%, input_norm=none). Winner is selected by val mIoU
(GridProbeEvaluator.select_metric="mIoU"); GridProbeWinnerSelector(skip_test=True)
writes grid_search_results.json only — the 5-task seed ensemble is a separate
MultiTaskSegmentorV2 job, generated after this sweep.

Frozen checkpoint: pretrain job 862680 / epoch_120 (W_SONATA).
Launch: scripts/sonata/sbatch_flair3d_lin_grid_h100.sh

Val uses stratified 2k subset capped at max_sample=100. No test split.
No coord_feat_scale (Sonata pretrain does not use one). segment="v20".
"""

_base_ = ["../../../../_base_/default_runtime.py"]

# Sonata pretrain ckpt (job 862680, epoch 120) — remap via CheckpointLoader.
weight = "/lustre/fsn1/projects/rech/unv/usi32yh/logs/pointcept_logs/slurm/862680/model/epoch_120.pth"

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------
grp_exp = 1
num_exp = 1

num_gpu = 1
batch_size_per_gpu = 24
batch_size = batch_size_per_gpu * num_gpu
batch_size_val = 1
batch_size_test = max(1, batch_size // 2)
num_worker = 16  # H100 Jean-Zay
num_worker_test = 2  # packed grid-probe test loader OOMs above this
mix_prob = 0.8
empty_cache = False
enable_amp = True
evaluate = True

grid_size = 0.1
point_max = 102400

# 10000 steps / 1000 per epoch -> 10 trainer epochs
total_iters = 10000
iter_per_epoch = 1000
eval_every = 2

feat_keys = ["coord", "color", "strength"]

# Encoder levels (enc_mode): 48+96+192+384+512 = 1232
backbone_out_channels = 1232

# -----------------------------------------------------------------------------
# Labels (segment v20) — hardcoded to avoid importing pointcept.datasets at
# config-parse time (that package pulls torch_cluster via models).
# Must match flair3d_label_remap segment/v20 (finer12; void = ignore_index).
# -----------------------------------------------------------------------------
label_definitions = dict(segment="v20")
num_classes = 15
ignore_index = 15
names = [
    "Building",
    "Greenhouse",
    "Impervious surface",
    "Other soil",
    "Herbaceous",
    "Vineyard",
    "Brushwood",
    "Other infrastructures",
    "Swimming pool",
    "Water",
    "Deciduous",
    "Coniferous",
    "Bridge",
    "Agricultural soil",
    "Soil under vegetation",
    "Void",
]

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

wandb_project = "flair3d_sonata"
wandb_run_name = (
    f"Sonata-v1m2 grid-probe Flair3D+ segment {grp_exp}.{num_exp} H100 | "
    f"{len(probes)} probes | bs={batch_size} | iters={total_iters}"
)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = dict(
    type="GridProbeSegmentorV2",
    probes=probes,
    num_classes=num_classes,
    ignore_index=ignore_index,
    target_key="segment",
    backbone_out_channels=backbone_out_channels,
    backbone=dict(
        type="PT-v3m2",
        in_channels=7,  # coord(3) + color(3) + strength(1)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(3, 3, 3, 3),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
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

# -----------------------------------------------------------------------------
# Trainer (no test pass)
# -----------------------------------------------------------------------------
train = dict(type="GridProbeTrainer")

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset_type = "Flair3DDataset"
data_root = "data/flair3d_plus"
csv_manifest = "data/flair3d_plus/raw/scene_split_manifest.csv"
min_points = {"train": 1000}
val_stratified_subset_manifest = "data/flair3d_plus/manifests/val_dev_subset_2000.csv"

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
        csv_manifest=csv_manifest,
        min_points=min_points,
        target_keys=["segment"],
        primary_target_key="segment",
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
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=feat_keys,
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        csv_manifest=csv_manifest,
        min_points=min_points,
        stratified_subset_manifest=val_stratified_subset_manifest,
        max_sample=100,
        target_keys=["segment"],
        primary_target_key="segment",
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
            ),
        ],
        test_mode=False,
    ),
)

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
# Order matters: GridProbeEvaluator before GridProbeCheckpointSaver and
# CheckpointSaver. GridProbeWinnerSelector last; skip_test=True (no tester).
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="GridProbeEvaluator", write_cls_iou=True, select_metric="mIoU"),
    dict(type="GridProbeCheckpointSaver"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="MetricsJsonWriter"),
    dict(type="GridProbeWinnerSelector", skip_test=True),
]
