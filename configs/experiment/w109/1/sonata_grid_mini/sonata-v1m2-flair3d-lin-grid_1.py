"""
Sonata-v1m2 mini grid-search linear probe on Flair3D+ segment (v20) —
Jean Zay experiment w109/1/sonata_grid_mini (_1), 1× H100, no test.

11 probes share one frozen PT-v3m2 forward per batch (GridProbeSegmentorV2 +
GridProbeTrainer). 8 CE probes: input_norm {linf, none} × lr {1e-3, 2e-3} ×
wd {0, 1e-3}. Plus 3 CE l2 probes at lr {1e-2, 2e-2, 5e-2} with wd=0.
Fixed: feat_norm=None, dropout=0, AdamW, OneCycleLR, grad_clip=3.0.

Launch the epoch 10…150 sweep with scripts/sonata/sbatch_lin_grid_probe_mini_h100.sh
(weight override via -w). Default checkpoint is pretrain job 862680 / epoch_150.

Val uses stratified 2k subset capped at max_sample=100. No test split:
GridProbeWinnerSelector(skip_test=True) writes grid_search_results.json only.
"""

_base_ = ["../../../../_base_/default_runtime.py"]

# Sonata pretrain ckpt (job 862680, epoch 150) — remap via CheckpointLoader.
# Sweep launcher overrides this with -w epoch_{10,20,...,150}.pth.
weight = "/lustre/fsn1/projects/rech/unv/usi32yh/logs/pointcept_logs/slurm/862680/model/epoch_150.pth"

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------
grp_exp = 1
num_exp = 1

num_gpu = 1
batch_size_per_gpu = 24
batch_size = batch_size_per_gpu * num_gpu
batch_size_val = 12 * num_gpu
num_worker = 16  # H100 Jean-Zay
mix_prob = 0.8
empty_cache = False
enable_amp = True
evaluate = True

grid_size = 0.1
point_max = 102400

# 10000 steps / 1000 per epoch → 10 trainer epochs
total_iters = 10000
iter_per_epoch = 1000
eval_every = 2

feat_keys = ["coord", "color", "strength"]

wandb_project = "flair3d_sonata"
wandb_run_name = (
    f"Sonata-v1m2 grid-mini FT {grp_exp}.{num_exp} H100 | 11 probes | "
    f"bs={batch_size} | iters={total_iters}"
)

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
# Grid-search probes — 8 CE (linf/none × 2 lr × 2 wd) + 3 CE l2 high-lr
# -----------------------------------------------------------------------------
_ce = [
    dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
]

_lrs = {
    "1e-3": 1e-3,
    "2e-3": 2e-3,
}
_wds = {
    "0": 0.0,
    "1e-3": 1e-3,
}
_norms = {
    "linf": "linf",
    "none": None,
}

probes = {}
for _lr_name, _lr in _lrs.items():
    for _wd_name, _wd in _wds.items():
        for _norm_name, _input_norm in _norms.items():
            _name = f"ce_lr{_lr_name}_wd{_wd_name}_{_norm_name}"
            probes[_name] = dict(
                criteria=_ce,
                input_norm=_input_norm,
                feat_norm=None,
                dropout=0.0,
                optimizer=dict(type="AdamW", lr=_lr, weight_decay=_wd),
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

_l2_lrs = {
    "1e-2": 1e-2,
    "2e-2": 2e-2,
    "5e-2": 5e-2,
}
for _lr_name, _lr in _l2_lrs.items():
    _name = f"ce_lr{_lr_name}_wd0_l2"
    probes[_name] = dict(
        criteria=_ce,
        input_norm="l2",
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

del _ce, _lrs, _wds, _norms, _l2_lrs
del _lr_name, _lr, _wd_name, _wd, _norm_name, _input_norm, _name

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = dict(
    type="GridProbeSegmentorV2",
    probes=probes,
    num_classes=num_classes,
    ignore_index=ignore_index,
    target_key="segment",
    backbone_out_channels=1232,
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
    dict(type="GridProbeEvaluator", write_cls_iou=True),
    dict(type="GridProbeCheckpointSaver"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="MetricsJsonWriter"),
    dict(type="GridProbeWinnerSelector", skip_test=True),
]
