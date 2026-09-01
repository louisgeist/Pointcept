"""
Sonata-v1m2 grid-search linear probing on Malibu3D+ semantic segmentation (v20).

Trains N independently-configured linear probe heads (see `probes` below) on
top of ONE shared frozen-PT-v3m2-encoder forward pass per batch, instead of
launching N separate jobs each redoing that forward pass — see
pointcept/models/grid_probe.py and pointcept/engines/train.py::GridProbeTrainer
for the mechanism. At the end of training, the probe with the best val mIoU is
selected automatically, reloaded from its own best checkpoint, tested via
SemSegTester, and the winning config + metrics are written to
save_path/grid_search_results.json (GridProbeWinnerSelector hook).

Adapted from sonata-v1m2-malibu3d-lin.py (single-probe version): same frozen
backbone / label definitions / train+val pipeline, plus a `data.test` split
(needed for the automatic final test pass, which the single-probe config
doesn't have) and `probes` in place of the single top-level `criteria`.
"""

_base_ = ["../../_base_/default_runtime.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------
num_gpu = 1
batch_size_per_gpu = 24
batch_size = batch_size_per_gpu * num_gpu
batch_size_val = 1
batch_size_test = max(1, batch_size // 2)
num_worker = 8 * num_gpu
num_worker_test = 2  # packed grid-probe test loader OOMs above this locally/on JZ
mix_prob = 0.8
empty_cache = False
enable_amp = True
evaluate = True

grid_size = 0.1
point_max = 102400

# Short probe: 1000 steps / 100 per epoch -> 10 trainer epochs
total_iters = 1000
iter_per_epoch = 100
eval_every = 2

feat_keys = ["coord", "color", "strength"]

wandb_project = "malibu3d_sonata"
wandb_run_name = (
    f"Sonata-v1m2 grid-probe Malibu3D+ segment | bs={batch_size} | "
    f"iters={total_iters}"
)

# -----------------------------------------------------------------------------
# Labels (segment v20) — hardcoded to avoid importing pointcept.datasets at
# config-parse time (that package pulls torch_cluster via models).
# Must match malibu3d_label_remap segment/v20 (finer12; void = ignore_index).
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
# Grid-search probes
# -----------------------------------------------------------------------------
# Each probe is one grid point: its own loss(es), input/feature normalization,
# dropout, optimizer, scheduler, grad_clip — fully independent from the other
# probes (see pointcept/engines/train.py::GridProbeTrainer for how optimizer/
# scheduler heterogeneity is handled).
#
# cartesian_probes crosses N axes of partial probe-config dicts (deep-merged
# onto a shared `base`) — here: loss type x learning rate, since lr is
# probably the single most important axis to sweep for every lin-probe
# experiment. No built-in auto-cartesian-expansion beyond what you ask for —
# a full cross-product over many axes explodes combinatorially, so cross only
# what you actually want swept and fix the rest in `base`.
from pointcept.utils.grid_probe_utils import cartesian_probes

_loss_axis = [
    dict(criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index)]),
    dict(criteria=[dict(type="FocalLoss", gamma=2.0, loss_weight=1.0, ignore_index=ignore_index)]),
]
_lr_axis = [
    dict(optimizer=dict(type="AdamW", lr=lr, weight_decay=0.02)) for lr in (1e-4, 5e-4, 2e-3)
]
probes = cartesian_probes(
    dict(
        input_norm=None,
        feat_norm=None,
        dropout=0.0,
        scheduler=dict(type="CosineAnnealingLR", eta_min=0.0),
        grad_clip=3.0,
    ),
    _loss_axis,
    _lr_axis,
)
del cartesian_probes, _loss_axis, _lr_axis  # avoid leaking a function object into
# the config namespace — Config.dump() pipes every non-dunder module attribute
# through yapf, which can't format a "<function ... at 0x...>" repr (same
# reason ptv3-v1m0-malibu3d.py deletes its imported task-config helpers).

# Composes freely with hand-curated one-off probes, e.g. combined losses or a
# different optimizer/scheduler type entirely:
probes["ce_lovasz_adamw_lr1e-3_ln"] = dict(
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=0.5, ignore_index=ignore_index),
    ],
    input_norm=None,
    feat_norm="layernorm",
    dropout=0.2,
    optimizer=dict(type="AdamW", lr=0.001, weight_decay=0.05),
    scheduler=dict(type="OneCycleLR", max_lr=0.001, pct_start=0.1),
    grad_clip=3.0,
)
probes["focal_g2_sgd_lr1e-2"] = dict(
    criteria=[dict(type="FocalLoss", gamma=2.0, loss_weight=1.0, ignore_index=ignore_index)],
    input_norm="l2",
    feat_norm="batchnorm",
    dropout=0.1,
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=1e-4),
    scheduler=dict(type="MultiStepLR", milestones=[0.6, 0.9], gamma=0.1),
    grad_clip=None,
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
# Trainer / tester
# -----------------------------------------------------------------------------
train = dict(type="GridProbeTrainer")
test_single_fragment = True
test = dict(type="SemSegTester", verbose=True)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset_type = "Malibu3DDataset"
data_root = "data/malibu3d_plus"
csv_manifest = "data/malibu3d_plus/raw/scene_split_manifest.csv"
min_points = {"train": 1000}
val_stratified_subset_manifest = "data/malibu3d_plus/manifests/val_dev_subset_2000.csv"

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=names,
    # Lets InformationWriter's existing per-task train-mIoU accumulation
    # (reads cfg.data.task_configs) pick up every probe for free — see
    # GridProbeSegmentorV2.forward()'s input_dict[probe_name] aliasing.
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
        max_sample=20,
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
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        csv_manifest=csv_manifest,
        min_points=min_points,
        target_keys=["segment"],
        primary_target_key="segment",
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
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ],
        ),
    ),
)

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
# Order matters: GridProbeEvaluator must run before GridProbeCheckpointSaver
# (reads this epoch's per-probe mIoU) and before CheckpointSaver (reads
# comm_info["current_metric_value"]). GridProbeWinnerSelector must be last
# (after_train-only; frees optimizers/schedulers before the final test pass).
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
    dict(type="GridProbeWinnerSelector"),
]
