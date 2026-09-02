"""
Local smoke-test variant of sonata-v1m2-malibu3d-lin-grid.py: D067 manifest
(the only region mirrored on local machine — see README_MALIBU3D.md "Local vs cluster data
availability"), tiny total_iters/batch/point budget, 3 probes with
deliberately different optimizer types/LRs (to visually confirm per-probe
optimizer heterogeneity is real — one SGD probe with an absurd LR should
diverge while the AdamW probes stay stable).

D067's manifest has no "test" split rows, so data.test below reuses "val"
purely so GridProbeWinnerSelector's automatic test pass has something to run
against locally — do not copy that substitution into a real experiment config.

Not meant to produce a good model — plumbing verification only.
"""

_base_ = ["../../_base_/default_runtime.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------
num_gpu = 1
batch_size_per_gpu = 2
batch_size = batch_size_per_gpu * num_gpu
batch_size_val = 1
batch_size_test = 1
num_worker = 4 * num_gpu
mix_prob = 0.8
empty_cache = False
enable_amp = True
evaluate = True

grid_size = 0.2
point_max = 20000

total_iters = 6
iter_per_epoch = 2
eval_every = 1

train_max_sample = 8
val_max_sample = 6

feat_keys = ["coord", "color", "strength"]

wandb_project = "malibu3d_sonata"
wandb_run_name = f"Sonata-v1m2 grid-probe TOY | bs={batch_size} | iters={total_iters}"
enable_wandb = False

# -----------------------------------------------------------------------------
# Labels (segment v20) — matches on-disk D067 tiles (meta.json label_definitions).
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
# cartesian_probes demo: loss type x learning rate (2 x 2 = 4 probes), plus one
# hand-added one-off probe with a deliberately absurd LR to prove per-probe
# optimizer heterogeneity (it should diverge/NaN-out independently of the
# cartesian_probes-generated ones).
from pointcept.utils.grid_probe_utils import cartesian_probes

_loss_axis = [
    dict(criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index)]),
    dict(criteria=[dict(type="FocalLoss", gamma=2.0, loss_weight=1.0, ignore_index=ignore_index)]),
]
_lr_axis = [
    dict(optimizer=dict(type="AdamW", lr=lr, weight_decay=0.02)) for lr in (0.002, 0.0002)
]
probes = cartesian_probes(
    dict(
        input_norm="l2",
        feat_norm="batchnorm",
        dropout=0.1,
        scheduler=dict(type="CosineAnnealingLR", eta_min=0.0),
        grad_clip=3.0,
    ),
    _loss_axis,
    _lr_axis,
)
del cartesian_probes, _loss_axis, _lr_axis  # avoid leaking a function object
# into the config namespace (Config.dump() -> yapf can't format it — see the
# production grid config for the full explanation).

probes["sgd_lr1e2_unstable"] = dict(
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index)],
    input_norm=None,
    feat_norm=None,
    dropout=0.0,
    optimizer=dict(type="SGD", lr=100.0, momentum=0.9, weight_decay=0.0),
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
        in_channels=7,
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
        # flash_attn isn't installed in this environment; the real config
        # keeps enable_flash=True (matches sonata-v1m2-malibu3d-lin.py).
        enable_flash=False,
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
# -----------------------------------------------------------------------------
dataset_type = "Malibu3DDataset"
data_root = "data/malibu3d"
csv_manifest = "data/malibu3d/raw/scene_split_manifest_D067.csv"
# D067's manifest lacks the n_points column min_points needs — skip it locally.
min_points = {}

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
        max_sample=train_max_sample,
        target_keys=["segment"],
        primary_target_key="segment",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Z_MinShift"),
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
        max_sample=val_max_sample,
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
    # D067 manifest has no "test" rows locally — reuse "val" purely so the
    # automatic winner test pass (GridProbeWinnerSelector) has data to run
    # against. Do not copy this into a real experiment config.
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        csv_manifest=csv_manifest,
        min_points=min_points,
        max_sample=val_max_sample,
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
hooks = [
    # No pretrained backbone locally (see module docstring) — this just
    # enables resume (cfg.weight=save_path/model/model_last.pth, cfg.resume=True).
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=1),
    dict(type="InformationWriter"),
    dict(type="GridProbeEvaluator", write_cls_iou=False),
    dict(type="GridProbeCheckpointSaver"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="MetricsJsonWriter"),
    dict(type="GridProbeWinnerSelector"),
]
