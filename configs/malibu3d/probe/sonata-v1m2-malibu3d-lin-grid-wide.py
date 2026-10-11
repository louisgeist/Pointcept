"""
Wide linear-probe grid search on a frozen Sonata encoder for MALiBU3D land-cover segmentation
(`segment`, v20). Same backbone, pipeline and winner selection as sonata-v1m2-malibu3d-lin-grid.py,
with 336 probes trained on one shared backbone forward pass:
  4 losses (CE + Lovasz, CE, focal gamma=2, focal gamma=1)
  x 7 learning rates (1e-5 to 5e-2)
  x 3 weight decays (0, 1e-6, 1e-4)
  x 4 input normalizations (l2, linf, l1, none),
all with AdamW, cosine annealing, no feature norm, no dropout, grad clip 3.
10k iterations; validation on 100 tiles of the stratified validation subset; the winning
probe gets a full test pass. Pass the checkpoint with --options weight=<path>.
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
num_worker_test = 2  # more test workers run out of host RAM (dense test clouds)
mix_prob = 0.8
empty_cache = False
enable_amp = True
evaluate = True

grid_size = 0.1
point_max = 102400

total_iters = 10000
iter_per_epoch = 100
eval_every = 5

feat_keys = ["coord", "color", "strength"]

wandb_project = "malibu3d_sonata"
wandb_run_name = (
    f"Sonata-v1m2 grid-wide MALiBU3D segment | bs={batch_size} | "
    f"iters={total_iters}"
)

# -----------------------------------------------------------------------------
# Labels (segment v20: 15 classes + void), must match malibu3d_label_remap v20.
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
    "Infrastructure",
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
# Grid-search probes: loss x lr x weight decay x input_norm
# -----------------------------------------------------------------------------
_losses = {
    "ce_lovasz": [
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=ignore_index,
        ),
    ],
    "ce": [
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
    ],
    "focal_g2": [
        dict(
            type="FocalLoss",
            gamma=2.0,
            loss_weight=1.0,
            ignore_index=ignore_index,
        ),
    ],
    "focal_g1": [
        dict(
            type="FocalLoss",
            gamma=1.0,
            loss_weight=1.0,
            ignore_index=ignore_index,
        ),
    ],
}

_lrs = {
    "1e-5": 1e-5,
    "1e-4": 1e-4,
    "5e-4": 5e-4,
    "1e-3": 1e-3,
    "5e-3": 5e-3,
    "1e-2": 1e-2,
    "5e-2": 5e-2,
}

_wds = {
    "0": 0.0,
    "1e-4": 1e-4,
    "1e-6": 1e-6,
}

# input_norm projects the features onto the l2 / linf / l1 unit sphere before the head.
_norms = {
    "l2": "l2",
    "linf": "linf",
    "l1": "l1",
    "none": None,
}

probes = {}
for _loss_name, _criteria in _losses.items():
    for _lr_name, _lr in _lrs.items():
        for _wd_name, _wd in _wds.items():
            for _norm_name, _input_norm in _norms.items():
                _name = f"{_loss_name}_lr{_lr_name}_wd{_wd_name}_{_norm_name}"
                probes[_name] = dict(
                    criteria=_criteria,
                    input_norm=_input_norm,
                    feat_norm=None,
                    dropout=0.0,
                    optimizer=dict(type="AdamW", lr=_lr, weight_decay=_wd),
                    scheduler=dict(type="CosineAnnealingLR", eta_min=0.0),
                    grad_clip=3.0,
                )

del _losses, _lrs, _wds, _norms, _loss_name, _criteria, _lr_name, _lr
del _wd_name, _wd, _norm_name, _input_norm, _name

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
dataset_type = "MALiBU3DDataset"
data_root = "data/malibu3d"
csv_manifest = "data/malibu3d/tiles.csv"
min_points = {"train": 1000}
val_stratified_subset_manifest = "data/malibu3d/manifests/val_dev_subset_2000.csv"

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
        # More val tiles than the other probes (20) for a stabler ranking of 336 probes.
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
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    # Per-class IoU off: 336 probes x 15 classes per evaluation would flood the log.
    dict(type="GridProbeEvaluator", write_cls_iou=False),
    dict(type="GridProbeCheckpointSaver"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="MetricsJsonWriter"),
    dict(type="GridProbeWinnerSelector"),
]
