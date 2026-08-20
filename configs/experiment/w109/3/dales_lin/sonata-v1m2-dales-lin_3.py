"""
Sonata-v1m2 linear probing on DALES (transfer from Flair3D+ SSL pretrain).
Epoch=100 variant of `sonata-v1m2-dales-lin_1.py`.

Frozen PT-v3m2 encoder (enc_mode) from the Flair3D+ Sonata pretrain job
862680, epoch_120. DALES has no RGB — Sonata was pretrained with scene-level
RandomDropColor/RandomDropStrength (drop_value=0.0) specifically so the
encoder tolerates all-zero color at transfer time, so `FillMissingFeat`
synthesizes a zero "color" channel here to keep in_channels=7. No learned
masked-feat mechanism was used at pretrain time (feature_mask_values unset),
so a literal zero fill is faithful to what the encoder saw.
"""

_base_ = ["../../../../_base_/default_runtime.py"]

grp_exp = 1
num_exp = 3

num_classes = 8
ignore_index = 8
grid_size = 0.1
point_max = 102400

num_gpu = 1
epoch = 100
eval_epoch = 10
lr = 2e-2
patch_size = 1024

test_single_fragment = True

# misc custom setting
batch_size_per_gpu = 24
batch_size = batch_size_per_gpu * num_gpu
batch_size_val = 1
batch_size_test = 1  # DALES test tiles are unchunked full scenes (~12M raw pts each);
# batch_size // 2 packs multiple into one forward pass and crashes spconv (CUBLAS/illegal
# memory access in the stem indice_conv). 1 mirrors batch_size_val, which already works.
num_worker = 8 * num_gpu
num_worker_test = 2
mix_prob = 0.8
empty_cache = False
enable_amp = False

# dataset settings
dataset_type = "DALESDataset"
data_root = "data/dales"

weight = "/lustre/fsn1/projects/rech/unv/usi32yh/logs/pointcept_logs/slurm/862680/model/epoch_120.pth"

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
    dict(type="GridProbeWinnerSelector", skip_test=False),
]

feat_keys = ["coord", "color", "strength"]

names = [
    "Ground",
    "Vegetation",
    "Cars",
    "Trucks",
    "Power lines",
    "Fences",
    "Poles",
    "Buildings",
    "Unknown",
]

# -----------------------------------------------------------------------------
# Grid-search probes — loss x lr x wd x input_norm (3 x 6 x 1 x 3 = 54 probes),
# each with its own AdamW + OneCycleLR (matches this config's original single-probe
# schedule shape) and grad_clip=3.0 (only meaningful under GridProbeTrainer, which
# has no top-level grad-clip equivalent to the classic Trainer's — added for
# stability across the sweep, following the Flair3D grid-probe configs' pattern).
# -----------------------------------------------------------------------------
_losses = {
    "ce": [dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index)],
    "ce_lovasz": [
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
    ],
    "focal_g2": [
        dict(type="FocalLoss", gamma=2.0, loss_weight=1.0, ignore_index=ignore_index),
    ],
}
_lrs = {"2e-3": 2e-3, "5e-3": 5e-3, "1e-2": 1e-2, "2e-2": 2e-2, "5e-2": 5e-2, "1e-1": 1e-1}
_wds = {"5e-3": 0.005}
_norms = {"none": None, "l2": "l2", "linf": "linf"}

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
del _losses, _lrs, _wds, _norms, _loss_name, _criteria, _lr_name, _lr
del _wd_name, _wd, _norm_name, _input_norm, _name

wandb_run_name = (
    f"Sonata GridProbe DALES {grp_exp}.{num_exp}) epoch_120, "
    f"{len(probes)} probes, epoch={epoch}"
)

# model settings
model = dict(
    type="GridProbeSegmentorV2",
    probes=probes,
    num_classes=num_classes,
    ignore_index=ignore_index,
    target_key="segment",
    backbone_out_channels=1232,
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
            dict(type="CenterShift", apply_z=True),
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
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
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
                ),
            ],
            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]],
        ),
    ),
)
