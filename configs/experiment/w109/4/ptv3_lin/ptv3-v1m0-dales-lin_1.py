"""
PT-v3-malibu grid-search linear probing on DALES — transfer from the
Flair3D+ multitask supervised pretrain at w109/2/ptv3_wd/multi-ptv3-v1m0-flair3d_5.py
(job 1052200, /lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/1052200/model/model_best.pth).
Mirrors w109/3/dales_lin/litept-b-v1m0-dales-lin_2.py's grid but for
PT-v3-malibu, and both `bn_eval_mode`/`drop_path_eval_mode` are set
explicitly (see pointcept/models/grid_probe.py) rather than left at their
default — even though PT-v3-malibu has no BatchNorm submodule at all (only
`nn.LayerNorm`, confirmed by grep across point_transformer_v3_malibu.py: no
`pdnorm_bn`/BatchNorm1d anywhere in that class, unlike PT-v3m1 or LitePT-v1),
so `bn_eval_mode` is a documented no-op here, kept for consistency with the
other grid-probe configs in this debug series (litept-b-v1m0-dales-lin_3/_4.py
in w109/4/dales_debug) rather than relying on the class default.

`drop_path_eval_mode=True` *does* matter here: PT-v3-malibu blocks use
`DropPath(drop_path)` same as LitePT-v1 (drop_path=0.3), so pinning it to
eval() keeps stochastic depth off during probe training — the actual grid
search should isolate hyperparameter effects, not entangle them with
backbone-side stochasticity (see w109/4/dales_debug discussion for why this
matters and how to ablate it if needed).

backbone_out_channels=64 matches the pretrain's own task heads: PT-v3-malibu
is enc_mode=False (decoder on) with no hypercolumn/traceable trick enabled,
so the backbone forward returns just the finest decoder stage's feature
(dec_channels[0]=64) — the same input every task head in the multitask
pretrain read from.

DALES has no RGB (same handling as the LitePT-B DALES configs):
`FillMissingFeat` synthesizes a zero "color" channel + `color_mask`, and
`feature_mask_values` makes the frozen model fill masked positions with its
pretrained learned `color_mask_value`/`strength_mask_value` instead of a raw
zero.
"""

_base_ = ["../../../../_base_/default_runtime.py"]

grp_exp = 1
num_exp = 1

num_classes = 8
ignore_index = 8
grid_size = 0.1
point_max = 102400
coord_feat_scale = 0.01  # must match Flair3D multitask pretrain
strength_feat_scale = 1 / 60000  # DALES raw intensity → Flair3D [0,1] convention

num_gpu = 1
epoch = 10
eval_epoch = 10
lr = 2e-2
patch_size = 1024

test_single_fragment = True

# misc custom setting
batch_size_per_gpu = 24
batch_size = batch_size_per_gpu * num_gpu
batch_size_val = 1
batch_size_test = 1
num_worker = 8 * num_gpu
num_worker_test = 2
mix_prob = 0.8
empty_cache = False
enable_amp = False

# dataset settings
dataset_type = "DALESDataset"
data_root = "data/dales"

weight = "/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/1052200/model/model_best.pth"

wandb_project = f"pointcept_{dataset_type[:-7].lower()}"

# Hooks
# Order matters: GridProbeEvaluator before GridProbeCheckpointSaver/CheckpointSaver;
# GridProbeWinnerSelector last (frees the per-probe optimizers/schedulers, then runs
# its own SemSegTester pass on the winning probe — replaces PreciseEvaluator, which
# never sets raw_model.active_probe and would break under GridProbeSegmentorV2).
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

backbone_out_channels = 64  # final decoder stage (dec_channels[0]), no hypercolumn trick

# -----------------------------------------------------------------------------
# Grid-search probes — loss x lr x wd x input_norm (3 x 6 x 1 x 3 = 54 probes),
# each with its own AdamW + OneCycleLR and grad_clip=3.0. Same grid shape as
# w109/3/dales_lin/litept-b-v1m0-dales-lin_2.py.
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
    f"PT-v3-malibu GridProbe DALES {grp_exp}.{num_exp}) final decoder 64ch, "
    f"{len(probes)} probes, epoch={epoch}"
)

# model settings
model = dict(
    type="GridProbeSegmentorV2",
    probes=probes,
    num_classes=num_classes,
    ignore_index=ignore_index,
    target_key="segment",
    backbone_out_channels=backbone_out_channels,
    backbone=dict(
        type="PT-v3-malibu",
        in_channels=7,  # coord(3) + color(3, fake) + strength(1)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(3, 3, 3, 3),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(patch_size, patch_size, patch_size, patch_size, patch_size),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(patch_size, patch_size, patch_size, patch_size),
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
        enc_mode=False,
    ),
    freeze_backbone=True,
    bn_eval_mode=True,  # no-op for PT-v3-malibu (LayerNorm only, no BatchNorm) — set explicitly anyway
    drop_path_eval_mode=True,  # keep DropPath(0.3) inactive during probe training
    feature_mask_values=dict(
        enable=True,
        masked_feat_keys=["color", "strength"],
    ),
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
                feat_scales=dict(coord=coord_feat_scale, strength=strength_feat_scale),
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
                feat_scales=dict(coord=coord_feat_scale, strength=strength_feat_scale),
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
                    feat_scales=dict(coord=coord_feat_scale, strength=strength_feat_scale),
                ),
            ],
            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]],
        ),
    ),
)
