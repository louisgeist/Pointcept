"""
SpUNet-v1m1 grid-search linear probing on DALES — combined encoder+decoder
hypercolumn variant (same frozen checkpoint as
spunet-v1m0-dales-lin-grid-{enc,dec}.py, job spunet_multitask, Malibu3D multitask
supervised pretrain: channels=(32,64,128,256,256,128,96,96),
layers=(2,3,4,6,2,2,2,2), stride=3).

Sets `point_mode=True` AND `dec_point_mode=True` together
(spconv_unet_v1m1_base.py): the decoder hypercolumn chain
(dec0(96)+dec1(96)+dec2(128)+dec3(256)+bottleneck(256) = 832ch, built by
`dec_point_mode` alone, mirroring LitePT/PT-v3's dec_traceable/traceable
convention of "decoder stages + encoder bottleneck") is concatenated with the
raw encoder multiscale (stem(32)+stage0(32)+stage1(64)+stage2(128) = 256ch,
bottleneck dropped here since dec_point_mode already carries it, to avoid
duplicating an identical 256ch block) — 832 + 256 = 1088ch total. Decoder-
hypercolumn only (832ch) is spunet-v1m0-dales-lin-grid-dec-hc.py; plain
single-scale decoder is spunet-v1m0-dales-lin-grid-dec.py (96ch); encoder-only
is spunet-v1m0-dales-lin-grid-enc.py (512ch). This closes
most of the channel-budget gap against LitePT-B's enc/dec hypercolumns
(1386/1404ch) and PT-v3-malibu's/Sonata's (992/1024/1232ch): SpUNet's
per-stage widths are simply narrower in this checkpoint, so tapping every
level of both the encoder and decoder (instead of the encoder XOR decoder) is
the way to reach a comparable feature budget without retraining. See
tests/test_spunet_point_mode.py (`test_combined_point_mode_and_dec_point_mode_shape_and_alignment`)
for the shape/row-alignment correctness check.

Same probe grid as litept-b-v1m0-dales-lin-grid-enc.py (ce_lovasz x 12 LRs,
AdamW/wd0/OneCycleLR warmup5%, epoch=400/eval_epoch=10) for cross-backbone
comparability. `bn_eval_mode=True` freezes SpUNet's BatchNorm running stats
(real BatchNorm1d); `drop_path_eval_mode=True` is a no-op (SpUNet has no
DropPath modules). Z_MinShift/Z_RandomOffset included in train/val/test per
the H3D/ECLAIR lin-grid convention. Same DALES-has-no-RGB handling as the
other DALES lin configs.
"""

_base_ = ["../_base_/default_runtime.py"]

grp_exp = 1
num_exp = 3

num_classes = 8
ignore_index = 8
grid_size = 0.1
point_max = 102400
coord_feat_scale = 0.01  # must match Malibu3D multitask pretrain
strength_feat_scale = 1 / 60000  # DALES raw intensity → Malibu3D [0,1] convention

num_gpu = 1
epoch = 400
eval_epoch = 10
lr = 5e-2

test_single_fragment = True

# misc custom setting
batch_size_per_gpu = 24
batch_size = batch_size_per_gpu * num_gpu
batch_size_val = 1
batch_size_test = 1
num_worker = 24 * num_gpu  num_worker_test = 2
mix_prob = 0.8
empty_cache = False
enable_amp = True

# dataset settings
dataset_type = "DALESDataset"
data_root = "data/dales"

weight = "ckpt/malibu3d/spunet_multitask/model_best.pth"

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
    dict(type="GridProbeWinnerSelector", skip_test=True),
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

# dec_point_mode chain (dec0+dec1+dec2+dec3+bottleneck = 832) + point_mode's
# raw encoder levels with the (already-counted) bottleneck dropped (stem+
# stage0+stage1+stage2 = 256) = 1088.
backbone_channels = (32, 64, 128, 256, 256, 128, 96, 96)
backbone_out_channels = (96 + 96 + 128 + 256 + 256) + (32 + 32 + 64 + 128)

# Grid-search probes — ce_lovasz, AdamW/wd0/OneCycleLR warmup5%, lr sweep only (12 probes).
_criteria = [
    dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
    dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
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

probes = {
    f"ce_lovasz_lr{lr_name}": dict(
        criteria=_criteria,
        input_norm=None,
        feat_norm=None,
        dropout=0.0,
        optimizer=dict(type="AdamW", lr=lr, weight_decay=0.0),
        scheduler=dict(
            type="OneCycleLR",
            max_lr=lr,
            pct_start=0.05,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=1000.0,
        ),
        grad_clip=3.0,
    )
    for lr_name, lr in _lrs.items()
}
del _criteria, _lrs

wandb_run_name = (
    f"SpUNet GridProbe DALES {grp_exp}.{num_exp}) enc+dec combined {backbone_out_channels}ch, "
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
        type="SpUNet-v1m1",
        in_channels=7,  # coord(3) + color(3, fake) + strength(1)
        num_classes=0,  # unused in point_mode/dec_point_mode (no final conv applied) — kept for checkpoint key compat
        channels=backbone_channels,
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        stride=3,
        point_mode=True,
        dec_point_mode=True,
    ),
    freeze_backbone=True,
    bn_eval_mode=True,  # freeze BatchNorm running stats during probe training (real BN here)
    drop_path_eval_mode=True,  # no-op — SpUNet has no DropPath modules
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
            dict(type="Z_MinShift"),
            dict(type="Z_RandomOffset"),
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
            dict(type="Z_MinShift"),
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
