"""
KPConvX grid-search linear probing on H3D — encoder multiscale variant
(Malibu3D+ multitask supervised pretrain, configs/malibu3d_default/multi-kpconvx-v1m0-malibu3d.py
recipe: input_channels=7, kp_radius=3.2, radius_scaling=3.0,
neighbor_limits=(12,16,20,20,20), layer_blocks=(3,3,9,12,3), init_channels=64,
channel_scaling=1.414, grid_size=0.1, point_max=40000, drop_path_rate=0.3 at
pretrain time).

`enc_mode=True` (new `kpconvx_base.py` flag) skips the decoder entirely and
returns the 5 raw encoder-stage features, each gathered back onto the
original points via KPConvX's pyramid `upsamples` cluster indices (chained
through intermediate levels — KPConvX only stores adjacent-level indices,
unlike PT-v3/LitePT's `pooling_parent`) and concatenated finest-first.

IMPORTANT — channel widths are NOT the nominal per-stage `layer_C`
([64,96,128,192,256]). With grid_pool=True, KPConvX bumps the LAST block of
each stage to the NEXT stage's channel width before pooling (so pooling never
changes channel count), so the captured stage features are already at
[96,128,192,256,256] (stage 5, the last, has no further transition so it
stays at layer_C[4]=256). Verified empirically via a synthetic forward pass +
manual replay (max abs diff 0.0 against both a 0-hop and 1-hop manual
gather) — see plan file / scratchpad/check_kpconvx_enc_mode.py.
backbone_out_channels = 96+128+192+256+256 = 928.

`drop_path_rate=0` on the backbone here is required, not optional: KPConvX's
stochastic depth uses its own `DropPathPack` class
(pointcept/models/kpconvx/utils/generic_blocks.py), not `timm.layers.DropPath`
— so `GridProbeSegmentorV2`'s `drop_path_eval_mode` (which only checks
`isinstance(m, timm.layers.DropPath)`) does NOT pin it to eval. Building with
drop_path_rate=0 makes every block use `nn.Identity()` instead (no
learnable params either way, so this is checkpoint-compatible with the
drop_path_rate=0.3 pretrain).

point_max=100_000 — deliberately larger than the KPConvX pretrain's own
SphereCrop budget (40000); roughly matches the ~102400 convention used by
the other backbones' GridProbe configs (LitePT/PT-v3/SpUNet/Sonata). Set by
user instruction. SphereCrop point_max is a data-pipeline parameter, not
part of the backbone architecture/checkpoint, so this doesn't break
checkpoint loading — but the frozen backbone was never trained on scenes
this large, so its features at this crop scale are unverified.

Frozen backbone: Malibu3D+ KPConvX multitask checkpoint (see ckpt/malibu3d/kpconvx_multitask/).

batch_size=24 on a single GPU (vs. the pretrain's global 24 spread over 6-8
GPUs) — set by hand per user instruction; the frozen-backbone, no-backward
probe forward is far cheaper than full multitask training, so this has not
been formally verified against a find_max_batch_size.py run at this batch
size, only inferred from that gap.

Same probe grid as litept-b-v1m0-h3d-lin_enc.py / spunet-v1m0-h3d-lin-grid-enc.py
(ce_lovasz, AdamW/wd0/OneCycleLR warmup5%, lr sweep {1e-4 … 5e-1}) for
cross-backbone comparability. epoch=2000 / eval_epoch=10. H3D has no real
intensity (FillMissingFeat strength=0.0), no strength_feat_scale, and
requires log_test_f1=True / skip_test=False on GridProbeEvaluator/
GridProbeWinnerSelector (required for H3D lin-grid configs).
"""

_base_ = ["../_base_/default_runtime.py"]

grp_exp = 1
num_exp = 1

num_classes = 11
ignore_index = 11
grid_size = 0.1
point_max = 100_000  # deliberately > pretrain's 40000 SphereCrop budget — see docstring
coord_feat_scale = 0.01  # must match Malibu3D multitask pretrain
kp_radius = 3.2
kp_sigma = kp_radius
radius_scaling = 3.0

num_gpu = 1
epoch = 2000
eval_epoch = 10
lr = 5e-2

test_single_fragment = True
log_test_f1 = True

# misc custom setting — see docstring above (set by hand, not calibrated).
batch_size_per_gpu = 24
batch_size = batch_size_per_gpu * num_gpu
batch_size_val = 1
batch_size_test = 1
num_worker = 24 * num_gpu
num_worker_test = 2
mix_prob = 0.8
empty_cache = False
enable_amp = True

# dataset settings
dataset_type = "H3DDataset"
data_root = "data/h3d"

weight = "ckpt/malibu3d/kpconvx_multitask/model_best.pth"

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
    dict(type="InformationWriter", log_interval=1),
    dict(type="GridProbeEvaluator", write_cls_iou=True, select_metric="macro_f1"),
    dict(type="GridProbeCheckpointSaver"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="GridProbeWinnerSelector", skip_test=False),
]

feat_keys = ["coord", "color", "strength"]

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

# Encoder levels (enc_mode): actual captured widths, see docstring — NOT layer_C.
enc_channels = (96, 128, 192, 256, 256)
backbone_out_channels = sum(enc_channels)

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

wandb_run_name = (
    f"KPConvX GridProbe H3D {grp_exp}.{num_exp}) enc multiscale {backbone_out_channels}ch, "
    f"AdamW/wd0/OneCycleLR warmup5%, {len(probes)} probes, epoch={epoch}"
)

# model settings
model = dict(
    type="GridProbeSegmentorV2",
    probes=probes,
    num_classes=num_classes,
    ignore_index=ignore_index,
    target_key="segment",
    backbone_out_channels=backbone_out_channels,
    channel_blocks=enc_channels,
    backbone=dict(
        type="kpconvx_base",
        input_channels=7,  # coord(3) + color(3) + strength(1, fake/zero)
        num_classes=0,
        dim=3,
        task="cloud_segmentation",
        kp_mode="kpconvx",
        shell_sizes=(1, 14, 28),
        kp_radius=kp_radius,
        kp_aggregation="nearest",
        kp_influence="constant",
        kp_sigma=kp_sigma,
        share_kp=False,
        conv_groups=-1,
        inv_groups=8,
        inv_act="sigmoid",
        inv_grp_norm=True,
        kpx_upcut=False,
        subsample_size=grid_size,
        neighbor_limits=(12, 16, 20, 20, 20),
        layer_blocks=(3, 3, 9, 12, 3),
        init_channels=64,
        channel_scaling=1.414,
        radius_scaling=radius_scaling,
        decoder_layer=True,
        grid_pool=True,
        upsample_n=3,
        first_inv_layer=1,
        drop_path_rate=0,  # see docstring — required for correct eval-pinning
        norm="batch",
        bn_momentum=0.1,
        smooth_labels=False,
        class_w=(),
        enc_mode=True,
    ),
    freeze_backbone=True,
    bn_eval_mode=True,  # freeze BatchNorm running stats during probe training
    drop_path_eval_mode=True,  # no-op for KPConvX here (drop_path_rate=0 already removes DropPathPack)
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
            dict(type="FillMissingFeat", feat_key="strength", feat_dim=1, fill_value=0.0),
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
                feat_scales=dict(coord=coord_feat_scale),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="FillMissingFeat", feat_key="strength", feat_dim=1, fill_value=0.0),
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
                feat_scales=dict(coord=coord_feat_scale),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="FillMissingFeat", feat_key="strength", feat_dim=1, fill_value=0.0),
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
                    optional_keys=("inverse",),  # for test_single_fragment broadcast
                    feat_keys=feat_keys,
                    feat_scales=dict(coord=coord_feat_scale),
                ),
            ],
            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]],
        ),
    ),
)
