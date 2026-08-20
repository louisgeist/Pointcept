"""
Sonata-v1m2 grid-search linear probing on H3D — same methodology/grid as the
DALES cross-dataset probe (w109/4/sonata_13h/sonata-v1m2-dales-lin_1.py),
budget scaled to H3D's much smaller split sizes (15 train / 4 val / 11 test
tiles vs DALES).

Frozen PT-v3m2 encoder (enc_mode=True -> multi-scale concat 1232ch) from the
same Flair3D+ Sonata pretrain job 862680, epoch_120, so results are directly
comparable to the DALES probe on an identical pretraining point.

H3D has no strength.npy (LAS `intensity` is all-zero for this release; the
only real intensity-like field, `Reflectance`, is on a different dB scale and
is NOT extracted by preprocess_h3d.py). Sonata was pretrained with scene-level
RandomDropColor/RandomDropStrength (drop_value=0.0, see comment in
pretrain-sonata-v1m2-flair3d.py: "robustness when transferring to datasets
missing color and/or strength"), so `FillMissingFeat` synthesizing a zero
"strength" channel is in-distribution, not a foreign filler (mirrors DALES's
zero "color" channel — same mechanism, opposite missing modality). No learned
masked-feat at pretrain time, so literal zero fill is faithful.

Grid (24 probes): ce_lovasz x {5e-2, 1e-1, 2e-1, 5e-1} lr x {5e-3, 5e-2} wd x
dropout {0, 0.3, 0.5} x input_norm=None. epoch=50 / eval_epoch=10.
skip_test=False -- H3D has a real train/val/test split (unlike DALES, which
reuses its test split as val and sets skip_test=True to avoid a redundant
pass on the same data). Winner is still picked on val mIoU only
(GridProbeWinnerSelector always selects via val; skip_test only gates the
final SemSegTester pass), so this adds a proper held-out test number without
leaking test into the hyperparameter choice.
"""

_base_ = ["../../../../_base_/default_runtime.py"]

grp_exp = 1
num_exp = 1

num_classes = 11
ignore_index = 11
grid_size = 0.1
point_max = 102400  # keep pretrain SphereCrop budget; do not raise for denser H3D


num_gpu = 1
epoch = 150
eval_epoch = 10
lr = 5e-2
patch_size = 1024

test_single_fragment = True
log_test_f1 = True

# misc custom setting
batch_size = 24
batch_size_val = 1
batch_size_test = 1
num_worker = 8 * num_gpu
num_worker_test = 2
mix_prob = 0.8
empty_cache = False
enable_amp = False

# dataset settings
dataset_type = "H3DDataset"
data_root = "data/h3d"

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
    dict(type="InformationWriter", log_interval=1),
    dict(type="GridProbeEvaluator", write_cls_iou=True),
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

# Encoder levels (enc_mode): 48+96+192+384+512 = 1232
backbone_out_channels = 1232

# -----------------------------------------------------------------------------
# Grid-search probes — ce_lovasz x lr x wd x dropout x input_norm
# (1 x 4 x 2 x 3 x 1 = 24 probes). Same grid as the DALES probe.
# -----------------------------------------------------------------------------
_losses = {
    "ce_lovasz": [
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
    ],
}
_lrs = {"5e-2": 5e-2, "1e-1": 1e-1, "2e-1": 2e-1, "5e-1": 5e-1}
_wds = {"5e-3": 0.005, "5e-2": 5e-2}
_dropouts = {"0": 0.0}# "03": 0.3}#, "05": 0.5}
_norms = {"none": None}

probes = {}
for _loss_name, _criteria in _losses.items():
    for _lr_name, _lr in _lrs.items():
        for _wd_name, _wd in _wds.items():
            for _do_name, _dropout in _dropouts.items():
                for _norm_name, _input_norm in _norms.items():
                    _name = f"{_loss_name}_lr{_lr_name}_wd{_wd_name}_do{_do_name}_{_norm_name}"
                    probes[_name] = dict(
                        criteria=_criteria,
                        input_norm=_input_norm,
                        feat_norm=None,
                        dropout=_dropout,
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
del _losses, _lrs, _wds, _dropouts, _norms, _loss_name, _criteria, _lr_name, _lr
del _wd_name, _wd, _do_name, _dropout, _norm_name, _input_norm, _name

wandb_run_name = (
    f"Sonata GridProbe H3D {grp_exp}.{num_exp}) epoch_120, enc multiscale 1232ch, "
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
        type="PT-v3m2",
        in_channels=7,  # coord(3) + color(3) + strength(1, fake/zero)
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
        loop=8,
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
                ),
            ],
            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]],
        ),
    ),
)
