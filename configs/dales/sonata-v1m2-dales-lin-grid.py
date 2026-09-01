"""
Sonata-v1m2 seed-ensemble linear probing on DALES — 10 probes with IDENTICAL
hyperparameters (the best lr from the earlier lr-grid sweep), differing only
by random init, to report test mIoU/mAcc/allAcc as mean±std across seeds
instead of a single point estimate. Reuses the GridProbe machinery (shared
frozen-backbone forward across all probes) purely to avoid paying the
backbone cost 10x, not for hyperparameter search.

Frozen PT-v3m2 encoder (enc_mode=True → multi-scale concat 1232ch) from the
Malibu3D+ Sonata pretrain job sonata_outdoor, epoch_120. DALES has no RGB — Sonata was
pretrained with scene-level RandomDropColor/RandomDropStrength (drop_value=0.0)
so `FillMissingFeat` synthesizes a zero "color" channel (in_channels=7). No
learned masked-feat at pretrain time, so literal zero fill is faithful.

10 probes, identical config, each getting its own random Linear init from
sequential construction in GridProbeSegmentorV2.__init__ (no seed field
needed). Test-set aggregation: GridProbeSeedEnsembleTester (in place of
GridProbeWinnerSelector) + GridProbeSemSegTester (test = dict(...) below)
reload every probe's best-val checkpoint, run one shared-backbone test pass
across all 10, and write save_path/seed_ensemble_results.json with
mean/std of test_mIoU/test_mAcc/test_allAcc.
"""

_base_ = ["../_base_/default_runtime.py"]

grp_exp = 1
num_exp = 4

num_classes = 8
ignore_index = 8
grid_size = 0.1
point_max = 102400
strength_feat_scale = 1 / 60000  # DALES raw intensity → Malibu3D [0,1] convention

num_gpu = 1
epoch = 400
eval_epoch = 10
lr = 0.02  # best lr from the earlier 12-value lr-grid sweep on DALES/Sonata
patch_size = 1024

test_single_fragment = True

# misc custom setting
batch_size_per_gpu = 24
batch_size = batch_size_per_gpu * num_gpu
batch_size_val = 1
batch_size_test = 1
num_worker = 24  num_worker_test = 2
mix_prob = 0.8
empty_cache = False
enable_amp = True

# dataset settings
dataset_type = "DALESDataset"
data_root = "data/dales"

weight = "ckpt/malibu3d/sonata_outdoor/epoch_120.pth"

wandb_project = f"pointcept_{dataset_type[:-7].lower()}"

# Hooks
# Order matters: GridProbeEvaluator before GridProbeCheckpointSaver/CheckpointSaver;
# GridProbeSeedEnsembleTester last (frees the per-probe optimizers/schedulers, then
# runs one shared-backbone GridProbeSemSegTester pass across ALL 10 probes and
# aggregates mean/std — replaces PreciseEvaluator, which never sets
# raw_model.active_probe and would break under GridProbeSegmentorV2, and replaces
# GridProbeWinnerSelector, which only tests a single "winner" — there is no winner
# concept for a seed ensemble).
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
    dict(type="GridProbeSeedEnsembleTester"),
]

# Override default_runtime.py's test = dict(type="SemSegTester", ...): the
# seed-ensemble test pass needs the shared-backbone, all-probes-at-once tester.
test = dict(type="GridProbeSemSegTester", verbose=True)

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

# Encoder levels (enc_mode): 48+96+192+384+512 = 1232
backbone_out_channels = 1232

# Seed-ensemble probes — ce_lovasz, AdamW/wd0/OneCycleLR warmup5%, lr fixed at the
# best value from the earlier lr-grid sweep; only the random init differs across
# probes (each ProbeHead.linear draws its own init from sequential construction in
# GridProbeSegmentorV2.__init__ — no explicit seed field needed).
_criteria = [
    dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
    dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
]
num_seeds = 10

probes = {
    f"ce_lovasz_init{i}": dict(
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
    for i in range(num_seeds)
}
del _criteria


wandb_run_name = (
    f"Sonata SeedEnsemble DALES {grp_exp}.{num_exp}) epoch_120, enc multiscale 1232ch, "
    f"{len(probes)} inits, lr={lr}, epoch={epoch}"
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
                feat_scales=dict(strength=strength_feat_scale),
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
                feat_scales=dict(strength=strength_feat_scale),
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
                    feat_scales=dict(strength=strength_feat_scale),
                ),
            ],
            aug_transform=[[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]],
        ),
    ),
)
