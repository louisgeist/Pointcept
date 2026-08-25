"""
DEBUG / SMOKE TEST -- not a real experiment, do not compare its mIoU numbers
to anything. Purpose: check that the seed-ensemble wandb rendering (mean/std/
max/min scalars + the per-probe wandb.Table added to GridProbeSeedEnsembleTester)
actually looks right in the UI, in a couple of minutes, fully local.

Derived from configs/experiment/w110/2/grid_seed_dales/spunet-v1m0-dales-lin-grid-seed_5.py
with everything shrunk for speed:
  - RANDOM-INIT backbone: no CheckpointLoader hook. SpUNet's real checkpoint
    (job 1052217) isn't mirrored locally (only 862680/873542 are) -- fine
    here since we only care about whether metrics/artifacts reach wandb, not
    whether they're any good. Swap in a CheckpointLoader + real `weight=` to
    also sanity-check against real features.
  - data.{train,val,test}.max_sample=2 -- a couple of DALES tiles instead of
    29 train / 11 test.
  - epoch=4, eval_epoch=2 -> loop=2 (classic mode, cfg.epoch % cfg.eval_epoch
    == 0 required) -- 2 short training epochs, 2 validations, so
    GridProbeCheckpointSaver/GridProbeEvaluator's resume-safe bookkeeping and
    GridProbeSeedEnsembleTester's best-checkpoint reload both actually get
    exercised before the final test pass.
  - batch_size_per_gpu=2, num_worker(_test)=2, log_interval=1: matched to the
    tiny sample counts above.
  - Still 10 probes / seeds -- that's the actual thing being smoke-tested.
  - wandb_project has a "_debug" suffix so this doesn't mix into the real
    pointcept_dales project history.
"""

_base_ = ["../../../../_base_/default_runtime.py"]

grp_exp = 2
num_exp = 1

num_classes = 8
ignore_index = 8
grid_size = 0.1
point_max = 102400
coord_feat_scale = 0.01  # must match Flair3D multitask pretrain (irrelevant here -- random init)
strength_feat_scale = 1 / 60000  # DALES raw intensity → Flair3D [0,1] convention

num_gpu = 1
epoch = 4
eval_epoch = 2
lr = 0.005  # same fixed lr as the real spunet-v1m0-dales-lin-grid-seed_5.py
patch_size = 1024

test_single_fragment = True

# misc custom setting -- shrunk for a tiny local smoke test
batch_size_per_gpu = 2
batch_size = batch_size_per_gpu * num_gpu
batch_size_val = 1
batch_size_test = 1
num_worker = 2
num_worker_test = 2
mix_prob = 0.8
empty_cache = False
enable_amp = True

# dataset settings
dataset_type = "DALESDataset"
data_root = "data/dales"

# No weight= / CheckpointLoader -- deliberately random-init backbone, see docstring.

wandb_project = f"pointcept_{dataset_type[:-7].lower()}_debug"

# Hooks
# No CheckpointLoader: fresh random-init backbone (see docstring). Order
# otherwise matches every other seed-ensemble config: GridProbeEvaluator
# before GridProbeCheckpointSaver/CheckpointSaver; GridProbeSeedEnsembleTester
# last (frees the per-probe optimizers/schedulers, then runs one
# shared-backbone GridProbeSemSegTester pass across ALL 10 probes and
# aggregates mean/std/max/min).
hooks = [
    # Local-only: hecate's spconv autotuner crashes on SpUNet's 2nd distinct
    # conv shape (see SpconvNativeConvAlgo docstring) -- not needed on JZ,
    # not present in the real (non-debug) spunet-*-lin-grid-seed_*.py configs.
    dict(type="SpconvNativeConvAlgo"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=1),
    dict(type="InformationWriter", log_interval=1),
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

# point_mode multiscale concat: stem(32) + stage0(32) + stage1(64) + stage2(128) + stage3/bottleneck(256)
backbone_channels = (32, 64, 128, 256, 256, 128, 96, 96)
backbone_out_channels = 32 + sum(backbone_channels[:4])  # 512

# Seed-ensemble probes — ce_lovasz, AdamW/wd0/OneCycleLR warmup5%, lr fixed;
# only the random init differs across probes (each ProbeHead.linear draws its
# own init from sequential construction in GridProbeSegmentorV2.__init__ —
# no explicit seed field needed). Same 10-probe pattern as the real config.
_criteria = [
    dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
    dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
]

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
    for i in range(10)
}
del _criteria

wandb_run_name = (
    f"[DEBUG] SpUNet SeedEnsemble DALES {grp_exp}.{num_exp}) RANDOM INIT, "
    f"max_sample=2, epoch={epoch} -- wandb render smoke test"
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
        num_classes=0,  # unused in point_mode (no final conv applied)
        channels=backbone_channels,
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        stride=3,
        point_mode=True,
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
        max_sample=2,
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
        max_sample=2,
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
        max_sample=2,
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
