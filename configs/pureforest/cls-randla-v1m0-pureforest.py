"""
RandLA-Net classification on PureForest.

Backbone keeps the RandLA-Net encoder/decoder and returns per-point features
(`num_classes=0`) so `DefaultClassifier` handles scene pooling and logits.
"""

_base_ = ["../_base_/default_runtime.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------
grp_exp = 1
num_exp = 1

num_gpu = 1
num_worker = 8 * num_gpu
enable_amp = True

batch_size = 2 * num_gpu
batch_size_val = batch_size // 2
batch_size_test = batch_size // 2

grid_size = 0.25
point_max = 5000

lr = 5e-4
epoch = 100
eval_epoch = 1
warmup_steps = 5000

feat_keys = ["coord", "color"]

wandb_run_name = f"PureForest RandLA classification ({grp_exp}.{num_exp}) lr={lr}"
wandb_project = "pointcept_pureforest"

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ClsEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

test = dict(type="ClsTester")

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = dict(
    type="DefaultClassifier",
    num_classes=13,
    pooling="mean",
    backbone_embed_dim=32,
    backbone=dict(
        type="RandLA-Net",
        input_channels=6,
        num_classes=0,
        task="cloud_segmentation",
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(128, 64, 32),
        neighbors=16,
        decimation=4,
        bn_momentum=0.01,
        dropout=0.1,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

optimizer = dict(type="AdamW", lr=lr, weight_decay=0.02)
scheduler = dict(
    type="LinearLR",
    start_factor=1 / 10,
    total_iters=warmup_steps,
)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset_type = "PureForestDataset"
data_root = "data/pureforest"

class_names = [
    "deciduous_oak",
    "evergreen_oak",
    "beech",
    "chestnut",
    "black_locust",
    "maritime_pine",
    "scotch_pine",
    "black_pine",
    "aleppo_pine",
    "fir",
    "spruce",
    "larch",
    "douglas",
]

_val_test_transform = [
    dict(type="CenterShift", apply_z=True),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_min_coord=True,
    ),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "category"),
        feat_keys=feat_keys,
        optional_keys=("name",),
    ),
]

data = dict(
    num_classes=13,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
            ),
            dict(type="SphereCrop", point_max=point_max, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "category"),
                feat_keys=feat_keys,
                optional_keys=("name",),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        class_names=class_names,
        transform=_val_test_transform,
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=class_names,
        transform=_val_test_transform,
        test_mode=False,
    ),
)
