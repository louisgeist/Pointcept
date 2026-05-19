# -----------------------------------------------------------------------------
# Default
# -----------------------------------------------------------------------------
_base_ = ["../../../../_base_/default_runtime.py"]


# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------

# Logging parameters
grp_exp = 2
num_exp = 4


# Hardware parameters
num_gpu = 1
num_worker = 8 * num_gpu
enable_amp = True

# Data parameters
batch_size = 128  # total batch size across all gpus
batch_size_val = batch_size // 2
batch_size_test = batch_size // 2

grid_size = 0.1
point_max = 100000
# mix_prob = 0.8 # Not used for classication

# Optimization parameters
lr = 1e-3
epoch = 100
eval_epoch = 1
warmup_steps = 2500

# Features
learned_masked_feat = True
feat_keys = ["coord", "color",]

# Wandb parameters
wandb_run_name = (
    f"PureForest SpUNet cls ({grp_exp}.{num_exp}) "
    f"lr={lr} bs={batch_size} epoch={epoch}"
)
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
    backbone_embed_dim=96,
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=6,
        num_classes=0,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        enc_mode=False,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# -----------------------------------------------------------------------------
# Optimizer / scheduler
# -----------------------------------------------------------------------------
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.005)
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

# Shared val/test pipeline (ClsTester — test_mode=False; same order as val eval).
_val_test_transform = [
    dict(type="CenterShift", apply_z=True),
    dict(
        type="GridSample",
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "category"),
        feat_keys=["coord", "color"],
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
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "color"],
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