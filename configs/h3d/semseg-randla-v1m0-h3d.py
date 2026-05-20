"""
RandLA-Net semantic segmentation on H3D.

The recipe mirrors Myria3D spirit (coarser voxel grid, simple geometric
augmentations) while remaining fully native to Pointcept transforms/evaluator.
"""

_base_ = ["../_base_/default_runtime.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------
grp_exp = 1
num_exp = 1

num_gpu = 1
num_worker = 8 * num_gpu
enable_amp = False

batch_size = 2 * num_gpu
batch_size_val = batch_size // 2

grid_size = 0.25
point_max = 40000

lr = 4e-3
epoch = 200
eval_epoch = epoch // 10

num_classes = 11
ignore_index = num_classes
feat_keys = ["coord", "color"]
test_single_fragment = True

wandb_run_name = f"H3D RandLA semseg ({grp_exp}.{num_exp}) lr={lr}"
wandb_project = "pointcept_h3d"
log_test_f1 = True

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=100),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

test = dict(type="SemSegTester", verbose=True)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="RandLA-Net",
        input_channels=6,
        num_classes=num_classes,
        task="cloud_segmentation",
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(128, 64, 32),
        neighbors=16,
        decimation=4,
        bn_momentum=0.01,
        dropout=0.1,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
    ],
)

optimizer = dict(type="AdamW", lr=lr, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=lr,
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=25.0,
    final_div_factor=1000.0,
)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset_type = "H3DDataset"
data_root = "data/h3d"

class_names = [
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

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
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
            dict(type="Collect", keys=("coord", "segment"), feat_keys=feat_keys),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "origin_segment", "inverse"),
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
                return_inverse=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index"),
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
