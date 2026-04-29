_base_ = ["../../../../_base_/default_runtime.py"] # level experiment/wXX/DD/subfolder/config.py

grp_exp = 1
num_exp = 1

# H3D settings
num_classes = 11
ignore_index = num_classes

# Test settings
tta = False
test_single_fragment = True


# Hooks
# Note: configs are imported as python modules before `_base_` is merged, so we
# must redefine `hooks` here instead of mutating it.
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=100),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# Parameters factorization for easier experiments
lr = 5e-4
warmup_steps = 5000*4
grid_size = 0.1

# minimal example settings
num_gpu = 1
num_worker = 8 * num_gpu  # total worker in all gpu
batch_size_per_gpu = 2
batch_size = batch_size_per_gpu * num_gpu
gradient_accumulation_steps = 1
mix_prob = 0.8
max_input_pts = 40000

# # Settings for multigpu training with 8 80GB gpus
# num_worker = 64  # total worker in all gpu
# batch_size = 12  
# mix_prob = 0.8
empty_cache = False
enable_amp = False
# sync_bn = True
# max_input_pts = 80000

# scheduler settings
# epoch = 1000
# eval_epoch = 200
epoch=100
eval_epoch=epoch//10

# dataset settings
dataset_type = "H3DDataset"
data_root = "data/h3d"

wandb_run_name = f"KPConvX on {dataset_type[:-7]}: {grp_exp}.{num_exp}) lr={lr}"

feat_keys = ["coord", "color"]

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="kpconvx_base",
        input_channels=6,  # xyz + RGB
        num_classes=num_classes,
        dim=3,
        task='cloud_segmentation',
        kp_mode='kpconvx',
        shell_sizes=(1, 14, 28),
        kp_radius=2.3,
        kp_aggregation='nearest',
        kp_influence='constant',
        kp_sigma=2.3,
        share_kp=False,
        conv_groups=-1,  # Only for old KPConv blocks
        inv_groups=8,
        inv_act='sigmoid',
        inv_grp_norm=True,
        kpx_upcut=False,
        subsample_size=grid_size,
        neighbor_limits=(12, 16, 20, 20, 20),
        layer_blocks=(3, 3, 9, 12, 3),
        init_channels=64,
        channel_scaling=1.414,
        radius_scaling=2.2,
        decoder_layer=True,
        grid_pool=True,
        upsample_n=3,  # Ignored if grid_pool is True
        first_inv_layer=1,
        drop_path_rate=0.3,
        norm='batch',
        bn_momentum=0.1,
        smooth_labels=False,  # True only for classification
        class_w=(),
    ),
    criteria=[
        dict(type="CrossEntropyLoss",
             loss_weight=1.0,
             ignore_index=ignore_index),
        dict(type="LovaszLoss",
             mode="multiclass",
             loss_weight=1.0,
             ignore_index=ignore_index)
    ]
)

optimizer = dict(type="AdamW", lr=lr, weight_decay=0.02)
scheduler = dict(type="OneCycleLR",
                 max_lr=optimizer["lr"],
                 pct_start=0.05,
                 anneal_strategy="cos",
                 div_factor=100.0,
                 final_div_factor=1000.0)
# scheduler = dict(type="LinearLR", 
#                  start_factor = 1/10, # start with lr/10
#                  total_iters = warmup_steps, # number of epochs before reaching lr (and plateauing) 
#                  # As we have 100 epoch for small training:
#                  # 1. 50 epochs of linear increase, 
#                  # 2. then 50 epochs of constant lr
#                 )


data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=[
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
        ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        # 19/03 : copied from the one I defined for LitePT on Flair3D (except for GridSample, Update grid size(specific LitePT) )
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5), # as waymo/nuscences
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),                                 # as waymo/nuscences
            dict(type="RandomFlip", p=0.5),                                             # as waymo/nuscences
            dict(type="RandomJitter", sigma=0.005, clip=0.02),                          # as waymo/nuscences
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="GridSample", 
                 grid_size=grid_size, 
                 hash_type="fnv", 
                 mode="train", 
                 return_min_coord=True, # return_min_coord here - return_grid_coord otherwise
                 ),
            dict(type="SphereCrop", point_max=max_input_pts, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment",), feat_keys=feat_keys)
        ],
        test_mode=False,
    ),

    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="GridSample", 
                 grid_size=grid_size, 
                 hash_type="fnv", 
                 mode="train", 
                 return_min_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="center"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment",), feat_keys=feat_keys)
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
            voxelize=dict(type="GridSample",
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
                dict(type="Collect", keys=("coord", "index"), optional_keys=("inverse",), feat_keys=feat_keys)
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomFlip", p=1)]
            ] if tta else [[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]],
        )
    ),
)
