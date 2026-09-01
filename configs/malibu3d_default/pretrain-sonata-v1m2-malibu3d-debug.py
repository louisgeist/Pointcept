"""
DEBUG smoke: Sonata-v1m2 pretrain on Malibu3D+ (2× A100, few iters).

Same model/data as pretrain-sonata-v1m2-malibu3d.py, but short schedule +
max_sample.
"""

_base_ = ["../_base_/default_runtime.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------
num_gpu = 2
batch_size_per_gpu = 4
batch_size = batch_size_per_gpu * num_gpu
num_worker = 4 * num_gpu
mix_prob = 0
clip_grad = 3.0
# Skip optimizer/scheduler when global grad L2 is NaN/Inf (all-reduced across ranks).
skip_nan_grad = True
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
evaluate = False
find_unused_parameters = False
grid_size = 0.1

# Short debug schedule: 2 trainer epochs × 20 steps
total_iters = 40
iter_per_epoch = 20
eval_every = 5

wandb_project = "malibu3d_sonata"
wandb_run_name = (
    f"DEBUG | Sonata-v1m2 pretrain Malibu3D+ | bs={batch_size} | grid={grid_size} | "
    f"iters={total_iters}"
)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = dict(
    type="Sonata-v1m2",
    backbone=dict(
        type="PT-v3m2",
        in_channels=7,  # coord(3) + color(3) + strength(1)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(3, 3, 3, 3),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
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
        traceable=True,
        enc_mode=True,
        mask_token=True,
    ),
    teacher_custom=dict(
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ),
    head_in_channels=1088,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=4,
    mask_size_start=0.1,
    mask_size_base=0.4,
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.01,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,
    mask_loss_weight=2 / 8,
    roll_mask_loss_weight=2 / 8,
    unmask_loss_weight=4 / 8,
    momentum_base=0.994,
    momentum_final=1,
    match_max_k=8,
    match_max_r=0.32,
    up_cast_level=2,
)

# -----------------------------------------------------------------------------
# Optimizer / scheduler
# -----------------------------------------------------------------------------
base_lr = 0.001  # Divided by 4, because we train on 8 GPUs instead of 32
lr_decay = 0.9  # layer-wise lr decay

base_wd = 0.04
final_wd = 0.2

dec_depths = model["backbone"]["enc_depths"]
param_dicts = [
    dict(
        keyword=f"enc{e}.block{b}.",
        lr=base_lr * lr_decay ** (sum(dec_depths) - sum(dec_depths[:e]) - b - 1),
    )
    for e in range(len(dec_depths))
    for b in range(dec_depths[e])
]
del dec_depths

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset_type = "Malibu3DDataset"
data_root = "data/malibu3d_plus"
csv_manifest = "data/malibu3d_plus/raw/scene_split_manifest.csv"
min_points = {"train": 1000}

transform = [
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train"),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord", "color", "strength"),
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=[
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="NormalizeColor"),
            # Scene-level modality dropout (shared by both global views): robustness
            # when transferring to datasets missing color and/or strength.
            dict(
                type="RandomDropColor",
                drop_ratio=1.0,
                drop_application_ratio=0.2,
            ),
            dict(
                type="RandomDropStrength",
                drop_ratio=1.0,
                drop_application_ratio=0.2,
            ),
        ],
        global_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
        ],
        local_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="NormalizeColor"),
        ],
        max_size=65_536,
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": grid_size}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_color",
            "global_strength",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_color",
            "local_strength",
            "local_offset",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", "global_color", "global_strength"),
        local_feat_keys=("local_coord", "local_color", "local_strength"),
    ),
]

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        csv_manifest=csv_manifest,
        min_points=min_points,
        max_sample=64,
        transform=transform,
        test_mode=False,
        loop=1,
    )
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=1),
]
