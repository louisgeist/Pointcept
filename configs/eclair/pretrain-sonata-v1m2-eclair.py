"""
Sonata-v1m2 self-supervised pretraining of a PT-v3m2 backbone on the ECLAIR train split (exploratory
experiment, not used in the MALiBU3D paper).

Adapted from configs/malibu3d/pretrain-sonata-v1m2-malibu3d.py: coord + color + strength inputs
(in_channels=7, raw intensity scaled by 1/60000), 0.1 m grid with stride 3, scene-level
color/strength dropout, iteration-based schedule (150k iterations, 1000 per trainer epoch), no
online evaluation (checkpoints are evaluated with probe/sonata-v1m2-eclair-lin.py).
"""

_base_ = ["../_base_/default_runtime.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------
# 24 GPUs x 4 scenes; 4 scenes per GPU fit with stride=(3, 3, 3, 3), as in the MALiBU3D pretraining.
num_gpu = 24
batch_size_per_gpu = 4
batch_size = batch_size_per_gpu * num_gpu  # → 96
num_worker = 8 * num_gpu
mix_prob = 0
clip_grad = 3.0
# Skip optimizer/scheduler when global grad L2 is NaN/Inf (all-reduced across ranks).
skip_nan_grad = True
# Log the global gradient norm and weight update every iteration (helps catch divergence).
# The per-task breakdown this flag also enables needs model.backbone_parameters, which
# Sonata does not have, so it is skipped.
log_task_gradient_norms = True
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
find_unused_parameters = False
grid_size = 0.1
stride = (3, 3, 3, 3)
max_size = 65_536  # points-per-view budget (MultiViewGenerator); bounds GPU memory
strength_feat_scale = 1 / 60000  # raw uint16 intensity → MALiBU3D [0,1] convention

# match_max_r must scale with the point spacing at the matching resolution
# (grid_size after up_cast_level=2 pooling stages), not be a fixed absolute
# distance: upstream Sonata uses match_max_r=0.32 with grid_size=0.02 and stride 2
# (voxel 0.08 m, a 4x margin). Keep the same 4x margin here (voxel 0.9 m).
match_max_r_factor = 4
match_max_r = match_max_r_factor * grid_size * stride[0] * stride[1]

# generate_mask buckets points via (coord - min_coord) // mask_size on metric
# coordinates, so mask_size/mask_jitter are absolute distances (meters).
# Scale linearly by grid_size so patches keep the same voxel footprint as
# upstream (grid_size=0.02).
grid_size_factor = grid_size / 0.02  # 0.02 is the default Sonata config grid_size
mask_size_start = 0.1 * grid_size_factor
mask_size_base = 0.4 * grid_size_factor
mask_jitter = 0.01 * grid_size_factor

# Iter-limited schedule (1 trainer epoch = 1000 optimizer steps)
total_iters = 150_000
iter_per_epoch = 1000

# No online evaluation: representations are evaluated afterwards with linear probes.
evaluate = False
eval_every = 1
# Checkpoints every trainer epoch (1000 iters).
ckpt_save_freq = 1

wandb_project = "eclair_sonata"
wandb_run_name = (
    f"Sonata-v1m2 pretrain ECLAIR | bs={batch_size} | grid={grid_size} | "
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
        stride=stride,
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
    mask_size_start=mask_size_start,
    mask_size_base=mask_size_base,
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=mask_jitter,
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
    match_max_r=match_max_r,
    up_cast_level=2,
)

# -----------------------------------------------------------------------------
# Optimizer / scheduler
# -----------------------------------------------------------------------------
base_lr = 0.004  # Sonata default
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
dataset_type = "ECLAIRDataset"
data_root = "data/eclair"
# A few pseudo-labeled train tiles are near-empty (well under 1500 points, while the
# next-smallest tile has ~5000). With local_view_scale=(0.1, 0.4) their local views
# shrink to ~150 points, which destabilizes teacher-student matching (NaN loss).
# 2000 excludes exactly these outliers.
min_points = {"train": 2000}

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
        max_size=max_size,
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
        feat_scales=dict(
            global_strength=strength_feat_scale,
            local_strength=strength_feat_scale,
        ),
    ),
]

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        include_pseudo=True,
        min_points=min_points,
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
    dict(type="InformationWriter", log_interval=1),
    dict(type="CheckpointSaver", save_freq=ckpt_save_freq),
]
