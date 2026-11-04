"""
Sonata (Sonata-v1m2) self-supervised pretraining of a LitePT-v2-S encoder on ShapeNetPart
(train + val, 14k shapes, no labels), coord-only input. Phase 2 of the hybrid-SSL project.

Backbone: LitePT-v2-S in enc_mode with the Sonata options (LayerNorm, linear stem, mask
token, trunc_normal init). Teacher and student heads read the encoder features up-cast over
the last up_cast_level + 1 stages: 144 + 252 + 504 = 900 channels for LitePT-S.

Data: the object recipe of Utonia (upstream configs/utonia/pretrain-utonia-v1m1-0-base_stagev1.py,
obj_transform): NormalizeCoord (unit sphere), random scale, global views of 80-100 % and local
views of 60-80 % of the points, full rotations. Two departures from it: a 0.02 grid instead of
0.01 (at 0.01, finer than the ~0.04 point spacing of the shapes, the first poolings barely
reduce the points and the heads run on half of them), and no elastic distortion by default.

Scales: the Sonata mask sizes (meters, see Sonata.generate_mask) and the matching radius are
rescaled from the upstream Sonata values (0.02 grid, stride 2) to grid_size, as in
configs/malibu3d/pretrain-sonata-v1m2-malibu3d.py: mask patches cover as many voxels as
upstream, and match_max_r keeps upstream's 4x margin over the voxel size at the matching
level (grid_size * stride ** up_cast_level).

Schedule: no cosine. Linear lr warmup then constant lr (LinearLR), constant weight decay,
constant EMA momentum. The Sonata warmups (mask size, mask ratio, teacher temperature) are
defined over a nominal horizon (schedule_iters), not over the run length (total_iters), so
that a run can be stopped when the periodic linear probe plateaus.
"""

_base_ = ["../_base_/default_runtime.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------
num_gpu = 1
batch_size_per_gpu = 32
batch_size = batch_size_per_gpu * num_gpu
num_worker = 8 * num_gpu
mix_prob = 0
clip_grad = 3.0
skip_nan_grad = True
log_grad_norm_every_step = True
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
find_unused_parameters = False

grid_size = 0.02
stride = (2, 2, 2, 2)
up_cast_level = 2
max_size = 65_536  # max points per view; ShapeNetPart shapes have ~2.6k points

# Matching radius at the up-cast level, upstream Sonata margin (0.32 m for 0.08 m voxels).
match_max_r_factor = 4
match_max_r = match_max_r_factor * grid_size * stride[0] * stride[1]  # 0.32
# Mask sizes in coordinate units, scaled from the upstream 0.02 grid.
grid_size_factor = grid_size / 0.02
mask_size_start = 0.1 * grid_size_factor  # 0.1
mask_size_base = 0.4 * grid_size_factor  # 0.4
mask_jitter = 0.01 * grid_size_factor  # 0.01

# Iter-limited schedule (1 trainer epoch = iter_per_epoch optimizer steps, ~1 pass over
# the 14k shapes at batch 32). total_iters is the run length; schedule_iters the nominal
# horizon of the Sonata warmups (5 % of it).
total_iters = 10_000  # POC; schedule_iters stays the nominal horizon
iter_per_epoch = 500
schedule_iters = 30_000
lr_warmup_iters = 1_500

evaluate = False
eval_every = 1  # also the checkpoint frequency in epochs (CheckpointSaver below)
checkpoint_save_freq = 5  # 2500 iterations

# Elastic distortion of the views, off by default. Utonia's object recipe uses
# [[0.2, 0.4], [0.8, 1.6]], which displaces points by ~0.14 on unit-sphere objects.
elastic_distortion_params = None

wandb_project = "hybrid_ssl"
wandb_run_name = (
    f"Sonata LitePT-v2-S pretrain ShapeNetPart | bs={batch_size} | iters={total_iters}"
)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
enc_depths = (2, 2, 2, 6, 2)
enc_channels = (36, 72, 144, 252, 504)

model = dict(
    type="Sonata-v1m2",
    backbone=dict(
        type="LitePT-v2",
        in_channels=3,  # coord
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=stride,
        enc_depths=enc_depths,
        enc_channels=enc_channels,
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=True,
        norm="ln",
        stem="linear",
        mask_token=True,
        init="trunc_normal",
        layer_scale=None,
        rope_shift=None,
        rope_jitter=None,
        rope_rescale=None,
    ),
    teacher_custom=dict(
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ),
    head_in_channels=sum(enc_channels[-(up_cast_level + 1) :]),  # 900
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
    # constant EMA momentum (momentum_final = momentum_base)
    momentum_base=0.996,
    momentum_final=0.996,
    match_max_k=8,
    match_max_r=match_max_r,
    up_cast_level=up_cast_level,
    schedule_iters=schedule_iters,
)

# -----------------------------------------------------------------------------
# Optimizer / scheduler
# -----------------------------------------------------------------------------
base_lr = 0.002
lr_decay = 0.9  # layer-wise lr decay
weight_decay = 0.04  # constant

param_dicts = [
    dict(
        keyword=f"enc{e}.block{b}.",
        lr=base_lr * lr_decay ** (sum(enc_depths) - sum(enc_depths[:e]) - b - 1),
    )
    for e in range(len(enc_depths))
    for b in range(enc_depths[e])
]

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=weight_decay)
# Linear warmup from 0.1x over lr_warmup_iters, then constant (every param group).
scheduler = dict(
    type="LinearLR", start_factor=0.1, end_factor=1.0, total_iters=lr_warmup_iters
)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset_type = "ShapeNetPartDataset"
data_root = "data/shapenetcore_partanno_segmentation_benchmark_v0_normal"

view_transform = [
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
    dict(type="RandomScale", scale=[0.5, 1.5]),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="y", p=0.8),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="ElasticDistortion", distortion_params=elastic_distortion_params),
]

transform = [
    dict(type="NormalizeCoord"),
    dict(type="RandomScale", scale=[0.5, 1.5]),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train"),
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord"),
        global_view_num=2,
        global_view_scale=(0.8, 1.0),
        local_view_num=4,
        local_view_scale=(0.6, 0.8),
        global_transform=view_transform,
        local_transform=view_transform,
        max_size=max_size,
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": grid_size}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_offset",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord",),
        local_feat_keys=("local_coord",),
    ),
]

data = dict(
    train=dict(
        type=dataset_type,
        split=["train", "val"],
        data_root=data_root,
        transform=transform,
        test_mode=False,
        loop=1,
    )
)

# Periodic linear probe of every probe_save_freq-th checkpoint, in the background
# (PeriodicProbeHook; results in save_path/probes/summary.csv). Example, on GPU 0:
#   probe_command = (
#       "CUDA_VISIBLE_DEVICES=0 python tools/train.py --config-file "
#       "configs/shapenet_part/probe-lin-litept-s-v2m0-shapenet-part.py --num-gpus 1 "
#       "--options weight={weight} save_path={probe_dir}"
#   )
probe_command = None
probe_save_freq = checkpoint_save_freq

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=10),
    dict(type="CheckpointSaver", save_freq=checkpoint_save_freq),
    # Reads probe_command, probe_save_freq and iter_per_epoch from this config.
    dict(type="PeriodicProbeHook"),
]
