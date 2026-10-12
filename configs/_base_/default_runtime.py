"""
Runtime defaults shared by every config of this repository (MALiBU3D, DALES, H3D, ECLAIR,
OpenGF, PureForest) through `_base_`: checkpointing, batch sizes and workers, training
schedule, gradient logging, multitask loss balancing, AMP, W&B, default hooks, trainer and
tester. Each config overrides what it needs.
"""

weight = None  # path to model weight
resume = False  # whether to resume training process
evaluate = True  # evaluate after each epoch training process
test_only = False  # test process

seed = None  # train process will init a random seed and record
save_path = "exp/default"
num_worker = 16  # total worker in all gpu
batch_size = 16  # total batch size in all gpu
gradient_accumulation_steps = 1  # total steps to accumulate gradients for
batch_size_val = None  # auto adapt to bs 1 for each gpu
batch_size_test = None  # auto adapt to bs 1 for each gpu
# Val/test batching by voxel budget (None = fixed batch_size_*). When set, the loader
# (VoxelBudgetBatchSampler) packs scenes up to this many voxels per batch, and
# batch_size_val / batch_size_test become the max number of scenes per batch.
# Scene sizes come from the n_voxels column of data.*.csv_manifest (tiles.csv).
# Independent of the train point_max; about 8x the median n_voxels is a reasonable
# start.
test_voxel_budget = None  # e.g. 1_200_000
val_voxel_budget = None  # e.g. 1_200_000
# Worker count of the test DataLoader (None = num_worker). Test samples keep the
# dense cloud and its fragment list, so a packed test loader (test_voxel_budget
# set, or batch_size_test > 1) needs much more host RAM per worker than train/val:
# lower this (e.g. 2) when the test loader runs out of memory.
num_worker_test = None

# -----------------------------------------------------------------------------
# Training schedule: two mutually exclusive modes (resolved in
# pointcept/engines/defaults.py)
# -----------------------------------------------------------------------------

# Classic mode (leave total_iters = None): each trainer epoch loops over the dataset.
epoch = 100  # total dataset passes over the full run
eval_epoch = 100  # number of validation epochs

# Iter-limited mode (set total_iters): each trainer epoch = iter_per_epoch random batch steps.
total_iters = None  # total optimizer steps; enables iter-limited mode when set
iter_per_epoch = None  # iteration steps per epoch (default 1000 when total_iters is set)
eval_every = None  # validate every N epochs (default 5 when total_iters is set)

clip_grad = None  # disable with None, enable with a float
# If True, skip the optimizer and scheduler step when the global grad L2 norm is
# non-finite (NaN/Inf). The decision is all-reduced so DDP ranks stay in sync.
skip_nan_grad = False
# The global L2 grad norm (train/gradient/global) and the weight-update norm
# (train/gradient/weight_update) are always logged as epoch averages.
# log_task_gradient_norms=True also logs them every step (train_batch/...) and
# adds a costly per-task breakdown: per-task L2 grad norms on the shared backbone
# and on each task head, and pairwise backbone cosine similarities between tasks
# (train/gradient/backbone_cos/{task_a}__{task_b}).
log_task_gradient_norms = False

# GradNormLite (multitask loss balancing): every grad_norm_lite_interval steps
# (the first time at that step, not at 0), measure each task's L2 grad norm on the
# last backbone layer, update an EMA of it, and divide each task loss by its EMA
# before the main backward. gradient/last_layer/* and loss_scale/* are logged on
# those steps only.
# Under DDP the EMA is averaged across ranks only when sync_bn=True (otherwise it
# stays per rank, like non-synced BatchNorm statistics).
grad_norm_lite = False
grad_norm_lite_interval = 100
grad_norm_lite_ema_alpha = 0.1
grad_norm_lite_eps = 1e-3

# GradNorm (Chen et al. 2018, "GradNorm: Gradient Normalization for Adaptive Loss
# Balancing in Deep Multitask Networks"). Mutually exclusive with grad_norm_lite.
# Learns one loss weight per task (or per group, via grad_norm_task_groups) by
# gradient descent on the GradNorm L1 objective, renormalized to sum to the number
# of groups. The weights, their Adam state and the initial-loss anchors L_g(0) are
# checkpointed. grad_norm/{weight,gw,target,loss_ratio}/* and grad_norm/loss are
# logged every step. Inputs are averaged across ranks so the weights stay
# identical on every rank.
grad_norm = False
grad_norm_alpha = 1.5  # restoring-force / asymmetry exponent
grad_norm_weight_lr = 1e-2  # Adam lr for the learnable weights
grad_norm_interval = 1  # update the weights every N optimizer steps (1 = faithful)

sync_bn = False
enable_amp = False
amp_dtype = "float16"
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False

enable_wandb = True
log_test_f1 = False  # if True, log per-class and macro F1 on test (console + wandb when enabled)
wandb_project = "pointcept"  # custom your project name e.g. Sonata, PTv3
wandb_key = None  # wandb token, default is None. If None, login with `wandb login` in your terminal

mix_prob = 0
param_dicts = None  # example: param_dicts = [dict(keyword="block", lr_scale=0.1)]

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# Trainer
train = dict(type="DefaultTrainer")

# Tester
test = dict(type="SemSegTester", verbose=True)
