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

# -----------------------------------------------------------------------------
# Training schedule — two mutually exclusive modes (see README_geist.md)
# -----------------------------------------------------------------------------

# Classic mode (leave total_iters = None): each trainer epoch loops over the dataset.
epoch = 100  # total dataset passes over the full run
eval_epoch = 100  # number of validation epochs

# Iter-limited mode (set total_iters): each trainer epoch = iter_per_epoch random batch steps.
total_iters = None  # total optimizer steps; enables iter-limited mode when set
iter_per_epoch = None  # iteration steps per epoch (default 1000 when total_iters is set)
eval_every = None  # validate every N epochs (default 5 when total_iters is set)

clip_grad = None  # disable with None, enable with a float
# Global L2 grad norm (train/gradient/global) and weight update norm
# (train/gradient/weight_update) are always logged at each optimizer step
# (TB per-step train_batch/... + epoch avg in W&B).
# log_task_gradient_norms enables additional per-task decomposition (costly):
# per-task L2 grad norm on shared backbone and task head, plus pairwise
# backbone cosine similarities between tasks (upper triangle, e.g. 6 metrics for
# 4 tasks: train/gradient/backbone_cos/{task_a}__{task_b}); logged to TB/W&B
# (per epoch 'train/gradient/...' and per-step 'train_batch/gradient/...').
log_task_gradient_norms = False

# GradNormLite (multitask): every N steps (first measure at N, not 0), measure
# per-task L2 grad norms on the backbone last layer only, update an EMA, and
# divide each task loss by that EMA before the main backward. Logged keys:
# gradient/last_layer/* and loss_scale/* (TB + W&B). Independent of
# log_task_gradient_norms. EMA sync follows sync_bn: all_reduce mean across
# ranks only when sync_bn=True (otherwise local per rank, like non-synced
# BatchNorm stats).
grad_norm_lite = False
grad_norm_lite_interval = 100
grad_norm_lite_ema_alpha = 0.1
grad_norm_lite_eps = 1e-3

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
