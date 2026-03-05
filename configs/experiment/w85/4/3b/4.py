_base_ = ["./1.py"]

num_gpu = 4

# Exp 4: lr=6e-5, effective_bs=12
lr = 6e-5
batch_size_per_gpu = 3
batch_size = batch_size_per_gpu * num_gpu
patch_size = 1024 

optimizer = dict(type="AdamW", lr=lr, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[lr, lr / 10],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=lr / 10)]

wandb_run_name = f"Exp 3b.4 | effective_bs={batch_size}, lr={lr}, patch_size={patch_size}"
