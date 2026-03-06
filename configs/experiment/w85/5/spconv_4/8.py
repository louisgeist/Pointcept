_base_ = ["./7.py"]

num_gpu = 4

# Exp 8: grid_size=0.2, lr=5e-3
num_exp = 8
lr = 5e-3

batch_size = 96
grid_size = 0.2
epoch = 800


optimizer = dict(type="SGD", lr=lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(
    type="OneCycleLR",
    max_lr=lr,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=10000.0,
)

wandb_run_name = f"SpUNet 4.{num_exp}| bs={batch_size}, lr={lr}, grid_size={grid_size}, epoch={epoch}"
