_base_ = ["./cls-spunet-v1m0-pureforest.py"]

data_root = "data/pureforest_toy"
epoch = 10
eval_epoch = 1

wandb_project = "pointcept-pureforest-toy"

data = dict(
    train=dict(data_root=data_root),
    val=dict(data_root=data_root),
    test=dict(data_root=data_root),
)
