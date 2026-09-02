# Anonymous default paths for Malibu3D benchmark reproduction.
# Override via --options or environment variables when running tools/train.py.

data_root = "data/malibu3d"
ckpt_root = "ckpt"
network_graphs_root = "data/network_graphs"

# Frozen-backbone checkpoints (Malibu3D multitask pretrain / Sonata SSL).
# Bundled in the supplementary archive under ckpt/.
ckpt_litept_b_multitask = f"{ckpt_root}/malibu3d/litept_b_multitask/model_best.pth"
ckpt_ptv3_multitask = f"{ckpt_root}/malibu3d/ptv3_multitask/model_best.pth"
ckpt_spunet_multitask = f"{ckpt_root}/malibu3d/spunet_multitask/model_best.pth"
ckpt_kpconvx_multitask = f"{ckpt_root}/malibu3d/kpconvx_multitask/model_best.pth"
ckpt_sonata_outdoor = f"{ckpt_root}/malibu3d/sonata_outdoor/epoch_120.pth"
