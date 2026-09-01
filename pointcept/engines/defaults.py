"""
Default training/testing logic

modified from detectron2(https://github.com/facebookresearch/detectron2)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import argparse
import warnings
import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


import pointcept.utils.comm as comm
from pointcept.utils.env import get_random_seed, set_seed
from pointcept.utils.config import Config, ConfigDict, DictAction


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.
    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if comm.get_world_size() == 1:
        return model
    # kwargs['find_unused_parameters'] = True
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
        if "output_device" not in kwargs:
            kwargs["output_device"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = None if seed is None else num_workers * rank + worker_id + seed
    set_seed(worker_seed)


def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog or f"""
    Examples:
    Run on single machine:
        $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
    Change some config options:
        $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
    Run on multiple machines:
        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    # port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        # default="tcp://127.0.0.1:{}".format(port),
        default="auto",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    return parser


def _apply_sphere_crop_point_max(obj, point_max: int) -> int:
    """Recursively set ``point_max`` on every ``type=="SphereCrop"`` dict.

    Used when ``override_point_max`` is set (VRAM probes). Real training
    configs do not set that key, so this is a no-op outside probes.
    """
    n = 0
    if isinstance(obj, dict):
        if obj.get("type") == "SphereCrop":
            obj["point_max"] = int(point_max)
            n += 1
        for value in obj.values():
            n += _apply_sphere_crop_point_max(value, point_max)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            n += _apply_sphere_crop_point_max(value, point_max)
    return n


def default_config_parser(file_path, options):
    """Load a config file and resolve training-schedule parameters.

    Classic mode :
        - Definition: 1 epoch means 1 full pass over the dataset
        - Activated by setting ``total_iters`` to ``None``
        - ``loop = epoch // eval_epoch`` on ``data.train``
        - ``eval_every`` defaults to 1 (validate every trainer epoch)

    Iter-limited mode:
        - Definition: 1 epoch means ``iter_per_epoch`` random batch steps
        - ``total_iters`` is the total optimizer-step budget
        - ``num_epochs = total_iters // iter_per_epoch`` (requires exact division)
        - ``loop`` is forced to 1; ``epoch`` / ``eval_epoch`` are unused (classic-only)
        - ``eval_every`` defaults to 5 (validate every N epochs, plus the last epoch)
        - Scheduler ``total_steps = total_iters``

    See README_MALIBU3D.md § Training schedule for examples.
    """
    # config name protocol: dataset_name/model_name-exp_name
    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find("-")
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1 :]))

    if options is not None:
        cfg.merge_from_dict(options)

    cfg_dict = object.__getattribute__(cfg, "_cfg_dict")
    if "override_point_max" in cfg_dict:
        point_max = int(cfg_dict["override_point_max"])
        cfg.point_max = point_max
        n_patched = _apply_sphere_crop_point_max(cfg_dict, point_max)
        if n_patched == 0:
            warnings.warn(
                f"override_point_max={point_max} was set but no SphereCrop "
                "transform was found in the config; crop size is unchanged "
                "(e.g. Sonata SSL uses max_size, and most val/test pipelines "
                "have no SphereCrop).",
                UserWarning,
                stacklevel=2,
            )

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    if getattr(cfg, "grad_norm", False) and getattr(cfg, "grad_norm_lite", False):
        raise ValueError(
            "grad_norm (real GradNorm) and grad_norm_lite are mutually "
            "exclusive multitask loss-balancing schemes; enable at most one."
        )

    if "data" not in cfg:
        cfg.data = ConfigDict()
    elif not isinstance(cfg.data, ConfigDict):
        cfg.data = ConfigDict(cfg.data)
    if "train" not in cfg.data:
        cfg.data.train = ConfigDict()
    elif not isinstance(cfg.data.train, ConfigDict):
        cfg.data.train = ConfigDict(cfg.data.train)

    total_iters = getattr(cfg, "total_iters", None)
    if total_iters is not None: # iter-limited mode
        if total_iters <= 0:
            raise ValueError("total_iters must be > 0")
        
        # eval_every (number of train epochs between validations)
        if getattr(cfg, "eval_every", None) is None:
            cfg.eval_every = 5
        if cfg.eval_every <= 0:
            raise ValueError("eval_every must be > 0 when total_iters is set")
        
        # iter_per_epoch
        if getattr(cfg, "iter_per_epoch", None) is None:
            cfg.iter_per_epoch = 1000
        if cfg.iter_per_epoch <= 0:
            raise ValueError("iter_per_epoch must be > 0 when total_iters is set")
        if total_iters % cfg.iter_per_epoch != 0:
            raise ValueError(
                f"total_iters ({total_iters}) must be divisible by iter_per_epoch "
                f"({cfg.iter_per_epoch})"
            )
        cfg.num_epochs = total_iters // cfg.iter_per_epoch

        # Default loop=1 (epoch length is iter_per_epoch). Preserve an explicit
        # train loop > 1 for single-scene overfit (repeat one tile in the dataset).
        if getattr(cfg.data.train, "loop", 1) <= 1:
            cfg.data.train.loop = 1
    else: # classic mode
        # Deactivate iter-limited mode parameters
        cfg.iter_per_epoch = None
        cfg.eval_every = None
        cfg.num_epochs = None
        cfg.data.train.loop = cfg.epoch // cfg.eval_epoch

    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
    return cfg


def default_setup(cfg):
    # scalar by world size
    world_size = comm.get_world_size()
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else mp.cpu_count()
    cfg.num_worker_per_gpu = cfg.num_worker // world_size
    assert cfg.batch_size % world_size == 0
    assert cfg.batch_size_val is None or cfg.batch_size_val % world_size == 0
    assert cfg.batch_size_test is None or cfg.batch_size_test % world_size == 0
    cfg.batch_size_per_gpu = cfg.batch_size // world_size
    cfg.batch_size_val_per_gpu = (
        cfg.batch_size_val // world_size if cfg.batch_size_val is not None else 1
    )
    cfg.batch_size_test_per_gpu = (
        cfg.batch_size_test // world_size if cfg.batch_size_test is not None else 1
    )
    # classic mode only: epoch must divide eval_epoch for loop resolution
    if getattr(cfg, "total_iters", None) is None:
        assert cfg.epoch % cfg.eval_epoch == 0
    # settle random seed
    rank = comm.get_rank()
    seed = None if cfg.seed is None else cfg.seed + rank * cfg.num_worker_per_gpu
    set_seed(seed)
    return cfg
