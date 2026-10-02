"""
Optimizer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import copy
import torch
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from .muon_kimi import *

OPTIMIZERS = Registry("optimizers")


OPTIMIZERS.register_module(module=torch.optim.SGD, name="SGD")
OPTIMIZERS.register_module(module=torch.optim.Adam, name="Adam")
OPTIMIZERS.register_module(module=torch.optim.AdamW, name="AdamW")
OPTIMIZERS.register_module(module=MuonKIMI, name="Muon_KIMI")


def _is_no_decay_param(name, param):
    """Standard rule (BERT / nanoGPT): biases and 1-D tensors (norm gains,
    learned mask/embedding vectors) get no weight decay."""
    return param.ndim <= 1 or name.endswith(".bias")


def build_optimizer(cfg, model=None, param_dicts=None, params=None):
    """
    Optional `cfg.no_decay_bias_norm=True`: split every parameter group in two,
    the second copy holding the biases / 1-D params with weight_decay=0. The
    no-decay groups are appended AFTER all regular groups, in the same order, so
    group i + n_groups has the same lr as group i. A scheduler taking a per-group
    list (e.g. OneCycleLR max_lr) must therefore list the lrs twice:
    max_lr=[lr, lr/10, lr, lr/10] for param_dicts=[block].
    """
    cfg = copy.deepcopy(cfg)
    no_decay_bias_norm = cfg.pop("no_decay_bias_norm", False)
    if no_decay_bias_norm and params is None and param_dicts is None:
        param_dicts = []  # force the grouped path so there is a group to split
    if params is not None:
        # Explicit parameter list (e.g. one probe head's own params in
        # GridProbeTrainer) takes precedence over model.parameters()/param_dicts.
        cfg.params = params
    elif param_dicts is None:
        cfg.params = model.parameters()
    else:
        cfg.params = [dict(names=[], params=[], lr=cfg.lr)]
        for i in range(len(param_dicts)):
            param_group = dict(names=[], params=[])
            if "lr" in param_dicts[i].keys():
                param_group["lr"] = param_dicts[i].lr
            if "momentum" in param_dicts[i].keys():
                param_group["momentum"] = param_dicts[i].momentum
            if "weight_decay" in param_dicts[i].keys():
                param_group["weight_decay"] = param_dicts[i].weight_decay
            if "force_adamw" in param_dicts[i].keys():
                param_group["force_adamw"] = param_dicts[i].force_adamw
            cfg.params.append(param_group)

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            flag = False
            for i in range(len(param_dicts)):
                if param_dicts[i].keyword in n:
                    cfg.params[i + 1]["names"].append(n)
                    cfg.params[i + 1]["params"].append(p)
                    flag = True
                    break
            if not flag:
                cfg.params[0]["names"].append(n)
                cfg.params[0]["params"].append(p)

        if no_decay_bias_norm:
            decay_groups = list(cfg.params)
            for group in decay_groups:
                kept = [
                    (n, p)
                    for n, p in zip(group["names"], group["params"])
                    if not _is_no_decay_param(n, p)
                ]
                dropped = [
                    (n, p)
                    for n, p in zip(group["names"], group["params"])
                    if _is_no_decay_param(n, p)
                ]
                group["names"] = [n for n, _ in kept]
                group["params"] = [p for _, p in kept]
                cfg.params.append(
                    {
                        **{k: v for k, v in group.items() if k not in ("names", "params")},
                        "weight_decay": 0.0,
                        "names": [n for n, _ in dropped],
                        "params": [p for _, p in dropped],
                    }
                )

        logger = get_root_logger()
        for i in range(len(cfg.params)):
            param_names = cfg.params[i].pop("names")
            message = ""
            for key in cfg.params[i].keys():
                if key != "params":
                    message += f" {key}: {cfg.params[i][key]};"
            logger.info(f"Params Group {i+1} -{message} Params: {param_names}.")
    return OPTIMIZERS.build(cfg=cfg)
