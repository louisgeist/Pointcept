"""
Misc Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
import glob
import json
import os
import math
import re
import shutil
import subprocess
import time
import gc
import wandb
import torch
import torch.utils.data
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

if sys.version_info >= (3, 10):
    from collections.abc import Sequence
else:
    from collections import Sequence
from pointcept.utils.epoch_timing import (
    begin_epoch_wall_timing,
    finalize_epoch_wall_timing,
)
from pointcept.utils.timer import Timer
from pointcept.utils.comm import is_main_process, synchronize
from pointcept.utils.network_apls import run_network_apls_eval_if_configured
from pointcept.utils.cache import shared_dict
from pointcept.utils.scheduler import CosineScheduler
import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu
from pointcept.utils.wandb_metrics import class_name_slug, iou_class_tag, metric_tag
from pointcept.utils.multilabel_metrics import (
    MultilabelStats,
    accumulate_multilabel_stats,
    compute_multilabel_metrics,
)

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        self._start_time = time.perf_counter()
        _remain_epoch = self.trainer.max_epoch - self.trainer.start_epoch
        self._remain_iter = _remain_epoch * len(self.trainer.train_loader)

    def before_epoch(self):
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self, log_interval=1, wandb_log_every_step=False, write_cls_iou=True):
        """
        Args:
            log_interval: Log to console/file every N steps (default 1).
                          TensorBoard/W&B are still updated every step.
            wandb_log_every_step: If True, log train_batch/* and params/lr to wandb
                                  every step. If False (default), log only at epoch
                                  level to keep .wandb small.
            write_cls_iou: If True, log per-class train IoU to TensorBoard/W&B each
                           epoch (rank-0 minibatches only; same scope as train/mIoU).
        """
        self.curr_iter = 0
        self.model_output_keys = []
        self.log_interval = log_interval
        self.wandb_log_every_step = wandb_log_every_step
        self.write_cls_iou = write_cls_iou
        # task_name -> running (intersection, union) tensors on CUDA
        self._train_seg_stats = {}
        # task_name -> MultilabelStats (epoch accumulation, rank 0 only)
        self._train_multilabel_stats = {}

    def before_epoch(self):
        self._train_seg_stats = {}
        self._train_multilabel_stats = {}
        self._epoch_scalar_keys = []
        self._epoch_scalar_key_set = set()
        self._train_epoch_start = begin_epoch_wall_timing()

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        self.curr_iter = self.trainer.start_epoch * len(self.trainer.train_loader)

    @staticmethod
    def _cfg_get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _semantic_task_specs(self, data_cfg):
        """Returns list of (task_name, task_config) for semantic multitask configs."""
        tc = self._cfg_get(data_cfg, "task_configs", None)
        if not isinstance(tc, dict) or len(tc) == 0:
            return []
        specs = []
        for name, cfg in tc.items():
            if not isinstance(cfg, dict):
                continue
            if cfg.get("task_type") != "semantic":
                continue
            specs.append((str(name), cfg))
        return specs

    def _multilabel_task_specs(self, data_cfg):
        """Returns list of (task_name, task_config) for multilabel classification tasks."""
        tc = self._cfg_get(data_cfg, "task_configs", None)
        if not isinstance(tc, dict) or len(tc) == 0:
            return []
        specs = []
        for name, cfg in tc.items():
            if not isinstance(cfg, dict):
                continue
            if cfg.get("task_type") != "multilabel_classification":
                continue
            specs.append((str(name), cfg))
        return specs

    def _accumulate_seg_batch(self, pred, target, num_classes, ignore_index, task_name):
        area_intersection, area_union, _ = intersection_and_union_gpu(
            pred.clone(), target, int(num_classes), ignore_index=int(ignore_index)
        )
        stats = self._train_seg_stats.get(task_name)
        if stats is None:
            self._train_seg_stats[task_name] = {
                "intersection": area_intersection.detach(),
                "union": area_union.detach(),
            }
        else:
            stats["intersection"] += area_intersection.detach()
            stats["union"] += area_union.detach()

    def _epoch_miou_for_task(self, task_name):
        if task_name not in self._train_seg_stats:
            return None
        intersection = self._train_seg_stats[task_name]["intersection"]
        union = self._train_seg_stats[task_name]["union"]
        valid = union > 0
        if valid.any():
            return (intersection[valid] / (union[valid] + 1e-10)).mean().item()
        return 0.0

    def _train_per_cls_iou_items(self, data_cfg, tc, use_task_in_tag):
        """Yields (tag, value) for each logged train class-IoU (epoch-level, rank 0).

        TensorBoard and W&B share the same tag path.
        """
        for task_name, stats in self._train_seg_stats.items():
            intersection = stats["intersection"].cpu().numpy()
            union = stats["union"].cpu().numpy()
            iou_class = intersection / (union + 1e-10)
            task_conf = tc.get(task_name) if isinstance(tc, dict) else None
            if isinstance(task_conf, dict):
                num_classes = int(task_conf["num_classes"])
                ignore_index = int(task_conf["ignore_index"])
                names = list(task_conf["names"])
                task_for_tag = task_name if use_task_in_tag else None
            else:
                num_classes = int(self._cfg_get(data_cfg, "num_classes"))
                ignore_index = int(self._cfg_get(data_cfg, "ignore_index", -1))
                names = list(self._cfg_get(data_cfg, "names"))
                task_for_tag = None
            for class_idx in range(num_classes):
                if class_idx == ignore_index:
                    continue
                slug = class_name_slug(names[class_idx])
                yield iou_class_tag("train", slug, task=task_for_tag), float(
                    iou_class[class_idx]
                )

    @staticmethod
    def _optimizer_lrs(trainer):
        """{probe_name: lr} for dict-valued trainer.optimizer (GridProbeTrainer,
        one entry per probe), or {"": lr} for a single optimizer (every other
        trainer)."""
        opt = trainer.optimizer
        if isinstance(opt, dict):
            return {
                name: o.state_dict()["param_groups"][0]["lr"] for name, o in opt.items()
            }
        return {"": opt.state_dict()["param_groups"][0]["lr"]}

    @staticmethod
    def _skip_console_scalar(key):
        key = str(key)
        return key.startswith("gradient/") or key.startswith("loss_scale/")

    _GLOBAL_GRAD_STEP_KEYS = frozenset(
        {"gradient/global", "gradient/weight_update"}
    )

    @classmethod
    def _include_in_train_batch(cls, key, log_task_gradient_norms):
        """Whether key should be written to per-step train_batch (TB/W&B).

        gradient/global and gradient/weight_update are always put_scalar'd for
        epoch averages, but only appear in train_batch when
        log_task_gradient_norms is True.
        """
        if key in cls._GLOBAL_GRAD_STEP_KEYS:
            return bool(log_task_gradient_norms)
        return True

    @classmethod
    def _wandb_step_keys(
        cls, model_output_keys, log_all_steps, log_task_grads, log_lite
    ):
        """Select keys for per-step W&B train_batch logging."""
        if log_all_steps:
            return [
                k
                for k in model_output_keys
                if cls._include_in_train_batch(k, log_task_grads)
            ]
        step_keys = []
        for key in model_output_keys:
            if key in cls._GLOBAL_GRAD_STEP_KEYS:
                if log_task_grads:
                    step_keys.append(key)
            elif key.startswith("gradient/last_layer/") or key.startswith(
                "loss_scale/"
            ):
                if log_lite or log_task_grads:
                    step_keys.append(key)
            elif key.startswith("gradient/"):
                if log_task_grads:
                    step_keys.append(key)
        return step_keys

    def before_step(self):
        self.curr_iter += 1
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            scalar_keys = []
            for key, value in model_output_dict.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        continue
                    value = value.item()
                if not isinstance(value, (int, float)):
                    continue
                self.trainer.storage.put_scalar(key, float(value))
                scalar_keys.append(key)
                
            loss_by_task = model_output_dict.get("loss_by_task")
            # Skip when only one task: scalar "loss" already summarizes training.
            if isinstance(loss_by_task, dict) and len(loss_by_task) > 1: # For logging
                for task_name, v in loss_by_task.items():
                    if isinstance(v, torch.Tensor):
                        if v.numel() != 1:
                            continue
                        v = v.item()
                    if not isinstance(v, (int, float)):
                        continue
                    subkey = f"loss/{task_name}"
                    self.trainer.storage.put_scalar(subkey, float(v))
                    scalar_keys.append(subkey)

            task_gradient_norms = self.trainer.comm_info.get("task_gradient_norms")
            if isinstance(task_gradient_norms, dict):
                norms_by_task = task_gradient_norms.get("norms")
                if norms_by_task is None:
                    norms_by_task = {
                        task_name: value
                        for task_name, value in task_gradient_norms.items()
                        if isinstance(value, dict) and "backbone" in value
                    }
                for task_name, norms in norms_by_task.items():
                    if not isinstance(norms, dict):
                        continue
                    for part in ("backbone", "head"):
                        value = norms.get(part)
                        if value is None:
                            continue
                        subkey = f"gradient/{part}/{task_name}"
                        self.trainer.storage.put_scalar(subkey, float(value))
                        scalar_keys.append(subkey)

                backbone_cos = task_gradient_norms.get("backbone_cos", {})
                if isinstance(backbone_cos, dict):
                    for pair_key, value in backbone_cos.items():
                        if value is None:
                            continue
                        subkey = f"gradient/backbone_cos/{pair_key}"
                        self.trainer.storage.put_scalar(subkey, float(value))
                        scalar_keys.append(subkey)

            grad_norm_lite = self.trainer.comm_info.get("grad_norm_lite")
            if isinstance(grad_norm_lite, dict):
                last_layer_norms = grad_norm_lite.get("last_layer_norms")
                # Log last_layer + loss_scale only on EMA update steps (interval).
                if isinstance(last_layer_norms, dict):
                    for task_name, value in last_layer_norms.items():
                        if value is None:
                            continue
                        value = float(value)
                        if not math.isfinite(value):
                            continue
                        subkey = f"gradient/last_layer/{task_name}"
                        self.trainer.storage.put_scalar(subkey, value)
                        scalar_keys.append(subkey)
                    loss_scales = grad_norm_lite.get("loss_scales")
                    if isinstance(loss_scales, dict):
                        for task_name, value in loss_scales.items():
                            if value is None:
                                continue
                            value = float(value)
                            if not math.isfinite(value):
                                continue
                            subkey = f"loss_scale/{task_name}"
                            self.trainer.storage.put_scalar(subkey, value)
                            scalar_keys.append(subkey)

            global_diag = self.trainer.comm_info.get("global_gradient_diag")
            if isinstance(global_diag, dict):
                for key, value in global_diag.items():
                    if value is None:
                        continue
                    self.trainer.storage.put_scalar(key, float(value))
                    scalar_keys.append(key)

            self.model_output_keys = scalar_keys
            if not hasattr(self, "_epoch_scalar_key_set"):
                self._epoch_scalar_keys = []
                self._epoch_scalar_key_set = set()
            for key in scalar_keys:
                if key not in self._epoch_scalar_key_set:
                    self._epoch_scalar_key_set.add(key)
                    self._epoch_scalar_keys.append(key)

            # Accumulate epoch-level confusion stats for segmentation (rank 0 only).
            if comm.is_main_process() and "input_dict" in self.trainer.comm_info:
                input_dict = self.trainer.comm_info["input_dict"]
                data_cfg = self.trainer.cfg.data
                pred_by_task = model_output_dict.get("pred_by_task")
                with torch.no_grad():
                    multitask_specs = self._semantic_task_specs(data_cfg)
                    if isinstance(pred_by_task, dict) and pred_by_task and multitask_specs:
                        for task_name, task_conf in multitask_specs:
                            if task_name not in pred_by_task or task_name not in input_dict:
                                continue
                            self._accumulate_seg_batch(
                                pred_by_task[task_name],
                                input_dict[task_name],
                                int(task_conf["num_classes"]),
                                int(task_conf["ignore_index"]),
                                task_name,
                            )
                    elif (
                        "pred" in model_output_dict
                        and "segment" in input_dict
                    ):
                        pred = model_output_dict["pred"]
                        target = input_dict["segment"]
                        num_classes = int(
                            self._cfg_get(data_cfg, "num_classes")
                        )
                        model = (
                            self.trainer.model.module
                            if hasattr(self.trainer.model, "module")
                            else self.trainer.model
                        )
                        if hasattr(model, "get_ignore_index"):
                            ignore_index = model.get_ignore_index()
                        else:
                            ignore_index = int(
                                self._cfg_get(data_cfg, "ignore_index", -1)
                            )
                        main_key = self._cfg_get(data_cfg, "main_task", None) or "segment"
                        self._accumulate_seg_batch(
                            pred, target, num_classes, ignore_index, main_key
                        )
                    elif (
                        "pred" in model_output_dict
                        and "category" in input_dict # classification
                    ):
                        pred = model_output_dict["pred"]
                        target = input_dict["category"].view(-1)
                        num_classes = int(
                            self._cfg_get(data_cfg, "num_classes")
                        )
                        ignore_index = int(
                            self._cfg_get(data_cfg, "ignore_index", -1)
                        )
                        main_key = self._cfg_get(data_cfg, "main_task", None) or "segment"
                        self._accumulate_seg_batch(
                            pred, target, num_classes, ignore_index, main_key
                        )

                    pred_cls_by_task = model_output_dict.get("pred_cls_by_task")
                    if isinstance(pred_cls_by_task, dict) and pred_cls_by_task:
                        for task_name, _task_conf in self._multilabel_task_specs(
                            data_cfg
                        ):
                            if (
                                task_name not in pred_cls_by_task
                                or task_name not in input_dict
                            ):
                                continue
                            pred = pred_cls_by_task[task_name].detach().cpu().numpy()
                            target_tensor = input_dict[task_name].float()
                            if target_tensor.ndim == 1:
                                target_tensor = target_tensor.unsqueeze(0)
                            target = target_tensor.detach().cpu().numpy()
                            stats = self._train_multilabel_stats.get(task_name)
                            if stats is None:
                                stats = MultilabelStats()
                                self._train_multilabel_stats[task_name] = stats
                            accumulate_multilabel_stats(
                                pred,
                                target,
                                stats,
                                ignore_index=_task_conf.get("ignore_index"),
                            )

        for key in self.model_output_keys:
            if self._skip_console_scalar(key):
                continue
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lrs = self._optimizer_lrs(self.trainer)
        if list(lrs.keys()) == [""]:
            self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lrs[""])
        else:
            self.trainer.comm_info["iter_info"] += " ".join(
                f"Lr/{name}: {lr:.5f}" for name, lr in lrs.items()
            )
        if self.curr_iter % self.log_interval == 0 and comm.is_main_process():
            self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            log_task_grads = getattr(
                self.trainer.cfg, "log_task_gradient_norms", False
            )
            log_lite = getattr(self.trainer.cfg, "grad_norm_lite", False)
            for name, lr in lrs.items():
                tag = "params/lr" if name == "" else f"params/lr/{name}"
                self.trainer.writer.add_scalar(tag, lr, self.curr_iter)
            for key in self.model_output_keys:
                if not self._include_in_train_batch(key, log_task_grads):
                    continue
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.curr_iter,
                )
            if self.trainer.cfg.enable_wandb:
                log_all_steps = self.wandb_log_every_step
                if log_all_steps or log_task_grads or log_lite:
                    wandb_payload = {"Iter": self.curr_iter}
                    if log_all_steps:
                        for name, lr in lrs.items():
                            key = "params/lr" if name == "" else f"params/lr/{name}"
                            wandb_payload[key] = lr
                    step_keys = self._wandb_step_keys(
                        self.model_output_keys,
                        log_all_steps,
                        log_task_grads,
                        log_lite,
                    )
                    for key in step_keys:
                        wandb_payload[f"train_batch/{key}"] = (
                            self.trainer.storage.history(key).val
                        )
                    if len(wandb_payload) > 1:
                        wandb.log(wandb_payload)

    def after_epoch(self):
        # Epoch-level metrics (rank 0 only): train/loss and train/mIoU are both
        # computed from rank 0's batches; not aggregated across GPUs.
        # train/mIoU uses subsampled voxel labels (no inverse remap); val/mIoU uses
        # full-resolution labels after inverse remap (see SemSegEvaluator).
        data_cfg = self.trainer.cfg.data
        main_task = self._cfg_get(data_cfg, "main_task", None) or "segment"
        epoch_miou_main = self._epoch_miou_for_task(main_task)
        train_miou_by_task = {
            task_name: self._epoch_miou_for_task(task_name)
            for task_name in self._train_seg_stats
        }

        epoch_info = "Train result: "
        epoch_keys = getattr(self, "_epoch_scalar_keys", None) or self.model_output_keys
        for key in epoch_keys:
            if self._skip_console_scalar(key):
                continue
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        if epoch_miou_main is not None:
            epoch_info += "mIoU: {:.4f} ".format(epoch_miou_main)
        for task_name, task_miou in train_miou_by_task.items():
            if task_name != main_task and task_miou is not None:
                epoch_info += f"{task_name}_mIoU: {task_miou:.4f} "
        tc = self._cfg_get(data_cfg, "task_configs", None)
        train_multilabel_macro_f1 = {}
        if comm.is_main_process() and self._train_multilabel_stats:
            for task_name, stats in self._train_multilabel_stats.items():
                task_conf = tc.get(task_name) if isinstance(tc, dict) else None
                names = (
                    list(task_conf["names"])
                    if isinstance(task_conf, dict)
                    else []
                )
                metrics = compute_multilabel_metrics(stats, names)
                train_multilabel_macro_f1[task_name] = float(metrics["macro_f1"])
                epoch_info += f"{task_name}_macro_f1: {metrics['macro_f1']:.4f} "
        self.trainer.logger.info(epoch_info)

        use_task_in_tag = isinstance(tc, dict) and len(tc) > 0
        epoch_step = self.trainer.epoch + 1
        wandb_dict = None

        if comm.is_main_process() and self.trainer.writer is not None:
            for key in epoch_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )
            if epoch_miou_main is not None:
                self.trainer.writer.add_scalar(
                    "train/mIoU", float(epoch_miou_main), self.trainer.epoch + 1
                )
            for task_name, task_miou in train_miou_by_task.items():
                if task_miou is not None:
                    self.trainer.writer.add_scalar(
                        metric_tag("train", "mIoU", task=task_name),
                        float(task_miou),
                        self.trainer.epoch + 1,
                    )
            for task_name, macro_f1 in train_multilabel_macro_f1.items():
                self.trainer.writer.add_scalar(
                    metric_tag("train", "macro_f1", task=task_name),
                    macro_f1,
                    self.trainer.epoch + 1,
                )

            if self.write_cls_iou:
                for tag, value in self._train_per_cls_iou_items(
                    data_cfg, tc, use_task_in_tag
                ):
                    self.trainer.writer.add_scalar(tag, value, epoch_step)

        if (
            comm.is_main_process()
            and self.trainer.cfg.enable_wandb
            and wandb.run is not None
        ):
            lrs = self._optimizer_lrs(self.trainer)
            wandb_dict = {
                "Epoch": epoch_step,
                "Iter": self.curr_iter,
            }
            for name, lr in lrs.items():
                key = "params/lr" if name == "" else f"params/lr/{name}"
                wandb_dict[key] = lr
            for key in epoch_keys:
                wandb_dict[f"train/{key}"] = self.trainer.storage.history(key).avg
            if epoch_miou_main is not None:
                wandb_dict["train/mIoU"] = float(epoch_miou_main)
            for task_name, task_miou in train_miou_by_task.items():
                if task_miou is not None:
                    wandb_dict[metric_tag("train", "mIoU", task=task_name)] = float(
                        task_miou
                    )
            for task_name, macro_f1 in train_multilabel_macro_f1.items():
                wandb_dict[metric_tag("train", "macro_f1", task=task_name)] = macro_f1
            if self.write_cls_iou:
                for tag, value in self._train_per_cls_iou_items(
                    data_cfg, tc, use_task_in_tag
                ):
                    wandb_dict[tag] = value

        finalize_epoch_wall_timing(
            self.trainer,
            self._train_epoch_start,
            epoch_step,
            "train",
            wandb_dict=wandb_dict,
        )
        if wandb_dict is not None:
            # Do not pass an explicit `step`: after a Slurm requeue the W&B
            # run is resumed and its internal step is already past the epoch
            # number, so `step=epoch_step` would be silently dropped.
            # Charts use the "Epoch" field as x-axis via define_metric.
            wandb.log(wandb_dict)


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None or int, None indicate only save model last

    def after_epoch(self):
        if is_main_process():
            is_best = False
            if self.should_evaluate():
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )

            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            # Per-hook running state (e.g. MultiTaskEvaluator's per-task best-so-far
            # trackers) that isn't captured by trainer.best_metric_value alone — see
            # HookBase.state_dict / CheckpointLoader.before_train.
            hook_states = {}
            for h in self.trainer.hooks:
                s = h.state_dict()
                if s:
                    hook_states[h.__class__.__name__] = s
            if isinstance(self.trainer.optimizer, dict):
                # GridProbeTrainer: one optimizer/scheduler per probe head.
                optimizer_state = {
                    name: o.state_dict() for name, o in self.trainer.optimizer.items()
                }
                scheduler_state = {
                    name: s.state_dict() for name, s in self.trainer.scheduler.items()
                }
            else:
                optimizer_state = self.trainer.optimizer.state_dict()
                scheduler_state = self.trainer.scheduler.state_dict()
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": optimizer_state,
                    "scheduler": scheduler_state,
                    "scaler": (
                        self.trainer.scaler.state_dict()
                        if self.trainer.cfg.enable_amp
                        else None
                    ),
                    "best_metric_value": self.trainer.best_metric_value,
                    "hook_states": hook_states,
                },
                filename + ".tmp",
            )
            os.replace(filename + ".tmp", filename)
            if is_best:
                shutil.copyfile(
                    filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )
            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"epoch_{self.trainer.epoch + 1}.pth",
                    ),
                )
            from pointcept.utils.runtime_state import flush_runtime_segment

            flush_runtime_segment(self.trainer.cfg.save_path)


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False, exclude_keys=None):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict
        # Exclude checkpoint keys by substring match before loading.
        # Example: exclude_keys=("seg_head",)
        self.exclude_keys = tuple(exclude_keys) if exclude_keys is not None else tuple()

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            if self.exclude_keys:
                self.trainer.logger.info(
                    f"Excluding checkpoint keys containing: {self.exclude_keys}"
                )
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if self.exclude_keys and any(k in key for k in self.exclude_keys):
                    continue # the key is not stored in `weight`
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement, 1)
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                if isinstance(self.trainer.optimizer, dict):
                    # GridProbeTrainer: one optimizer/scheduler per probe head.
                    for name, o in self.trainer.optimizer.items():
                        o.load_state_dict(checkpoint["optimizer"][name])
                    for name, s in self.trainer.scheduler.items():
                        s.load_state_dict(checkpoint["scheduler"][name])
                else:
                    self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                    self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
                # Restore per-hook running state (older checkpoints predate this key,
                # hence .get) — see HookBase.state_dict / CheckpointSaver.after_epoch.
                hook_states = checkpoint.get("hook_states", {})
                for h in self.trainer.hooks:
                    state = hook_states.get(h.__class__.__name__)
                    if state:
                        h.load_state_dict(state)
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")


@HOOKS.register_module()
class PreciseEvaluator(HookBase):
    def __init__(self, test_last=False):
        self.test_last = test_last

    def after_train(self):
        from pointcept.engines.test import TESTERS

        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Precise Evaluation >>>>>>>>>>>>>>>>"
        )
        if getattr(self.trainer, "optimizer", None) is not None:
            del self.trainer.optimizer
            self.trainer.optimizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        test_cfg = dict(cfg=cfg, model=self.trainer.model, **cfg.test)
        tester = TESTERS.build(test_cfg)
        if self.test_last:
            self.trainer.logger.info("=> Testing on model_last ...")
        else:
            self.trainer.logger.info("=> Testing on model_best ...")
            best_path = os.path.join(
                self.trainer.cfg.save_path, "model", "model_best.pth"
            )
            checkpoint = torch.load(best_path, weights_only=False)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value
            tester.model.load_state_dict(weight, strict=True)
        tester.test()


@HOOKS.register_module()
class NetworkAPLSEvaluator(HookBase):
    """After training, score the network-prediction task with APLS (see
    tools/eval_network_apls.py). Opt-in via cfg.network_apls_eval; no-op otherwise.

    Reuses the `{patch_id}_logits_network.npy` rasters written by `PreciseEvaluator`'s
    in-process test run -- must be listed *after* `PreciseEvaluator` in `cfg.hooks` so those
    files already exist on disk by the time this hook's `after_train` runs.
    """

    def after_train(self):
        t0 = time.time()
        payload = run_network_apls_eval_if_configured(self.trainer.cfg, self.trainer.logger)
        elapsed_s = time.time() - t0
        self.trainer.logger.info(f"Network APLS eval took {elapsed_s:.1f}s")

        current_epoch = self.trainer.epoch + 1
        tag = metric_tag("test", "network_apls_eval_time_s")
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(tag, elapsed_s, current_epoch)
        if self.trainer.cfg.enable_wandb:
            wandb.log({"Epoch": current_epoch, tag: elapsed_s})

        if payload is None:
            return

        scalars = {}
        macro_apls = float(payload["macro_apls"])
        if math.isfinite(macro_apls):
            scalars[metric_tag("test", "APLS", task="network")] = macro_apls
        for ch_name, value in payload["per_channel"].items():
            value = float(value)
            if not math.isfinite(value):
                continue
            ch_slug = class_name_slug(ch_name)
            scalars[metric_tag("test", f"APLS/{ch_slug}", task="network")] = value

        # Unidirectional halves (already aggregated in the APLS payload).
        for direction, key in (
            ("gt_to_pred", "per_channel_gt_to_pred"),
            ("pred_to_gt", "per_channel_pred_to_gt"),
        ):
            per_ch = payload.get(key) or {}
            finite_vals = []
            for ch_name, value in per_ch.items():
                value = float(value)
                if not math.isfinite(value):
                    continue
                finite_vals.append(value)
                ch_slug = class_name_slug(ch_name)
                scalars[
                    metric_tag("test", f"APLS_{direction}/{ch_slug}", task="network")
                ] = value
            if finite_vals:
                scalars[metric_tag("test", f"APLS_{direction}", task="network")] = (
                    sum(finite_vals) / len(finite_vals)
                )

        if self.trainer.writer is not None:
            for k, v in scalars.items():
                self.trainer.writer.add_scalar(k, v, current_epoch)
        if self.trainer.cfg.enable_wandb and scalars:
            wandb.log({"Epoch": current_epoch, **scalars})


@HOOKS.register_module()
class DataCacheOperator(HookBase):
    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        self.data_list = self.get_data_list()

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split))
        else:
            raise NotImplementedError
        return data_list

    def get_cache_name(self, data_path):
        data_name = data_path.replace(os.path.dirname(self.data_root), "")
        return "pointcept" + data_name.replace(os.path.sep, "-")

    def before_train(self):
        self.trainer.logger.info(
            f"=> Caching dataset: {self.data_root}, split: {self.split} ..."
        )
        if is_main_process():
            dataset = self.trainer.train_loader.dataset
            for i in range(len(dataset)):
                data_dict = dataset[i]
                name = data_dict["name"]
                shared_dict(f"Pointcept-{name}", data_dict)
        synchronize()


@HOOKS.register_module()
class RuntimeProfiler(HookBase):
    def __init__(
        self,
        forward=True,
        backward=True,
        interrupt=False,
        warm_up=2,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.forward = forward
        self.backward = backward
        self.interrupt = interrupt
        self.warm_up = warm_up
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import profile, record_function, ProfilerActivity

        for i, input_dict in enumerate(self.trainer.train_loader):
            if i == self.warm_up + 1:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            if self.forward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as forward_prof:
                    with record_function("model_inference"):
                        output_dict = self.trainer.model(input_dict)
            else:
                output_dict = self.trainer.model(input_dict)
            loss = output_dict["loss"]
            if self.backward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as backward_prof:
                    with record_function("model_inference"):
                        loss.backward()
            self.trainer.logger.info(f"Profile: [{i + 1}/{self.warm_up + 1}]")
        if self.forward:
            self.trainer.logger.info(
                "Forward profile: \n"
                + str(
                    forward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            forward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "forward_trace.json")
            )

        if self.backward:
            self.trainer.logger.info(
                "Backward profile: \n"
                + str(
                    backward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            backward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "backward_trace.json")
            )
        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class RuntimeProfilerV2(HookBase):
    def __init__(
        self,
        interrupt=False,
        wait=1,
        warmup=1,
        active=10,
        repeat=1,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.interrupt = interrupt
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import (
            profile,
            record_function,
            ProfilerActivity,
            schedule,
            tensorboard_trace_handler,
        )

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=tensorboard_trace_handler(self.trainer.cfg.save_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
        for i, input_dict in enumerate(self.trainer.train_loader):
            if i >= (self.wait + self.warmup + self.active) * self.repeat:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with record_function("model_forward"):
                output_dict = self.trainer.model(input_dict)
                loss = output_dict["loss"]
            with record_function("model_backward"):
                loss.backward()
            prof.step()
            self.trainer.logger.info(
                f"Profile: [{i + 1}/{(self.wait + self.warmup + self.active) * self.repeat}]"
            )
        self.trainer.logger.info(
            "Profile: \n"
            + str(
                prof.key_averages().table(
                    sort_by=self.sort_by, row_limit=self.row_limit
                )
            )
        )
        prof.stop()

        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class WeightDecaySchedular(HookBase):
    def __init__(
        self,
        base_value=0.04,
        final_value=0.2,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.scheduler = None

    def before_train(self):
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        self.scheduler = CosineScheduler(
            base_value=self.base_value,
            final_value=self.final_value,
            total_iters=self.trainer.cfg.scheduler.total_steps,
        )
        self.scheduler.iter = curr_step

    def before_step(self):
        wd = self.scheduler.step()
        for param_group in self.trainer.optimizer.param_groups:
            param_group["weight_decay"] = wd
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/wd", wd, self.scheduler.iter)


@HOOKS.register_module()
class GarbageHandler(HookBase):
    def __init__(self, interval=150, disable_auto=True, empty_cache=False):
        self.interval = interval
        self.disable_auto = disable_auto
        self.empty_cache = empty_cache
        self.iter = 1

    def before_train(self):
        if self.disable_auto:
            gc.disable()
            self.trainer.logger.info("Disable automatic garbage collection")

    def before_epoch(self):
        self.iter = 1

    def after_step(self):
        if self.iter % self.interval == 0:
            gc.collect()
            if self.empty_cache:
                torch.cuda.empty_cache()
            self.trainer.logger.info("Garbage collected")
        self.iter += 1

    def after_train(self):
        gc.collect()
        torch.cuda.empty_cache()


@HOOKS.register_module()
class MetricsJsonWriter(HookBase):
    """Write best validation metrics to ``save_path/metrics.json`` after training.

    Intended for evaluate=True runs (e.g. linear probes). The periodic Sonata
    probe watcher reads this file as the primary mIoU source.
    """

    def __init__(self, filename="metrics.json"):
        self.filename = filename

    def after_train(self):
        if not is_main_process():
            return
        if not getattr(self.trainer.cfg, "evaluate", False):
            return

        metric_name = self.trainer.comm_info.get("current_metric_name", "mIoU")
        best_value = float(self.trainer.best_metric_value)
        payload = {
            "metric_name": str(metric_name),
            "best_metric_value": best_value,
            "epoch": int(self.trainer.epoch + 1),
        }
        if str(metric_name) == "mIoU":
            payload["best_val_mIoU"] = best_value

        out_path = os.path.join(self.trainer.cfg.save_path, self.filename)
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, out_path)
        self.trainer.logger.info("Wrote metrics to: %s", out_path)


@HOOKS.register_module()
class LinProbeSbatchHook(HookBase):
    """Submit a Slurm linear-probe job after each periodic ``epoch_N.pth`` save.

    Non-blocking: failures to submit are logged and training continues.
    Place this hook **after** ``CheckpointSaver`` so the epoch checkpoint exists.
    Outside Slurm (no ``sbatch``), the hook is a no-op after a one-time warning.
    """

    def __init__(
        self,
        enable=True,
        save_freq=5,
        sbatch_script="scripts/sonata/sbatch_lin_probe.sh",
        iter_per_epoch=1000,
        state_filename="lin_probe_state.json",
    ):
        self.enable = bool(enable)
        self.save_freq = int(save_freq) if save_freq else 0
        self.sbatch_script = sbatch_script
        self.iter_per_epoch = int(iter_per_epoch)
        self.state_filename = state_filename
        self._warned_no_sbatch = False

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _state_path(self):
        return os.path.join(self.trainer.cfg.save_path, self.state_filename)

    def _load_state(self):
        path = self._state_path()
        if not os.path.isfile(path):
            return {"completed": {}, "in_flight": {}}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {"completed": {}, "in_flight": {}}
        data.setdefault("completed", {})
        data.setdefault("in_flight", {})
        return data

    def _save_state(self, state):
        path = self._state_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)

    def _resolve_sbatch_script(self):
        script = Path(self.sbatch_script)
        if script.is_file():
            return script.resolve()
        # Prefer repo root (cwd is usually the Pointcept root under train.sh).
        cand = Path.cwd() / self.sbatch_script
        if cand.is_file():
            return cand.resolve()
        return None

    def after_epoch(self):
        if not is_main_process():
            return
        if not self.enable or self.save_freq <= 0:
            return

        epoch_1based = int(self.trainer.epoch) + 1
        if epoch_1based % self.save_freq != 0:
            return

        save_path = Path(self.trainer.cfg.save_path).resolve()
        ckpt_name = f"epoch_{epoch_1based}.pth"
        ckpt_path = save_path / "model" / ckpt_name
        if not ckpt_path.is_file():
            self.trainer.logger.warning(
                "LinProbeSbatchHook: expected checkpoint missing: %s", ckpt_path
            )
            return

        state = self._load_state()
        if ckpt_name in state["completed"] or ckpt_name in state["in_flight"]:
            self.trainer.logger.info(
                "LinProbeSbatchHook: skip %s (already submitted/completed)", ckpt_name
            )
            return

        if shutil.which("sbatch") is None:
            if not self._warned_no_sbatch:
                self.trainer.logger.warning(
                    "LinProbeSbatchHook: sbatch not found; linear-probe submit disabled "
                    "(use scripts/sonata/periodic_lin_probe.py --mode local for offline probing)."
                )
                self._warned_no_sbatch = True
            return

        script = self._resolve_sbatch_script()
        if script is None:
            self.trainer.logger.warning(
                "LinProbeSbatchHook: sbatch script not found: %s", self.sbatch_script
            )
            return

        exp_name = f"sonata_lin_ep{epoch_1based}"
        pretrain_iters = epoch_1based * self.iter_per_epoch
        env = os.environ.copy()
        # Drop the pretrain job's wandb service token: --export=ALL would
        # otherwise leak it into the probe job, which then tries to reuse a
        # Unix socket that only lives inside the pretrain job's allocation
        # (WandbServiceConnectionError: FileNotFoundError on the stale socket).
        env.pop("WANDB_SERVICE", None)
        env["WEIGHT"] = str(ckpt_path)
        env["EXP_NAME"] = exp_name
        env["PRETRAIN_JOB_DIR"] = str(save_path)
        env["PRETRAIN_EPOCH"] = str(epoch_1based)
        env["PRETRAIN_ITERS"] = str(pretrain_iters)

        # IMAGINE compute-accounting wraps `sbatch` with a narrow argparse CLI
        # (script path only; no native Slurm flags like --comment/--export).
        # Tags come from the script's `#SBATCH --comment=...`; WEIGHT/EXP_NAME/…
        # are already in `env` and Slurm's default --export=ALL propagates them.
        cmd = ["sbatch", str(script)]
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                cwd=str(Path.cwd()),
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            self.trainer.logger.warning(
                "LinProbeSbatchHook: failed to run sbatch for %s: %s", ckpt_name, exc
            )
            return

        if proc.returncode != 0:
            self.trainer.logger.warning(
                "LinProbeSbatchHook: sbatch failed for %s (rc=%s): %s%s",
                ckpt_name,
                proc.returncode,
                proc.stdout.strip(),
                (" | " + proc.stderr.strip()) if proc.stderr else "",
            )
            return

        job_id = None
        m = re.search(r"(\d+)", (proc.stdout or "").strip())
        if m:
            job_id = m.group(1)

        state["in_flight"][ckpt_name] = {
            "job_id": job_id or "",
            "ckpt": str(ckpt_path),
            "pretrain_epoch": epoch_1based,
            "pretrain_iters": pretrain_iters,
            "status": "submitted",
            "submitted_at": self._utc_now(),
            "exp_name": exp_name,
        }
        self._save_state(state)
        self.trainer.logger.info(
            "LinProbeSbatchHook: submitted %s -> job_id=%s exp=%s",
            ckpt_name,
            job_id,
            exp_name,
        )
