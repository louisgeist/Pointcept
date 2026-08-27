"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import copy
import math
import os
import sys
import weakref
import time
import wandb
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
from packaging import version
from functools import partial
from pathlib import Path
import itertools

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.datasets.utils import build_voxel_budget_batch_sampler
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.config import ConfigDict
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.wandb_metrics import define_wandb_metrics
from pointcept.utils.wandb_resume import bump_wandb_step_on_resume, read_local_last_step
from pointcept.utils.gradient_norm import (
    GradNormLiteEMA,
    all_reduce_mean_task_norms,
    build_grad_norm_state,
    combine_weighted_task_losses,
    compute_task_gradient_norms,
    compute_task_last_layer_grad_norms,
    group_task_losses,
    resolve_grad_norm_lite_scales,
    l2_model_grad_norm,
    l2_model_update_norm,
    snapshot_trainable_params,
)
from pointcept.utils.registry import Registry
from pointcept.datasets.dataloader import (
    DistributedImbalancedSampler,
    IterLimitedDistributedSampler,
    IterLimitedSampler,
)

TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


def _log_iter_limited_training_mode(cfg, logger):
    if getattr(cfg, "total_iters", None) is None:
        return
    message = (
        "Iter-limited training enabled: total_iters=%d optimizer steps, "
        "iter_per_epoch=%d, num_epochs=%d, eval_every=%d, loop=1. "
        "Validation runs every eval_every epochs. "
        "Classic params epoch/eval_epoch are unused."
    )
    logger.info(
        message,
        cfg.total_iters,
        cfg.iter_per_epoch,
        cfg.num_epochs,
        cfg.eval_every,
    )


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter
        # End-to-end runtime timer from trainer creation to final after_train hooks.
        self.total_runtime_start_time = time.perf_counter()

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if torch.cuda.is_available(): # synchronize GPU before measuring runtime
            torch.cuda.synchronize()
        total_runtime_seconds = time.perf_counter() - self.total_runtime_start_time
        if (
            comm.is_main_process()
            and getattr(self, "cfg", None) is not None
        ):
            from pointcept.utils.runtime_state import (
                get_total_runtime_s,
                runtime_state_path,
            )

            state_path = runtime_state_path(self.cfg.save_path)
            if os.path.isfile(state_path):
                total_runtime_seconds = get_total_runtime_s(self.cfg.save_path)
        if (
            comm.is_main_process()
            and getattr(self, "cfg", None) is not None
            and self.cfg.enable_wandb
            and wandb.run is not None
        ):
            wandb.log(
                {
                    "Epoch": self.max_epoch,
                    "runtime/total_s": float(total_runtime_seconds),
                    "runtime/total_h": float(total_runtime_seconds / 3600.0),
                }
            )
        if hasattr(self, "logger"):
            self.logger.info(
                "Total end-to-end runtime (all segments, model init -> test end): "
                f"{total_runtime_seconds:.2f}s ({total_runtime_seconds / 3600.0:.4f}h)"
            )
        if comm.is_main_process() and hasattr(self, "model"):
            model_for_debug = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            if hasattr(model_for_debug, "_print_learned_masked_feat"):
                model_for_debug._print_learned_masked_feat()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = (
            cfg.num_epochs
            if getattr(cfg, "total_iters", None) is not None
            else cfg.eval_epoch
        )
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        _log_iter_limited_training_mode(cfg, self.logger)
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)
        self._gradient_accumulation_counter = 0
        self._grad_norm_lite_ema = None
        # Real GradNorm (Chen et al. 2018): learnable per-group loss weights +
        # their Adam state + L_g(0) anchors. Built eagerly so CheckpointLoader
        # can restore into it before the first step. Mutually exclusive with
        # grad_norm_lite (enforced in default_config_parser).
        self._grad_norm_state = None
        if getattr(cfg, "grad_norm", False):
            _gn_model = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            self._grad_norm_state = build_grad_norm_state(cfg, _gn_model)

    def before_train(self):
        if comm.is_main_process():
            from pointcept.utils.runtime_state import init_runtime_segment

            init_runtime_segment(self.cfg.save_path)
        super().before_train()

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                sampler = getattr(self.train_loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        # Only clear gradients on first accumulation step
        if self._gradient_accumulation_counter == 0:
            self.optimizer.zero_grad()

        # Forward pass
        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(input_dict)
            loss = (
                output_dict["loss"] / self.cfg.gradient_accumulation_steps
            )  # scale loss

        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        loss_by_task = output_dict.get("loss_by_task")

        if getattr(self.cfg, "grad_norm_lite", False):
            if isinstance(loss_by_task, dict) and loss_by_task:
                if not hasattr(model, "last_backbone_layer_parameters"):
                    raise AttributeError(
                        "grad_norm_lite requires a model with "
                        "last_backbone_layer_parameters() (e.g. MultiTaskSegmentorV2)."
                    )
                if self._grad_norm_lite_ema is None:
                    self._grad_norm_lite_ema = GradNormLiteEMA(
                        alpha=getattr(self.cfg, "grad_norm_lite_ema_alpha", 0.1),
                        eps=getattr(self.cfg, "grad_norm_lite_eps", 1e-3),
                    )
                interval = int(getattr(self.cfg, "grad_norm_lite_interval", 100))
                lite_info = {}
                iter_idx = int(self.comm_info["iter"])
                # Scene-level losses underflow in fp16 without a probe scale.
                amp_probe_scale = float(
                    getattr(
                        self.cfg,
                        "grad_norm_amp_probe_scale",
                        1024.0 if self.cfg.enable_amp else 1.0,
                    )
                )
                # Optional task_name -> group_name map: tasks sharing a group
                # get one pooled EMA scale instead of independent ones.
                task_groups = getattr(self.cfg, "grad_norm_lite_task_groups", None)
                if iter_idx > 0 and iter_idx % interval == 0:
                    norms = compute_task_last_layer_grad_norms(
                        model,
                        loss_by_task,
                        probe_scale=amp_probe_scale,
                        task_groups=task_groups,
                    )
                    # Align with SyncBatchNorm: sync across ranks only when sync_bn.
                    if getattr(self.cfg, "sync_bn", False):
                        if task_groups:
                            task_order = sorted(
                                set(
                                    task_groups.get(t, t)
                                    for t in loss_by_task.keys()
                                )
                            )
                        else:
                            task_order = getattr(model, "tasks", None) or sorted(
                                loss_by_task.keys()
                            )
                        norms = all_reduce_mean_task_norms(
                            norms, task_names=task_order
                        )
                    self._grad_norm_lite_ema.update(norms)
                    lite_info["last_layer_norms"] = norms

                scales = resolve_grad_norm_lite_scales(
                    self._grad_norm_lite_ema, loss_by_task.keys(), task_groups
                )
                total_loss, scales = combine_weighted_task_losses(
                    loss_by_task,
                    getattr(model, "task_weights", {}),
                    scales,
                )
                loss = total_loss / self.cfg.gradient_accumulation_steps
                output_dict["loss"] = total_loss
                lite_info["loss_scales"] = scales
                self.comm_info["grad_norm_lite"] = lite_info
            else:
                self.comm_info.pop("grad_norm_lite", None)
        else:
            self.comm_info.pop("grad_norm_lite", None)

        # Real GradNorm (Chen et al. 2018): learn per-group loss weights w_g by
        # gradient descent on the GradNorm L1 objective, then reweight the main
        # loss with w_g.detach() before the normal backward.
        if getattr(self.cfg, "grad_norm", False):
            if isinstance(loss_by_task, dict) and loss_by_task:
                if not hasattr(model, "last_backbone_layer_parameters"):
                    raise AttributeError(
                        "grad_norm requires a model with "
                        "last_backbone_layer_parameters() (e.g. MultiTaskSegmentorV2)."
                    )
                if self._grad_norm_state is None:
                    self._grad_norm_state = build_grad_norm_state(self.cfg, model)
                gn_state = self._grad_norm_state
                task_groups = getattr(self.cfg, "grad_norm_task_groups", None)
                interval = int(getattr(self.cfg, "grad_norm_interval", 1))
                amp_probe_scale = float(
                    getattr(
                        self.cfg,
                        "grad_norm_amp_probe_scale",
                        1024.0 if self.cfg.enable_amp else 1.0,
                    )
                )
                iter_idx = int(self.comm_info["iter"])
                gn_info = {}
                if interval > 0 and iter_idx % interval == 0:
                    grad_norms = compute_task_last_layer_grad_norms(
                        model,
                        loss_by_task,
                        probe_scale=amp_probe_scale,
                        task_groups=task_groups,
                    )
                    loss_by_group = group_task_losses(loss_by_task, task_groups)
                    # Keep w_g bit-identical across ranks: average the scalar
                    # inputs (not gated on sync_bn -- unlike GradNormLite, the
                    # weights themselves must not diverge).
                    order = gn_state.group_names
                    grad_norms = all_reduce_mean_task_norms(
                        grad_norms, task_names=order
                    )
                    loss_by_group = all_reduce_mean_task_norms(
                        loss_by_group, task_names=order
                    )
                    gn_state.capture_initial(loss_by_group)
                    gn_info = gn_state.update(grad_norms, loss_by_group)
                    gn_info["last_layer_norms"] = grad_norms

                scales = gn_state.per_task_scales(loss_by_task.keys(), task_groups)
                total_loss, scales = combine_weighted_task_losses(
                    loss_by_task,
                    getattr(model, "task_weights", {}),
                    scales,
                )
                loss = total_loss / self.cfg.gradient_accumulation_steps
                output_dict["loss"] = total_loss
                gn_info["loss_scales"] = scales
                self.comm_info["grad_norm"] = gn_info
            else:
                self.comm_info.pop("grad_norm", None)
        else:
            self.comm_info.pop("grad_norm", None)

        if getattr(self.cfg, "log_task_gradient_norms", False):
            if isinstance(loss_by_task, dict) and loss_by_task:
                if hasattr(model, "backbone_parameters"):
                    task_weights = getattr(model, "task_weights", {})
                    amp_probe_scale = float(
                        getattr(
                            self.cfg,
                            "grad_norm_amp_probe_scale",
                            1024.0 if self.cfg.enable_amp else 1.0,
                        )
                    )
                    print(f"amp_probe_scale: {amp_probe_scale}")
                    self.comm_info["task_gradient_norms"] = compute_task_gradient_norms(
                        model,
                        loss_by_task,
                        task_weights,
                        self.cfg.gradient_accumulation_steps,
                        probe_scale=amp_probe_scale,
                    )
                else:
                    self.comm_info.pop("task_gradient_norms", None)
            else:
                self.comm_info.pop("task_gradient_norms", None)
        else:
            self.comm_info.pop("task_gradient_norms", None)

        # Backward pass
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self._gradient_accumulation_counter += 1

        # Perform optimizer step only when enough gradients have accumulated
        if self._gradient_accumulation_counter >= self.cfg.gradient_accumulation_steps:
            global_gradient_diag = {}
            if self.cfg.enable_amp:
                self.scaler.unscale_(self.optimizer)

            if self.cfg.clip_grad is not None:
                grad_l2 = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
                global_gradient_diag["gradient/global"] = float(grad_l2)
            else:
                global_gradient_diag["gradient/global"] = l2_model_grad_norm(
                    self.model
                )

            valid_gradients = True
            if getattr(self.cfg, "skip_nan_grad", False):
                valid_gradients = math.isfinite(
                    global_gradient_diag["gradient/global"]
                )
                if comm.get_world_size() > 1:
                    # MIN so any rank with bad grads forces every rank to skip
                    # (avoids DDP desync).
                    flag = torch.tensor(
                        1 if valid_gradients else 0,
                        device="cuda",
                        dtype=torch.int32,
                    )
                    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                    valid_gradients = bool(flag.item())
                if not valid_gradients:
                    if comm.is_main_process():
                        self.logger.warning(
                            "Detected inf or nan values in gradients; "
                            "skipping optimizer and scheduler step"
                        )
                    self.optimizer.zero_grad()

            if valid_gradients:
                if self.cfg.enable_amp:
                    param_snapshot = snapshot_trainable_params(self.model)
                    self.scaler.step(self.optimizer)

                    # When enable amp, optimizer.step call are skipped if the
                    # loss scaling factor is too large. Fix torch warning
                    # scheduler step before optimizer step.
                    scale = self.scaler.get_scale()
                    self.scaler.update()
                    if scale <= self.scaler.get_scale():
                        global_gradient_diag["gradient/weight_update"] = (
                            l2_model_update_norm(self.model, param_snapshot)
                        )
                        self.scheduler.step()
                else:
                    param_snapshot = snapshot_trainable_params(self.model)
                    self.optimizer.step()
                    global_gradient_diag["gradient/weight_update"] = (
                        l2_model_update_norm(self.model, param_snapshot)
                    )
                    self.scheduler.step()
            elif self.cfg.enable_amp:
                # Still advance GradScaler state after a skipped step.
                self.scaler.update()

            self.comm_info["global_gradient_diag"] = global_gradient_diag

            # Reset grad accumulation counter
            self._gradient_accumulation_counter = 0
        else:
            self.comm_info.pop("global_gradient_diag", None)

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.n_free_parameters = n_parameters
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb and comm.is_main_process():
            tag, name = Path(self.cfg.save_path).parts[-2:]
            # Allow overriding the W&B run name from config if provided
            run_name = getattr(self.cfg, "wandb_run_name", f"{tag}/{name}")
            target_keys = getattr(self.cfg, "target_keys", None)
            if target_keys:
                if isinstance(target_keys, (list, tuple)):
                    tags = [str(x) for x in target_keys]
                else:
                    tags = [str(target_keys)]
            else:
                tags = None
            wandb_run_id_path = os.path.join(self.cfg.save_path, "wandb_run_id.txt")
            run_id = None
            if os.path.isfile(wandb_run_id_path):
                with open(wandb_run_id_path, "r") as f:
                    run_id = f.read().strip() or None

            init_kw = dict(
                project=self.cfg.wandb_project,
                name=run_name,
                dir=self.cfg.save_path,
                settings=wandb.Settings(api_key=self.cfg.wandb_key),
                config=self.cfg,
            )
            if tags:
                init_kw["tags"] = tags
            # Optional run grouping (e.g. tools/grid_then_seeds.py puts the grid
            # sweep and its seed-ensemble follow-up in one group). No-op for
            # configs that don't set wandb_group.
            wandb_group = getattr(self.cfg, "wandb_group", None)
            if wandb_group:
                init_kw["group"] = str(wandb_group)
            if run_id:
                init_kw["id"] = run_id
                init_kw["resume"] = "allow"
                self.logger.info("Resuming W&B run: %s", run_id)
            wandb.init(**init_kw)
            # New-process resume (manual relaunch or Slurm requeue) can restart
            # the local _step near zero and collide with already-uploaded
            # history. Bump past lastHistoryStep before the first wandb.log.
            if run_id:
                bump_wandb_step_on_resume(
                    wandb.run,
                    project=self.cfg.wandb_project,
                    run_id=run_id,
                    logger=self.logger,
                    entity=getattr(wandb.run, "entity", None),
                    local_last_step=read_local_last_step(self.cfg.save_path),
                )
            task_configs = getattr(self.cfg.data, "task_configs", None) or {}
            task_names = []
            if isinstance(task_configs, dict):
                task_names = [str(name) for name in task_configs]
            define_wandb_metrics(task_names=task_names)
            with open(wandb_run_id_path, "w") as f:
                f.write(wandb.run.id)
            wandb.log({"model/free_params": self.n_free_parameters})
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)
        iter_per_epoch = getattr(self.cfg, "iter_per_epoch", None)
        seed = self.cfg.seed if self.cfg.seed is not None else 0

        if iter_per_epoch is not None:
            if comm.get_world_size() > 1:
                train_sampler = IterLimitedDistributedSampler(
                    train_data,
                    iter_per_epoch=iter_per_epoch,
                    batch_size_per_gpu=self.cfg.batch_size_per_gpu,
                    num_replicas=comm.get_world_size(),
                    rank=comm.get_rank(),
                    shuffle=True,
                    seed=seed,
                )
            else:
                train_sampler = IterLimitedSampler(
                    train_data,
                    iter_per_epoch=iter_per_epoch,
                    batch_size=self.cfg.batch_size_per_gpu,
                    shuffle=True,
                    seed=seed,
                )
        elif comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        drop_last = len(train_data) > self.cfg.batch_size
        if iter_per_epoch is not None:
            drop_last = True

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=drop_last,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            voxel_budget = getattr(self.cfg, "val_voxel_budget", None)
            if voxel_budget is not None:
                max_batch_size = int(self.cfg.batch_size_val_per_gpu)
                batch_sampler, sizes, missing_csv = build_voxel_budget_batch_sampler(
                    dataset=val_data,
                    voxel_budget=int(voxel_budget),
                    max_batch_size=max_batch_size,
                    rank=comm.get_rank(),
                    world_size=comm.get_world_size(),
                )
                size_source = getattr(val_data, "csv_manifest", None) or "<mmap fallback>"
                self.logger.info(
                    "Val VoxelBudgetBatchSampler: budget=%d max_bs=%d "
                    "batches_this_rank=%d dataset=%d missing_sizes=%d source=%s",
                    int(voxel_budget),
                    max_batch_size,
                    len(batch_sampler),
                    len(sizes),
                    missing_csv,
                    size_source,
                )
                val_loader = torch.utils.data.DataLoader(
                    val_data,
                    batch_sampler=batch_sampler,
                    num_workers=self.cfg.num_worker_per_gpu,
                    pin_memory=True,
                    collate_fn=collate_fn,
                )
            else:
                if comm.get_world_size() > 1:
                    val_sampler = torch.utils.data.distributed.DistributedSampler(
                        val_data
                    )
                else:
                    val_sampler = None
                val_loader = torch.utils.data.DataLoader(
                    val_data,
                    batch_size=self.cfg.batch_size_val_per_gpu,
                    shuffle=False,
                    num_workers=self.cfg.num_worker_per_gpu,
                    pin_memory=True,
                    sampler=val_sampler,
                    collate_fn=collate_fn,
                )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        if getattr(self.cfg, "total_iters", None) is not None:
            self.cfg.scheduler.total_steps = (
                self.cfg.total_iters // self.cfg.gradient_accumulation_steps
            )
        else:
            self.cfg.scheduler.total_steps = (
                len(self.train_loader)
                * self.cfg.eval_epoch
                // self.cfg.gradient_accumulation_steps
            )
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("PartialSampledTrainer")
class PartialSampledTrainer(Trainer):
    def __init__(self, cfg):
        if getattr(cfg, "total_iters", None) is not None:
            raise ValueError(
                "total_iters is not supported with PartialSampledTrainer; "
                "use DefaultTrainer instead."
            )
        super().__init__(cfg)

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)
        sampled_dataset_index = self.cfg.data.sampled_dataset_index
        sampled_dataset_limit = self.cfg.data.sampled_dataset_limit

        train_sampler = DistributedImbalancedSampler(
            dataset=train_data,
            sampled_dataset_index=sampled_dataset_index,
            sampled_dataset_limit=sampled_dataset_limit,
            num_replicas=comm.get_world_size(),
            rank=comm.get_rank(),
            shuffle=True,
            seed=self.cfg.seed if self.cfg.seed is not None else 0,
        )

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=False,
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def __init__(self, cfg):
        if getattr(cfg, "total_iters", None) is not None:
            raise ValueError(
                "total_iters is not supported with MultiDatasetTrainer; "
                "use DefaultTrainer instead."
            )
        super().__init__(cfg)

    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader


@TRAINERS.register_module("GridProbeTrainer")
class GridProbeTrainer(Trainer):
    """Trainer for GridProbeSegmentorV2: N linear probe heads, each with its
    own optimizer/scheduler/grad_clip type and hyperparameters, sharing one
    frozen-backbone forward per batch, then one shared backward over the sum
    of every active probe's loss (heads share no trainable params, so this
    gives every head exactly its own gradient in one autograd pass — cheaper
    than one backward() per probe, but keeps every active probe's forward+loss
    graph alive simultaneously until that point: ~440MB/probe measured on the
    wide-grid config (batch_size=12, point_max=102400) — cap probe count to
    what fits (~100 is a safe margin on an 80GB GPU; verify per-config with
    scripts/sonata/diagnose_grid_probe_vram.py rather than assuming).

    self.optimizer / self.scheduler become dict[str, ...] here (one entry per
    probe) instead of the base Trainer's single instance — hooks that assume
    a single optimizer/scheduler (InformationWriter's Lr: line, CheckpointSaver,
    CheckpointLoader) branch on `isinstance(trainer.optimizer, dict)`.

    Precision (enable_amp/amp_dtype) stays a single global setting: the
    backbone forward (the expensive part) runs once under one autocast
    context shared by every probe head, so per-probe precision would gain
    nothing while complicating the shared-backward argument this class
    relies on (see module docstring in models/grid_probe.py).

    Diagnostics tied to MultiTaskSegmentorV2's heterogeneous-task-weight
    machinery (grad_norm_lite, log_task_gradient_norms, skip_nan_grad,
    global_gradient_diag logging) are intentionally not implemented here —
    those cfg flags are simply never read, no crash risk, just an omission.
    """

    def _raw_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def before_train(self):
        super().before_train()
        n_probes = len(self._raw_model().probe_names)
        self.logger.info(
            "Grid probe: %d linear heads trained in parallel "
            "(one frozen-backbone forward per batch).",
            n_probes,
        )
        if (
            self.cfg.enable_wandb
            and comm.is_main_process()
            and wandb.run is not None
        ):
            wandb.run.summary["grid_probe/num_probes"] = n_probes
            wandb.log({"grid_probe/num_probes": n_probes})

    def build_optimizer(self):
        raw_model = self._raw_model()
        optimizers = {}
        for name in raw_model.probe_names:
            probe_cfg = raw_model.probe_configs[name]
            opt_cfg = ConfigDict(copy.deepcopy(dict(probe_cfg["optimizer"])))
            optimizers[name] = build_optimizer(
                opt_cfg, params=raw_model.probe_head_parameters(name)
            )
        return optimizers

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        if getattr(self.cfg, "total_iters", None) is not None:
            total_steps = self.cfg.total_iters // self.cfg.gradient_accumulation_steps
        else:
            total_steps = (
                len(self.train_loader)
                * self.cfg.eval_epoch
                // self.cfg.gradient_accumulation_steps
            )
        raw_model = self._raw_model()
        schedulers = {}
        for name, opt in self.optimizer.items():
            probe_cfg = raw_model.probe_configs[name]
            sched_cfg = ConfigDict(copy.deepcopy(dict(probe_cfg["scheduler"])))
            sched_cfg.total_steps = total_steps
            schedulers[name] = build_scheduler(sched_cfg, opt)
        return schedulers

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        # Only clear gradients on first accumulation step (every probe optimizer).
        if self._gradient_accumulation_counter == 0:
            for opt in self.optimizer.values():
                opt.zero_grad()

        # One frozen-backbone forward, then one shared backward over the sum
        # of every active probe's loss: heads share zero trainable params
        # (only the frozen backbone), so one backward() on the sum gives every
        # head exactly its own gradient. Cheaper in wall-clock than one
        # backward() per probe (no repeated autograd-engine dispatch), at the
        # cost of keeping every active probe's forward+loss graph alive at
        # once until this point -- ~440MB/probe measured (batch_size=12,
        # point_max=102400, wide-grid config), so cap probe count accordingly
        # (see GridProbeSegmentorV2 docstring / scripts/sonata/diagnose_grid_probe_vram.py).
        raw_model = self._raw_model()
        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            feat_by_norm, point, active = raw_model.prepare_batch(input_dict)
            # Backbone-internal-only state (serialized order/inverse for every
            # order, per-stage pad/unpad/cu_seqlens, sparse_conv_feat, ...) --
            # nothing below reads `point` again, but the reference would
            # otherwise sit in this frame's locals (Python doesn't drop it
            # until the function returns) for the whole per-probe loop.
            del point
            target = input_dict.get(raw_model.target_key)
            loss_by_task = {}
            pred_by_task = {}
            task_losses = []
            for name in active:
                logits = raw_model.probe_logits(name, feat_by_norm)
                if target is not None:
                    task_loss = raw_model.criteria_by_task[name](logits, target)
                    task_losses.append(task_loss)
                    loss_by_task[name] = task_loss.detach()
                    pred_by_task[name] = logits.argmax(dim=1).detach()
                    input_dict.setdefault(name, target)
                del logits

            if task_losses:
                total = sum(task_losses) / self.cfg.gradient_accumulation_steps
                if self.cfg.enable_amp:
                    self.scaler.scale(total).backward()
                else:
                    total.backward()

        output_dict = {}
        if loss_by_task:
            output_dict["loss"] = sum(loss_by_task.values())
            output_dict["loss_by_task"] = loss_by_task
            output_dict["pred_by_task"] = pred_by_task
        self._gradient_accumulation_counter += 1

        # Perform optimizer step only when enough gradients have accumulated
        if self._gradient_accumulation_counter >= self.cfg.gradient_accumulation_steps:
            raw_model = self._raw_model()
            probe_configs = raw_model.probe_configs
            scale_before = self.scaler.get_scale() if self.cfg.enable_amp else None

            for name, opt in self.optimizer.items():
                if self.cfg.enable_amp:
                    self.scaler.unscale_(opt)
                grad_clip = probe_configs[name].get("grad_clip")
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        raw_model.probe_head_parameters(name), grad_clip
                    )
                if self.cfg.enable_amp:
                    self.scaler.step(opt)
                else:
                    opt.step()

            if self.cfg.enable_amp:
                self.scaler.update()
                # Skip the scheduler step this round if AMP detected an inf/nan
                # and shrank the scale (mirrors the base Trainer's single-optimizer
                # guard; GradScaler has one global scale regardless of how many
                # optimizers share it, so one check covers every probe).
                step_schedulers = scale_before <= self.scaler.get_scale()
            else:
                step_schedulers = True
            if step_schedulers:
                for sched in self.scheduler.values():
                    sched.step()

            # Reset grad accumulation counter
            self._gradient_accumulation_counter = 0

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict
