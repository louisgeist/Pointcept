"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import wandb
import torch
import torch.distributed as dist
import pointops
from uuid import uuid4

import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class ClsEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["cls_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                label,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss_avg,
                        "val/mIoU": m_iou,
                        "val/mAcc": m_acc,
                        "val/allAcc": all_acc,
                    },
                    step=wandb.run.step,
                )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_acc  # save for saver
        self.trainer.comm_info["current_metric_name"] = "allAcc"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    """Single-task semantic segmentation validation (seg_logits vs segment)."""

    def __init__(self, write_cls_iou=False):
        self.write_cls_iou = write_cls_iou

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "inverse" in input_dict.keys():
                assert "origin_segment" in input_dict.keys()
                pred = pred[input_dict["inverse"]]
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        m_iou_best = max(self.trainer.best_metric_value, m_iou)
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU_best", m_iou_best, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss_avg,
                        "val/mIoU": m_iou,
                        "val/mIoU_best": m_iou_best,
                        "val/mAcc": m_acc,
                        "val/allAcc": all_acc,
                    },
                    step=wandb.run.step,
                )
            if self.write_cls_iou:
                for i in range(self.trainer.cfg.data.num_classes):
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
                        iou_class[i],
                        current_epoch,
                    )
                if self.trainer.cfg.enable_wandb:
                    for i in range(self.trainer.cfg.data.num_classes):
                        wandb.log(
                            {
                                "Epoch": current_epoch,
                                f"val/iou_{self.trainer.cfg.data.names[i]}": iou_class[
                                    i
                                ],
                            },
                            step=wandb.run.step,
                        )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )

@HOOKS.register_module()
class MultiTaskEvaluator(HookBase):
    """Multi-task validation: semantic IoU from seg_logits_by_task; regression from reg_pred_by_task.

    Reads cfg.data.task_configs and cfg.data.main_task. Every task_configs entry must set "task_type"
    to "semantic" or "regression". Task targets are read from input_dict[task_name] for both
    semantic and regression tasks, together with reg_pred_by_task[task_name] for regression.

    Checkpoint selection (comm_info) follows mIoU of main_task, which must be a semantic task.
    Regression metrics (MAE/RMSE) are logged per regression task name.

    The "main task" (checkpoint / mIoU selection) is cfg.data.main_task when multiple tasks are
    configured; with a single task in task_configs, main_task may be omitted. TensorBoard / W&B
    semantic metrics use val/<task_name>/... for every semantic task; mIoU_best is logged only
    for main_task.
    """

    def __init__(self, write_cls_iou=False):
        self.write_cls_iou = write_cls_iou
        self._best_neg_rmse = float("-inf")

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")
            wandb.define_metric("val/reg/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    @staticmethod
    def _cfg_get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _get_task_configs(self):
        data_cfg = self.trainer.cfg.data
        task_configs = self._cfg_get(data_cfg, "task_configs", None)
        if isinstance(task_configs, dict) and len(task_configs) > 0:
            return {str(k): dict(v) for k, v in task_configs.items()}
        raise ValueError(
            "MultiTaskEvaluator requires cfg.data.task_configs to be a non-empty dict."
        )

    def _main_task_name(self, task_configs):
        """Resolve cfg.data.main_task; required when len(task_configs) > 1."""
        data_cfg = self.trainer.cfg.data
        main_task = self._cfg_get(data_cfg, "main_task", None)
        keys = list(task_configs.keys())
        
        if main_task is not None :
            if str(main_task) in keys:
                return str(main_task)
            else:
                raise ValueError(
                    f"cfg.data.main_task is {main_task!r} but that key is not in task_configs "
                    f"(keys: {keys})."
                )
        else:
            if len(keys) == 1:
                return keys[0]
            else:
                raise ValueError(
                    "cfg.data.main_task is not defined (and cannot be inferred as there are multiple tasks)"
                )

    @staticmethod
    def _task_origin_target_key(task_name):
        return f"origin_{task_name}"

    @staticmethod
    def _gather_masked(pred, target):
        mask = torch.isfinite(pred) & torch.isfinite(target)
        if mask.sum() == 0:
            return None, None
        return pred[mask], target[mask]

    @staticmethod
    def _semantic_task_names(task_configs):
        return [
            k
            for k, task_config in task_configs.items()
            if task_config.get("task_type") == "semantic"
        ]

    @staticmethod
    def _regression_task_names(task_configs):
        return [
            k
            for k, task_config in task_configs.items()
            if task_config.get("task_type") == "regression"
        ]

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        task_configs = self._get_task_configs()
        main_task = self._main_task_name(task_configs)
        semantic_tasks = self._semantic_task_names(task_configs)
        regression_tasks = self._regression_task_names(task_configs)

        reg_sums = {t: {"mae": 0.0, "mse": 0.0, "count": 0.0} for t in regression_tasks}

        for i, input_dict in enumerate(self.trainer.val_loader):
            # --------- Model forward ---------
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            
            loss = output_dict["loss"]
            logits_by_task = output_dict.get("seg_logits_by_task", None)
            if not isinstance(logits_by_task, dict):
                logits_by_task = {"segment": output_dict["seg_logits"]}


            # --------- Evaluate Semantic Tasks ---------
            for task_name in semantic_tasks:
                task_config = task_configs[task_name]
                if task_name not in input_dict or task_name not in logits_by_task:
                    continue
                pred = logits_by_task[task_name].max(1)[1]
                target_tensor = input_dict[task_name]
                if "inverse" in input_dict.keys():
                    origin_target_key = self._task_origin_target_key(task_name)
                    if origin_target_key in input_dict:
                        pred = pred[input_dict["inverse"]]
                        target_tensor = input_dict[origin_target_key]
                intersection, union, target = intersection_and_union_gpu(
                    pred,
                    target_tensor,
                    int(task_config["num_classes"]),
                    int(task_config["ignore_index"]),
                )
                if comm.get_world_size() > 1:
                    dist.all_reduce(intersection)
                    dist.all_reduce(union)
                    dist.all_reduce(target)
                intersection, union, target = (
                    intersection.cpu().numpy(),
                    union.cpu().numpy(),
                    target.cpu().numpy(),
                )
                self.trainer.storage.put_scalar(
                    f"val_intersection/{task_name}", intersection
                )
                self.trainer.storage.put_scalar(f"val_union/{task_name}", union)
                self.trainer.storage.put_scalar(f"val_target/{task_name}", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            
            
            # --------- Evaluate Regression Tasks ---------
            reg_pred_by_task = output_dict.get("reg_pred_by_task") or {}
            for task_name in regression_tasks:
                pred = reg_pred_by_task.get(task_name)
                if pred is None:
                    continue
                if task_name not in input_dict:
                    continue
                pred = pred.reshape(-1).float()
                target = input_dict[task_name].reshape(-1).float()
                if "inverse" in input_dict.keys():
                    origin_key = self._task_origin_target_key(task_name)
                    if origin_key in input_dict:
                        pred = pred[input_dict["inverse"]]
                        target = input_dict[origin_key].reshape(-1).float()
                p, t = self._gather_masked(pred, target)
                if p is not None:
                    err = (p - t).abs()
                    err2 = (p - t) ** 2
                    reg_sums[task_name]["mae"] += float(err.sum().item())
                    reg_sums[task_name]["mse"] += float(err2.sum().item())
                    reg_sums[task_name]["count"] += float(err.numel())

            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        if comm.get_world_size() > 1 and regression_tasks:
            flat = []
            for task_name in regression_tasks:
                s = reg_sums[task_name]
                flat.extend([s["mae"], s["mse"], s["count"]])
            buf = torch.tensor(flat, dtype=torch.float64, device="cuda")
            dist.all_reduce(buf)
            flat = buf.cpu().tolist()
            for ti, task_name in enumerate(regression_tasks):
                base = ti * 3
                reg_sums[task_name]["mae"] = flat[base]
                reg_sums[task_name]["mse"] = flat[base + 1]
                reg_sums[task_name]["count"] = flat[base + 2]

        per_task_metrics = {}
        for task_name in semantic_tasks:
            task_config = task_configs[task_name]
            intersection = self.trainer.storage.history(
                f"val_intersection/{task_name}"
            ).total
            union = self.trainer.storage.history(f"val_union/{task_name}").total
            target = self.trainer.storage.history(f"val_target/{task_name}").total
            iou_class = intersection / (union + 1e-10)
            acc_class = intersection / (target + 1e-10)
            m_iou = np.mean(iou_class)
            m_acc = np.mean(acc_class)
            all_acc = sum(intersection) / (sum(target) + 1e-10)
            per_task_metrics[task_name] = dict(
                iou_class=iou_class,
                acc_class=acc_class,
                m_iou=m_iou,
                m_acc=m_acc,
                all_acc=all_acc,
                names=list(task_config["names"]),
            )
            self.trainer.logger.info(
                "[task={}] Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                    task_name, m_iou, m_acc, all_acc
                )
            )
            for class_idx in range(int(task_config["num_classes"])):
                class_name = per_task_metrics[task_name]["names"][class_idx]
                self.trainer.logger.info(
                    "[task={}] Class_{}-{} Result: iou/accuracy {:.4f}/{:.4f}".format(
                        task_name,
                        class_idx,
                        class_name,
                        iou_class[class_idx],
                        acc_class[class_idx],
                    )
                )

        current_epoch = self.trainer.epoch + 1
        main_m_iou = per_task_metrics[main_task]["m_iou"]
        m_iou_best = max(self.trainer.best_metric_value, main_m_iou)
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            wandb_log = None
            if self.trainer.cfg.enable_wandb:
                wandb_log = {"Epoch": current_epoch, "val/loss": loss_avg}
                
            # General metrics
            for task_name, metric in per_task_metrics.items():
                prefix = f"val/{task_name}"
                self.trainer.writer.add_scalar(
                    f"{prefix}/mIoU", metric["m_iou"], current_epoch
                )
                if task_name == main_task:
                    self.trainer.writer.add_scalar(
                        f"{prefix}/mIoU_best", m_iou_best, current_epoch
                    )
                self.trainer.writer.add_scalar(
                    f"{prefix}/mAcc", metric["m_acc"], current_epoch
                )
                self.trainer.writer.add_scalar(
                    f"{prefix}/allAcc", metric["all_acc"], current_epoch
                )
                if wandb_log is not None:
                    wandb_log[f"{prefix}/mIoU"] = float(metric["m_iou"])
                    if task_name == main_task:
                        wandb_log[f"{prefix}/mIoU_best"] = float(m_iou_best)
                    wandb_log[f"{prefix}/mAcc"] = float(metric["m_acc"])
                    wandb_log[f"{prefix}/allAcc"] = float(metric["all_acc"])
            if wandb_log is not None:
                wandb.log(wandb_log, step=wandb.run.step)

            # Per-class metrics
            if self.write_cls_iou:
                cls_log = (
                    {"Epoch": current_epoch}
                    if self.trainer.cfg.enable_wandb
                    else None
                )
                for task_name, metric in per_task_metrics.items():
                    task_config = task_configs[task_name]
                    for class_idx in range(int(task_config["num_classes"])):
                        if class_idx == task_config["ignore_index"]:
                            continue
                        class_name = task_config["names"][class_idx]
                        slug = "".join(
                            c if (c.isalnum() or c in "._-") else "_"
                            for c in str(class_name).strip().replace(" ", "_")
                        )
                        tag = f"val/{task_name}/iou/{class_idx}_{slug}"
                        self.trainer.writer.add_scalar(
                            tag,
                            metric["iou_class"][class_idx],
                            current_epoch,
                        )
                        if cls_log is not None:
                            cls_log[tag] = float(metric["iou_class"][class_idx])
                if cls_log is not None:
                    wandb.log(cls_log, step=wandb.run.step)

        reg_wandb = {"Epoch": current_epoch}
        best_neg_rmse_epoch = float("-inf")
        writer = self.trainer.writer
        enable_wandb = self.trainer.cfg.enable_wandb
        for task_name in regression_tasks:
            s = reg_sums[task_name]
            cnt = s["count"]
            if cnt <= 1e-8:
                continue
            mae = s["mae"] / cnt
            rmse = (s["mse"] / cnt) ** 0.5
            best_neg_rmse_epoch = max(best_neg_rmse_epoch, -rmse)
            self.trainer.logger.info(
                "[task={}] Val regression: MAE {:.6f} RMSE {:.6f} (n={:.0f}).".format(
                    task_name, mae, rmse, cnt
                )
            )
            if writer is not None:
                writer.add_scalar(
                    f"val/reg/{task_name}/mae", mae, current_epoch
                )
                writer.add_scalar(
                    f"val/reg/{task_name}/rmse", rmse, current_epoch
                )
            if enable_wandb:
                reg_wandb[f"val/reg/{task_name}/mae"] = float(mae)
                reg_wandb[f"val/reg/{task_name}/rmse"] = float(rmse)
        if best_neg_rmse_epoch > float("-inf"):
            self._best_neg_rmse = max(self._best_neg_rmse, best_neg_rmse_epoch)
            if writer is not None:
                writer.add_scalar(
                    "val/reg/rmse_best_neg", self._best_neg_rmse, current_epoch
                )
            if enable_wandb:
                reg_wandb["val/reg/rmse_best_neg"] = float(self._best_neg_rmse)
        if enable_wandb and len(reg_wandb) > 1:
            wandb.log(reg_wandb, step=wandb.run.step)

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = main_m_iou
        self.trainer.comm_info["current_metric_name"] = f"mIoU/{main_task}"

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format(
                self.trainer.comm_info.get("current_metric_name", "mIoU"),
                self.trainer.best_metric_value,
            )
        )


@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self, segment_ignore_index=(-1,), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.trainer.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.trainer.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.trainer.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.trainer.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        thresholds, unique_indices = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = {}
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert (
                len(input_dict["offset"]) == 1
            )  # currently only support bs 1 for each GPU
            data_name = input_dict.pop("name")[0]
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]

            segment = input_dict["segment"]
            instance = input_dict["instance"]
            # map to origin
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                idx = idx.cpu().flatten().long()
                output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes[data_name] = dict(gt=gt_instances, pred=pred_instance)

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)

        if comm.is_main_process():
            scenes = {}
            for _ in range(len(scenes_sync)):
                r = scenes_sync.pop()
                scenes.update(r)
                del r
            scenes = list(scenes.values())
            ap_scores = self.evaluate_matches(scenes)
            all_ap = ap_scores["all_ap"]
            all_ap_50 = ap_scores["all_ap_50%"]
            all_ap_25 = ap_scores["all_ap_25%"]
            self.trainer.logger.info(
                "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                    all_ap, all_ap_50, all_ap_25
                )
            )
            for i, label_name in enumerate(self.valid_class_names):
                ap = ap_scores["classes"][label_name]["ap"]
                ap_50 = ap_scores["classes"][label_name]["ap50%"]
                ap_25 = ap_scores["classes"][label_name]["ap25%"]
                self.trainer.logger.info(
                    "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                        idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                    )
                )
            current_epoch = self.trainer.epoch + 1
            if self.trainer.writer is not None:
                self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
                self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
                self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
                self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
                if self.trainer.cfg.enable_wandb:
                    wandb.log(
                        {
                            "Epoch": current_epoch,
                            "val/loss": loss_avg,
                            "val/mAP": all_ap,
                            "val/AP50": all_ap_50,
                            "val/AP25": all_ap_25,
                        },
                        step=wandb.run.step,
                    )
            self.trainer.logger.info(
                "<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<"
            )
            self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
            self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver


@HOOKS.register_module()
class ShapeNetPartSegEvaluator(HookBase):
    def __init__(self, write_cls_iou=False):
        self.write_cls_iou = write_cls_iou

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Part Segmentation Evaluation >>>>>>>>>>>>>>>>"
        )
        self.trainer.model.eval()

        # Initialize numpy arrays to aggregate results over the entire validation set.
        num_categories = len(self.trainer.val_loader.dataset.categories)
        total_iou_category = torch.zeros(num_categories, device="cuda")
        total_iou_count = torch.zeros(num_categories, device="cuda")

        # Iterate over all batches in the validation loader
        for i, input_dict in enumerate(self.trainer.val_loader):
            # Move all tensor data to the GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            # Perform model forward pass without gradient computation
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            pred_scores = output_dict["seg_logits"]
            pred_labels = torch.argmax(pred_scores, dim=-1)

            segment = input_dict["segment"]
            cls_token = input_dict["cls_token"][0].cpu().numpy()

            if "inverse" in input_dict.keys():
                assert (
                    "origin_segment" in input_dict.keys()
                ), "origin_segment must be provided with inverse"
                pred_labels = pred_labels[input_dict["inverse"]]
                segment = input_dict["origin_segment"]

            category_name = self.trainer.val_loader.dataset.categories[cls_token]
            parts_idx = self.trainer.val_loader.dataset.category2part[category_name]
            parts_iou = torch.zeros(len(parts_idx), device="cuda")
            for k, part_id in enumerate(parts_idx):
                if (torch.sum(segment == part_id) == 0) and (
                    torch.sum(pred_labels == part_id) == 0
                ):
                    parts_iou[k] = (
                        1.0  # This part is correctly not predicted and not present
                    )
                else:
                    intersection = torch.sum(
                        (segment == part_id) & (pred_labels == part_id)
                    )
                    union = torch.sum((segment == part_id) | (pred_labels == part_id))
                    parts_iou[k] = intersection / (union + 1e-10)

            # Calculate the mean IoU for this specific sample over its relevant parts
            sample_miou = parts_iou.mean()

            # Aggregate the result into the corresponding category
            total_iou_category[cls_token] += sample_miou
            total_iou_count[cls_token] += 1

        if comm.get_world_size() > 1:
            dist.all_reduce(total_iou_category), dist.all_reduce(total_iou_count)
        total_iou_count = total_iou_count.cpu().numpy()
        total_iou_category = total_iou_category.cpu().numpy()
        # Instance-wise mIoU: average of all sample mIoUs
        ins_mIoU = total_iou_category.sum() / (total_iou_count.sum() + 1e-10)
        # Category-wise mIoU: average of per-category mIoUs
        iou_per_cat = total_iou_category / (total_iou_count + 1e-10)
        # Only average over categories that were actually present in the validation set
        cat_mIoU = np.mean(iou_per_cat[total_iou_count > 0])

        self.trainer.logger.info(
            "Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.".format(ins_mIoU, cat_mIoU)
        )

        # Log detailed results for each category
        for i in range(num_categories):
            if total_iou_count[i] > 0:
                self.trainer.logger.info(
                    "Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.0f}".format(
                        idx=i,
                        name=self.trainer.val_loader.dataset.categories[i],
                        iou_cat=iou_per_cat[i],
                        iou_count=total_iou_count[i],
                    )
                )

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/ins_mIoU", ins_mIoU, current_epoch)
            self.trainer.writer.add_scalar("val/cat_mIoU", cat_mIoU, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/ins_mIoU": ins_mIoU,
                        "val/cat_mIoU": cat_mIoU,
                    },
                    step=wandb.run.step,
                )

            if self.write_cls_iou:
                for i in range(num_categories):
                    if total_iou_count[i] > 0:
                        category_name = self.trainer.val_loader.dataset.categories[i]
                        self.trainer.writer.add_scalar(
                            f"val/cls_{i}-{category_name}_IoU",
                            iou_per_cat[i],
                            current_epoch,
                        )

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

        # Save the primary metric for checkpointing logic (e.g., saving the best model)
        # Category mIoU is often a more robust metric for this.
        self.trainer.comm_info["current_metric_value"] = cat_mIoU
        self.trainer.comm_info["current_metric_name"] = "cat_mIoU"

    def after_train(self):
        """
        Log the best performing metric at the very end of training.
        """
        self.trainer.logger.info(
            "Best {}: {:.4f}".format(
                self.trainer.comm_info.get("current_metric_name", "metric"),
                self.trainer.best_metric_value,
            )
        )


@HOOKS.register_module()
class PartNetEPartSegEvaluator(HookBase):
    def __init__(self, num_parts=None, write_part_iou=False):
        self.num_parts = sum(num_parts)
        self.write_part_iou = write_part_iou

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Part Segmentation Evaluation >>>>>>>>>>>>>>>>"
        )
        self.trainer.model.eval()

        # Initialize numpy arrays to aggregate results over the entire validation set.
        num_categories = len(self.trainer.val_loader.dataset.categories)
        total_iou_parts = torch.zeros(self.num_parts, device="cuda")
        total_iou_count = torch.zeros(self.num_parts, device="cuda")

        # Iterate over all batches in the validation loader
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert len(input_dict["offset"]) == 1
            # Move all tensor data to the GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            # Perform model forward pass without gradient computation
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            pred_scores = output_dict["seg_logits"]
            segment = input_dict["segment"]
            cls_token = input_dict["cls_token"][0].cpu().numpy()
            category_name = self.trainer.val_loader.dataset.categories[cls_token]
            parts_idx = self.trainer.val_loader.dataset.category2part[category_name]
            pred_labels = torch.argmax(pred_scores, dim=-1)
            # pred_labels = torch.argmax(pred_scores[:, parts_idx], dim=-1)

            if "inverse" in input_dict.keys():
                assert (
                    "origin_segment" in input_dict.keys()
                ), "origin_segment must be provided with inverse"
                pred_labels = pred_labels[input_dict["inverse"]]
                segment = input_dict["origin_segment"]

            for k, part_id in enumerate(parts_idx):
                if k == 0:
                    continue
                if (segment == part_id).sum() == 0:
                    continue
                if (torch.sum(segment == part_id) == 0) and (
                    torch.sum(pred_labels == part_id) == 0
                ):
                    continue
                else:
                    intersection = torch.sum(
                        (segment == part_id) & (pred_labels == part_id)
                    )
                    union = torch.sum((segment == part_id) | (pred_labels == part_id))
                    total_iou_parts[
                        k + self.trainer.val_loader.dataset.num_part_offset[cls_token]
                    ] += intersection / (union + 1e-10)
                    total_iou_count[
                        k + self.trainer.val_loader.dataset.num_part_offset[cls_token]
                    ] += 1
        if comm.get_world_size() > 1:
            dist.all_reduce(total_iou_parts), dist.all_reduce(total_iou_count)
        total_iou_count = total_iou_count.cpu().numpy()
        total_iou_parts = total_iou_parts.cpu().numpy()
        current_iou_count = total_iou_count[total_iou_count > 0]
        current_iou_parts = total_iou_parts[total_iou_count > 0]
        # part-wise mIoU: average of all sample mIoUs
        part_mIoU = (current_iou_parts / current_iou_count).mean()

        self.trainer.logger.info("Val result: part mIoU {:.4f}.".format(part_mIoU))

        # Log detailed results for each category
        for i in range(self.num_parts):
            if total_iou_count[i] > 0:
                self.trainer.logger.info(
                    "Class_{idx}-{name} Result: iou_part/num_sample {iou_part:.4f}/{iou_count:.0f}".format(
                        idx=i,
                        name=self.trainer.val_loader.dataset.parts[i],
                        iou_part=total_iou_parts[i] / total_iou_count[i],
                        iou_count=total_iou_count[i],
                    )
                )

        # --- Log metrics to TensorBoard / WandB ---
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/part_mIoU", part_mIoU, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/part_mIoU": part_mIoU,
                    },
                    step=wandb.run.step,
                )

            if self.write_part_iou:
                for i in range(self.num_parts):
                    if total_iou_count[i] > 0:
                        part_name = self.trainer.val_loader.dataset.parts[i]
                        self.trainer.writer.add_scalar(
                            f"val/part_{i}-{part_name}_IoU",
                            total_iou_parts[i] / total_iou_count[i],
                            current_epoch,
                        )
                # (Similar logging block can be added for WandB if needed)

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

        # Save the primary metric for checkpointing logic (e.g., saving the best model)
        self.trainer.comm_info["current_metric_value"] = part_mIoU
        self.trainer.comm_info["current_metric_name"] = "part_mIoU"

    def after_train(self):
        """
        Log the best performing metric at the very end of training.
        """
        self.trainer.logger.info(
            "Best {}: {:.4f}".format(
                self.trainer.comm_info.get("current_metric_name", "metric"),
                self.trainer.best_metric_value,
            )
        )
