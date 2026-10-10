"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import wandb
import torch
import torch.distributed as dist

import pointcept.utils.comm as comm
from pointcept.utils.misc import (
    abs_freq_error_rows,
    accumulate_regression_errors,
    intersection_and_union_gpu,
    kl_divergence_rows,
    mean_acc_from_hist,
    mean_iou_from_hist,
    pool_axis_distribution_from_probs,
    tv_from_abs_errors,
)
from pointcept.utils.regression import (
    denorm_regression_prediction,
    get_regression_target_scale,
)
from pointcept.utils.epoch_timing import (
    begin_epoch_wall_timing,
    finalize_epoch_wall_timing,
)
from pointcept.utils.progress import EvaluationProgressBar
from pointcept.utils.multilabel_metrics import (
    MultilabelStats,
    accumulate_multilabel_stats,
    compute_multilabel_metrics,
    multilabel_stats_from_tensors,
    multilabel_stats_to_tensors,
)
from pointcept.utils.wandb_metrics import (
    best_miou_tag,
    class_name_slug,
    iou_class_tag,
    metric_tag,
)
from pointcept.utils.dilated_metrics import (
    dilated_precision_recall_counts,
    dilated_prf_enabled,
    precision_recall_f1,
)
from pointcept.utils.gradient_norm import (
    combine_weighted_task_losses,
    resolve_grad_norm_lite_scales,
)

from .default import HookBase
from .builder import HOOKS


def _offset_to_list(offset):
    if offset is None:
        return None
    if isinstance(offset, torch.Tensor):
        return offset.tolist()
    return list(offset)


def remap_pred_with_inverse(
    pred_voxel, inverse, offset=None, origin_offset=None
):
    """Map voxel-level predictions back to full-resolution points.

    When multiple scenes are collated for validation, each scene's ``inverse`` indices
    are local (0 .. num_voxels-1) while ``pred_voxel`` is concatenated scene-by-scene.
    ``offset`` marks voxel boundaries; ``origin_offset`` marks full-resolution boundaries.
    """
    if origin_offset is None:
        return pred_voxel[inverse]

    voxel_ends = _offset_to_list(offset)
    point_ends = _offset_to_list(origin_offset)
    if voxel_ends is None or point_ends is None:
        return pred_voxel[inverse]
    if len(voxel_ends) != len(point_ends):
        raise ValueError(
            f"offset and origin_offset must have the same batch length, "
            f"got {len(voxel_ends)} vs {len(point_ends)}"
        )

    pred_full = []
    voxel_start = 0
    point_start = 0
    for voxel_end, point_end in zip(voxel_ends, point_ends):
        voxel_end = int(voxel_end)
        point_end = int(point_end)
        inv = inverse[point_start:point_end]
        pred_scene = pred_voxel[voxel_start:voxel_end]
        pred_full.append(pred_scene[inv])
        voxel_start = voxel_end
        point_start = point_end
    pred_full = torch.cat(pred_full)
    assert pred_full.shape[0] == inverse.shape[0], (
        f"remapped pred length {pred_full.shape[0]} != inverse length {inverse.shape[0]}"
    )
    return pred_full


def begin_val_epoch_timing():
    return begin_epoch_wall_timing()


def finalize_val_epoch_timing(trainer, start_time, current_epoch, wandb_dict=None):
    return finalize_epoch_wall_timing(
        trainer,
        start_time,
        current_epoch,
        "val",
        wandb_dict=wandb_dict,
        log_console=True,
    )


def sync_confusion_hist_totals(intersection, union, target):
    """All-reduce local confusion-histogram totals once after the val loop.

    Per-batch all_reduce deadlocks when ranks have unequal loader lengths
    (e.g. VoxelBudgetBatchSampler stride sharding). Summing locally then
    reducing once is mathematically equivalent.
    """
    intersection = np.asarray(intersection, dtype=np.float64).reshape(-1)
    union = np.asarray(union, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if intersection.shape != union.shape or intersection.shape != target.shape:
        raise ValueError(
            "intersection/union/target must share the same shape, got "
            f"{intersection.shape}, {union.shape}, {target.shape}"
        )
    if comm.get_world_size() <= 1:
        return intersection, union, target
    flat = np.concatenate([intersection, union, target])
    buf = torch.tensor(flat, dtype=torch.float64, device="cuda")
    dist.all_reduce(buf)
    flat = buf.cpu().numpy()
    c = intersection.size
    return flat[0:c], flat[c : 2 * c], flat[2 * c : 3 * c]


def local_task_confusion_hist_totals(storage, task_name, num_classes):
    """Local val_intersection|union|target totals for a named task (or zeros)."""
    histories = storage.histories()
    zeros = np.zeros(int(num_classes), dtype=np.float64)
    ik = f"val_intersection/{task_name}"
    uk = f"val_union/{task_name}"
    tk = f"val_target/{task_name}"
    intersection = histories[ik].total if ik in histories else zeros.copy()
    union = histories[uk].total if uk in histories else zeros.copy()
    target = histories[tk].total if tk in histories else zeros.copy()
    return (
        np.asarray(intersection, dtype=np.float64).reshape(-1),
        np.asarray(union, dtype=np.float64).reshape(-1),
        np.asarray(target, dtype=np.float64).reshape(-1),
    )


def local_dilated_prf_totals(storage, task_name, ch_name):
    """Local buffer-P/R hit/denominator sums for one pixel_semantic task channel (or zeros)."""
    histories = storage.histories()
    base = f"val_dilated_prf/{task_name}/{ch_name}"

    def _total(key):
        return float(histories[key].total) if key in histories else 0.0

    return (
        _total(f"{base}/p_num"),
        _total(f"{base}/p_denom"),
        _total(f"{base}/r_num"),
        _total(f"{base}/r_denom"),
    )


def sync_dilated_prf_totals(p_num, p_denom, r_num, r_denom):
    """All-reduce buffer-P/R totals once after the val loop (see sync_confusion_hist_totals)."""
    if comm.get_world_size() <= 1:
        return p_num, p_denom, r_num, r_denom
    buf = torch.tensor(
        [p_num, p_denom, r_num, r_denom], dtype=torch.float64, device="cuda"
    )
    dist.all_reduce(buf)
    flat = buf.cpu().tolist()
    return flat[0], flat[1], flat[2], flat[3]


def local_semseg_confusion_hist_totals(storage, num_classes):
    """Local val_intersection|union|target totals for SemSegEvaluator (or zeros)."""
    histories = storage.histories()
    zeros = np.zeros(int(num_classes), dtype=np.float64)
    intersection = (
        histories["val_intersection"].total
        if "val_intersection" in histories
        else zeros.copy()
    )
    union = histories["val_union"].total if "val_union" in histories else zeros.copy()
    target = (
        histories["val_target"].total if "val_target" in histories else zeros.copy()
    )
    return (
        np.asarray(intersection, dtype=np.float64).reshape(-1),
        np.asarray(union, dtype=np.float64).reshape(-1),
        np.asarray(target, dtype=np.float64).reshape(-1),
    )


@HOOKS.register_module()
class ClsEvaluator(HookBase):
    _METRICS = ("mIoU", "mAcc", "allAcc")

    def __init__(self, metric="allAcc"):
        if metric not in self._METRICS:
            raise ValueError(
                f"ClsEvaluator metric must be one of {self._METRICS}, got {metric!r}"
            )
        self.metric = metric
        self._best_m_iou = -np.inf

    def before_train(self):
        # On resume, CheckpointLoader.before_train (registered first in cfg.hooks)
        # already restored self._best_m_iou via load_state_dict — don't clobber it.
        if not self.trainer.cfg.resume:
            self._best_m_iou = -np.inf

    def state_dict(self):
        return {"best_m_iou": float(self._best_m_iou)}

    def load_state_dict(self, state):
        self._best_m_iou = float(state.get("best_m_iou", self._best_m_iou))

    def after_epoch(self):
        if self.should_evaluate():
            self.eval()

    def _metric_value(self, m_iou, m_acc, all_acc):
        if self.metric == "mIoU":
            return m_iou
        if self.metric == "mAcc":
            return m_acc
        return all_acc

    def eval(self):
        val_start = begin_val_epoch_timing()
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
        current_metric_value = self._metric_value(m_iou, m_acc, all_acc)
        self._best_m_iou = max(self._best_m_iou, m_iou)
        if self.metric == "mIoU":
            metric_best = self._best_m_iou
        else:
            metric_best = max(self.trainer.best_metric_value, current_metric_value)
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            self.trainer.writer.add_scalar(
                "val/mIoU_best", self._best_m_iou, current_epoch
            )
            if self.metric != "mIoU":
                self.trainer.writer.add_scalar(
                    f"val/{self.metric}_best", metric_best, current_epoch
                )
        wandb_log = None
        if self.trainer.cfg.enable_wandb and comm.is_main_process():
            wandb_log = {
                "Epoch": current_epoch,
                "val/loss": loss_avg,
                "val/mIoU": m_iou,
                "val/mAcc": m_acc,
                "val/allAcc": all_acc,
                "val/mIoU_best": self._best_m_iou,
            }
            if self.metric != "mIoU":
                wandb_log[f"val/{self.metric}_best"] = metric_best
        finalize_val_epoch_timing(
            self.trainer, val_start, current_epoch, wandb_dict=wandb_log
        )
        if wandb_log is not None:
            wandb.log(wandb_log)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = current_metric_value
        self.trainer.comm_info["current_metric_name"] = self.metric

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format(self.metric, self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    """Single-task semantic segmentation validation (seg_logits vs segment)."""

    def __init__(self, write_cls_iou=False):
        self.write_cls_iou = write_cls_iou

    def after_epoch(self):
        if self.should_evaluate():
            self.eval()

    def eval(self):
        val_start = begin_val_epoch_timing()
        if comm.is_main_process():
            self.trainer.logger.info(
                ">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>"
            )
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
                pred = remap_pred_with_inverse(
                    pred,
                    input_dict["inverse"],
                    input_dict.get("offset"),
                    input_dict.get("origin_offset"),
                )
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            # Accumulate locally; all_reduce once after the loop.
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            if comm.is_main_process():
                info = "Test: [{iter}/{max_iter}] ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader)
                )
                if "origin_coord" in input_dict.keys():
                    info = "Interp. " + info
                self.trainer.logger.info(
                    info
                    + "Loss {loss:.4f} ".format(
                        iter=i + 1,
                        max_iter=len(self.trainer.val_loader),
                        loss=loss.item(),
                    )
                )
        val_histories = self.trainer.storage.histories()
        loss_avg = (
            val_histories["val_loss"].avg if "val_loss" in val_histories else 0.0
        )
        intersection, union, target = local_semseg_confusion_hist_totals(
            self.trainer.storage, self.trainer.cfg.data.num_classes
        )
        intersection, union, target = sync_confusion_hist_totals(
            intersection, union, target
        )
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = mean_iou_from_hist(intersection, union)
        valid = union != 0
        m_acc = float(np.mean(acc_class[valid])) if valid.any() else 0.0
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        if comm.is_main_process():
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
            wandb_log = None
            if self.trainer.cfg.enable_wandb:
                wandb_log = {
                    "Epoch": current_epoch,
                    "val/loss": loss_avg,
                    "val/mIoU": m_iou,
                    "val/mIoU_best": m_iou_best,
                    "val/mAcc": m_acc,
                    "val/allAcc": all_acc,
                }
            finalize_val_epoch_timing(
                self.trainer, val_start, current_epoch, wandb_dict=wandb_log
            )
            if wandb_log is not None:
                wandb.log(wandb_log)
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
                        )
        else:
            finalize_val_epoch_timing(self.trainer, val_start, current_epoch)
        if comm.is_main_process():
            self.trainer.logger.info(
                "<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<"
            )
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class RegressionEvaluator(HookBase):
    """Single-task point-wise regression validation (reg_pred vs cfg.data.target_key)."""

    def __init__(self):
        self._best_neg_rmse = float("-inf")

    def state_dict(self):
        return {"best_neg_rmse": float(self._best_neg_rmse)}

    def load_state_dict(self, state):
        self._best_neg_rmse = float(state.get("best_neg_rmse", self._best_neg_rmse))

    def after_epoch(self):
        if self.should_evaluate():
            self.eval()

    @staticmethod
    def _cfg_get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _target_key(self):
        data_cfg = self.trainer.cfg.data
        target_key = self._cfg_get(data_cfg, "target_key", None)
        if target_key is not None:
            return str(target_key)
        raise ValueError("RegressionEvaluator requires cfg.data.target_key.")

    @staticmethod
    def _origin_target_key(target_key):
        return f"origin_{target_key}"

    @staticmethod
    def _gather_masked(pred, target):
        mask = torch.isfinite(pred) & torch.isfinite(target)
        if mask.sum() == 0:
            return None, None
        return pred[mask], target[mask]

    def eval(self):
        val_start = begin_val_epoch_timing()
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        target_key = self._target_key()
        origin_key = self._origin_target_key(target_key)
        target_scales = self._cfg_get(self.trainer.cfg.data, "target_scales", {}) or {}
        reg_sums = {"mae": 0.0, "mse": 0.0, "count": 0.0}

        loader = self.trainer.val_loader
        n_batches = len(loader)
        with EvaluationProgressBar(
            n_batches,
            desc="Val regression",
        ) as prog:
            for i, input_dict in enumerate(loader):
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    output_dict = self.trainer.model(input_dict)
                loss = output_dict["loss"]
                pred = output_dict["reg_pred"].reshape(-1).float()
                target = input_dict[target_key].reshape(-1).float()
                if "inverse" in input_dict.keys():
                    if origin_key in input_dict:
                        pred = remap_pred_with_inverse(
                            pred,
                            input_dict["inverse"],
                            input_dict.get("offset"),
                            input_dict.get("origin_offset"),
                        )
                        target = input_dict[origin_key].reshape(-1).float()
                        scale = get_regression_target_scale(target_scales, target_key)
                        pred = denorm_regression_prediction(pred, scale)
                p, t = self._gather_masked(pred, target)
                if p is not None:
                    mae_b, mse_b, cnt_b = accumulate_regression_errors(p, t)
                    reg_sums["mae"] += mae_b
                    reg_sums["mse"] += mse_b
                    reg_sums["count"] += cnt_b
                self.trainer.storage.put_scalar("val_loss", loss.item())
                postfix = dict(loss=float(loss.item()))
                if "origin_coord" in input_dict.keys():
                    postfix["interp"] = True
                prog.step(**postfix)

        loss_avg = self.trainer.storage.history("val_loss").avg
        if comm.get_world_size() > 1:
            buf = torch.tensor(
                [reg_sums["mae"], reg_sums["mse"], reg_sums["count"]],
                dtype=torch.float64,
                device="cuda",
            )
            dist.all_reduce(buf)
            flat = buf.cpu().tolist()
            reg_sums["mae"] = flat[0]
            reg_sums["mse"] = flat[1]
            reg_sums["count"] = flat[2]

        current_epoch = self.trainer.epoch + 1
        cnt = reg_sums["count"]
        writer = self.trainer.writer
        enable_wandb = self.trainer.cfg.enable_wandb
        current_metric_value = float("-inf")
        reg_wandb = {"Epoch": current_epoch, "val/loss": loss_avg}
        if cnt > 1e-8:
            mae = reg_sums["mae"] / cnt
            rmse = (reg_sums["mse"] / cnt) ** 0.5
            current_metric_value = -rmse
            self.trainer.logger.info(
                "Val regression: MAE {:.6f} RMSE {:.6f} (n={:.0f}).".format(
                    mae, rmse, cnt
                )
            )
            if writer is not None:
                writer.add_scalar("val/loss", loss_avg, current_epoch)
                writer.add_scalar(f"val/reg/{target_key}/mae", mae, current_epoch)
                writer.add_scalar(f"val/reg/{target_key}/rmse", rmse, current_epoch)
            reg_wandb[f"val/reg/{target_key}/mae"] = float(mae)
            reg_wandb[f"val/reg/{target_key}/rmse"] = float(rmse)

        if current_metric_value > float("-inf"):
            self._best_neg_rmse = max(self._best_neg_rmse, current_metric_value)
            if writer is not None:
                writer.add_scalar(
                    "val/reg/rmse_best_neg", self._best_neg_rmse, current_epoch
                )
            reg_wandb["val/reg/rmse_best_neg"] = float(self._best_neg_rmse)
        elif writer is not None:
            writer.add_scalar("val/loss", loss_avg, current_epoch)

        wandb_log = (
            reg_wandb
            if enable_wandb and comm.is_main_process() and len(reg_wandb) > 1
            else None
        )
        finalize_val_epoch_timing(
            self.trainer, val_start, current_epoch, wandb_dict=wandb_log
        )
        if wandb_log is not None:
            wandb.log(wandb_log)

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = current_metric_value
        self.trainer.comm_info["current_metric_name"] = "neg_RMSE"

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("neg_RMSE", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class MultiTaskEvaluator(HookBase):
    """Multi-task validation: semantic IoU from seg_logits_by_task; regression from reg_pred_by_task.

    Reads cfg.data.task_configs and cfg.data.main_task. Every task_configs entry must set "task_type"
    to "semantic" or "regression". Task targets are read from input_dict[task_name] for both
    semantic and regression tasks, together with reg_pred_by_task[task_name] for regression.

    Checkpoint selection (comm_info) follows mIoU of main_task, which must be a semantic task.
    Regression metrics (MAE, RMSE) are logged per regression task name.

    The "main task" (checkpoint / mIoU selection) is cfg.data.main_task when multiple tasks are
    configured; with a single task in task_configs, main_task may be omitted. TensorBoard / W&B
    semantic metrics use val/<task_name>/... for every semantic task, including mIoU_best.
    """

    def __init__(self, write_cls_iou=False):
        self.write_cls_iou = write_cls_iou
        self._best_neg_rmse = float("-inf")
        self._best_miou_by_task = {}

    def state_dict(self):
        return {
            "best_neg_rmse": float(self._best_neg_rmse),
            "best_miou_by_task": dict(self._best_miou_by_task),
        }

    def load_state_dict(self, state):
        self._best_neg_rmse = float(state.get("best_neg_rmse", self._best_neg_rmse))
        self._best_miou_by_task = dict(state.get("best_miou_by_task", {}))

    def after_epoch(self):
        if self.should_evaluate():
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
    def _pixel_semantic_task_names(task_configs):
        return [
            k
            for k, task_config in task_configs.items()
            if task_config.get("task_type") == "pixel_semantic"
        ]

    @staticmethod
    def _regression_task_names(task_configs):
        return [
            k
            for k, task_config in task_configs.items()
            if task_config.get("task_type") == "regression"
        ]

    @staticmethod
    def _classification_task_names(task_configs):
        return [
            k
            for k, task_config in task_configs.items()
            if task_config.get("task_type") == "classification"
        ]

    @staticmethod
    def _multilabel_classification_task_names(task_configs):
        return [
            k
            for k, task_config in task_configs.items()
            if task_config.get("task_type") == "multilabel_classification"
        ]

    @staticmethod
    def _tile_distribution_task_names(task_configs):
        return [
            k
            for k, task_config in task_configs.items()
            if task_config.get("task_type") == "tile_distribution"
        ]

    @staticmethod
    def _tile_distribution_pointwise_miou_task_names(task_configs):
        """tile_distribution axes trained with CE that also report point-wise mIoU."""
        return [
            k
            for k, task_config in task_configs.items()
            if task_config.get("task_type") == "tile_distribution"
            and task_config.get("pointwise_supervision")
        ]

    @staticmethod
    def _sync_multilabel_stats(stats: MultilabelStats):
        if comm.get_world_size() <= 1:
            return stats
        tp, fp, fn, hamming, subset = multilabel_stats_to_tensors(stats)
        flat = np.concatenate(
            [tp.reshape(-1), fp.reshape(-1), fn.reshape(-1), hamming, subset]
        )
        buf = torch.tensor(flat, dtype=torch.float64, device="cuda")
        dist.all_reduce(buf)
        flat = buf.cpu().numpy()
        c = tp.size
        tp = flat[0:c]
        fp = flat[c : 2 * c]
        fn = flat[2 * c : 3 * c]
        hamming = flat[3 * c : 3 * c + 2]
        subset = flat[3 * c + 2 : 3 * c + 4]
        return multilabel_stats_from_tensors(tp, fp, fn, hamming, subset)

    def _log_multilabel_task_metrics(
        self,
        task_name,
        metrics,
        current_epoch,
        wandb_log,
    ):
        prefix = f"val/{task_name}"
        self.trainer.logger.info(
            "[task={}] Val multilabel: macro_f1/micro_f1/subset_acc/hamming_acc "
            "{:.4f}/{:.4f}/{:.4f}/{:.4f}.".format(
                task_name,
                metrics["macro_f1"],
                metrics["micro_f1"],
                metrics["subset_accuracy"],
                metrics["hamming_accuracy"],
            )
        )
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                f"{prefix}/macro_f1", metrics["macro_f1"], current_epoch
            )
            self.trainer.writer.add_scalar(
                f"{prefix}/micro_f1", metrics["micro_f1"], current_epoch
            )
            self.trainer.writer.add_scalar(
                f"{prefix}/subset_acc", metrics["subset_accuracy"], current_epoch
            )
            self.trainer.writer.add_scalar(
                f"{prefix}/hamming_acc", metrics["hamming_accuracy"], current_epoch
            )
        if wandb_log is not None:
            wandb_log[f"{prefix}/macro_f1"] = float(metrics["macro_f1"])
            wandb_log[f"{prefix}/micro_f1"] = float(metrics["micro_f1"])
            wandb_log[f"{prefix}/subset_acc"] = float(metrics["subset_accuracy"])
            wandb_log[f"{prefix}/hamming_acc"] = float(metrics["hamming_accuracy"])
        for label_name, label_metrics in metrics["per_label"].items():
            slug = "".join(
                c if (c.isalnum() or c in "._-") else "_"
                for c in str(label_name).strip().replace(" ", "_")
            )
            self.trainer.logger.info(
                "[task={}] Label {}: precision/recall/f1 {:.4f}/{:.4f}/{:.4f}".format(
                    task_name,
                    label_name,
                    label_metrics["precision"],
                    label_metrics["recall"],
                    label_metrics["f1"],
                )
            )
            if self.trainer.writer is not None:
                self.trainer.writer.add_scalar(
                    f"{prefix}/{slug}/f1", label_metrics["f1"], current_epoch
                )
                self.trainer.writer.add_scalar(
                    f"{prefix}/{slug}/precision",
                    label_metrics["precision"],
                    current_epoch,
                )
                self.trainer.writer.add_scalar(
                    f"{prefix}/{slug}/recall",
                    label_metrics["recall"],
                    current_epoch,
                )
            if wandb_log is not None:
                wandb_log[f"{prefix}/{slug}/f1"] = float(label_metrics["f1"])
                wandb_log[f"{prefix}/{slug}/precision"] = float(
                    label_metrics["precision"]
                )
                wandb_log[f"{prefix}/{slug}/recall"] = float(label_metrics["recall"])

    def eval(self):
        val_start = begin_val_epoch_timing()
        if comm.is_main_process():
            self.trainer.logger.info(
                ">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>"
            )
        self.trainer.model.eval()
        task_configs = self._get_task_configs()
        main_task = self._main_task_name(task_configs)
        semantic_tasks = self._semantic_task_names(task_configs)
        pixel_semantic_tasks = self._pixel_semantic_task_names(task_configs)
        regression_tasks = self._regression_task_names(task_configs)
        classification_tasks = self._classification_task_names(task_configs)
        multilabel_tasks = self._multilabel_classification_task_names(task_configs)
        tile_distribution_tasks = self._tile_distribution_task_names(task_configs)
        tile_pw_miou_tasks = self._tile_distribution_pointwise_miou_task_names(
            task_configs
        )

        reg_sums = {
            t: {"mae": 0.0, "mse": 0.0, "count": 0.0} for t in regression_tasks
        }
        td_sums = {
            t: {
                "kl_weighted": 0.0,
                "weight": 0.0,
                "abs_weighted": np.zeros(
                    int(task_configs[t]["num_classes"]), dtype=np.float64
                ),
            }
            for t in tile_distribution_tasks
        }
        target_scales = self._cfg_get(self.trainer.cfg.data, "target_scales", {}) or {}
        multilabel_stats = {t: MultilabelStats() for t in multilabel_tasks}

        loader = self.trainer.val_loader
        n_batches = len(loader)
        with EvaluationProgressBar(
            n_batches,
            desc="Val multitask",
        ) as prog:
            for i, input_dict in enumerate(loader):
                # --------- Model forward ---------
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    output_dict = self.trainer.model(input_dict)

                loss = output_dict["loss"]
                loss_by_task = output_dict.get("loss_by_task")
                if (
                    getattr(self.trainer.cfg, "grad_norm_lite", False)
                    and isinstance(loss_by_task, dict)
                    and loss_by_task
                ):
                    model = self.trainer.model
                    if hasattr(model, "module"):
                        model = model.module
                    ema = getattr(self.trainer, "_grad_norm_lite_ema", None)
                    task_groups = getattr(
                        self.trainer.cfg, "grad_norm_lite_task_groups", None
                    )
                    scales = (
                        resolve_grad_norm_lite_scales(
                            ema, loss_by_task.keys(), task_groups
                        )
                        if ema is not None
                        else None
                    )
                    loss, _ = combine_weighted_task_losses(
                        loss_by_task,
                        getattr(model, "task_weights", {}),
                        scales,
                    )
                elif (
                    getattr(self.trainer.cfg, "grad_norm", False)
                    and getattr(self.trainer, "_grad_norm_state", None) is not None
                    and isinstance(loss_by_task, dict)
                    and loss_by_task
                ):
                    # Real GradNorm: report val/loss under the learned weights so
                    # it tracks the training objective.
                    model = self.trainer.model
                    if hasattr(model, "module"):
                        model = model.module
                    task_groups = getattr(
                        self.trainer.cfg, "grad_norm_task_groups", None
                    )
                    scales = self.trainer._grad_norm_state.per_task_scales(
                        loss_by_task.keys(), task_groups
                    )
                    loss, _ = combine_weighted_task_losses(
                        loss_by_task,
                        getattr(model, "task_weights", {}),
                        scales,
                    )

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
                            pred = remap_pred_with_inverse(
                                pred,
                                input_dict["inverse"],
                                input_dict.get("offset"),
                                input_dict.get("origin_offset"),
                            )
                            target_tensor = input_dict[origin_target_key]
                    intersection, union, target = intersection_and_union_gpu(
                        pred,
                        target_tensor,
                        int(task_config["num_classes"]),
                        int(task_config["ignore_index"]),
                    )
                    # Accumulate locally; all_reduce once after the loop (avoids
                    # NCCL deadlock when ranks have unequal val batch counts).
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

                # --------- Evaluate Pixel-Semantic Tasks (network 2D) ---------
                pixel_logits_by_task = output_dict.get("pixel_seg_logits_by_task") or {}
                pixel_targets_by_task = output_dict.get("pixel_seg_target_by_task") or {}
                pixel_dense_by_task = (
                    output_dict.get("pixel_seg_logits_dense_by_task") or {}
                )
                pixel_dense_target_by_task = (
                    output_dict.get("pixel_seg_target_dense_by_task") or {}
                )
                for task_name in pixel_semantic_tasks:
                    task_config = task_configs[task_name]
                    if (
                        task_name not in pixel_logits_by_task
                        or task_name not in pixel_targets_by_task
                    ):
                        continue
                    logits = pixel_logits_by_task[task_name]  # (P, r, C)
                    targets = pixel_targets_by_task[task_name]  # (P, r)
                    num_classes = int(task_config["num_classes"])
                    ignore_index = int(task_config["ignore_index"])
                    num_networks = int(task_config.get("num_networks", logits.shape[1]))
                    channel_names = task_config.get("channel_names") or [
                        f"ch{c}" for c in range(num_networks)
                    ]
                    # Aggregate confusion over all network channels (binary like forest).
                    inter_sum = None
                    union_sum = None
                    target_sum = None
                    for c in range(num_networks):
                        pred_c = logits[:, c, :].max(1)[1]
                        target_c = targets[:, c]
                        intersection, union, target = intersection_and_union_gpu(
                            pred_c,
                            target_c,
                            num_classes,
                            ignore_index,
                        )
                        intersection = intersection.cpu().numpy()
                        union = union.cpu().numpy()
                        target = target.cpu().numpy()
                        if inter_sum is None:
                            inter_sum, union_sum, target_sum = (
                                intersection,
                                union,
                                target,
                            )
                        else:
                            inter_sum = inter_sum + intersection
                            union_sum = union_sum + union
                            target_sum = target_sum + target
                        # Per-channel foreground confusion storage for logging.
                        ch_name = channel_names[c] if c < len(channel_names) else f"ch{c}"
                        self.trainer.storage.put_scalar(
                            f"val_intersection/{task_name}/{ch_name}", intersection
                        )
                        self.trainer.storage.put_scalar(
                            f"val_union/{task_name}/{ch_name}", union
                        )
                        self.trainer.storage.put_scalar(
                            f"val_target/{task_name}/{ch_name}", target
                        )
                    if inter_sum is not None:
                        self.trainer.storage.put_scalar(
                            f"val_intersection/{task_name}", inter_sum
                        )
                        self.trainer.storage.put_scalar(
                            f"val_union/{task_name}", union_sum
                        )
                        self.trainer.storage.put_scalar(
                            f"val_target/{task_name}", target_sum
                        )

                    # ---- Buffer ("relaxed") precision/recall hit accumulation ----
                    # Uses the dense per-scene grids (reconstructed on the GT Lambert
                    # grid) so the tolerance dilation has real 2D neighborhoods to work
                    # with; the (P, r) pooled arrays above have no spatial layout.
                    # Skipped entirely for tasks that opt out (e.g. forest_2d): the
                    # buffer tolerance is a diagnostic for thin curvilinear masks
                    # (road/rail), not meaningful for a blobby area-coverage mask.
                    if dilated_prf_enabled(task_config):
                        dense_pred_list = pixel_dense_by_task.get(task_name) or []
                        dense_gt_list = pixel_dense_target_by_task.get(task_name) or []
                        radius_px = int(task_config.get("buffer_radius_px", 3))
                        for dense_pred, dense_gt in zip(dense_pred_list, dense_gt_list):
                            if dense_pred is None or dense_gt is None:
                                continue
                            dense_pred_np = dense_pred.detach().cpu().numpy()
                            dense_gt_np = dense_gt.detach().cpu().numpy()
                            for c in range(num_networks):
                                ch_name = (
                                    channel_names[c] if c < len(channel_names) else f"ch{c}"
                                )
                                pred_prob = dense_pred_np[c]
                                gt = dense_gt_np[c]
                                # Unobserved cells: pred_prob is NaN, gt is -1 (see
                                # models/default.py _compute_pixel_logits). Void GT cells
                                # (ignore_index) are excluded like plain IoU excludes them.
                                observed = ~np.isnan(pred_prob) & (gt != -1)
                                valid = observed & (gt != ignore_index)
                                pred_fg = (pred_prob > 0.5) & valid
                                gt_fg = (gt == 1) & valid
                                p_num, p_denom, r_num, r_denom = (
                                    dilated_precision_recall_counts(
                                        pred_fg, gt_fg, valid, radius_px=radius_px
                                    )
                                )
                                base = f"val_dilated_prf/{task_name}/{ch_name}"
                                self.trainer.storage.put_scalar(f"{base}/p_num", p_num)
                                self.trainer.storage.put_scalar(f"{base}/p_denom", p_denom)
                                self.trainer.storage.put_scalar(f"{base}/r_num", r_num)
                                self.trainer.storage.put_scalar(f"{base}/r_denom", r_denom)

                cls_logits_by_task = output_dict.get("cls_logits_by_task") or {}
                for task_name in classification_tasks:
                    task_config = task_configs[task_name]
                    if task_name not in input_dict or task_name not in cls_logits_by_task:
                        continue
                    pred = cls_logits_by_task[task_name].max(1)[1]
                    target_tensor = input_dict[task_name].reshape(-1).long()
                    intersection, union, target = intersection_and_union_gpu(
                        pred,
                        target_tensor,
                        int(task_config["num_classes"]),
                        int(task_config["ignore_index"]),
                    )
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

                for task_name in multilabel_tasks:
                    task_config = task_configs[task_name]
                    if task_name not in input_dict or task_name not in cls_logits_by_task:
                        continue
                    threshold = float(task_config.get("threshold", 0.5))
                    logits = cls_logits_by_task[task_name]
                    pred = (torch.sigmoid(logits) >= threshold).cpu().numpy()
                    target_tensor = input_dict[task_name].float()
                    if target_tensor.ndim == 1:
                        target_tensor = target_tensor.unsqueeze(0)
                    target = target_tensor.cpu().numpy()
                    accumulate_multilabel_stats(
                        pred,
                        target,
                        multilabel_stats[task_name],
                        ignore_index=task_config.get("ignore_index"),
                    )

                # --------- Evaluate Tile-Distribution Tasks (nathab axes) ---------
                for task_name in tile_distribution_tasks:
                    task_config = task_configs[task_name]
                    if task_name not in input_dict or task_name not in logits_by_task:
                        continue
                    target_tensor = input_dict[task_name].reshape(-1).long()
                    num_classes = int(task_config["num_classes"])
                    ignore_index = int(task_config["ignore_index"])
                    probs = torch.softmax(logits_by_task[task_name].float(), dim=-1)
                    indptr = torch.nn.functional.pad(input_dict["offset"], (1, 0))
                    pi_hat, q_t, n_t = pool_axis_distribution_from_probs(
                        probs, target_tensor, indptr, ignore_index, num_classes
                    )
                    keep = n_t > 0
                    if keep.any():
                        kl = kl_divergence_rows(q_t[keep], pi_hat[keep])
                        td_sums[task_name]["kl_weighted"] += float(
                            (n_t[keep] * kl).sum().item()
                        )
                        td_sums[task_name]["weight"] += float(n_t[keep].sum().item())
                        abs_err = abs_freq_error_rows(pi_hat[keep], q_t[keep])
                        td_sums[task_name]["abs_weighted"] += (
                            (n_t[keep].unsqueeze(-1) * abs_err)
                            .sum(0)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )

                    # Optional point-wise mIoU (CE / Lovasz supervision mode).
                    if task_name in tile_pw_miou_tasks:
                        pred = logits_by_task[task_name].max(1)[1]
                        miou_target = input_dict[task_name]
                        if "inverse" in input_dict.keys():
                            origin_target_key = self._task_origin_target_key(task_name)
                            if origin_target_key in input_dict:
                                pred = remap_pred_with_inverse(
                                    pred,
                                    input_dict["inverse"],
                                    input_dict.get("offset"),
                                    input_dict.get("origin_offset"),
                                )
                                miou_target = input_dict[origin_target_key]
                        intersection, union, target = intersection_and_union_gpu(
                            pred,
                            miou_target,
                            num_classes,
                            ignore_index,
                        )
                        intersection, union, target = (
                            intersection.cpu().numpy(),
                            union.cpu().numpy(),
                            target.cpu().numpy(),
                        )
                        self.trainer.storage.put_scalar(
                            f"val_intersection/{task_name}", intersection
                        )
                        self.trainer.storage.put_scalar(
                            f"val_union/{task_name}", union
                        )
                        self.trainer.storage.put_scalar(
                            f"val_target/{task_name}", target
                        )

                self.trainer.storage.put_scalar("val_loss", loss.item())
                # Skip when only one task: scalar "loss" already logged
                if isinstance(loss_by_task, dict) and len(loss_by_task) > 1:
                    for task_name, v in loss_by_task.items():
                        if isinstance(v, torch.Tensor):
                            if v.numel() != 1:
                                continue
                            self.trainer.storage.put_scalar(
                                f"val_loss/{task_name}", float(v.item())
                            )
                        elif isinstance(v, (int, float)):
                            self.trainer.storage.put_scalar(
                                f"val_loss/{task_name}", float(v)
                            )

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
                            pred = remap_pred_with_inverse(
                                pred,
                                input_dict["inverse"],
                                input_dict.get("offset"),
                                input_dict.get("origin_offset"),
                            )
                            target = input_dict[origin_key].reshape(-1).float()
                            scale = get_regression_target_scale(
                                target_scales, task_name
                            )
                            pred = denorm_regression_prediction(pred, scale)
                    p, t = self._gather_masked(pred, target)
                    if p is not None:
                        mae_b, mse_b, cnt_b = accumulate_regression_errors(p, t)
                        reg_sums[task_name]["mae"] += mae_b
                        reg_sums[task_name]["mse"] += mse_b
                        reg_sums[task_name]["count"] += cnt_b

                postfix = dict(loss=float(loss.item()))
                if "origin_coord" in input_dict.keys():
                    postfix["interp"] = True
                prog.step(**postfix)
        val_histories = self.trainer.storage.histories()
        loss_avg = (
            val_histories["val_loss"].avg if "val_loss" in val_histories else 0.0
        )
        if comm.get_world_size() > 1 and regression_tasks:
            flat = []
            for task_name in regression_tasks:
                s = reg_sums[task_name]
                flat.extend([s["mae"], s["mse"], s["count"]])
            buf = torch.tensor(flat, dtype=torch.float64, device="cuda")
            dist.all_reduce(buf)
            flat = buf.cpu().tolist()
            for task_index, task_name in enumerate(regression_tasks):
                base = task_index * 3
                reg_sums[task_name]["mae"] = flat[base]
                reg_sums[task_name]["mse"] = flat[base + 1]
                reg_sums[task_name]["count"] = flat[base + 2]

        if comm.get_world_size() > 1 and tile_distribution_tasks:
            flat = []
            for task_name in tile_distribution_tasks:
                s = td_sums[task_name]
                flat.extend([s["kl_weighted"], s["weight"]])
                flat.extend(s["abs_weighted"].tolist())
            buf = torch.tensor(flat, dtype=torch.float64, device="cuda")
            dist.all_reduce(buf)
            flat = buf.cpu().tolist()
            offset = 0
            for task_name in tile_distribution_tasks:
                C = len(td_sums[task_name]["abs_weighted"])
                td_sums[task_name]["kl_weighted"] = flat[offset]
                td_sums[task_name]["weight"] = flat[offset + 1]
                td_sums[task_name]["abs_weighted"] = np.asarray(
                    flat[offset + 2 : offset + 2 + C], dtype=np.float64
                )
                offset += 2 + C

        final_kl_by_task = {}
        final_mae_by_task = {}
        final_tv_by_task = {}
        for task_name in tile_distribution_tasks:
            s = td_sums[task_name]
            if s["weight"] > 1e-8:
                final_kl_by_task[task_name] = s["kl_weighted"] / s["weight"]
                mae = s["abs_weighted"] / s["weight"]
                final_mae_by_task[task_name] = mae
                final_tv_by_task[task_name] = float(
                    tv_from_abs_errors(torch.as_tensor(mae))
                )

        per_task_metrics = {}
        metric_task_names = (
            semantic_tasks
            + classification_tasks
            + pixel_semantic_tasks
            + tile_pw_miou_tasks
        )
        for task_name in metric_task_names:
            task_config = task_configs[task_name]
            num_classes = int(task_config["num_classes"])
            intersection, union, target = local_task_confusion_hist_totals(
                self.trainer.storage, task_name, num_classes
            )
            intersection, union, target = sync_confusion_hist_totals(
                intersection, union, target
            )
            iou_class = intersection / (union + 1e-10)
            acc_class = intersection / (target + 1e-10)
            m_iou = mean_iou_from_hist(intersection, union)
            m_acc = mean_acc_from_hist(intersection, target, union=union)
            all_acc = sum(intersection) / (sum(target) + 1e-10)
            per_task_metrics[task_name] = dict(
                iou_class=iou_class,
                acc_class=acc_class,
                m_iou=m_iou,
                m_acc=m_acc,
                all_acc=all_acc,
                names=list(task_config["names"]),
            )
            if comm.is_main_process():
                self.trainer.logger.info(
                    "[task={}] Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                        task_name, m_iou, m_acc, all_acc
                    )
                )
                for class_idx in range(num_classes):
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

        # Per-channel precision/recall/F1 for pixel-semantic tasks (e.g. network
        # ROADS / RAILROADS): plain IoU is not a good diagnostic for thin curvilinear
        # masks (a 1 px lateral offset destroys IoU on a 1 px line while still being a
        # good prediction), so we report exact P/R/F1 (derived from the confusion
        # histograms, no new accumulation needed) alongside buffer ("relaxed") P/R/F1
        # with a configurable pixel tolerance (accumulated during the val loop).
        # Everything here was already stored per channel during the val loop; sync on
        # all ranks (all_reduce) then log. Must not be gated on is_main_process.
        per_task_channel_metrics = {}
        for task_name in pixel_semantic_tasks:
            task_config = task_configs[task_name]
            num_classes = int(task_config["num_classes"])
            num_networks = int(task_config.get("num_networks", 1))
            channel_names = task_config.get("channel_names") or [
                f"ch{c}" for c in range(num_networks)
            ]
            names = list(task_config["names"])
            fg_idx = names.index("Foreground") if "Foreground" in names else 1
            radius_px = int(task_config.get("buffer_radius_px", 3))
            per_task_channel_metrics[task_name] = {}
            for c in range(num_networks):
                ch_name = (
                    channel_names[c] if c < len(channel_names) else f"ch{c}"
                )
                intersection, union, target = local_task_confusion_hist_totals(
                    self.trainer.storage,
                    f"{task_name}/{ch_name}",
                    num_classes,
                )
                intersection, union, target = sync_confusion_hist_totals(
                    intersection, union, target
                )
                tp = intersection[fg_idx]
                area_pred = union[fg_idx] - target[fg_idx] + tp
                exact_precision, exact_recall, exact_f1 = precision_recall_f1(
                    tp, area_pred, tp, target[fg_idx]
                )

                dilated_enabled = dilated_prf_enabled(task_config)
                channel_metrics = dict(
                    precision=exact_precision,
                    recall=exact_recall,
                    f1=exact_f1,
                )
                if dilated_enabled:
                    p_num, p_denom, r_num, r_denom = local_dilated_prf_totals(
                        self.trainer.storage, task_name, ch_name
                    )
                    p_num, p_denom, r_num, r_denom = sync_dilated_prf_totals(
                        p_num, p_denom, r_num, r_denom
                    )
                    dilated_precision, dilated_recall, dilated_f1 = precision_recall_f1(
                        p_num, p_denom, r_num, r_denom
                    )
                    channel_metrics.update(
                        dilated_precision=dilated_precision,
                        dilated_recall=dilated_recall,
                        dilated_f1=dilated_f1,
                    )
                per_task_channel_metrics[task_name][ch_name] = channel_metrics
                if comm.is_main_process():
                    if dilated_enabled:
                        self.trainer.logger.info(
                            "[task={}] Channel_{}-{} Result: precision/recall/f1 "
                            "{:.4f}/{:.4f}/{:.4f} | dilated(r={}px) precision/recall/f1 "
                            "{:.4f}/{:.4f}/{:.4f}".format(
                                task_name,
                                c,
                                ch_name,
                                exact_precision,
                                exact_recall,
                                exact_f1,
                                radius_px,
                                dilated_precision,
                                dilated_recall,
                                dilated_f1,
                            )
                        )
                    else:
                        self.trainer.logger.info(
                            "[task={}] Channel_{}-{} Result: precision/recall/f1 "
                            "{:.4f}/{:.4f}/{:.4f}".format(
                                task_name,
                                c,
                                ch_name,
                                exact_precision,
                                exact_recall,
                                exact_f1,
                            )
                        )

        multilabel_metrics_by_task = {}
        for task_name in multilabel_tasks:
            stats = self._sync_multilabel_stats(multilabel_stats[task_name])
            multilabel_metrics_by_task[task_name] = compute_multilabel_metrics(
                stats, task_configs[task_name]["names"]
            )

        current_epoch = self.trainer.epoch + 1
        main_task_type = task_configs[main_task].get("task_type")
        main_m_iou = per_task_metrics[main_task]["m_iou"] if main_task in per_task_metrics else None
        miou_best_by_task = {}
        for task_name, metric in per_task_metrics.items():
            task_m_iou = metric["m_iou"]
            prev_best = self._best_miou_by_task.get(task_name, float("-inf"))
            if task_name == main_task:
                prev_best = max(prev_best, self.trainer.best_metric_value)
            task_m_iou_best = max(prev_best, task_m_iou)
            self._best_miou_by_task[task_name] = task_m_iou_best
            miou_best_by_task[task_name] = task_m_iou_best

        wandb_log = None
        if (
            comm.is_main_process()
            and self.trainer.cfg.enable_wandb
            and wandb.run is not None
        ):
            wandb_log = {"Epoch": current_epoch, "val/loss": loss_avg}
            val_histories = self.trainer.storage.histories()
            for task_name in task_configs:
                vk = f"val_loss/{task_name}"
                if vk not in val_histories:
                    continue
                task_loss_avg = val_histories[vk].avg
                wandb_log[metric_tag("val", "loss", task=task_name)] = float(
                    task_loss_avg
                )
            for task_name, metric in per_task_metrics.items():
                wandb_log[metric_tag("val", "mIoU", task=task_name)] = float(
                    metric["m_iou"]
                )
                wandb_log[best_miou_tag("val", task=task_name)] = float(
                    miou_best_by_task[task_name]
                )
                wandb_log[metric_tag("val", "mAcc", task=task_name)] = float(
                    metric["m_acc"]
                )
                wandb_log[metric_tag("val", "allAcc", task=task_name)] = float(
                    metric["all_acc"]
                )

        if comm.is_main_process() and self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            val_histories = self.trainer.storage.histories()
            for task_name in task_configs:
                vk = f"val_loss/{task_name}"
                if vk not in val_histories:
                    continue
                task_loss_avg = val_histories[vk].avg
                self.trainer.writer.add_scalar(
                    metric_tag("val", "loss", task=task_name),
                    task_loss_avg,
                    current_epoch,
                )

            for task_name, metric in per_task_metrics.items():
                task_m_iou_best = miou_best_by_task[task_name]
                self.trainer.writer.add_scalar(
                    metric_tag("val", "mIoU", task=task_name),
                    metric["m_iou"],
                    current_epoch,
                )
                self.trainer.writer.add_scalar(
                    best_miou_tag("val", task=task_name),
                    task_m_iou_best,
                    current_epoch,
                )
                self.trainer.writer.add_scalar(
                    metric_tag("val", "mAcc", task=task_name),
                    metric["m_acc"],
                    current_epoch,
                )
                self.trainer.writer.add_scalar(
                    metric_tag("val", "allAcc", task=task_name),
                    metric["all_acc"],
                    current_epoch,
                )

        if comm.is_main_process():
            for task_name, metrics in multilabel_metrics_by_task.items():
                self._log_multilabel_task_metrics(
                    task_name, metrics, current_epoch, wandb_log=wandb_log
                )

            if self.write_cls_iou:
                for task_name, metric in per_task_metrics.items():
                    task_config = task_configs[task_name]
                    for class_idx in range(int(task_config["num_classes"])):
                        if class_idx == task_config["ignore_index"]:
                            continue
                        slug = class_name_slug(task_config["names"][class_idx])
                        tag = iou_class_tag("val", slug, task=task_name)
                        value = float(metric["iou_class"][class_idx])
                        if self.trainer.writer is not None:
                            self.trainer.writer.add_scalar(
                                tag, value, current_epoch
                            )
                        if wandb_log is not None:
                            wandb_log[tag] = value
                for task_name, channel_metrics in per_task_channel_metrics.items():
                    for ch_name, metric in channel_metrics.items():
                        ch_slug = class_name_slug(ch_name)
                        for metric_name, value in metric.items():
                            tag = metric_tag(
                                "val", f"{ch_slug}/{metric_name}", task=task_name
                            )
                            value = float(value)
                            if self.trainer.writer is not None:
                                self.trainer.writer.add_scalar(
                                    tag, value, current_epoch
                                )
                            if wandb_log is not None:
                                wandb_log[tag] = value

        best_neg_rmse_epoch = float("-inf")
        writer = self.trainer.writer
        for task_name in regression_tasks:
            s = reg_sums[task_name]
            cnt = s["count"]
            if cnt <= 1e-8:
                continue
            mae = s["mae"] / cnt
            rmse = (s["mse"] / cnt) ** 0.5
            best_neg_rmse_epoch = max(best_neg_rmse_epoch, -rmse)
            if comm.is_main_process():
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
            if wandb_log is not None:
                wandb_log[f"val/reg/{task_name}/mae"] = float(mae)
                wandb_log[f"val/reg/{task_name}/rmse"] = float(rmse)
        if best_neg_rmse_epoch > float("-inf"):
            self._best_neg_rmse = max(self._best_neg_rmse, best_neg_rmse_epoch)
            if writer is not None:
                writer.add_scalar(
                    "val/reg/rmse_best_neg", self._best_neg_rmse, current_epoch
                )
            if wandb_log is not None:
                wandb_log["val/reg/rmse_best_neg"] = float(self._best_neg_rmse)

        if tile_distribution_tasks:
            for task_name in tile_distribution_tasks:
                kl = final_kl_by_task.get(task_name)
                if kl is None:
                    continue
                tv = final_tv_by_task[task_name]
                mae = final_mae_by_task[task_name]
                task_config = task_configs[task_name]
                if comm.is_main_process():
                    self.trainer.logger.info(
                        "[task={}] Val tile-distribution: weighted KL {:.6f} "
                        "TV {:.6f} (N={:.0f}).".format(
                            task_name, kl, tv, td_sums[task_name]["weight"]
                        )
                    )
                    for class_idx in range(int(task_config["num_classes"])):
                        class_name = task_config["names"][class_idx]
                        self.trainer.logger.info(
                            "[task={}] Class_{}-{} Result: mae {:.6f}".format(
                                task_name,
                                class_idx,
                                class_name,
                                float(mae[class_idx]),
                            )
                        )
                if writer is not None:
                    writer.add_scalar(
                        metric_tag("val", f"weighted_kl/{task_name}", task="nathab"),
                        kl,
                        current_epoch,
                    )
                    writer.add_scalar(
                        metric_tag("val", f"tv/{task_name}", task="nathab"),
                        tv,
                        current_epoch,
                    )
                    for class_idx in range(int(task_config["num_classes"])):
                        slug = class_name_slug(task_config["names"][class_idx])
                        writer.add_scalar(
                            metric_tag(
                                "val", f"mae/{task_name}/{slug}", task="nathab"
                            ),
                            float(mae[class_idx]),
                            current_epoch,
                        )
                if wandb_log is not None:
                    wandb_log[
                        metric_tag("val", f"weighted_kl/{task_name}", task="nathab")
                    ] = float(kl)
                    wandb_log[
                        metric_tag("val", f"tv/{task_name}", task="nathab")
                    ] = float(tv)
                    for class_idx in range(int(task_config["num_classes"])):
                        slug = class_name_slug(task_config["names"][class_idx])
                        wandb_log[
                            metric_tag(
                                "val", f"mae/{task_name}/{slug}", task="nathab"
                            )
                        ] = float(mae[class_idx])
            if final_kl_by_task:
                kl_sum = sum(final_kl_by_task.values())
                # Unweighted mean over axes of the per-axis TV (each in [0, 1]).
                tv_mean = sum(final_tv_by_task.values()) / len(final_tv_by_task)
                if writer is not None:
                    writer.add_scalar(
                        metric_tag("val", "weighted_kl/sum", task="nathab"),
                        kl_sum,
                        current_epoch,
                    )
                    writer.add_scalar(
                        metric_tag("val", "tv/mean", task="nathab"),
                        tv_mean,
                        current_epoch,
                    )
                if wandb_log is not None:
                    wandb_log[metric_tag("val", "weighted_kl/sum", task="nathab")] = (
                        float(kl_sum)
                    )
                    wandb_log[metric_tag("val", "tv/mean", task="nathab")] = float(
                        tv_mean
                    )

        if comm.is_main_process():
            finalize_val_epoch_timing(
                self.trainer, val_start, current_epoch, wandb_dict=wandb_log
            )
            if wandb_log is not None:
                wandb.log(wandb_log)
        else:
            finalize_val_epoch_timing(self.trainer, val_start, current_epoch)

        if comm.is_main_process():
            self.trainer.logger.info(
                "<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<"
            )
        if main_task_type == "tile_distribution":
            main_kl = final_kl_by_task.get(main_task)
            if main_kl is not None:
                self.trainer.comm_info["current_metric_value"] = -main_kl
                self.trainer.comm_info["current_metric_name"] = f"-weighted_kl/{main_task}"
        else:
            self.trainer.comm_info["current_metric_value"] = main_m_iou
            self.trainer.comm_info["current_metric_name"] = f"mIoU/{main_task}"

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format(
                self.trainer.comm_info.get("current_metric_name", "mIoU"),
                self.trainer.best_metric_value,
            )
        )


