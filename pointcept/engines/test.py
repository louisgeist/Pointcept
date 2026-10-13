"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from contextlib import nullcontext
import os
import time
import traceback
import numpy as np
import wandb
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data

from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.datasets.utils import build_voxel_budget_batch_sampler
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    abs_freq_error_rows,
    accumulate_regression_errors,
    binary_prf_counts,
    f1_scores_from_hist,
    intersection_and_union,
    intersection_and_union_gpu,
    kl_divergence_rows,
    make_dirs,
    mean_acc_from_hist,
    mean_iou_from_hist,
    pool_axis_distribution_from_probs,
    tv_from_abs_errors,
)
from pointcept.utils.dilated_metrics import (
    dilated_prf_enabled,
    dilated_precision_recall_counts,
    precision_recall_f1,
)
from pointcept.utils.progress import EvaluationProgressBar
from pointcept.utils.regression import (
    denorm_regression_prediction,
    get_regression_target_scale,
)
from pointcept.utils.wandb_metrics import class_name_slug, iou_class_tag, metric_tag

TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        self.checkpoint_epoch = None
        if self.verbose and model is None:
            # if model is not none, trigger tester with trainer, no need to print config
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, weights_only=False)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.checkpoint_epoch = checkpoint.get("epoch")
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def _test_loader_worker_kwargs(self, packed):
        """DataLoader worker settings for the test loader.

        Test samples keep the full-resolution cloud plus its ``fragment_list``, so a
        packed loader (voxel budget or batch_size_test > 1) with the train worker
        count can exhaust CPU RAM (the process is killed without a CUDA error).
        ``cfg.num_worker_test`` overrides the worker count (e.g. 2 in the grid-probe
        configs); otherwise ``num_worker_per_gpu`` is used. Packed loaders also use
        ``prefetch_factor=1``.
        """
        override = getattr(self.cfg, "num_worker_test", None)
        num_workers = (
            int(override) if override is not None else int(self.cfg.num_worker_per_gpu)
        )
        kwargs = dict(num_workers=num_workers, pin_memory=True)
        if packed and num_workers > 0:
            kwargs["prefetch_factor"] = 1
            self.logger.info(
                "Test DataLoader: packed full-res tiles, "
                "num_workers=%d prefetch_factor=1 "
                "(cfg.num_worker_per_gpu=%d, cfg.num_worker_test=%s)",
                num_workers,
                self.cfg.num_worker_per_gpu,
                override,
            )
        return kwargs

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        voxel_budget = getattr(self.cfg, "test_voxel_budget", None)
        packed = voxel_budget is not None or int(self.cfg.batch_size_test_per_gpu) > 1
        worker_kwargs = self._test_loader_worker_kwargs(packed)

        if voxel_budget is not None:
            max_batch_size = int(self.cfg.batch_size_test_per_gpu)
            batch_sampler, sizes, missing_csv = build_voxel_budget_batch_sampler(
                dataset=test_dataset,
                voxel_budget=int(voxel_budget),
                max_batch_size=max_batch_size,
                rank=comm.get_rank(),
                world_size=comm.get_world_size(),
            )
            size_source = getattr(test_dataset, "csv_manifest", None) or "<unknown>"
            self.logger.info(
                "Test VoxelBudgetBatchSampler: budget=%d max_bs=%d "
                "batches_this_rank=%d dataset=%d missing_sizes=%d source=%s",
                int(voxel_budget),
                max_batch_size,
                len(batch_sampler),
                len(sizes),
                missing_csv,
                size_source,
            )
            return torch.utils.data.DataLoader(
                test_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.__class__.collate_fn,
                **worker_kwargs,
            )

        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
            **worker_kwargs,
        )

    def _uses_multi_scene_batches(self):
        """True when the loader packs multiple scenes per step.

        With a batch_sampler (e.g. VoxelBudgetBatchSampler), DataLoader.batch_size
        is None; treat that as multi-scene batching.
        """
        bs = self.test_loader.batch_size
        return bs is None or bs > 1

    def begin_test_timing(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._test_wall_start = time.perf_counter()

    def end_test_timing_and_log(self, extra_log_dict=None):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        comm.synchronize()
        total_s = time.perf_counter() - self._test_wall_start
        if comm.is_main_process():
            self.logger.info(
                "Test wall time (start -> end): %.2fs (%.4fh)",
                total_s,
                total_s / 3600.0,
            )
            timing = {
                "test/s": float(total_s),
                "test/h": float(total_s / 3600.0),
            }
            if getattr(self.cfg, "enable_wandb", False) and wandb.run is not None:
                from pointcept.utils.wandb_resume import write_local_last_step

                payload = {**(extra_log_dict or {}), **timing}
                if "Epoch" not in payload:
                    epoch = self.checkpoint_epoch
                    if epoch is None:
                        epoch = getattr(self.cfg, "epoch", None)
                    if epoch is not None:
                        payload["Epoch"] = int(epoch)
                wandb.log(payload)
                write_local_last_step(self.cfg.save_path, wandb.run.step)
        return total_s

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        if self._uses_multi_scene_batches():
            logger.info(
                "Batch size > 1 detected. Assuming test_single_fragment=True "
                "for SemSegTester."
            )
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.begin_test_timing()

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        comm.synchronize()
        record = {}
        packed = self._uses_multi_scene_batches()
        bar_cm = (
            EvaluationProgressBar(len(self.test_loader), desc="Test semseg")
            if packed
            else nullcontext()
        )
        # fragment inference
        with bar_cm as prog:
            for idx, data_dict in enumerate(self.test_loader):
                start = time.time()
                if packed:
                    self._test_packed_semseg_batch(
                        batch=data_dict,
                        save_path=save_path,
                        intersection_meter=intersection_meter,
                        union_meter=union_meter,
                        target_meter=target_meter,
                        record=record,
                        prog=prog,
                    )
                    continue
                data_dict = data_dict[0]  # current assume batch size is 1
                fragment_list = data_dict.pop("fragment_list")
                segment = data_dict.pop("segment")
                data_name = data_dict.pop("name")
                pred_save_path = os.path.join(
                    save_path, "{}_pred.npy".format(data_name)
                )
                if os.path.isfile(pred_save_path):
                    logger.info(
                        "{}/{}: {}, loaded pred and label.".format(
                            idx + 1, len(self.test_loader), data_name
                        )
                    )
                    pred = np.load(pred_save_path)
                    if "origin_segment" in data_dict.keys():
                        segment = data_dict["origin_segment"]
                else:
                    pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                    # Single-fragment mode: one point per voxel, broadcast via inverse
                    use_voxel_broadcast = "inverse" in fragment_list[0]
                    for i in range(len(fragment_list)):
                        fragment_batch_size = 1
                        s_i, e_i = i * fragment_batch_size, min(
                            (i + 1) * fragment_batch_size, len(fragment_list)
                        )
                        input_dict = collate_fn(fragment_list[s_i:e_i])
                        for key in input_dict.keys():
                            if isinstance(input_dict[key], torch.Tensor):
                                input_dict[key] = input_dict[key].cuda(
                                    non_blocking=True
                                )
                        idx_part = input_dict["index"]
                        with torch.no_grad():
                            pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                            pred_part = F.softmax(pred_part, -1)
                            if self.cfg.empty_cache:
                                torch.cuda.empty_cache()
                            if use_voxel_broadcast:
                                inv = fragment_list[s_i:e_i][0]["inverse"]
                                inv = (
                                    torch.from_numpy(inv).long().cuda()
                                    if isinstance(inv, np.ndarray)
                                    else inv.long().cuda()
                                )
                                pred += pred_part[inv, :]
                            else:
                                bs = 0
                                for be in input_dict["offset"]:
                                    pred[idx_part[bs:be], :] += pred_part[bs:be]
                                    bs = be

                        logger.info(
                            "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                                idx + 1,
                                len(self.test_loader),
                                data_name=data_name,
                                batch_idx=i,
                                batch_num=len(fragment_list),
                            )
                        )
                    pred = pred.max(1)[1].data.cpu().numpy()
                    if "origin_segment" in data_dict.keys():
                        assert "inverse" in data_dict.keys()
                        pred = pred[data_dict["inverse"]]
                        segment = data_dict["origin_segment"]
                    np.save(pred_save_path, pred)

                intersection, union, target = intersection_and_union(
                    pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
                )
                intersection_meter.update(intersection)
                union_meter.update(union)
                target_meter.update(target)
                record[data_name] = dict(
                    intersection=intersection, union=union, target=target
                )

                mask = union != 0
                iou_class = intersection / (union + 1e-10)
                iou = np.mean(iou_class[mask])
                acc = sum(intersection) / (sum(target) + 1e-10)

                m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
                m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

                batch_time.update(time.time() - start)
                logger.info(
                    "Test: {} [{}/{}]-{} "
                    "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    "Accuracy {acc:.4f} ({m_acc:.4f}) "
                    "mIoU {iou:.4f} ({m_iou:.4f})".format(
                        data_name,
                        idx + 1,
                        len(self.test_loader),
                        segment.size,
                        batch_time=batch_time,
                        acc=acc,
                        m_acc=m_acc,
                        iou=iou,
                        m_iou=m_iou,
                    )
                )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Test result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )

            log_test_f1 = getattr(self.cfg, "log_test_f1", False)
            if log_test_f1:
                f1_class, macro_f1 = f1_scores_from_hist(intersection, union, target)
                logger.info("Test result: macro-F1 {:.4f}".format(macro_f1))
                for i in range(self.cfg.data.num_classes):
                    logger.info(
                        "Class_{idx} - {name} Result: f1 {f1:.4f}".format(
                            idx=i,
                            name=self.cfg.data.names[i],
                            f1=f1_class[i],
                        )
                    )

            log_dict = {
                "test/mIoU": float(mIoU),
                "test/mAcc": float(mAcc),
                "test/allAcc": float(allAcc),
            }
            for i in range(self.cfg.data.num_classes):
                cls_name = self.cfg.data.names[i]
                log_dict[f"test/iou_{cls_name}"] = float(iou_class[i])
                log_dict[f"test/acc_{cls_name}"] = float(accuracy_class[i])
            if log_test_f1:
                log_dict["test/f1_macro"] = float(macro_f1)
                for i in range(self.cfg.data.num_classes):
                    cls_name = self.cfg.data.names[i]
                    log_dict[f"test/f1_{cls_name}"] = float(f1_class[i])
        else:
            log_dict = None

        # Final metrics for in-process callers (e.g. GridProbeWinnerSelector);
        # None on non-main ranks.
        self.test_metrics = log_dict

        self.end_test_timing_and_log(extra_log_dict=log_dict)
        if comm.is_main_process():
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    def _test_packed_semseg_batch(
        self,
        batch,
        save_path,
        intersection_meter,
        union_meter,
        target_meter,
        record,
        prog,
    ):
        """Packed multi-scene test step (batch_size > 1 or VoxelBudget sampler).

        Requires test_single_fragment=True (one fragment per scene). Collates
        those fragments, runs a single forward, and splits logits by offset.
        """
        assert all(len(d["fragment_list"]) == 1 for d in batch), (
            "Batch size > 1 is only supported with test_single_fragment=True "
            "(1 fragment per scene)."
        )

        batch_data_names = []
        batch_segments = []
        batch_scene_extra = []
        batch_fragment_lists = []
        batch_pred_paths = []
        batch_cached = []

        for data_dict in batch:
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            batch_data_names.append(data_name)
            batch_segments.append(segment)
            batch_scene_extra.append(data_dict)
            batch_fragment_lists.append(fragment_list)
            batch_pred_paths.append(pred_save_path)
            batch_cached.append(os.path.isfile(pred_save_path))

        if all(batch_cached):
            batch_pred = []
            for b_idx, pred_save_path in enumerate(batch_pred_paths):
                pred = np.load(pred_save_path)
                segment = batch_segments[b_idx]
                if "origin_segment" in batch_scene_extra[b_idx]:
                    segment = batch_scene_extra[b_idx]["origin_segment"]
                batch_pred.append(pred)
                batch_segments[b_idx] = segment
        else:
            fragments = [fl[0] for fl in batch_fragment_lists]
            use_voxel_broadcast = "inverse" in fragments[0]
            num_classes = self.cfg.data.num_classes
            pred_sum = [
                torch.zeros((int(np.asarray(seg).size), num_classes), device="cuda")
                for seg in batch_segments
            ]

            input_dict = collate_fn(fragments)
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            idx_part = input_dict["index"]

            with torch.no_grad():
                pred_part_all = self.model(input_dict)["seg_logits"]  # (n, k)
                pred_part_all = F.softmax(pred_part_all, -1)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()

                bs = 0
                for b_idx, be in enumerate(input_dict["offset"]):
                    pred_part = pred_part_all[bs:be]
                    if use_voxel_broadcast:
                        inv = fragments[b_idx]["inverse"]
                        inv = (
                            torch.from_numpy(inv).long().cuda()
                            if isinstance(inv, np.ndarray)
                            else inv.long().cuda()
                        )
                        pred_sum[b_idx] += pred_part[inv, :]
                    else:
                        pred_sum[b_idx][idx_part[bs:be], :] += pred_part
                    bs = be

            batch_pred = []
            for b_idx in range(len(batch)):
                pred = pred_sum[b_idx]
                pred = pred.max(1)[1].data.cpu().numpy()
                scene_extra = batch_scene_extra[b_idx]
                if "origin_segment" in scene_extra:
                    assert "inverse" in scene_extra
                    pred = pred[scene_extra["inverse"]]
                    batch_segments[b_idx] = scene_extra["origin_segment"]
                np.save(batch_pred_paths[b_idx], pred)
                batch_pred.append(pred)

        for b_idx in range(len(batch)):
            self._emit_semseg_scene_result(
                pred=batch_pred[b_idx],
                segment=batch_segments[b_idx],
                data_name=batch_data_names[b_idx],
                save_path=save_path,
                intersection_meter=intersection_meter,
                union_meter=union_meter,
                target_meter=target_meter,
                record=record,
                log_scene=False,
            )
        prog.step(cached=all(batch_cached))

    def _emit_semseg_scene_result(
        self,
        pred,
        segment,
        data_name,
        save_path,
        intersection_meter,
        union_meter,
        target_meter,
        record,
        logger=None,
        loader_idx=0,
        start=None,
        batch_time=None,
        update_batch_time=True,
        log_scene=True,
    ):
        intersection, union, target = intersection_and_union(
            pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
        )
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        record[data_name] = dict(intersection=intersection, union=union, target=target)

        if not log_scene:
            return

        mask = union != 0
        iou_class = intersection / (union + 1e-10)
        iou = np.mean(iou_class[mask])
        acc = sum(intersection) / (sum(target) + 1e-10)

        m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
        m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

        if update_batch_time and batch_time is not None and start is not None:
            batch_time.update(time.time() - start)
        logger.info(
            "Test: {} [{}/{}]-{} "
            "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
            "Accuracy {acc:.4f} ({m_acc:.4f}) "
            "mIoU {iou:.4f} ({m_iou:.4f})".format(
                data_name,
                loader_idx + 1,
                len(self.test_loader),
                segment.size if hasattr(segment, "size") else len(segment),
                batch_time=batch_time,
                acc=acc,
                m_acc=m_acc,
                iou=iou,
                m_iou=m_iou,
            )
        )

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class GridProbeSemSegTester(TesterBase):
    """SemSegTester variant for GridProbeSegmentorV2 that tests every probe head
    with one backbone forward per fragment (``raw_model.active_probe`` must be None,
    i.e. all-probes mode, so the model returns ``seg_logits_by_task``).

    Only supports ``batch_size_test_per_gpu == 1``. A scene that fails (e.g. OOM)
    is logged and skipped instead of aborting the test.

    Sets ``self.test_metrics_by_probe``: {probe_name: log_dict}, each log_dict shaped
    like SemSegTester's ``self.test_metrics`` ("test/mIoU", "test/iou_<cls>", ...).
    """

    def test(self):
        logger = get_root_logger()
        if self._uses_multi_scene_batches():
            raise NotImplementedError(
                "GridProbeSemSegTester only supports batch_size_test_per_gpu=1 "
                "(packed multi-scene batches not implemented)."
            )
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if raw_model.active_probe is not None:
            raise RuntimeError(
                "GridProbeSemSegTester requires raw_model.active_probe=None "
                "(all-probes mode), got %r." % (raw_model.active_probe,)
            )
        probe_names = raw_model.probe_names
        num_classes = self.cfg.data.num_classes
        log_test_f1 = getattr(self.cfg, "log_test_f1", False)

        logger.info(
            ">>>>>>>>>>>>>>>> Start Evaluation (GridProbe, %d probes) >>>>>>>>>>>>>>>>",
            len(probe_names),
        )
        self.begin_test_timing()

        batch_time = AverageMeter()
        self.model.eval()

        record = {name: {} for name in probe_names}
        skipped_scenes = []
        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_dict = data_dict[0]  # batch size 1 (asserted above)
            data_name_hint = data_dict.get("name", f"scene_idx_{idx}")
            try:
                fragment_list = data_dict.pop("fragment_list")
                segment = data_dict.pop("segment")
                data_name = data_dict.pop("name")

                pred = {
                    name: torch.zeros((segment.size, num_classes)).cuda()
                    for name in probe_names
                }
                use_voxel_broadcast = "inverse" in fragment_list[0]
                for i in range(len(fragment_list)):
                    input_dict = collate_fn(fragment_list[i : i + 1])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        logits_by_task = self.model(input_dict)["seg_logits_by_task"]
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()
                    if use_voxel_broadcast:
                        inv = fragment_list[i]["inverse"]
                        inv = (
                            torch.from_numpy(inv).long().cuda()
                            if isinstance(inv, np.ndarray)
                            else inv.long().cuda()
                        )
                        for name in probe_names:
                            pred_part = F.softmax(logits_by_task[name], -1)
                            pred[name] += pred_part[inv, :]
                    else:
                        for name in probe_names:
                            pred_part = F.softmax(logits_by_task[name], -1)
                            bs = 0
                            for be in input_dict["offset"]:
                                pred[name][idx_part[bs:be], :] += pred_part[bs:be]
                                bs = be

                    logger.info(
                        "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name=data_name,
                            batch_idx=i,
                            batch_num=len(fragment_list),
                        )
                    )

                if "origin_segment" in data_dict.keys():
                    assert "inverse" in data_dict.keys()
                    segment = data_dict["origin_segment"]

                for name in probe_names:
                    pred_name = pred[name].max(1)[1].data.cpu().numpy()
                    if "origin_segment" in data_dict.keys():
                        pred_name = pred_name[data_dict["inverse"]]
                    intersection, union, target = intersection_and_union(
                        pred_name, segment, num_classes, self.cfg.data.ignore_index
                    )
                    record[name][data_name] = dict(
                        intersection=intersection, union=union, target=target
                    )

                batch_time.update(time.time() - start)
                logger.info(
                    "Test: {} [{}/{}]-{} Batch {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        data_name,
                        idx + 1,
                        len(self.test_loader),
                        segment.size,
                        batch_time=batch_time,
                    )
                )
            except Exception:
                # A failing scene (e.g. OOM: the accumulation buffers grow with
                # the number of probes) is dropped from the totals so the other
                # scenes' results are kept.
                logger.error(
                    "GridProbeSemSegTester: scene %r failed (%d/%d), skipping -- "
                    "results for other scenes/probes are kept.\n%s",
                    data_name_hint,
                    idx + 1,
                    len(self.test_loader),
                    traceback.format_exc(),
                )
                skipped_scenes.append(data_name_hint)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        if skipped_scenes:
            logger.warning(
                "GridProbeSemSegTester: %d/%d scene(s) skipped due to errors: %s",
                len(skipped_scenes),
                len(self.test_loader),
                skipped_scenes,
            )
        self.skipped_scenes = skipped_scenes

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        test_metrics_by_probe = None
        if comm.is_main_process():
            merged_record = {name: {} for name in probe_names}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                for name in probe_names:
                    merged_record[name].update(r[name])
                del r

            test_metrics_by_probe = {}
            for name in probe_names:
                per_scene = merged_record[name]
                if not per_scene:
                    # Every scene failed: leave this probe out of
                    # test_metrics_by_probe instead of summing an empty list.
                    logger.error(
                        "[probe=%s] No scenes produced results (all %d skipped) -- "
                        "no test metrics for this probe.",
                        name,
                        len(skipped_scenes),
                    )
                    continue
                intersection = np.sum(
                    [m["intersection"] for m in per_scene.values()], axis=0
                )
                union = np.sum([m["union"] for m in per_scene.values()], axis=0)
                target = np.sum([m["target"] for m in per_scene.values()], axis=0)

                iou_class = intersection / (union + 1e-10)
                accuracy_class = intersection / (target + 1e-10)
                mIoU = np.mean(iou_class)
                mAcc = np.mean(accuracy_class)
                allAcc = sum(intersection) / (sum(target) + 1e-10)

                logger.info(
                    "[probe=%s] Test result: mIoU/mAcc/allAcc %.4f/%.4f/%.4f",
                    name,
                    mIoU,
                    mAcc,
                    allAcc,
                )

                log_dict = {
                    "test/mIoU": float(mIoU),
                    "test/mAcc": float(mAcc),
                    "test/allAcc": float(allAcc),
                }
                for i in range(num_classes):
                    cls_name = self.cfg.data.names[i]
                    log_dict[f"test/iou_{cls_name}"] = float(iou_class[i])
                    log_dict[f"test/acc_{cls_name}"] = float(accuracy_class[i])
                if log_test_f1:
                    f1_class, macro_f1 = f1_scores_from_hist(
                        intersection, union, target
                    )
                    log_dict["test/f1_macro"] = float(macro_f1)
                    for i in range(num_classes):
                        cls_name = self.cfg.data.names[i]
                        log_dict[f"test/f1_{cls_name}"] = float(f1_class[i])
                test_metrics_by_probe[name] = log_dict

        # Read by in-process callers (GridProbeSeedEnsembleTester); None on
        # non-main ranks.
        self.test_metrics_by_probe = test_metrics_by_probe

        self.end_test_timing_and_log()
        if comm.is_main_process():
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class RegressionTester(TesterBase):
    """Fragment-based test for DefaultRegressorV2 (point-wise MAE/RMSE).

    Expects cfg.data.target_key. Ground-truth must be present at full resolution
    in the sample dict (see MALiBU3DDataset.prepare_test_data).
    """

    @staticmethod
    def _cfg_get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _target_key(self):
        target_key = self._cfg_get(self.cfg.data, "target_key", None)
        if target_key is not None:
            return str(target_key)
        raise ValueError("RegressionTester requires cfg.data.target_key.")

    @staticmethod
    def _origin_target_key(target_key):
        return f"origin_{target_key}"

    @staticmethod
    def _gather_masked(pred, target):
        mask = torch.isfinite(pred) & torch.isfinite(target)
        if mask.sum() == 0:
            return None, None
        return pred[mask], target[mask]

    @staticmethod
    def _apply_inverse_origin_np(pred_np, scene_extra, target_key):
        origin_key = RegressionTester._origin_target_key(target_key)
        if "inverse" not in scene_extra or origin_key not in scene_extra:
            return pred_np
        inv = scene_extra["inverse"]
        if isinstance(inv, torch.Tensor):
            inv = inv.cpu().numpy()
        return np.asarray(pred_np).reshape(-1)[inv]

    @staticmethod
    def _target_for_metrics(target_np, scene_extra, target_key):
        origin_key = RegressionTester._origin_target_key(target_key)
        if origin_key in scene_extra:
            return np.asarray(scene_extra[origin_key]).reshape(-1)
        return np.asarray(target_np).reshape(-1)

    def _denorm_regression_pred_np(self, pred_np, target_key):
        target_scales = self._cfg_get(self.cfg.data, "target_scales", {}) or {}
        scale = get_regression_target_scale(target_scales, target_key)
        return denorm_regression_prediction(
            np.asarray(pred_np, dtype=np.float64), scale
        )

    def test(self):
        if self._uses_multi_scene_batches():
            logger = get_root_logger()
            logger.info(
                "Batch size > 1 detected. Assuming test_single_fragment=True "
                "for RegressionTester."
            )

        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Regression Evaluation >>>>>>>>>>>>>>>>")
        self.begin_test_timing()

        target_key = self._target_key()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        comm.synchronize()

        reg_sums = {"mae": 0.0, "mse": 0.0, "count": 0.0}
        n_batches = len(self.test_loader)

        def _process_one_regression_batch(batch, loader_idx, prog):
            if self._uses_multi_scene_batches():
                assert all(len(d["fragment_list"]) == 1 for d in batch), (
                    "Batch size > 1 is only supported with test_single_fragment=True "
                    "(1 fragment per scene)."
                )

            batch_data_names = []
            batch_targets = []
            batch_scene_extra = []
            batch_fragment_lists = []
            batch_pred_paths = []
            batch_all_cached = []

            for data_dict in batch:
                fragment_list = data_dict.pop("fragment_list")
                target = data_dict.pop(target_key)
                data_name = data_dict.pop("name")

                batch_data_names.append(data_name)
                batch_targets.append(target)
                batch_scene_extra.append(data_dict)
                batch_fragment_lists.append(fragment_list)
                pred_save_path = os.path.join(
                    save_path, f"{data_name}_reg_{target_key}.npy"
                )
                batch_pred_paths.append(pred_save_path)
                batch_all_cached.append(os.path.isfile(pred_save_path))

            if all(batch_all_cached):
                batch_pred_np = [np.load(p) for p in batch_pred_paths]
            elif self._uses_multi_scene_batches():
                fragments = [fl[0] for fl in batch_fragment_lists]
                use_voxel_broadcast = "inverse" in fragments[0]

                batch_reg_sum = []
                batch_reg_cnt = []
                for target in batch_targets:
                    n_ref = int(np.asarray(target).size)
                    batch_reg_sum.append(torch.zeros((n_ref,), device="cuda"))
                    batch_reg_cnt.append(torch.zeros((n_ref,), device="cuda"))

                input_dict = collate_fn(fragments)
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]

                with torch.no_grad():
                    output_dict = self.model(input_dict)
                    pred_part_all = output_dict["reg_pred"].reshape(-1).float()
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()

                    bs = 0
                    for b_idx, be in enumerate(input_dict["offset"]):
                        pred_part = pred_part_all[bs:be]
                        if use_voxel_broadcast:
                            inv = fragments[b_idx]["inverse"]
                            inv = (
                                torch.from_numpy(inv).long().cuda()
                                if isinstance(inv, np.ndarray)
                                else inv.long().cuda()
                            )
                            gathered = pred_part[inv]
                            batch_reg_sum[b_idx] += gathered
                            batch_reg_cnt[b_idx] += torch.ones_like(gathered)
                        else:
                            ip = idx_part[bs:be]
                            batch_reg_sum[b_idx][ip] += pred_part
                            batch_reg_cnt[b_idx][ip] += 1
                        bs = be

                batch_pred_np = []
                for b_idx in range(len(batch)):
                    pred_np = (
                        (batch_reg_sum[b_idx] / batch_reg_cnt[b_idx].clamp(min=1))
                        .cpu()
                        .numpy()
                    )
                    pred_np = self._apply_inverse_origin_np(
                        pred_np, batch_scene_extra[b_idx], target_key
                    )
                    pred_np = self._denorm_regression_pred_np(pred_np, target_key)
                    np.save(batch_pred_paths[b_idx], pred_np)
                    batch_pred_np.append(pred_np)
            else:
                batch_pred_np = []
                for b_idx in range(len(batch)):
                    data_name = batch_data_names[b_idx]
                    target = batch_targets[b_idx]
                    scene_extra = batch_scene_extra[b_idx]
                    fragment_list = batch_fragment_lists[b_idx]
                    pred_save_path = batch_pred_paths[b_idx]

                    if batch_all_cached[b_idx]:
                        pred_np = np.load(pred_save_path)
                    else:
                        n_ref = int(np.asarray(target).size)
                        reg_sum = torch.zeros((n_ref,), device="cuda")
                        reg_cnt = torch.zeros((n_ref,), device="cuda")
                        use_voxel_broadcast = "inverse" in fragment_list[0]

                        for i in range(len(fragment_list)):
                            fragment_batch_size = 1
                            s_i, e_i = i * fragment_batch_size, min(
                                (i + 1) * fragment_batch_size, len(fragment_list)
                            )
                            input_dict = collate_fn(fragment_list[s_i:e_i])
                            for key in input_dict.keys():
                                if isinstance(input_dict[key], torch.Tensor):
                                    input_dict[key] = input_dict[key].cuda(
                                        non_blocking=True
                                    )
                            idx_part = input_dict["index"]
                            with torch.no_grad():
                                output_dict = self.model(input_dict)
                                pred_part_all = (
                                    output_dict["reg_pred"].reshape(-1).float()
                                )
                                if self.cfg.empty_cache:
                                    torch.cuda.empty_cache()

                                bs = 0
                                for be in input_dict["offset"]:
                                    pred_part = pred_part_all[bs:be]
                                    if use_voxel_broadcast:
                                        inv = fragment_list[s_i:e_i][0]["inverse"]
                                        inv = (
                                            torch.from_numpy(inv).long().cuda()
                                            if isinstance(inv, np.ndarray)
                                            else inv.long().cuda()
                                        )
                                        gathered = pred_part[inv]
                                        reg_sum += gathered
                                        reg_cnt += torch.ones_like(gathered)
                                    else:
                                        ip = idx_part[bs:be]
                                        reg_sum[ip] += pred_part
                                        reg_cnt[ip] += 1
                                    bs = be

                            if self.verbose:
                                logger.info(
                                    "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                                        loader_idx + 1,
                                        len(self.test_loader),
                                        data_name=data_name,
                                        batch_idx=i,
                                        batch_num=len(fragment_list),
                                    )
                                )

                        pred_np = (reg_sum / reg_cnt.clamp(min=1)).cpu().numpy()
                        pred_np = self._apply_inverse_origin_np(
                            pred_np, scene_extra, target_key
                        )
                        pred_np = self._denorm_regression_pred_np(pred_np, target_key)
                        np.save(pred_save_path, pred_np)

                    batch_pred_np.append(pred_np)

            for b_idx in range(len(batch)):
                pred_np = batch_pred_np[b_idx]
                tgt_np = self._target_for_metrics(
                    batch_targets[b_idx], batch_scene_extra[b_idx], target_key
                )
                p_t = torch.from_numpy(np.asarray(pred_np, dtype=np.float64)).float()
                t_t = torch.from_numpy(np.asarray(tgt_np, dtype=np.float64)).float()
                p_m, t_m = self._gather_masked(p_t, t_t)
                if p_m is not None:
                    mae_b, mse_b, cnt_b = accumulate_regression_errors(p_m, t_m)
                    reg_sums["mae"] += mae_b
                    reg_sums["mse"] += mse_b
                    reg_sums["count"] += cnt_b

                prog.step(cached=batch_all_cached[b_idx])

        with EvaluationProgressBar(
            n_batches,
            desc="Test regression",
        ) as prog:
            for loader_idx, batch in enumerate(self.test_loader):
                _process_one_regression_batch(batch, loader_idx, prog)

        logger.info("Syncing ...")
        comm.synchronize()

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

        log_dict = None
        if comm.is_main_process():
            cnt = reg_sums["count"]
            if cnt > 1e-8:
                mae = reg_sums["mae"] / cnt
                rmse = (reg_sums["mse"] / cnt) ** 0.5
                logger.info(
                    "Test regression: MAE {:.6f} RMSE {:.6f} (n={:.0f}).".format(
                        mae, rmse, cnt
                    )
                )
                log_dict = {
                    f"test/reg/{target_key}/mae": float(mae),
                    f"test/reg/{target_key}/rmse": float(rmse),
                }

        self.end_test_timing_and_log(extra_log_dict=log_dict)
        if comm.is_main_process():
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class MultiTaskTester(TesterBase):
    """Test for MultiTaskSegmentorV2, with the same per-task metrics as MultiTaskEvaluator.

    Semantic tasks: mIoU/mAcc/allAcc; pixel_semantic tasks (2D rasters): per-channel
    exact and buffered precision/recall/F1 plus mIoU; tile_distribution tasks: weighted
    KL, TV and per-class frequency MAE (optional point-wise mIoU); regression: MAE/RMSE.
    Expects cfg.data.task_configs and cfg.data.main_task (required with several tasks),
    and one fragment per scene (test_single_fragment=True). Ground truth must be present
    at full resolution in the sample dict (see MALiBU3DDataset.prepare_test_data).
    Per-scene predictions are cached in ``save_path/result`` and reused on rerun; the
    ``{name}_logits_<task>.npy`` rasters are read by the network APLS evaluation.
    """

    def __init__(self, write_cls_iou=False, **kwargs):
        super().__init__(**kwargs)
        self.write_cls_iou = write_cls_iou

    @staticmethod
    def _cfg_get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _get_task_configs(self):
        data_cfg = self.cfg.data
        task_configs = self._cfg_get(data_cfg, "task_configs", None)
        if isinstance(task_configs, dict) and len(task_configs) > 0:
            return {str(k): dict(v) for k, v in task_configs.items()}
        raise ValueError(
            "MultiTaskTester requires cfg.data.task_configs to be a non-empty dict."
        )

    def _main_task_name(self, task_configs):
        data_cfg = self.cfg.data
        main_task = self._cfg_get(data_cfg, "main_task", None)
        keys = list(task_configs.keys())
        if main_task is not None:
            if str(main_task) in keys:
                return str(main_task)
            raise ValueError(
                f"cfg.data.main_task is {main_task!r} but that key is not in task_configs "
                f"(keys: {keys})."
            )
        if len(keys) == 1:
            return keys[0]
        raise ValueError(
            "cfg.data.main_task is not defined (and cannot be inferred as there are "
            "multiple tasks)"
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
    def _tile_distribution_cache_paths(save_path, data_name, task_name):
        return dict(
            pred=os.path.join(save_path, f"{data_name}_pred_{task_name}.npy"),
            tile=os.path.join(save_path, f"{data_name}_pred_{task_name}_tile.npy"),
            dist=os.path.join(save_path, f"{data_name}_dist_{task_name}.npy"),
        )

    @staticmethod
    def _empirical_tile_distribution(target_np, ignore_index, num_classes):
        """Empirical per-tile class histogram from point labels (void excluded)."""
        target = np.asarray(target_np, dtype=np.int64).reshape(-1)
        valid = target != int(ignore_index)
        n_t = int(valid.sum())
        q = np.zeros((int(num_classes),), dtype=np.float64)
        if n_t > 0:
            cls = np.clip(target[valid], 0, int(num_classes) - 1)
            q = np.bincount(cls, minlength=int(num_classes)).astype(np.float64)
            q /= float(n_t)
        return q, n_t

    @staticmethod
    def _tile_distribution_metrics_from_dist(
        dist_np, target_np, ignore_index, num_classes
    ):
        q, n_t = MultiTaskTester._empirical_tile_distribution(
            target_np, ignore_index, num_classes
        )
        if n_t <= 0:
            return None
        pi = torch.from_numpy(np.asarray(dist_np, dtype=np.float64).reshape(1, -1))
        q_t = torch.from_numpy(q.reshape(1, -1))
        kl = float(kl_divergence_rows(q_t, pi).item())
        abs_err = abs_freq_error_rows(pi, q_t)
        n_val = float(n_t)
        return dict(
            kl_weighted=n_val * kl,
            weight=n_val,
            abs_weighted=(n_val * abs_err.squeeze(0).detach().cpu().numpy()).astype(
                np.float64
            ),
        )

    @staticmethod
    def _finalize_and_save_tile_distribution(
        avg_probs,
        target_np,
        ignore_index,
        num_classes,
        cache_paths,
    ):
        """Argmax-per-point, pooled tile class, and (C,) distribution; write cache files.

        ``avg_probs`` is (N, C) softmaxed per-point probabilities, already inverse-
        mapped onto the origin cloud when that remap applies.
        """
        pred_np = avg_probs.argmax(dim=-1).detach().cpu().numpy().astype(np.int32)
        n_points = int(pred_np.shape[0])
        metrics = None
        target_ok = False
        if target_np is not None:
            tgt = np.asarray(target_np, dtype=np.int64).reshape(-1)
            target_ok = tgt.shape[0] == avg_probs.shape[0]
        if target_ok:
            target_t = torch.from_numpy(tgt).long().to(avg_probs.device)
            indptr = torch.tensor([0, avg_probs.shape[0]], device=avg_probs.device)
            pi_hat, q_t, n_t = pool_axis_distribution_from_probs(
                avg_probs, target_t, indptr, int(ignore_index), int(num_classes)
            )
            n_val = float(n_t.item())
            dist_np = pi_hat.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if n_val > 0:
                tile_cls = int(pi_hat.squeeze(0).argmax().item())
                abs_weighted = (
                    n_val
                    * abs_freq_error_rows(pi_hat, q_t).squeeze(0).detach().cpu().numpy()
                ).astype(np.float64)
                metrics = dict(
                    kl_weighted=n_val * float(kl_divergence_rows(q_t, pi_hat).item()),
                    weight=n_val,
                    abs_weighted=abs_weighted,
                )
            else:
                tile_cls = int(ignore_index)
        elif n_points == 0:
            dist_np = np.zeros((int(num_classes),), dtype=np.float32)
            tile_cls = int(ignore_index)
        else:
            dist_np = avg_probs.mean(dim=0).detach().cpu().numpy().astype(np.float32)
            tile_cls = int(np.argmax(dist_np))
        tile_pred_np = np.full((n_points,), tile_cls, dtype=np.int32)
        np.save(cache_paths["pred"], pred_np)
        np.save(cache_paths["tile"], tile_pred_np)
        np.save(cache_paths["dist"], dist_np)
        return pred_np, tile_pred_np, dist_np, metrics

    @staticmethod
    def _apply_inverse_origin_probs(pred_probs, scene_extra, task_name):
        """(N, C) analogue of _apply_inverse_origin_np, for tile_distribution predictions."""
        origin_key = MultiTaskTester._task_origin_target_key(task_name)
        if "inverse" not in scene_extra or origin_key not in scene_extra:
            return pred_probs
        inv = scene_extra["inverse"]
        inv = (
            torch.from_numpy(inv).long().to(pred_probs.device)
            if isinstance(inv, np.ndarray)
            else inv.long().to(pred_probs.device)
        )
        return pred_probs[inv, :]

    @staticmethod
    def _apply_inverse_origin_np(pred_np, target_np, scene_extra, task_name):
        origin_key = MultiTaskTester._task_origin_target_key(task_name)
        if "inverse" not in scene_extra or origin_key not in scene_extra:
            return pred_np, target_np
        inv = scene_extra["inverse"]
        if isinstance(inv, torch.Tensor):
            inv = inv.cpu().numpy()
        pred_np = np.asarray(pred_np).reshape(-1)[inv]
        target_np = np.asarray(scene_extra[origin_key]).reshape(-1)
        return pred_np, target_np

    @staticmethod
    def _target_for_metrics(task_name, targets_by_task, scene_extra):
        """Ground-truth aligned with saved predictions (after optional inverse/origin remap)."""
        origin_key = MultiTaskTester._task_origin_target_key(task_name)
        if origin_key in scene_extra:
            return np.asarray(scene_extra[origin_key]).reshape(-1)
        return np.asarray(targets_by_task[task_name]).reshape(-1)

    def _denorm_regression_pred_np(self, pred_np, task_name):
        target_scales = self._cfg_get(self.cfg.data, "target_scales", {}) or {}
        scale = get_regression_target_scale(target_scales, task_name)
        return denorm_regression_prediction(
            np.asarray(pred_np, dtype=np.float64), scale
        )

    def test(self):
        if self._uses_multi_scene_batches():
            logger = get_root_logger()
            logger.info(
                "Batch size > 1 detected. Assuming test_single_fragment=True for MultiTaskTester."
            )

        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Multi-task Evaluation >>>>>>>>>>>>>>>>")
        self.begin_test_timing()

        task_configs = self._get_task_configs()
        main_task = self._main_task_name(task_configs)
        semantic_tasks = self._semantic_task_names(task_configs)
        pixel_semantic_tasks = self._pixel_semantic_task_names(task_configs)
        regression_tasks = self._regression_task_names(task_configs)
        tile_distribution_tasks = self._tile_distribution_task_names(task_configs)
        tile_pw_miou_tasks = self._tile_distribution_pointwise_miou_task_names(
            task_configs
        )

        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        comm.synchronize()

        reg_sums_global = {
            t: {"mae": 0.0, "mse": 0.0, "count": 0.0} for t in regression_tasks
        }
        td_sums_global = {
            t: {
                "kl_weighted": 0.0,
                "weight": 0.0,
                "abs_weighted": np.zeros(
                    int(task_configs[t]["num_classes"]), dtype=np.float64
                ),
            }
            for t in tile_distribution_tasks
        }

        running_iou = {t: AverageMeter() for t in semantic_tasks}

        record = {}
        n_batches = len(self.test_loader)

        def _process_one_multitask_batch(batch, prog):
            if self._uses_multi_scene_batches():
                assert all(
                    len(d["fragment_list"]) == 1 for d in batch
                ), "Batch size > 1 is only supported with test_single_fragment=True (1 fragment per scene)."

            batch_data_names = []
            batch_targets = []
            batch_scene_extra = []
            batch_n_refs = []
            batch_pred_sem = []
            batch_reg_sum = []
            batch_reg_cnt = []
            batch_axis_pred_sum = []
            batch_axis_pred_cnt = []
            batch_sem_cache_paths = []
            batch_reg_cache_paths = []
            batch_pixel_logits_cache_paths = []
            batch_td_cache_paths = []
            batch_all_cached = []
            batch_pixel_dense_sum = []
            batch_pixel_dense_cnt = []

            for b_idx, data_dict in enumerate(batch):
                data_name = data_dict.pop("name")
                batch_data_names.append(data_name)

                targets_by_task = {}
                for tn in task_configs:
                    if tn in data_dict:
                        targets_by_task[tn] = data_dict.pop(tn)
                batch_targets.append(targets_by_task)
                batch_scene_extra.append(data_dict)

                ref_arr = None
                for cand in ("segment", main_task, *semantic_tasks):
                    cs = str(cand)
                    if (
                        cs in targets_by_task
                        and task_configs.get(cs, {}).get("task_type")
                        != "pixel_semantic"
                    ):
                        ref_arr = targets_by_task[cs]
                        break
                if ref_arr is None:
                    for cand in semantic_tasks:
                        if cand in targets_by_task:
                            ref_arr = targets_by_task[cand]
                            break
                if ref_arr is None and targets_by_task:
                    # Prefer a point-wise target size; fall back to any target.
                    ref_arr = next(iter(targets_by_task.values()))
                    if (
                        main_task in targets_by_task
                        and task_configs.get(main_task, {}).get("task_type")
                        == "pixel_semantic"
                    ):
                        # Use fragment point count later; placeholder size 1.
                        ref_arr = np.zeros(1, dtype=np.int32)
                if ref_arr is None:
                    raise RuntimeError(
                        f"Scene {data_name}: no labels found for any task in task_configs."
                    )
                n_ref = int(np.asarray(ref_arr).reshape(-1).size)
                batch_n_refs.append(n_ref)

                sem_cache_paths = {
                    t: os.path.join(save_path, f"{data_name}_pred_{t}.npy")
                    for t in semantic_tasks
                }
                reg_cache_paths = {
                    t: os.path.join(save_path, f"{data_name}_reg_{t}.npy")
                    for t in regression_tasks
                }
                pixel_logits_cache_paths = {
                    # Soft foreground probs (r, H, W) in [0, 1]; NaN where no
                    # LiDAR point fell in the raster cell (unobserved).
                    t: os.path.join(save_path, f"{data_name}_logits_{t}.npy")
                    for t in pixel_semantic_tasks
                }
                td_cache_paths = {
                    t: self._tile_distribution_cache_paths(save_path, data_name, t)
                    for t in tile_distribution_tasks
                }
                batch_sem_cache_paths.append(sem_cache_paths)
                batch_reg_cache_paths.append(reg_cache_paths)
                batch_pixel_logits_cache_paths.append(pixel_logits_cache_paths)
                batch_td_cache_paths.append(td_cache_paths)

                all_sem_cached = all(
                    os.path.isfile(p) for p in sem_cache_paths.values()
                )
                all_reg_cached = all(
                    os.path.isfile(p) for p in reg_cache_paths.values()
                )
                all_pixel_cached = all(
                    os.path.isfile(p) for p in pixel_logits_cache_paths.values()
                )
                all_td_cached = all(
                    os.path.isfile(p["pred"])
                    and os.path.isfile(p["tile"])
                    and os.path.isfile(p["dist"])
                    for p in td_cache_paths.values()
                )
                batch_all_cached.append(
                    all_sem_cached
                    and (len(regression_tasks) == 0 or all_reg_cached)
                    and (len(pixel_semantic_tasks) == 0 or all_pixel_cached)
                    and (len(tile_distribution_tasks) == 0 or all_td_cached)
                )

                batch_pred_sem.append(
                    {
                        t: torch.zeros(
                            (n_ref, int(task_configs[t]["num_classes"])), device="cuda"
                        )
                        for t in semantic_tasks
                    }
                )
                batch_reg_sum.append(
                    {t: torch.zeros((n_ref,), device="cuda") for t in regression_tasks}
                )
                batch_reg_cnt.append(
                    {t: torch.zeros((n_ref,), device="cuda") for t in regression_tasks}
                )
                batch_axis_pred_sum.append(
                    {
                        t: torch.zeros(
                            (n_ref, int(task_configs[t]["num_classes"])), device="cuda"
                        )
                        for t in tile_distribution_tasks
                    }
                )
                batch_axis_pred_cnt.append(
                    {
                        t: torch.zeros((n_ref,), device="cuda")
                        for t in tile_distribution_tasks
                    }
                )
                batch_pixel_dense_sum.append({t: None for t in pixel_semantic_tasks})
                batch_pixel_dense_cnt.append({t: None for t in pixel_semantic_tasks})

            # Skip the forward when every cached prediction file exists.
            batch_td_dist_np = []
            batch_td_pred_np = []
            batch_td_metrics = []
            if all(batch_all_cached):
                batch_pred_cls_np = []
                batch_pred_reg_np = []
                batch_pixel_logits_np = []
                for b_idx in range(len(batch)):
                    pred_cls_np = {}
                    pred_reg_np = {}
                    pixel_logits_np = {}
                    for t in semantic_tasks:
                        pred_cls_np[t] = np.load(batch_sem_cache_paths[b_idx][t])
                    for t in regression_tasks:
                        pred_reg_np[t] = np.load(batch_reg_cache_paths[b_idx][t])
                    for t in pixel_semantic_tasks:
                        pixel_logits_np[t] = np.load(
                            batch_pixel_logits_cache_paths[b_idx][t]
                        )
                    td_dist = {}
                    td_pred = {}
                    for t in tile_distribution_tasks:
                        paths = batch_td_cache_paths[b_idx][t]
                        td_dist[t] = np.load(paths["dist"])
                        if t in tile_pw_miou_tasks:
                            td_pred[t] = np.load(paths["pred"])
                    batch_pred_cls_np.append(pred_cls_np)
                    batch_pred_reg_np.append(pred_reg_np)
                    batch_pixel_logits_np.append(pixel_logits_np)
                    batch_td_dist_np.append(td_dist)
                    batch_td_pred_np.append(td_pred)
                    batch_td_metrics.append(None)
            else:
                # Extract the single fragment from each scene. Only fragment 0 is
                # forwarded below, so more than one would leave the other points'
                # accumulators at zero (-> argmax class 0): fail loudly instead.
                fragment_lists = [d.pop("fragment_list") for d in batch]
                n_fragments = [len(fl) for fl in fragment_lists]
                if any(n != 1 for n in n_fragments):
                    raise NotImplementedError(
                        "MultiTaskTester supports exactly one fragment per scene "
                        "(test_single_fragment=True, test_cfg.crop=None); got "
                        f"{n_fragments} for {batch_data_names}."
                    )
                fragments = [fl[0] for fl in fragment_lists]
                use_voxel_broadcast = "inverse" in fragments[0]

                input_dict = collate_fn(fragments)
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]

                with torch.no_grad():
                    output_dict = self.model(input_dict)
                    logits_by_task = output_dict.get("seg_logits_by_task") or {}
                    reg_pred_by_task = output_dict.get("reg_pred_by_task") or {}
                    pixel_dense_by_task = (
                        output_dict.get("pixel_seg_logits_dense_by_task") or {}
                    )

                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()

                    for task_name in semantic_tasks:
                        if task_name not in logits_by_task:
                            continue
                        pred_part_all = F.softmax(logits_by_task[task_name], dim=-1)

                        bs = 0
                        for b_idx, be in enumerate(input_dict["offset"]):
                            pred_part = pred_part_all[bs:be]
                            if use_voxel_broadcast:
                                inv = fragments[b_idx]["inverse"]
                                inv = (
                                    torch.from_numpy(inv).long().cuda()
                                    if isinstance(inv, np.ndarray)
                                    else inv.long().cuda()
                                )
                                batch_pred_sem[b_idx][task_name] += pred_part[inv, :]
                            else:
                                batch_pred_sem[b_idx][task_name][
                                    idx_part[bs:be], :
                                ] += pred_part
                            bs = be

                    for task_name in regression_tasks:
                        if task_name not in reg_pred_by_task:
                            continue
                        pred_part_all = reg_pred_by_task[task_name].reshape(-1).float()

                        bs = 0
                        for b_idx, be in enumerate(input_dict["offset"]):
                            pred_part = pred_part_all[bs:be]
                            if use_voxel_broadcast:
                                inv = fragments[b_idx]["inverse"]
                                inv = (
                                    torch.from_numpy(inv).long().cuda()
                                    if isinstance(inv, np.ndarray)
                                    else inv.long().cuda()
                                )
                                gathered = pred_part[inv]
                                batch_reg_sum[b_idx][task_name] += gathered
                                batch_reg_cnt[b_idx][task_name] += torch.ones_like(
                                    gathered
                                )
                            else:
                                sl = slice(bs, be)
                                ip = idx_part[sl]
                                batch_reg_sum[b_idx][task_name][ip] += pred_part
                                batch_reg_cnt[b_idx][task_name][ip] += 1
                            bs = be

                    # Probabilities are averaged with an explicit count: the plain
                    # sum used for semantic tasks is fine for argmax, not for KL.
                    for task_name in tile_distribution_tasks:
                        if task_name not in logits_by_task:
                            continue
                        pred_part_all = F.softmax(
                            logits_by_task[task_name].float(), dim=-1
                        )

                        bs = 0
                        for b_idx, be in enumerate(input_dict["offset"]):
                            pred_part = pred_part_all[bs:be]
                            if use_voxel_broadcast:
                                inv = fragments[b_idx]["inverse"]
                                inv = (
                                    torch.from_numpy(inv).long().cuda()
                                    if isinstance(inv, np.ndarray)
                                    else inv.long().cuda()
                                )
                                gathered = pred_part[inv, :]
                                batch_axis_pred_sum[b_idx][task_name] += gathered
                                batch_axis_pred_cnt[b_idx][task_name] += torch.ones(
                                    gathered.shape[0], device=gathered.device
                                )
                            else:
                                sl = slice(bs, be)
                                ip = idx_part[sl]
                                batch_axis_pred_sum[b_idx][task_name][ip] += pred_part
                                batch_axis_pred_cnt[b_idx][task_name][ip] += 1
                            bs = be

                    for task_name in pixel_semantic_tasks:
                        dense_list = pixel_dense_by_task.get(task_name) or []
                        for b_idx, dense in enumerate(dense_list):
                            if dense is None:
                                continue
                            dense = dense.float()
                            if batch_pixel_dense_sum[b_idx][task_name] is None:
                                batch_pixel_dense_sum[b_idx][task_name] = (
                                    torch.zeros_like(dense)
                                )
                                batch_pixel_dense_cnt[b_idx][task_name] = (
                                    torch.zeros_like(dense)
                                )
                            # dense is soft foreground probs (r, H, W); NaN = unobserved.
                            observed = torch.isfinite(dense)
                            filled = torch.nan_to_num(dense, nan=0.0)
                            batch_pixel_dense_sum[b_idx][task_name] += filled * observed
                            batch_pixel_dense_cnt[b_idx][task_name] += observed.to(
                                dense.dtype
                            )

                batch_pred_cls_np = []
                batch_pred_reg_np = []
                batch_pixel_logits_np = []
                for b_idx in range(len(batch)):
                    pred_cls_np = {}
                    pred_reg_np = {}

                    for task_name in semantic_tasks:
                        if task_name not in batch_targets[b_idx]:
                            continue
                        pred_raw = (
                            batch_pred_sem[b_idx][task_name].max(1)[1].cpu().numpy()
                        )
                        pred_np, _ = self._apply_inverse_origin_np(
                            pred_raw,
                            np.asarray(batch_targets[b_idx][task_name]).reshape(-1),
                            batch_scene_extra[b_idx],
                            task_name,
                        )
                        pred_cls_np[task_name] = pred_np
                        np.save(batch_sem_cache_paths[b_idx][task_name], pred_np)

                    for task_name in regression_tasks:
                        if task_name not in batch_targets[b_idx]:
                            continue
                        pred_raw = (
                            (
                                batch_reg_sum[b_idx][task_name]
                                / batch_reg_cnt[b_idx][task_name].clamp(min=1)
                            )
                            .cpu()
                            .numpy()
                        )
                        pred_np, _ = self._apply_inverse_origin_np(
                            pred_raw,
                            np.asarray(
                                batch_targets[b_idx][task_name], dtype=np.float64
                            ).reshape(-1),
                            batch_scene_extra[b_idx],
                            task_name,
                        )
                        pred_np = self._denorm_regression_pred_np(pred_np, task_name)
                        pred_reg_np[task_name] = pred_np
                        np.save(batch_reg_cache_paths[b_idx][task_name], pred_np)

                    pixel_logits_np = {}
                    for task_name in pixel_semantic_tasks:
                        dense_sum = batch_pixel_dense_sum[b_idx][task_name]
                        dense_cnt = batch_pixel_dense_cnt[b_idx][task_name]
                        if dense_sum is None:
                            # Fall back to NaN grid from GT mask shape if available.
                            gt = batch_targets[b_idx].get(task_name)
                            if gt is None:
                                continue
                            gt_arr = np.asarray(gt)
                            r = int(task_configs[task_name]["num_networks"])
                            if gt_arr.ndim == 3:
                                _, h, w = gt_arr.shape
                            else:
                                h = w = 1
                            logits_np = np.full((r, h, w), np.nan, dtype=np.float32)
                        else:
                            avg = dense_sum / dense_cnt.clamp(min=1.0)
                            unobserved = dense_cnt < 0.5
                            avg = avg.masked_fill(unobserved, float("nan"))
                            logits_np = avg.detach().cpu().numpy().astype(np.float32)
                        pixel_logits_np[task_name] = logits_np
                        np.save(
                            batch_pixel_logits_cache_paths[b_idx][task_name],
                            logits_np,
                        )

                    td_dist = {}
                    td_pred = {}
                    td_metrics = {}
                    for task_name in tile_distribution_tasks:
                        if task_name not in batch_axis_pred_sum[b_idx]:
                            continue
                        cnt = (
                            batch_axis_pred_cnt[b_idx][task_name]
                            .clamp(min=1)
                            .unsqueeze(-1)
                        )
                        avg_probs = batch_axis_pred_sum[b_idx][task_name] / cnt
                        avg_probs = self._apply_inverse_origin_probs(
                            avg_probs, batch_scene_extra[b_idx], task_name
                        )
                        tgt_np = None
                        tgt_aligned = False
                        if task_name in batch_targets[b_idx]:
                            tgt_np = self._target_for_metrics(
                                task_name,
                                batch_targets[b_idx],
                                batch_scene_extra[b_idx],
                            )
                            tgt_aligned = (
                                np.asarray(tgt_np).reshape(-1).shape[0]
                                == avg_probs.shape[0]
                            )
                            if not tgt_aligned:
                                logger.warning(
                                    "Scene %s: tile_distribution task %s pred/target length "
                                    "mismatch (%d vs %d); skipping metrics.",
                                    batch_data_names[b_idx],
                                    task_name,
                                    avg_probs.shape[0],
                                    int(np.asarray(tgt_np).reshape(-1).shape[0]),
                                )
                                tgt_np = None
                        tc = task_configs[task_name]
                        pred_np, _, dist_np, td_m = (
                            self._finalize_and_save_tile_distribution(
                                avg_probs,
                                tgt_np,
                                int(tc["ignore_index"]),
                                int(tc["num_classes"]),
                                batch_td_cache_paths[b_idx][task_name],
                            )
                        )
                        td_dist[task_name] = dist_np
                        if task_name in tile_pw_miou_tasks:
                            td_pred[task_name] = pred_np
                        if tgt_aligned:
                            td_metrics[task_name] = td_m
                    batch_td_dist_np.append(td_dist)
                    batch_td_pred_np.append(td_pred)
                    batch_td_metrics.append(td_metrics)

                    batch_pred_cls_np.append(pred_cls_np)
                    batch_pred_reg_np.append(pred_reg_np)
                    batch_pixel_logits_np.append(pixel_logits_np)
            # Compute metrics for each scene in the batch
            for b_idx in range(len(batch)):
                data_name = batch_data_names[b_idx]
                targets_by_task = batch_targets[b_idx]
                scene_extra = batch_scene_extra[b_idx]
                pred_cls_np = batch_pred_cls_np[b_idx]
                pred_reg_np = batch_pred_reg_np[b_idx]
                pixel_logits_np = batch_pixel_logits_np[b_idx]

                sem_metrics_scene = {}
                for task_name in semantic_tasks:
                    if task_name not in targets_by_task:
                        logger.warning(
                            "Scene %s: skip semantic task %s (missing ground truth).",
                            data_name,
                            task_name,
                        )
                        continue
                    if task_name not in pred_cls_np:
                        continue
                    pred_np = np.asarray(pred_cls_np[task_name]).reshape(-1)
                    tgt_np = self._target_for_metrics(
                        task_name, targets_by_task, scene_extra
                    )
                    tc = task_configs[task_name]
                    intersection, union, target_hist = intersection_and_union(
                        pred_np,
                        tgt_np,
                        int(tc["num_classes"]),
                        int(tc["ignore_index"]),
                    )
                    sem_metrics_scene[task_name] = dict(
                        intersection=intersection,
                        union=union,
                        target=target_hist,
                    )
                    mask = union != 0
                    iou_class = intersection / (union + 1e-10)
                    scene_m_iou = np.mean(iou_class[mask]) if mask.any() else 0.0
                    running_iou[task_name].update(scene_m_iou)

                # Precision/recall/F1 for pixel_semantic tasks (foreground class).
                # The GT is already the full-tile (r, H, W) raster (taken before any
                # per-fragment transform in MALiBU3DDataset.prepare_test_data), and
                # the prediction was merged into a dense raster above.
                pixel_prf_metrics_scene = {}
                for task_name in pixel_semantic_tasks:
                    if task_name not in targets_by_task:
                        continue
                    if task_name not in pixel_logits_np:
                        continue
                    tc = task_configs[task_name]
                    ignore_index = int(tc["ignore_index"])
                    num_networks = int(tc.get("num_networks", 1))
                    names = list(tc["names"])
                    fg_idx = names.index("Foreground") if "Foreground" in names else 1
                    channel_names = tc.get("channel_names") or [
                        f"ch{c}" for c in range(num_networks)
                    ]
                    target_arr = np.asarray(targets_by_task[task_name])
                    pred_arr = np.asarray(pixel_logits_np[task_name])
                    if pred_arr.shape != target_arr.shape:
                        raise ValueError(
                            f"{task_name} scene {data_name}: prediction shape {pred_arr.shape} "
                            f"!= ground-truth shape {target_arr.shape}"
                        )
                    channel_stats = {}
                    dilated_enabled = dilated_prf_enabled(tc)
                    radius_px = int(tc.get("buffer_radius_px", 3))
                    for c in range(num_networks):
                        ch_name = (
                            channel_names[c] if c < len(channel_names) else f"ch{c}"
                        )
                        tp, fp, fn = binary_prf_counts(
                            pred_arr[c], target_arr[c], ignore_index, fg_idx
                        )
                        channel_stats[ch_name] = dict(tp=tp, fp=fp, fn=fn)
                        if dilated_enabled:
                            # Same 0.5-threshold masks as binary_prf_counts / val.
                            valid = np.isfinite(pred_arr[c]) & (
                                target_arr[c] != ignore_index
                            )
                            pred_fg = (pred_arr[c] > 0.5) & valid
                            gt_fg = (target_arr[c] == fg_idx) & valid
                            p_num, p_denom, r_num, r_denom = (
                                dilated_precision_recall_counts(
                                    pred_fg, gt_fg, valid, radius_px=radius_px
                                )
                            )
                            channel_stats[ch_name].update(
                                p_num=p_num,
                                p_denom=p_denom,
                                r_num=r_num,
                                r_denom=r_denom,
                            )
                    pixel_prf_metrics_scene[task_name] = channel_stats

                    # Task-level confusion histogram summed over channels (as in
                    # MultiTaskEvaluator), aggregated and logged with the semantic
                    # tasks below.
                    num_classes = int(tc["num_classes"])
                    inter_sum = union_sum = target_sum = None
                    for c in range(num_networks):
                        valid = np.isfinite(pred_arr[c])
                        pred_cls = (pred_arr[c] > 0.5).astype(np.int64)
                        target_cls = target_arr[c].astype(np.int64).copy()
                        target_cls[~valid] = ignore_index
                        intersection, union, target_hist = intersection_and_union(
                            pred_cls, target_cls, num_classes, ignore_index
                        )
                        if inter_sum is None:
                            inter_sum, union_sum, target_sum = (
                                intersection,
                                union,
                                target_hist,
                            )
                        else:
                            inter_sum = inter_sum + intersection
                            union_sum = union_sum + union
                            target_sum = target_sum + target_hist
                    if inter_sum is not None:
                        sem_metrics_scene[task_name] = dict(
                            intersection=inter_sum,
                            union=union_sum,
                            target=target_sum,
                        )

                for task_name in regression_tasks:
                    if task_name not in targets_by_task:
                        logger.warning(
                            "Scene %s: skip regression task %s (missing ground truth).",
                            data_name,
                            task_name,
                        )
                        continue
                    if task_name not in pred_reg_np:
                        continue
                    pred_np = np.asarray(
                        pred_reg_np[task_name], dtype=np.float64
                    ).reshape(-1)
                    tgt_np = np.asarray(
                        self._target_for_metrics(
                            task_name, targets_by_task, scene_extra
                        ),
                        dtype=np.float64,
                    ).reshape(-1)
                    p_t = torch.from_numpy(pred_np).float()
                    t_t = torch.from_numpy(tgt_np).float()
                    p_m, t_m = self._gather_masked(p_t, t_t)
                    if p_m is not None:
                        mae_b, mse_b, cnt_b = accumulate_regression_errors(p_m, t_m)
                        reg_sums_global[task_name]["mae"] += mae_b
                        reg_sums_global[task_name]["mse"] += mse_b
                        reg_sums_global[task_name]["count"] += cnt_b

                td_metrics_scene = {}
                for task_name in tile_distribution_tasks:
                    if task_name not in targets_by_task:
                        logger.warning(
                            "Scene %s: skip tile_distribution task %s (missing ground truth).",
                            data_name,
                            task_name,
                        )
                        continue
                    forwarded = batch_td_metrics[b_idx]
                    if forwarded is not None:
                        if task_name not in forwarded:
                            continue
                        if forwarded[task_name] is not None:
                            td_metrics_scene[task_name] = forwarded[task_name]
                        continue
                    dist_np = batch_td_dist_np[b_idx].get(task_name)
                    if dist_np is None:
                        continue
                    tgt_np = np.asarray(
                        self._target_for_metrics(
                            task_name, targets_by_task, scene_extra
                        ),
                        dtype=np.int64,
                    ).reshape(-1)
                    td_m = self._tile_distribution_metrics_from_dist(
                        dist_np,
                        tgt_np,
                        int(task_configs[task_name]["ignore_index"]),
                        int(task_configs[task_name]["num_classes"]),
                    )
                    if td_m is not None:
                        td_metrics_scene[task_name] = td_m

                # Point-wise mIoU for tile_distribution tasks with pointwise_supervision.
                for task_name in tile_pw_miou_tasks:
                    if task_name not in targets_by_task:
                        continue
                    pred_np = batch_td_pred_np[b_idx].get(task_name)
                    if pred_np is None:
                        continue
                    pred_np = np.asarray(pred_np).reshape(-1)
                    tgt_np = np.asarray(
                        self._target_for_metrics(
                            task_name, targets_by_task, scene_extra
                        ),
                        dtype=np.int64,
                    ).reshape(-1)
                    if pred_np.shape[0] != tgt_np.shape[0]:
                        logger.warning(
                            "Scene %s: tile_distribution mIoU task %s pred/target length "
                            "mismatch (%d vs %d); skipping mIoU.",
                            data_name,
                            task_name,
                            pred_np.shape[0],
                            tgt_np.shape[0],
                        )
                        continue
                    tc = task_configs[task_name]
                    intersection, union, target_hist = intersection_and_union(
                        pred_np,
                        tgt_np,
                        int(tc["num_classes"]),
                        int(tc["ignore_index"]),
                    )
                    sem_metrics_scene[task_name] = dict(
                        intersection=intersection,
                        union=union,
                        target=target_hist,
                    )

                record[data_name] = dict(
                    semantic=sem_metrics_scene,
                    tile_distribution=td_metrics_scene,
                    pixel_semantic=pixel_prf_metrics_scene,
                )

            all_cached_here = all(batch_all_cached)
            prog.step(cached=all_cached_here)

        with EvaluationProgressBar(
            n_batches,
            desc="Test multitask",
        ) as prog:
            for _, batch in enumerate(self.test_loader):
                _process_one_multitask_batch(batch, prog)

        logger.info("Syncing ...")
        comm.synchronize()

        if comm.get_world_size() > 1 and regression_tasks:
            flat = []
            for task_name in regression_tasks:
                s = reg_sums_global[task_name]
                flat.extend([s["mae"], s["mse"], s["count"]])
            buf = torch.tensor(flat, dtype=torch.float64, device="cuda")
            dist.all_reduce(buf)
            flat = buf.cpu().tolist()
            for task_index, task_name in enumerate(regression_tasks):
                base = task_index * 3
                reg_sums_global[task_name]["mae"] = flat[base]
                reg_sums_global[task_name]["mse"] = flat[base + 1]
                reg_sums_global[task_name]["count"] = flat[base + 2]

        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            merged = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                merged.update(r)
                del r

            hist_tasks = semantic_tasks + pixel_semantic_tasks + tile_pw_miou_tasks
            per_task_sem = {t: None for t in hist_tasks}
            per_task_pixel_prf = {t: {} for t in pixel_semantic_tasks}
            for _, payload in merged.items():
                for task_name, meters in payload["semantic"].items():
                    if task_name not in per_task_sem:
                        continue
                    if per_task_sem[task_name] is None:
                        per_task_sem[task_name] = {
                            "intersection": meters["intersection"].copy(),
                            "union": meters["union"].copy(),
                            "target": meters["target"].copy(),
                        }
                    else:
                        per_task_sem[task_name]["intersection"] += meters[
                            "intersection"
                        ]
                        per_task_sem[task_name]["union"] += meters["union"]
                        per_task_sem[task_name]["target"] += meters["target"]
                for task_name, meters in payload.get("tile_distribution", {}).items():
                    if task_name not in td_sums_global:
                        continue
                    td_sums_global[task_name]["kl_weighted"] += meters["kl_weighted"]
                    td_sums_global[task_name]["weight"] += meters["weight"]
                    td_sums_global[task_name]["abs_weighted"] += np.asarray(
                        meters["abs_weighted"], dtype=np.float64
                    )
                for task_name, channel_stats in payload.get(
                    "pixel_semantic", {}
                ).items():
                    if task_name not in per_task_pixel_prf:
                        continue
                    for ch_name, counts in channel_stats.items():
                        acc = per_task_pixel_prf[task_name].setdefault(
                            ch_name, {"tp": 0, "fp": 0, "fn": 0}
                        )
                        acc["tp"] += counts["tp"]
                        acc["fp"] += counts["fp"]
                        acc["fn"] += counts["fn"]
                        if "p_num" in counts:
                            acc["p_num"] = acc.get("p_num", 0.0) + counts["p_num"]
                            acc["p_denom"] = acc.get("p_denom", 0.0) + counts["p_denom"]
                            acc["r_num"] = acc.get("r_num", 0.0) + counts["r_num"]
                            acc["r_denom"] = acc.get("r_denom", 0.0) + counts["r_denom"]

            per_task_metrics = {}
            for task_name in hist_tasks:
                hist = per_task_sem[task_name]
                if hist is None:
                    logger.warning(
                        "No aggregated semantic histograms for task %s.", task_name
                    )
                    continue
                intersection = hist["intersection"]
                union = hist["union"]
                target_hist = hist["target"]
                task_config = task_configs[task_name]
                iou_class = intersection / (union + 1e-10)
                acc_class = intersection / (target_hist + 1e-10)
                m_iou = mean_iou_from_hist(intersection, union)
                m_acc = mean_acc_from_hist(intersection, target_hist, union=union)
                all_acc = sum(intersection) / (sum(target_hist) + 1e-10)
                per_task_metrics[task_name] = dict(
                    iou_class=iou_class,
                    acc_class=acc_class,
                    m_iou=m_iou,
                    m_acc=m_acc,
                    all_acc=all_acc,
                    names=list(task_config["names"]),
                )
                logger.info(
                    "[task={}] Test result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                        task_name,
                        m_iou,
                        m_acc,
                        all_acc,
                    )
                )
                for class_idx in range(int(task_config["num_classes"])):
                    class_name = per_task_metrics[task_name]["names"][class_idx]
                    logger.info(
                        "[task={}] Class_{}-{} Result: iou/accuracy {:.4f}/{:.4f}".format(
                            task_name,
                            class_idx,
                            class_name,
                            iou_class[class_idx],
                            acc_class[class_idx],
                        )
                    )

            log_dict = {}
            for task_name, metric in per_task_metrics.items():
                log_dict[metric_tag("test", "mIoU", task=task_name)] = float(
                    metric["m_iou"]
                )
                log_dict[metric_tag("test", "mAcc", task=task_name)] = float(
                    metric["m_acc"]
                )
                log_dict[metric_tag("test", "allAcc", task=task_name)] = float(
                    metric["all_acc"]
                )
                if self.write_cls_iou:
                    task_config = task_configs[task_name]
                    for class_idx in range(int(task_config["num_classes"])):
                        if class_idx == task_config["ignore_index"]:
                            continue
                        slug = class_name_slug(metric["names"][class_idx])
                        log_dict[iou_class_tag("test", slug, task=task_name)] = float(
                            metric["iou_class"][class_idx]
                        )

            for task_name in pixel_semantic_tasks:
                channel_stats = per_task_pixel_prf.get(task_name) or {}
                task_config = task_configs[task_name]
                radius_px = int(task_config.get("buffer_radius_px", 3))
                dilated_enabled = dilated_prf_enabled(task_config)
                for ch_name, counts in channel_stats.items():
                    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
                    if tp + fp + fn == 0:
                        logger.warning(
                            "[task=%s] Channel %s: no test-set pixel_semantic "
                            "samples accumulated; skipping P/R/F1.",
                            task_name,
                            ch_name,
                        )
                        continue
                    precision, recall, f1 = precision_recall_f1(
                        tp, tp + fp, tp, tp + fn
                    )
                    ch_slug = class_name_slug(ch_name)
                    log_dict[
                        metric_tag("test", f"{ch_slug}/precision", task=task_name)
                    ] = float(precision)
                    log_dict[
                        metric_tag("test", f"{ch_slug}/recall", task=task_name)
                    ] = float(recall)
                    log_dict[metric_tag("test", f"{ch_slug}/f1", task=task_name)] = (
                        float(f1)
                    )
                    if dilated_enabled and "p_num" in counts:
                        dilated_precision, dilated_recall, dilated_f1 = (
                            precision_recall_f1(
                                counts["p_num"],
                                counts["p_denom"],
                                counts["r_num"],
                                counts["r_denom"],
                            )
                        )
                        logger.info(
                            "[task={}] Channel {} Test result: precision/recall/f1 "
                            "{:.4f}/{:.4f}/{:.4f} (tp={}, fp={}, fn={}) | "
                            "dilated(r={}px) precision/recall/f1 "
                            "{:.4f}/{:.4f}/{:.4f}.".format(
                                task_name,
                                ch_name,
                                precision,
                                recall,
                                f1,
                                tp,
                                fp,
                                fn,
                                radius_px,
                                dilated_precision,
                                dilated_recall,
                                dilated_f1,
                            )
                        )
                        log_dict[
                            metric_tag(
                                "test", f"{ch_slug}/dilated_precision", task=task_name
                            )
                        ] = float(dilated_precision)
                        log_dict[
                            metric_tag(
                                "test", f"{ch_slug}/dilated_recall", task=task_name
                            )
                        ] = float(dilated_recall)
                        log_dict[
                            metric_tag("test", f"{ch_slug}/dilated_f1", task=task_name)
                        ] = float(dilated_f1)
                    else:
                        logger.info(
                            "[task={}] Channel {} Test result: precision/recall/f1 "
                            "{:.4f}/{:.4f}/{:.4f} (tp={}, fp={}, fn={}).".format(
                                task_name, ch_name, precision, recall, f1, tp, fp, fn
                            )
                        )

            for task_name in regression_tasks:
                s = reg_sums_global[task_name]
                cnt = s["count"]
                if cnt <= 1e-8:
                    continue
                mae = s["mae"] / cnt
                rmse = (s["mse"] / cnt) ** 0.5
                logger.info(
                    "[task={}] Test regression: MAE {:.6f} RMSE {:.6f} (n={:.0f}).".format(
                        task_name, mae, rmse, cnt
                    )
                )
                log_dict[f"test/reg/{task_name}/mae"] = float(mae)
                log_dict[f"test/reg/{task_name}/rmse"] = float(rmse)

            final_kl_by_task = {}
            final_tv_by_task = {}
            for task_name in tile_distribution_tasks:
                s = td_sums_global[task_name]
                if s["weight"] <= 1e-8:
                    logger.warning(
                        "[task=%s] Test tile-distribution: no samples accumulated; "
                        "skipping metrics.",
                        task_name,
                    )
                    continue
                final_kl = s["kl_weighted"] / s["weight"]
                mae = s["abs_weighted"] / s["weight"]
                tv = float(tv_from_abs_errors(torch.as_tensor(mae)))
                final_kl_by_task[task_name] = final_kl
                final_tv_by_task[task_name] = tv
                task_config = task_configs[task_name]
                logger.info(
                    "[task={}] Test tile-distribution: weighted KL {:.6f} "
                    "TV {:.6f} (N={:.0f}).".format(task_name, final_kl, tv, s["weight"])
                )
                for class_idx in range(int(task_config["num_classes"])):
                    class_name = task_config["names"][class_idx]
                    logger.info(
                        "[task={}] Class_{}-{} Result: mae {:.6f}".format(
                            task_name,
                            class_idx,
                            class_name,
                            float(mae[class_idx]),
                        )
                    )
                log_dict[
                    metric_tag("test", f"weighted_kl/{task_name}", task="nathab")
                ] = float(final_kl)
                log_dict[metric_tag("test", f"tv/{task_name}", task="nathab")] = tv
                for class_idx in range(int(task_config["num_classes"])):
                    slug = class_name_slug(task_config["names"][class_idx])
                    log_dict[
                        metric_tag("test", f"mae/{task_name}/{slug}", task="nathab")
                    ] = float(mae[class_idx])
            if final_kl_by_task:
                log_dict[metric_tag("test", "weighted_kl/sum", task="nathab")] = float(
                    sum(final_kl_by_task.values())
                )
                log_dict[metric_tag("test", "tv/mean", task="nathab")] = float(
                    sum(final_tv_by_task.values()) / len(final_tv_by_task)
                )

            log_test_f1 = getattr(self.cfg, "log_test_f1", False)
            if log_test_f1:
                for task_name, metric in per_task_metrics.items():
                    f1_class, macro_f1 = f1_scores_from_hist(
                        per_task_sem[task_name]["intersection"],
                        per_task_sem[task_name]["union"],
                        per_task_sem[task_name]["target"],
                    )
                    logger.info(
                        "[task={}] Test result: macro-F1 {:.4f}".format(
                            task_name, macro_f1
                        )
                    )
                    task_config = task_configs[task_name]
                    for class_idx in range(int(task_config["num_classes"])):
                        class_name = metric["names"][class_idx]
                        logger.info(
                            "[task={}] Class_{}-{} Result: f1 {:.4f}".format(
                                task_name,
                                class_idx,
                                class_name,
                                f1_class[class_idx],
                            )
                        )
                    log_dict[metric_tag("test", "macro_f1", task=task_name)] = float(
                        macro_f1
                    )
                    for i in range(len(f1_class)):
                        slug = class_name_slug(metric["names"][i])
                        log_dict[f"test/{task_name}/f1/{slug}"] = float(f1_class[i])
        else:
            log_dict = None

        self.end_test_timing_and_log(extra_log_dict=log_dict)
        if comm.is_main_process():
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.begin_test_timing()
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        record = {}
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            data_name = input_dict.get("name", None)
            if data_name is None:
                raise RuntimeError(
                    "ClsTester requires sample `name` for deduplicated evaluation."
                )
            if isinstance(data_name, str):
                data_name = [data_name]
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            if len(data_name) != pred.shape[0]:
                raise RuntimeError(
                    "Number of sample names does not match batch size in ClsTester."
                )
            for b, name in enumerate(data_name):
                sample_pred = pred[b : b + 1].reshape(-1)
                sample_label = label[b : b + 1].reshape(-1)
                sample_intersection, sample_union, sample_target = (
                    intersection_and_union_gpu(
                        sample_pred,
                        sample_label,
                        self.cfg.data.num_classes,
                        self.cfg.data.ignore_index,
                    )
                )
                record[name] = dict(
                    intersection=sample_intersection.cpu().numpy(),
                    union=sample_union.cpu().numpy(),
                    target=sample_target.cpu().numpy(),
                )
            intersection, union, target = intersection_and_union_gpu(
                pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_meter.update(intersection), union_meter.update(
                union
            ), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info(
                "Test: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f} ".format(
                    i + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    accuracy=accuracy,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)
            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                    mIoU, mAcc, allAcc
                )
            )

            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )

            log_test_f1 = getattr(self.cfg, "log_test_f1", False)
            if log_test_f1:
                f1_class, macro_f1 = f1_scores_from_hist(intersection, union, target)
                logger.info("Test result: macro-F1 {:.4f}".format(macro_f1))
                for i in range(self.cfg.data.num_classes):
                    logger.info(
                        "Class_{idx} - {name} Result: f1 {f1:.4f}".format(
                            idx=i,
                            name=self.cfg.data.names[i],
                            f1=f1_class[i],
                        )
                    )

            log_dict = {
                "test/mIoU": float(mIoU),
                "test/mAcc": float(mAcc),
                "test/allAcc": float(allAcc),
            }
            for i in range(self.cfg.data.num_classes):
                cls_name = self.cfg.data.names[i]
                log_dict[f"test/iou_{cls_name}"] = float(iou_class[i])
                log_dict[f"test/acc_{cls_name}"] = float(accuracy_class[i])
            if log_test_f1:
                log_dict["test/f1_macro"] = float(macro_f1)
                for i in range(self.cfg.data.num_classes):
                    cls_name = self.cfg.data.names[i]
                    log_dict[f"test/f1_{cls_name}"] = float(f1_class[i])
        else:
            log_dict = None

        # Same contract as SemSegTester.test_metrics.
        self.test_metrics = log_dict

        self.end_test_timing_and_log(extra_log_dict=log_dict)
        if comm.is_main_process():
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)
