"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import json
from uuid import uuid4
import os
import time
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
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    f1_scores_from_hist,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)

try:
    import pointops
except:
    pointops = None

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
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegTester(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
            or self.cfg.data.test.type == "ScanNetPPDataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
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
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
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
                if self.cfg.data.test.type == "ScanNetPPDataset":
                    pred = pred.topk(3, dim=1)[1].data.cpu().numpy()
                else:
                    summed = pred.sum(1)
                    pred = pred.max(1)[1].data.cpu().numpy()
                if "origin_segment" in data_dict.keys():
                    assert "inverse" in data_dict.keys()
                    pred = pred[data_dict["inverse"]]
                    segment = data_dict["origin_segment"]
                np.save(pred_save_path, pred)
            if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "ScanNetPPDataset":
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    pred.astype(np.int32),
                    delimiter=",",
                    fmt="%d",
                )
                pred = pred[:, 0]  # for mIoU, TODO: support top3 mIoU
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                submit = pred.astype(np.uint32)
                submit = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(submit).astype(np.uint32)
                submit.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

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

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

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
                f1_class, macro_f1 = f1_scores_from_hist(
                    intersection, union, target
                )
                logger.info(
                    "Test result: macro-F1 {:.4f}".format(macro_f1)
                )
                for i in range(self.cfg.data.num_classes):
                    logger.info(
                        "Class_{idx} - {name} Result: f1 {f1:.4f}".format(
                            idx=i,
                            name=self.cfg.data.names[i],
                            f1=f1_class[i],
                        )
                    )

            # Optional logging to Weights & Biases for test metrics
            if getattr(self.cfg, "enable_wandb", False) and wandb.run is not None:
                log_dict = {
                    "test/mIoU": float(mIoU),
                    "test/mAcc": float(mAcc),
                    "test/allAcc": float(allAcc),
                }
                for i in range(self.cfg.data.num_classes):
                    cls_name = self.cfg.data.names[i]
                    log_dict[f"test/iou_{cls_name}"] = float(iou_class[i])
                    log_dict[f"test/acc_{cls_name}"] = float(
                        accuracy_class[i]
                    )
                if log_test_f1:
                    log_dict["test/f1_macro"] = float(macro_f1)
                    for i in range(self.cfg.data.num_classes):
                        cls_name = self.cfg.data.names[i]
                        log_dict[f"test/f1_{cls_name}"] = float(f1_class[i])
                wandb.log(log_dict)

            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class MultiTaskTester(TesterBase):
    """Fragment-based test for MultiTaskSegmentorV2 (semantic IoU + regression MAE/RMSE).

    Expects cfg.data.task_configs and cfg.data.main_task (required when multiple tasks).
    Ground-truth tensors for each task must be present at full resolution in the sample dict
    (see Flair3DDataset.prepare_test_data).
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
    def _regression_task_names(task_configs):
        return [
            k
            for k, task_config in task_configs.items()
            if task_config.get("task_type") == "regression"
        ]

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

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Multi-task Evaluation >>>>>>>>>>>>>>>>")

        task_configs = self._get_task_configs()
        main_task = self._main_task_name(task_configs)
        semantic_tasks = self._semantic_task_names(task_configs)
        regression_tasks = self._regression_task_names(task_configs)

        batch_time = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        comm.synchronize()

        reg_sums_global = {
            t: {"mae": 0.0, "mse": 0.0, "count": 0.0} for t in regression_tasks
        }

        running_iou = {t: AverageMeter() for t in semantic_tasks}

        record = {}

        for idx, batch in enumerate(self.test_loader):
            start = time.time()
            data_dict = batch[0]
            fragment_list = data_dict.pop("fragment_list")
            data_name = data_dict.pop("name")

            # Pop every configured task label still present (includes segment).
            targets_by_task = {}
            for tn in task_configs:
                if tn in data_dict:
                    targets_by_task[tn] = data_dict.pop(tn)
            scene_extra = data_dict

            ref_arr = None
            for cand in ("segment", main_task, *semantic_tasks):
                cs = str(cand)
                if cs in targets_by_task:
                    ref_arr = targets_by_task[cs]
                    break
            if ref_arr is None and targets_by_task:
                ref_arr = next(iter(targets_by_task.values()))
            if ref_arr is None:
                raise RuntimeError(
                    f"Scene {data_name}: no labels found for any task in task_configs."
                )
            n_ref = int(np.asarray(ref_arr).size)

            sem_cache_paths = {
                t: os.path.join(save_path, f"{data_name}_pred_{t}.npy")
                for t in semantic_tasks
            }
            reg_cache_paths = {
                t: os.path.join(save_path, f"{data_name}_reg_{t}.npy")
                for t in regression_tasks
            }
            all_sem_cached = all(os.path.isfile(p) for p in sem_cache_paths.values())
            all_reg_cached = all(os.path.isfile(p) for p in reg_cache_paths.values())

            pred_cls_np = {}
            pred_reg_np = {}

            if all_sem_cached and (
                len(regression_tasks) == 0 or all_reg_cached
            ):
                logger.info(
                    "{}/{}: {}, loaded cached multitask predictions.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                for t in semantic_tasks:
                    pred_cls_np[t] = np.load(sem_cache_paths[t])
                for t in regression_tasks:
                    pred_reg_np[t] = np.load(reg_cache_paths[t])
            else:
                pred_sem = {
                    t: torch.zeros(
                        (n_ref, int(task_configs[t]["num_classes"])),
                        device="cuda",
                    )
                    for t in semantic_tasks
                }
                reg_sum = {
                    t: torch.zeros((n_ref,), device="cuda") for t in regression_tasks
                }
                reg_cnt = {
                    t: torch.zeros((n_ref,), device="cuda") for t in regression_tasks
                }

                use_voxel_broadcast = "inverse" in fragment_list[0]

                for fi in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = fi * fragment_batch_size, min(
                        (fi + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]

                    with torch.no_grad():
                        output_dict = self.model(input_dict)
                        logits_by_task = output_dict.get("seg_logits_by_task") or {}
                        reg_pred_by_task = output_dict.get("reg_pred_by_task") or {}

                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()

                        for task_name in semantic_tasks:
                            if task_name not in logits_by_task:
                                continue
                            pred_part = F.softmax(logits_by_task[task_name], dim=-1)
                            if use_voxel_broadcast:
                                inv = fragment_list[s_i:e_i][0]["inverse"]
                                inv = (
                                    torch.from_numpy(inv).long().cuda()
                                    if isinstance(inv, np.ndarray)
                                    else inv.long().cuda()
                                )
                                pred_sem[task_name] += pred_part[inv, :]
                            else:
                                bs = 0
                                for be in input_dict["offset"]:
                                    pred_sem[task_name][idx_part[bs:be], :] += pred_part[
                                        bs:be
                                    ]
                                    bs = be

                        for task_name in regression_tasks:
                            if task_name not in reg_pred_by_task:
                                continue
                            pred_part = reg_pred_by_task[task_name].reshape(-1).float()
                            if use_voxel_broadcast:
                                inv = fragment_list[s_i:e_i][0]["inverse"]
                                inv = (
                                    torch.from_numpy(inv).long().cuda()
                                    if isinstance(inv, np.ndarray)
                                    else inv.long().cuda()
                                )
                                # Match semantic path: pred_part has one row per voxelized point;
                                # inv maps full-scene indices -> voxel rows (same as softmax broadcast).
                                gathered = pred_part[inv]
                                reg_sum[task_name] += gathered
                                reg_cnt[task_name] += torch.ones_like(gathered)
                            else:
                                bs = 0
                                for be in input_dict["offset"]:
                                    sl = slice(bs, be)
                                    ip = idx_part[sl]
                                    reg_sum[task_name][ip] += pred_part[sl]
                                    reg_cnt[task_name][ip] += 1
                                    bs = be

                    logger.info(
                        "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name=data_name,
                            batch_idx=fi,
                            batch_num=len(fragment_list),
                        )
                    )

                for task_name in semantic_tasks:
                    if task_name not in targets_by_task:
                        continue
                    pred_raw = pred_sem[task_name].max(1)[1].cpu().numpy()
                    pred_np, _ = self._apply_inverse_origin_np(
                        pred_raw,
                        np.asarray(targets_by_task[task_name]).reshape(-1),
                        scene_extra,
                        task_name,
                    )
                    pred_cls_np[task_name] = pred_np
                    np.save(sem_cache_paths[task_name], pred_np)

                for task_name in regression_tasks:
                    if task_name not in targets_by_task:
                        continue
                    pred_raw = (
                        reg_sum[task_name] / reg_cnt[task_name].clamp(min=1)
                    ).cpu().numpy()
                    pred_np, _ = self._apply_inverse_origin_np(
                        pred_raw,
                        np.asarray(targets_by_task[task_name], dtype=np.float64).reshape(
                            -1
                        ),
                        scene_extra,
                        task_name,
                    )
                    pred_reg_np[task_name] = pred_np
                    np.save(reg_cache_paths[task_name], pred_np)

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
                pred_np = np.asarray(pred_reg_np[task_name], dtype=np.float64).reshape(
                    -1
                )
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
                    err = (p_m - t_m).abs()
                    reg_sums_global[task_name]["mae"] += float(err.sum().item())
                    reg_sums_global[task_name]["mse"] += float(
                        ((p_m - t_m) ** 2).sum().item()
                    )
                    reg_sums_global[task_name]["count"] += float(err.numel())

            record[data_name] = dict(semantic=sem_metrics_scene)

            batch_time.update(time.time() - start)
            msg_parts = [
                f"Test: {data_name} [{idx + 1}/{len(self.test_loader)}]-{n_ref} "
                f"Batch {batch_time.val:.3f} ({batch_time.avg:.3f})"
            ]
            for task_name in semantic_tasks:
                if task_name in sem_metrics_scene:
                    msg_parts.append(
                        f"[{task_name}] mIoU {running_iou[task_name].val:.4f} "
                        f"({running_iou[task_name].avg:.4f})"
                    )
            logger.info(" ".join(msg_parts))

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
            for ti, task_name in enumerate(regression_tasks):
                base = ti * 3
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

            per_task_sem = {t: None for t in semantic_tasks}
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
                        per_task_sem[task_name]["intersection"] += meters["intersection"]
                        per_task_sem[task_name]["union"] += meters["union"]
                        per_task_sem[task_name]["target"] += meters["target"]

            per_task_metrics = {}
            for task_name in semantic_tasks:
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
                m_iou = np.mean(iou_class)
                m_acc = np.mean(acc_class)
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

            if getattr(self.cfg, "enable_wandb", False) and wandb.run is not None:
                wandb_log = {}
                for task_name, metric in per_task_metrics.items():
                    wandb_log[f"test/{task_name}/mIoU"] = float(metric["m_iou"])
                    wandb_log[f"test/{task_name}/mAcc"] = float(metric["m_acc"])
                    wandb_log[f"test/{task_name}/allAcc"] = float(metric["all_acc"])
                    if self.write_cls_iou:
                        task_config = task_configs[task_name]
                        for class_idx in range(int(task_config["num_classes"])):
                            if class_idx == task_config["ignore_index"]:
                                continue
                            class_name = metric["names"][class_idx]
                            slug = "".join(
                                c if (c.isalnum() or c in "._-") else "_"
                                for c in str(class_name).strip().replace(" ", "_")
                            )
                            wandb_log[
                                f"test/{task_name}/iou/{class_idx}_{slug}"
                            ] = float(metric["iou_class"][class_idx])
                if wandb_log:
                    wandb.log(wandb_log)

            for task_name in regression_tasks:
                s = reg_sums_global[task_name]
                cnt = s["count"]
                if cnt <= 1e-8:
                    continue
                mae = s["mae"] / cnt
                rmse = (s["mse"] / cnt) ** 0.5
                logger.info(
                    "[task={}] Test regression: MAE {:.6f} RMSE {:.6f} (n={:.0f}).".format(
                        task_name,
                        mae,
                        rmse,
                        cnt,
                    )
                )

            if getattr(self.cfg, "enable_wandb", False) and wandb.run is not None:
                reg_wandb = {}
                for task_name in regression_tasks:
                    s = reg_sums_global[task_name]
                    cnt = s["count"]
                    if cnt <= 1e-8:
                        continue
                    mae = s["mae"] / cnt
                    rmse = (s["mse"] / cnt) ** 0.5
                    reg_wandb[f"test/reg/{task_name}/mae"] = float(mae)
                    reg_wandb[f"test/reg/{task_name}/rmse"] = float(rmse)
                if reg_wandb:
                    wandb.log(reg_wandb)

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
                    if getattr(self.cfg, "enable_wandb", False) and wandb.run is not None:
                        f1_extra = {f"test/{task_name}/f1_macro": float(macro_f1)}
                        for i in range(len(f1_class)):
                            cn = metric["names"][i]
                            slug = "".join(
                                c if (c.isalnum() or c in "._-") else "_"
                                for c in str(cn).strip().replace(" ", "_")
                            )
                            f1_extra[f"test/{task_name}/f1_{i}_{slug}"] = float(
                                f1_class[i]
                            )
                        wandb.log(f1_extra)

            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class DINOSemSegTester(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
            or self.cfg.data.test.type == "ScanNetPPDataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            dino_coord = data_dict.pop("dino_coord").cuda(non_blocking=True)
            dino_feat = data_dict.pop("dino_feat").cuda(non_blocking=True)
            dino_offset = data_dict.pop("dino_offset").cuda(non_blocking=True)
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
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
                use_voxel_broadcast = "inverse" in fragment_list[0]
                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    input_dict["dino_coord"] = dino_coord
                    input_dict["dino_feat"] = dino_feat
                    input_dict["dino_offset"] = dino_offset
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
                if self.cfg.data.test.type == "ScanNetPPDataset":
                    pred = pred.topk(3, dim=1)[1].data.cpu().numpy()
                else:
                    pred = pred.max(1)[1].data.cpu().numpy()
                if "origin_segment" in data_dict.keys():
                    assert "inverse" in data_dict.keys()
                    pred = pred[data_dict["inverse"]]
                    segment = data_dict["origin_segment"]
                np.save(pred_save_path, pred)
            if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "ScanNetPPDataset":
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    pred.astype(np.int32),
                    delimiter=",",
                    fmt="%d",
                )
                pred = pred[:, 0]  # for mIoU, TODO: support top3 mIoU
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                submit = pred.astype(np.uint32)
                submit = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(submit).astype(np.uint32)
                submit.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

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

            batch_time.update(time.time() - end)
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

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
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
                f1_class, macro_f1 = f1_scores_from_hist(
                    intersection, union, target
                )
                logger.info(
                    "Test result: macro-F1 {:.4f}".format(macro_f1)
                )
                for i in range(self.cfg.data.num_classes):
                    logger.info(
                        "Class_{idx} - {name} Result: f1 {f1:.4f}".format(
                            idx=i,
                            name=self.cfg.data.names[i],
                            f1=f1_class[i],
                        )
                    )

            if getattr(self.cfg, "enable_wandb", False) and wandb.run is not None:
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
                wandb.log(log_dict)

            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
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
                f1_class, macro_f1 = f1_scores_from_hist(
                    intersection, union, target
                )
                logger.info(
                    "Test result: macro-F1 {:.4f}".format(macro_f1)
                )
                for i in range(self.cfg.data.num_classes):
                    logger.info(
                        "Class_{idx} - {name} Result: f1 {f1:.4f}".format(
                            idx=i,
                            name=self.cfg.data.names[i],
                            f1=f1_class[i],
                        )
                    )

            if getattr(self.cfg, "enable_wandb", False) and wandb.run is not None:
                log_dict = {
                    "test/mIoU": float(mIoU),
                    "test/mAcc": float(mAcc),
                    "test/allAcc": float(allAcc),
                }
                for i in range(self.cfg.data.num_classes):
                    cls_name = self.cfg.data.names[i]
                    log_dict[f"test/iou_{cls_name}"] = float(iou_class[i])
                    log_dict[f"test/acc_{cls_name}"] = float(
                        accuracy_class[i]
                    )
                if log_test_f1:
                    log_dict["test/f1_macro"] = float(macro_f1)
                    for i in range(self.cfg.data.num_classes):
                        cls_name = self.cfg.data.names[i]
                        log_dict[f"test/f1_{cls_name}"] = float(f1_class[i])
                wandb.log(log_dict)

        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class ClsVotingTester(TesterBase):
    def __init__(
        self,
        num_repeat=100,
        metric="allAcc",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_repeat = num_repeat
        self.metric = metric
        self.best_idx = 0
        self.best_record = None
        self.best_metric = 0

    def test(self):
        for i in range(self.num_repeat):
            logger = get_root_logger()
            logger.info(f">>>>>>>>>>>>>>>> Start Evaluation {i + 1} >>>>>>>>>>>>>>>>")
            record = self.test_once()
            if comm.is_main_process():
                if record[self.metric] > self.best_metric:
                    self.best_record = record
                    self.best_idx = i
                    self.best_metric = record[self.metric]
                info = f"Current best record is Evaluation {i + 1}: "
                for m in self.best_record.keys():
                    info += f"{m}: {self.best_record[m]:.4f} "
                logger.info(info)

    def test_once(self):
        logger = get_root_logger()
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        target_meter = AverageMeter()
        record = {}
        self.model.eval()

        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            voting_list = data_dict.pop("voting_list")
            category = data_dict.pop("category")
            data_name = data_dict.pop("name")
            # pred = torch.zeros([1, self.cfg.data.num_classes]).cuda()
            # for i in range(len(voting_list)):
            #     input_dict = voting_list[i]
            #     for key in input_dict.keys():
            #         if isinstance(input_dict[key], torch.Tensor):
            #             input_dict[key] = input_dict[key].cuda(non_blocking=True)
            #     with torch.no_grad():
            #         pred += F.softmax(self.model(input_dict)["cls_logits"], -1)
            input_dict = collate_fn(voting_list)
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                pred = F.softmax(self.model(input_dict)["cls_logits"], -1).sum(
                    0, keepdim=True
                )
            pred = pred.max(1)[1].cpu().numpy()
            intersection, union, target = intersection_and_union(
                pred, category, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            target_meter.update(target)
            record[data_name] = dict(intersection=intersection, target=target)
            acc = sum(intersection) / (sum(target) + 1e-10)
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
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
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
            accuracy_class = intersection / (target + 1e-10)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info("Val result: mAcc/allAcc {:.4f}/{:.4f}".format(mAcc, allAcc))
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        accuracy=accuracy_class[i],
                    )
                )
            return dict(mAcc=mAcc, allAcc=allAcc)

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class ShapeNetPartSegTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        num_categories = len(self.test_loader.dataset.categories)
        record = {}
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)

        comm.synchronize()
        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
                pred = torch.from_numpy(pred).cuda()
                if "origin_segment" in data_dict.keys():
                    segment = data_dict["origin_segment"]
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            bs = be
                    logger.info(
                        "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name=data_name,
                            batch_idx=i + 1,
                            batch_num=len(fragment_list),
                        )
                    )
                pred = pred.max(1)[1].data
                pred_np = pred.cpu().numpy()
                if "origin_segment" in data_dict.keys():
                    assert "inverse" in data_dict.keys()
                    pred = pred[data_dict["inverse"]]
                    segment = data_dict["origin_segment"]
                np.save(pred_save_path, pred_np)

            category_index = fragment_list[0]["cls_token"]
            category = self.test_loader.dataset.categories[category_index]
            parts_idx = self.test_loader.dataset.category2part[category]
            parts_iou = torch.zeros(len(parts_idx), device="cuda")

            segment = torch.from_numpy(segment).cuda()
            for j, part in enumerate(parts_idx):
                if (torch.sum(segment == part) == 0) and (torch.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (segment == part) & (pred == part)
                    u = (segment == part) | (pred == part)
                    parts_iou[j] = torch.sum(i) / (torch.sum(u) + 1e-10)
            parts_iou_mean = parts_iou.mean().item()
            record[data_name] = dict(
                category_index=int(category_index),
                parts_iou_mean=parts_iou_mean,
            )

            batch_time.update(time.time() - start)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} "
                "({batch_time.avg:.3f}) "
                "Mean IoU {iou:.3f}".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    iou=parts_iou_mean,
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

            iou_category = np.zeros(num_categories, dtype=np.float64)
            iou_count = np.zeros(num_categories, dtype=np.float64)
            for _, meters in record.items():
                category_index = meters["category_index"]
                iou_category[category_index] += meters["parts_iou_mean"]
                iou_count[category_index] += 1

            ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
            iou_per_cat = iou_category / (iou_count + 1e-10)
            cat_mIoU = (
                np.mean(iou_per_cat[iou_count > 0])
                if np.any(iou_count > 0)
                else float("nan")
            )
            logger.info(
                "Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.".format(
                    ins_mIoU, cat_mIoU
                )
            )
            for i in range(num_categories):
                if iou_count[i] == 0:
                    continue
                logger.info(
                    "Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}".format(
                        idx=i,
                        name=self.test_loader.dataset.categories[i],
                        iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                        iou_count=int(iou_count[i]),
                    )
                )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class PartNetEPartSegTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)

        record = {}
        num_parts_total = sum(self.test_loader.dataset.num_parts)
        local_total_iou_parts = np.zeros(num_parts_total, dtype=np.float64)
        local_total_iou_count = np.zeros(num_parts_total, dtype=np.float64)

        comm.synchronize()
        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            cls_token = fragment_list[0]["cls_token"]
            category = self.test_loader.dataset.categories[cls_token]
            parts_idx = self.test_loader.dataset.category2part[category]
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
                pred = torch.from_numpy(pred).cuda()
                if "origin_segment" in data_dict.keys():
                    segment = data_dict["origin_segment"]
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            bs = be
                    logger.info(
                        "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name=data_name,
                            batch_idx=i + 1,
                            batch_num=len(fragment_list),
                        )
                    )
                pred = pred.max(1)[1].data
                pred_np = pred.cpu().numpy()
                if "origin_segment" in data_dict.keys():
                    assert "inverse" in data_dict.keys()
                    pred = pred[data_dict["inverse"]]
                    segment = data_dict["origin_segment"]
                np.save(pred_save_path, pred_np)

            segment = torch.from_numpy(segment).cuda()
            sample_part_record = {}
            for k, part_id in enumerate(parts_idx):
                if k == 0:
                    continue
                if (segment == part_id).sum() == 0:
                    continue
                if (torch.sum(segment == part_id) == 0) and (
                    torch.sum(pred == part_id) == 0
                ):
                    continue
                else:
                    intersection = torch.sum((segment == part_id) & (pred == part_id))
                    union = torch.sum((segment == part_id) | (pred == part_id))
                    part_idx = int(
                        k + self.test_loader.dataset.num_part_offset[cls_token]
                    )
                    sample_part_record[part_idx] = float(
                        (intersection / (union + 1e-10)).item()
                    )
            record[data_name] = dict(part_iou=sample_part_record)
            for part_idx, part_iou in sample_part_record.items():
                local_total_iou_parts[int(part_idx)] += part_iou
                local_total_iou_count[int(part_idx)] += 1

            current_iou_count = local_total_iou_count[local_total_iou_count > 0]
            current_iou_parts = local_total_iou_parts[local_total_iou_count > 0]
            current_iou = current_iou_parts / current_iou_count
            current_iou_mean = (
                current_iou.mean() if current_iou.shape[0] > 0 else float("nan")
            )
            batch_time.update(time.time() - start)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} "
                "({batch_time.avg:.3f}) "
                "Mean IoU {iou:.3f}".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    iou=current_iou_mean,
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

            total_iou_parts = np.zeros(num_parts_total, dtype=np.float64)
            total_iou_count = np.zeros(num_parts_total, dtype=np.float64)
            for _, meters in record.items():
                for part_idx, part_iou in meters["part_iou"].items():
                    total_iou_parts[int(part_idx)] += part_iou
                    total_iou_count[int(part_idx)] += 1

            current_iou_count = total_iou_count[total_iou_count > 0]
            current_iou_parts = total_iou_parts[total_iou_count > 0]
            # part-wise mIoU: average of all sample mIoUs
            part_mIoU = (
                (current_iou_parts / (current_iou_count + 1e-10)).mean()
                if current_iou_count.shape[0] > 0
                else float("nan")
            )
            logger.info("Val result: part mIoU {:.4f}.".format(part_mIoU))
            for i in range(sum(self.test_loader.dataset.num_parts)):
                part_name = self.test_loader.dataset.parts[i]
                if total_iou_count[i] == 0:
                    continue
                logger.info(
                    "Class_{idx}-{name} Result: iou_part/num_sample {iou_part:.4f}/{iou_count:.4f}".format(
                        idx=i,
                        name=part_name,
                        iou_part=total_iou_parts[i] / (total_iou_count[i] + 1e-10),
                        iou_count=int(total_iou_count[i]),
                    )
                )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class InsSegTester(TesterBase):
    def __init__(
        self,
        segment_ignore_index,
        instance_ignore_index,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.valid_class_names = [
            self.cfg.data.names[i]
            for i in range(self.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        self.model.eval()
        scenes = {}

        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()
            data_name = data_dict.pop("name")
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.model(data_dict)
                segment = data_dict["segment"]
                instance = data_dict["instance"]

                if "origin_coord" in data_dict.keys():
                    reverse, _ = pointops.knn_query(
                        1,
                        data_dict["coord"].float(),
                        data_dict["offset"].int(),
                        data_dict["origin_coord"].float(),
                        data_dict["origin_offset"].int(),
                    )
                    reverse = reverse.cpu().flatten().long()
                    output_dict["pred_masks"] = output_dict["pred_masks"][:, reverse]
                    segment = data_dict["origin_segment"]
                    instance = data_dict["origin_instance"]

                gt_instances, pred_instance = self.associate_instances(
                    output_dict, segment, instance
                )

            scenes[data_name] = dict(gt=gt_instances, pred=pred_instance)
            batch_time.update(time.time() - start)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                )
            )
            if self.cfg.data.test.type == "ScanNetPPDataset":
                self.write_scannetpp_results(
                    output_dict["pred_scores"],
                    output_dict["pred_masks"],
                    output_dict["pred_classes"],
                    data_name,
                )

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
            logger.info(
                "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                    all_ap, all_ap_50, all_ap_25
                )
            )
            for i, label_name in enumerate(self.valid_class_names):
                ap = ap_scores["classes"][label_name]["ap"]
                ap_50 = ap_scores["classes"][label_name]["ap50%"]
                ap_25 = ap_scores["classes"][label_name]["ap25%"]
                logger.info(
                    "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                        idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    def write_scannetpp_results(
        self,
        pred_scores,
        pred_masks,
        pred_classes,
        data_name,
    ):
        pred_scores[pred_scores < 0] = 0
        pred_scores[pred_scores >= 0] = 1

        save_dir = os.path.join(self.cfg.save_path, "result", "submit")
        mask_dir = os.path.join(save_dir, "predicted_masks")
        make_dirs(mask_dir)

        result_path = os.path.join(save_dir, f"{data_name}.txt")
        result_file = open(result_path, "w")
        for i, (score, mask, cls) in enumerate(
            zip(
                pred_scores.cpu().numpy(),
                pred_masks.cpu().numpy(),
                pred_classes.cpu().numpy(),
            )
        ):
            mask = mask.astype(np.uint8)
            length = mask.shape[0]
            mask = np.concatenate([[0], mask, [0]])
            runs = np.where(mask[1:] != mask[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            counts = " ".join(str(x) for x in runs)
            rle = dict(length=length, counts=counts)

            mask_path = os.path.join(mask_dir, f"{data_name}_{i:03d}.json")
            relative_path = os.path.join("predicted_masks", f"{data_name}_{i:03d}.json")
            with open(mask_path, "w") as mask_file:
                json.dump(rle, mask_file, indent=2)
            result_file.write(f"{relative_path} {cls} {score:.3f}\n")
        result_file.close()

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
        for i in range(self.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.cfg.data.names[i]] = []
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
            gt_instances[self.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.cfg.data.names[i]] = []
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
            segment_name = self.cfg.data.names[pred_inst["segment_id"]]
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

    @staticmethod
    def collate_fn(batch):
        # Restrict to bs 1
        return batch[0]
