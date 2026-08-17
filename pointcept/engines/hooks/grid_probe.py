"""
Grid-Search Linear Probing Hooks

Per-probe validation, per-probe best-checkpoint saving, and end-of-training
winner selection + automatic test pass for GridProbeSegmentorV2 /
GridProbeTrainer (see pointcept/models/grid_probe.py,
pointcept/engines/train.py).
"""

import json
import os

import numpy as np
import torch
import wandb

import pointcept.utils.comm as comm
from pointcept.utils.comm import is_main_process
from pointcept.utils.misc import (
    intersection_and_union_gpu,
    mean_acc_from_hist,
    mean_iou_from_hist,
)

from .builder import HOOKS
from .default import HookBase
from .evaluator import (
    begin_val_epoch_timing,
    finalize_val_epoch_timing,
    local_task_confusion_hist_totals,
    remap_pred_with_inverse,
    sync_confusion_hist_totals,
)


def _raw_model(trainer):
    return trainer.model.module if hasattr(trainer.model, "module") else trainer.model


def _find_hook(trainer, cls):
    for h in trainer.hooks:
        if isinstance(h, cls):
            return h
    return None


def _atomic_write_json(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp_path, path)


def _flatten_for_wandb(d, prefix):
    """Recursively flatten a (possibly nested) config dict into dotted-path
    wandb summary keys; lists (e.g. `criteria`) are JSON-stringified since
    wandb summary fields are meant to be scalar/string, not arbitrary
    Python objects."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(_flatten_for_wandb(v, key))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v, default=str)
        else:
            out[key] = v
    return out


@HOOKS.register_module()
class GridProbeEvaluator(HookBase):
    """Per-probe validation: one shared backbone forward per val batch,
    confusion-matrix mIoU computed independently per probe head, resume-safe
    running-best tracking per probe (via HookBase.state_dict/load_state_dict).

    comm_info["current_metric_value"] is set to THIS epoch's max mIoU across
    probes (not a running best) so CheckpointSaver's own max-over-epochs
    bookkeeping ends up computing max_p(max_e m_iou(p, e)) into
    trainer.best_metric_value — which equals max_e(max_p m_iou(p, e)), i.e.
    exactly the eventual winning probe's own best value (max commutes over
    the epoch/probe axes). MetricsJsonWriter's existing best_val_mIoU output
    is therefore already correct for grid runs with zero changes to that hook.
    """

    def __init__(self, write_cls_iou=False):
        self.write_cls_iou = write_cls_iou
        self._best_miou_by_probe = {}
        # This epoch's raw (non-running-best) values; read by
        # GridProbeCheckpointSaver to decide which probes to save. Not
        # persisted across resume — recomputed every eval() call.
        self._last_miou_by_probe = {}

    def state_dict(self):
        return {"best_miou_by_probe": dict(self._best_miou_by_probe)}

    def load_state_dict(self, state):
        self._best_miou_by_probe = dict(state.get("best_miou_by_probe", {}))

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
        raw_model = _raw_model(self.trainer)
        probe_names = raw_model.probe_names
        num_classes = raw_model.num_classes
        ignore_index = raw_model.ignore_index

        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            logits_by_task = output_dict["seg_logits_by_task"]
            segment = input_dict["segment"]
            has_inverse = "inverse" in input_dict.keys()
            if has_inverse:
                assert "origin_segment" in input_dict.keys()
                origin_segment = input_dict["origin_segment"]

            for name in probe_names:
                pred = logits_by_task[name].max(1)[1]
                target = segment
                if has_inverse:
                    pred = remap_pred_with_inverse(
                        pred,
                        input_dict["inverse"],
                        input_dict.get("offset"),
                        input_dict.get("origin_offset"),
                    )
                    target = origin_segment
                intersection, union, target_hist = intersection_and_union_gpu(
                    pred, target, num_classes, ignore_index
                )
                intersection, union, target_hist = (
                    intersection.cpu().numpy(),
                    union.cpu().numpy(),
                    target_hist.cpu().numpy(),
                )
                self.trainer.storage.put_scalar(
                    f"val_intersection/{name}", intersection
                )
                self.trainer.storage.put_scalar(f"val_union/{name}", union)
                self.trainer.storage.put_scalar(f"val_target/{name}", target_hist)

            if comm.is_main_process():
                self.trainer.logger.info(
                    "Test: [{iter}/{max_iter}]".format(
                        iter=i + 1, max_iter=len(self.trainer.val_loader)
                    )
                )

        current_epoch = self.trainer.epoch + 1
        m_iou_by_probe = {}
        m_acc_by_probe = {}
        all_acc_by_probe = {}
        for name in probe_names:
            intersection, union, target_hist = local_task_confusion_hist_totals(
                self.trainer.storage, name, num_classes
            )
            intersection, union, target_hist = sync_confusion_hist_totals(
                intersection, union, target_hist
            )
            m_iou_by_probe[name] = mean_iou_from_hist(intersection, union)
            m_acc_by_probe[name] = mean_acc_from_hist(intersection, target_hist, union)
            all_acc_by_probe[name] = float(
                np.sum(intersection) / (np.sum(target_hist) + 1e-10)
            )

        if comm.is_main_process():
            for name in probe_names:
                self.trainer.logger.info(
                    "[probe={}] Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                        name, m_iou_by_probe[name], m_acc_by_probe[name], all_acc_by_probe[name]
                    )
                )

        self._last_miou_by_probe = dict(m_iou_by_probe)
        for name, m_iou in m_iou_by_probe.items():
            prev = self._best_miou_by_probe.get(name, float("-inf"))
            self._best_miou_by_probe[name] = max(prev, m_iou)

        wandb_log = None
        if self.trainer.writer is not None:
            for name in probe_names:
                self.trainer.writer.add_scalar(
                    f"val/mIoU/{name}", m_iou_by_probe[name], current_epoch
                )
                self.trainer.writer.add_scalar(
                    f"val/mIoU_best/{name}",
                    self._best_miou_by_probe[name],
                    current_epoch,
                )
            if self.trainer.cfg.enable_wandb:
                wandb_log = {"Epoch": current_epoch}
                for name in probe_names:
                    wandb_log[f"val/mIoU/{name}"] = m_iou_by_probe[name]
                    wandb_log[f"val/mIoU_best/{name}"] = self._best_miou_by_probe[name]
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

        # See class docstring: this is THIS epoch's max, not a running best —
        # CheckpointSaver does the running-max itself via best_metric_value.
        self.trainer.comm_info["current_metric_value"] = max(m_iou_by_probe.values())
        self.trainer.comm_info["current_metric_name"] = "mIoU"

    def after_train(self):
        if not self._best_miou_by_probe:
            return
        winner = max(self._best_miou_by_probe, key=self._best_miou_by_probe.get)
        self.trainer.logger.info(
            "Best val mIoU per probe: %s; overall best: %r (%.4f)",
            {k: round(v, 4) for k, v in self._best_miou_by_probe.items()},
            winner,
            self._best_miou_by_probe[winner],
        )


@HOOKS.register_module()
class GridProbeCheckpointSaver(HookBase):
    """Save each probe head's own best-val weights separately (a few KB —
    just that head's state_dict, not the whole shared model) whenever it
    improves this epoch. Must be registered after GridProbeEvaluator.
    """

    def after_epoch(self):
        if not is_main_process():
            return
        evaluator = _find_hook(self.trainer, GridProbeEvaluator)
        if evaluator is None or not evaluator._last_miou_by_probe:
            return
        raw_model = _raw_model(self.trainer)
        model_dir = os.path.join(self.trainer.cfg.save_path, "model")
        os.makedirs(model_dir, exist_ok=True)
        epoch = self.trainer.epoch + 1
        for name, m_iou in evaluator._last_miou_by_probe.items():
            best = evaluator._best_miou_by_probe.get(name)
            if best is None or m_iou < best:
                continue  # this epoch didn't (re)set the probe's own best
            ckpt_path = os.path.join(model_dir, f"probe_best_{name}.pth")
            torch.save(raw_model.heads[name].state_dict(), ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            meta_path = os.path.join(model_dir, f"probe_best_{name}.json")
            _atomic_write_json(
                meta_path,
                {
                    "probe_name": name,
                    "probe_config": dict(raw_model.probe_configs[name]),
                    "best_val_mIoU": float(best),
                    "epoch": epoch,
                },
            )


@HOOKS.register_module()
class GridProbeWinnerSelector(HookBase):
    """After training: pick the probe with the best val mIoU over the whole
    run, reload its best-epoch weights (the live weights at end of training
    are that probe's LAST epoch, not its BEST, since other probes may have
    kept training after it peaked), optionally run a precise test pass
    restricted to just that probe, and write save_path/grid_search_results.json.

    Must be registered after GridProbeEvaluator/GridProbeCheckpointSaver, and
    should be last in the hooks list (frees the per-probe optimizers/
    schedulers before testing).

    skip_test: if True, skip the tester (val-only sweeps). The winner JSON
    is still written; test_mIoU / test_mAcc / test_allAcc are null.
    """

    def __init__(self, skip_test=False):
        self.skip_test = bool(skip_test)

    def after_train(self):
        evaluator = _find_hook(self.trainer, GridProbeEvaluator)
        if evaluator is None or not evaluator._best_miou_by_probe:
            self.trainer.logger.warning(
                "GridProbeWinnerSelector: no per-probe val results found, skipping."
            )
            return

        winner_name = max(
            evaluator._best_miou_by_probe, key=evaluator._best_miou_by_probe.get
        )
        self.trainer.logger.info(
            "Grid search winner: %r (best val mIoU=%.4f)",
            winner_name,
            evaluator._best_miou_by_probe[winner_name],
        )

        raw_model = _raw_model(self.trainer)
        test_metrics = {}
        if self.skip_test:
            self.trainer.logger.info(
                "GridProbeWinnerSelector: skip_test=True, writing val-only results."
            )
        else:
            model_dir = os.path.join(self.trainer.cfg.save_path, "model")
            best_head_path = os.path.join(model_dir, f"probe_best_{winner_name}.pth")
            if os.path.isfile(best_head_path):
                state = torch.load(best_head_path, map_location="cpu", weights_only=False)
                raw_model.heads[winner_name].load_state_dict(state)
            else:
                self.trainer.logger.warning(
                    "No probe_best_%s.pth found; testing with the probe's last-epoch "
                    "weights instead of its best-epoch weights.",
                    winner_name,
                )
            raw_model.active_probe = winner_name

            if getattr(self.trainer, "optimizer", None) is not None:
                del self.trainer.optimizer
                self.trainer.optimizer = None
            if getattr(self.trainer, "scheduler", None) is not None:
                del self.trainer.scheduler
                self.trainer.scheduler = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            from pointcept.engines.test import TESTERS

            cfg = self.trainer.cfg
            test_cfg = dict(cfg=cfg, model=self.trainer.model, **cfg.test)
            tester = TESTERS.build(test_cfg)
            tester.test()
            test_metrics = getattr(tester, "test_metrics", None) or {}

        if not is_main_process():
            return

        cfg = self.trainer.cfg
        leaderboard = {
            name: {
                "probe_config": dict(raw_model.probe_configs[name]),
                "best_val_mIoU": float(best),
            }
            for name, best in evaluator._best_miou_by_probe.items()
        }
        payload = {
            "num_probes": len(raw_model.probe_names),
            "winner": {
                "probe_name": winner_name,
                "probe_config": dict(raw_model.probe_configs[winner_name]),
                "best_val_mIoU": float(evaluator._best_miou_by_probe[winner_name]),
                "test_mIoU": test_metrics.get("test/mIoU"),
                "test_mAcc": test_metrics.get("test/mAcc"),
                "test_allAcc": test_metrics.get("test/allAcc"),
            },
            "leaderboard": leaderboard,
        }
        out_path = os.path.join(cfg.save_path, "grid_search_results.json")
        _atomic_write_json(out_path, payload)
        self.trainer.logger.info("Wrote grid search results to: %s", out_path)

        if getattr(cfg, "enable_wandb", False) and wandb.run is not None:
            summary = {
                "winner/probe_name": winner_name,
                "winner/best_val_mIoU": float(evaluator._best_miou_by_probe[winner_name]),
            }
            summary.update(
                _flatten_for_wandb(raw_model.probe_configs[winner_name], "winner/config")
            )
            summary.update({f"winner/{k}": v for k, v in test_metrics.items()})
            for key, value in summary.items():
                wandb.run.summary[key] = value
            wandb.log(summary)
