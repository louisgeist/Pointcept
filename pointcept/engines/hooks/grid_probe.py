"""
Grid-Search Linear Probing Hooks

Hooks for GridProbeTrainer with GridProbeSegmentorV2
(pointcept/models/grid_probe.py): per-probe validation (GridProbeEvaluator),
per-probe best-checkpoint saving (GridProbeCheckpointSaver) and, after training,
either selection and test of the winning probe (GridProbeWinnerSelector) or a test
of every probe with mean/std reporting for seed ensembles (GridProbeSeedEnsembleTester).
"""

import csv
import json
import os
import traceback

import numpy as np
import torch
import wandb

import pointcept.utils.comm as comm
from pointcept.utils.comm import is_main_process
from pointcept.utils.misc import (
    f1_scores_from_hist,
    intersection_and_union_gpu,
    mean_acc_from_hist,
    mean_iou_from_hist,
)

from .builder import HOOKS
from .default import HookBase
from .misc import InformationWriter
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


# Validation metrics that can drive per-probe best-checkpoint and winner selection
# (GridProbeEvaluator.select_metric). "macro_f1" uses the same estimator as
# test/f1_macro.
_SELECT_METRICS = ("mIoU", "macro_f1")

_HISTORY_CSV_FIELDNAMES = (
    "epoch",
    "probe_name",
    "mIoU",
    "mIoU_best",
    "f1_macro",
    "f1_macro_best",
)


def _atomic_write_history_csv(path, rows):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        # restval="" lets rows without the f1_macro keys be written; unknown keys
        # still raise (extrasaction default).
        writer = csv.DictWriter(
            f, fieldnames=_HISTORY_CSV_FIELDNAMES, restval=""
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def _metric_stats(metrics_by_probe, metric_key):
    """Mean/std/min/max of ``metric_key`` across probes (np.std ddof=0)."""
    values = [
        m[metric_key]
        for m in metrics_by_probe.values()
        if metric_key in m and m[metric_key] is not None
    ]
    if not values:
        return dict(mean=None, std=None, max=None, min=None)
    arr = np.array(values, dtype=float)
    return dict(
        mean=float(arr.mean()),
        std=float(arr.std()),
        max=float(arr.max()),
        min=float(arr.min()),
    )


def _discover_class_metric_keys(
    metrics_by_probe, prefix, names=None, exclude=()
):
    """Per-class keys like ``test/iou_Ground``.

    Prefer ``names`` order (``cfg.data.names``); fall back to first-seen
    insertion order across probes so this stays dataset-agnostic.
    """
    metrics = list(metrics_by_probe.values())
    exclude = frozenset(exclude)
    keys = []
    for cls_name in names or []:
        key = f"{prefix}{cls_name}"
        if key not in exclude and any(key in m for m in metrics):
            keys.append(key)
    if keys:
        return keys
    seen = set()
    for m in metrics:
        for key in m:
            if key.startswith(prefix) and key not in exclude and key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def _flatten_class_stats(keys, stats_by_key, prefix, *, with_spread=False):
    """Flatten per-class stats into JSON keys ``test_iou_Ground_mean``, …

    ``prefix`` is the per-probe key prefix (``test/iou_`` / ``test/f1_``).
    """
    json_stem = prefix.replace("/", "_")
    out = {}
    for key in keys:
        cls_name = key[len(prefix) :]
        stats = stats_by_key[key]
        out[f"{json_stem}{cls_name}_mean"] = stats["mean"]
        if with_spread:
            out[f"{json_stem}{cls_name}_std"] = stats["std"]
            out[f"{json_stem}{cls_name}_min"] = stats["min"]
            out[f"{json_stem}{cls_name}_max"] = stats["max"]
    return out


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
    """Per-probe validation for GridProbeTrainer.

    One backbone forward per val batch; mIoU, mAcc, allAcc and macro-F1 are computed
    from each probe's own confusion histogram. Running bests are checkpointed
    (state_dict / load_state_dict).

    ``select_metric`` ("mIoU" or "macro_f1") is the metric that drives per-probe
    best-checkpoint saving (GridProbeCheckpointSaver) and winner selection
    (GridProbeWinnerSelector); both metrics are always computed and logged. It must
    not change when resuming a run.

    ``comm_info["current_metric_value"]`` is this epoch's max of the selected metric
    over probes (not a running best). The running max kept by CheckpointSaver in
    ``trainer.best_metric_value`` is then the winning probe's best value, which
    MetricsJsonWriter reports.

    Also writes ``save_path/grid_probe_miou_history.csv`` (one row per epoch and
    probe: mIoU, f1_macro and their running bests), rewritten after every evaluation.
    """

    def __init__(self, write_cls_iou=False, select_metric="mIoU"):
        assert select_metric in _SELECT_METRICS, (
            f"GridProbeEvaluator.select_metric must be one of {_SELECT_METRICS}, "
            f"got {select_metric!r}"
        )
        self.select_metric = select_metric
        self.write_cls_iou = write_cls_iou
        # Checkpointed state: running best of both metrics per probe (whatever
        # select_metric is), per-class IoU/accuracy/F1 at each probe's best epoch
        # for the selected metric (the epoch whose weights GridProbeCheckpointSaver
        # saved; reported for the winner by GridProbeWinnerSelector), and the
        # (epoch, probe) rows of grid_probe_miou_history.csv.
        self._best_miou_by_probe = {}
        self._best_f1_by_probe = {}
        self._best_cls_iou_by_probe = {}
        self._best_cls_acc_by_probe = {}
        self._best_cls_f1_by_probe = {}
        # This epoch's values, read by GridProbeCheckpointSaver (not checkpointed).
        self._last_miou_by_probe = {}
        self._last_f1_by_probe = {}
        self._history = []

    def _selected_best_by_probe(self):
        """Running-best dict for the metric that drives selection."""
        return (
            self._best_f1_by_probe
            if self.select_metric == "macro_f1"
            else self._best_miou_by_probe
        )

    def _selected_last_by_probe(self):
        """This-epoch value dict for the metric that drives selection."""
        return (
            self._last_f1_by_probe
            if self.select_metric == "macro_f1"
            else self._last_miou_by_probe
        )

    def best_miou(self, name):
        """Probe's running-best val mIoU, or nan if not tracked yet."""
        v = self._best_miou_by_probe.get(name)
        return float(v) if v is not None else float("nan")

    def best_macro_f1(self, name):
        """Probe's running-best val macro-F1, or nan if not tracked yet."""
        v = self._best_f1_by_probe.get(name)
        return float(v) if v is not None else float("nan")

    def _update_bests(
        self, m_iou_by_probe, macro_f1_by_probe, f1_cls_by_probe, cls_hist_by_probe
    ):
        """Fold this epoch's per-probe metrics into the running bests.

        Both running maxima are always updated. The per-class snapshots are
        refreshed when the selected metric reaches its running best (``>=``, so the
        latest epoch wins ties): the same rule GridProbeCheckpointSaver uses to save
        the probe weights.
        """
        selected_now = (
            macro_f1_by_probe
            if self.select_metric == "macro_f1"
            else m_iou_by_probe
        )
        selected_best = self._selected_best_by_probe()
        for name in m_iou_by_probe:
            if selected_now[name] >= selected_best.get(name, float("-inf")):
                intersection, union, target_hist = cls_hist_by_probe[name]
                self._best_cls_iou_by_probe[name] = (
                    intersection / (union + 1e-10)
                ).tolist()
                self._best_cls_acc_by_probe[name] = (
                    intersection / (target_hist + 1e-10)
                ).tolist()
                self._best_cls_f1_by_probe[name] = f1_cls_by_probe[name].tolist()
            self._best_miou_by_probe[name] = max(
                self._best_miou_by_probe.get(name, float("-inf")),
                m_iou_by_probe[name],
            )
            self._best_f1_by_probe[name] = max(
                self._best_f1_by_probe.get(name, float("-inf")),
                macro_f1_by_probe[name],
            )

    def state_dict(self):
        return {
            "select_metric": self.select_metric,
            "best_miou_by_probe": dict(self._best_miou_by_probe),
            "best_f1_by_probe": dict(self._best_f1_by_probe),
            "best_cls_iou_by_probe": dict(self._best_cls_iou_by_probe),
            "best_cls_acc_by_probe": dict(self._best_cls_acc_by_probe),
            "best_cls_f1_by_probe": dict(self._best_cls_f1_by_probe),
            "history": list(self._history),
        }

    def load_state_dict(self, state):
        # Called by CheckpointLoader (the first hook) on resume, so raising here
        # aborts before any training step.
        persisted = state.get("select_metric")
        if persisted is not None and persisted != self.select_metric:
            raise RuntimeError(
                f"GridProbeEvaluator.select_metric changed on resume: checkpoint "
                f"was trained with select_metric={persisted!r}, config now says "
                f"{self.select_metric!r}. Per-probe best-checkpoint state and "
                f"trainer.best_metric_value are metric-specific and cannot be "
                f"reinterpreted. Start a fresh run (new save_path) or restore "
                f"select_metric={persisted!r}."
            )
        if persisted is None and self.select_metric != "mIoU":
            raise RuntimeError(
                f"Resuming a checkpoint without macro-F1 tracking state with "
                f"select_metric={self.select_metric!r}: validation macro-F1 "
                f"history for epochs before this checkpoint is unavailable, so "
                f"probe_best_*.pth selection would be wrong. Start a fresh run, "
                f"or set select_metric='mIoU' to resume."
            )
        self._best_miou_by_probe = dict(state.get("best_miou_by_probe", {}))
        self._best_f1_by_probe = dict(state.get("best_f1_by_probe", {}))
        self._best_cls_iou_by_probe = dict(state.get("best_cls_iou_by_probe", {}))
        self._best_cls_acc_by_probe = dict(state.get("best_cls_acc_by_probe", {}))
        self._best_cls_f1_by_probe = dict(state.get("best_cls_f1_by_probe", {}))
        self._history = list(state.get("history", []))

    def before_train(self):
        # A resumed checkpoint without GridProbeEvaluator state never calls
        # load_state_dict (so the select_metric check is skipped), and its restored
        # trainer.best_metric_value may be on another metric's scale. With
        # select_metric="macro_f1", reset it so best-checkpoint selection restarts
        # on the configured metric.
        best = getattr(self.trainer, "best_metric_value", None)
        if (
            getattr(self.trainer.cfg, "resume", False)
            and self.select_metric != "mIoU"
            and not self._best_f1_by_probe
            and best is not None
            and best != float("-inf")
        ):
            self.trainer.logger.warning(
                "GridProbeEvaluator: resumed a checkpoint with no macro-F1 "
                "tracking state; resetting trainer.best_metric_value (%.4f, "
                "likely mIoU-scale) to -inf so selection restarts on %s.",
                best,
                self.select_metric,
            )
            self.trainer.best_metric_value = float("-inf")

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
            target_key = getattr(raw_model, "target_key", "segment")
            if target_key not in input_dict:
                raise KeyError(
                    f"GridProbeEvaluator: target_key={target_key!r} missing from "
                    f"val batch (keys={sorted(input_dict.keys())})."
                )
            segment = input_dict[target_key]
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
        macro_f1_by_probe = {}
        f1_cls_by_probe = {}
        cls_hist_by_probe = {}
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
            # Same estimator as test/f1_macro (unmasked mean over all classes),
            # from the already-synced confusion histogram.
            f1_cls, macro_f1 = f1_scores_from_hist(intersection, union, target_hist)
            macro_f1_by_probe[name] = macro_f1
            f1_cls_by_probe[name] = f1_cls
            cls_hist_by_probe[name] = (intersection, union, target_hist)

        if comm.is_main_process():
            for name in probe_names:
                self.trainer.logger.info(
                    "[probe={}] Val result: mIoU/mAcc/allAcc/macroF1 "
                    "{:.4f}/{:.4f}/{:.4f}/{:.4f}.".format(
                        name,
                        m_iou_by_probe[name],
                        m_acc_by_probe[name],
                        all_acc_by_probe[name],
                        macro_f1_by_probe[name],
                    )
                )

        self._last_miou_by_probe = dict(m_iou_by_probe)
        self._last_f1_by_probe = dict(macro_f1_by_probe)
        self._update_bests(
            m_iou_by_probe, macro_f1_by_probe, f1_cls_by_probe, cls_hist_by_probe
        )

        if comm.is_main_process():
            for name in probe_names:
                self._history.append(
                    {
                        "epoch": current_epoch,
                        "probe_name": name,
                        "mIoU": float(m_iou_by_probe[name]),
                        "mIoU_best": float(self._best_miou_by_probe[name]),
                        "f1_macro": float(macro_f1_by_probe[name]),
                        "f1_macro_best": float(self._best_f1_by_probe[name]),
                    }
                )
            history_path = os.path.join(
                self.trainer.cfg.save_path, "grid_probe_miou_history.csv"
            )
            _atomic_write_history_csv(history_path, self._history)

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
                self.trainer.writer.add_scalar(
                    f"val/f1_macro/{name}", macro_f1_by_probe[name], current_epoch
                )
                self.trainer.writer.add_scalar(
                    f"val/f1_macro_best/{name}",
                    self._best_f1_by_probe[name],
                    current_epoch,
                )
            if self.trainer.cfg.enable_wandb:
                # Only the best-over-probes values go to W&B; per-probe curves
                # stay in the console and TensorBoard. GridProbeWinnerSelector
                # replays the winner's curve from _history after training.
                wandb_log = {
                    "Epoch": current_epoch,
                    "val/mIoU/epoch_best": max(m_iou_by_probe.values()),
                    "val/mIoU_best/running_best": max(
                        self._best_miou_by_probe.values()
                    ),
                    "val/f1_macro/epoch_best": max(macro_f1_by_probe.values()),
                    "val/f1_macro_best/running_best": max(
                        self._best_f1_by_probe.values()
                    ),
                }
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

        # This epoch's max of the selected metric over probes, not a running best
        # (CheckpointSaver keeps the running max; see class docstring).
        if self.select_metric == "macro_f1":
            self.trainer.comm_info["current_metric_value"] = max(
                macro_f1_by_probe.values()
            )
            self.trainer.comm_info["current_metric_name"] = "macro_f1"
        else:
            self.trainer.comm_info["current_metric_value"] = max(
                m_iou_by_probe.values()
            )
            self.trainer.comm_info["current_metric_name"] = "mIoU"

    def after_train(self):
        sel_best = self._selected_best_by_probe()
        if not sel_best:
            return
        winner = max(sel_best, key=sel_best.get)
        self.trainer.logger.info(
            "Best val %s per probe: %s; overall best: %r "
            "(mIoU=%.4f, macro_f1=%.4f)",
            self.select_metric,
            {k: round(v, 4) for k, v in sel_best.items()},
            winner,
            self.best_miou(winner),
            self.best_macro_f1(winner),
        )


@HOOKS.register_module()
class GridProbeCheckpointSaver(HookBase):
    """Save a probe head's weights to ``model/probe_best_<name>.pth`` (head
    state_dict only, a few KB) plus a JSON sidecar whenever the probe reaches its
    best value of GridProbeEvaluator.select_metric. Must be registered after
    GridProbeEvaluator.
    """

    def after_epoch(self):
        if not is_main_process():
            return
        evaluator = _find_hook(self.trainer, GridProbeEvaluator)
        if evaluator is None:
            return
        last_by_probe = evaluator._selected_last_by_probe()
        best_by_probe = evaluator._selected_best_by_probe()
        if not last_by_probe:
            return
        raw_model = _raw_model(self.trainer)
        model_dir = os.path.join(self.trainer.cfg.save_path, "model")
        os.makedirs(model_dir, exist_ok=True)
        epoch = self.trainer.epoch + 1
        for name, value in last_by_probe.items():
            best = best_by_probe.get(name)
            if best is None or value < best:
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
                    "select_metric": evaluator.select_metric,
                    "best_val_mIoU": float(evaluator._best_miou_by_probe[name]),
                    "best_val_macro_f1": float(evaluator._best_f1_by_probe[name]),
                    "epoch": epoch,
                },
            )


@HOOKS.register_module()
class GridProbeWinnerSelector(HookBase):
    """After training, pick the probe with the best val ``select_metric`` over the
    run, reload its best-epoch head (the live weights are from the last epoch), test
    that probe alone and write ``save_path/grid_search_results.json`` (winner and
    leaderboard). With W&B, also replays the winner's val/train curves, which are
    not logged per probe during training.

    Must be registered after GridProbeEvaluator / GridProbeCheckpointSaver, last in
    the hook list (it frees the per-probe optimizers and schedulers before testing).

    skip_test: skip the test pass (val-only sweeps); the test_* fields are then null.
    """

    def __init__(self, skip_test=False):
        self.skip_test = bool(skip_test)

    def after_train(self):
        evaluator = _find_hook(self.trainer, GridProbeEvaluator)
        sel_best = evaluator._selected_best_by_probe() if evaluator is not None else {}
        if evaluator is None or not sel_best:
            self.trainer.logger.warning(
                "GridProbeWinnerSelector: no per-probe val results found, skipping."
            )
            return

        winner_name = max(sel_best, key=sel_best.get)
        self.trainer.logger.info(
            "Grid search winner: %r (select_metric=%s, best val mIoU=%.4f, "
            "best val macro_f1=%.4f)",
            winner_name,
            evaluator.select_metric,
            evaluator.best_miou(winner_name),
            evaluator.best_macro_f1(winner_name),
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

        def _num(d, name):
            # None (JSON null) when a probe has no value for this metric.
            v = d.get(name)
            return float(v) if v is not None else None

        leaderboard = {
            name: {
                "probe_config": dict(raw_model.probe_configs[name]),
                "best_val_mIoU": _num(evaluator._best_miou_by_probe, name),
                "best_val_macro_f1": _num(evaluator._best_f1_by_probe, name),
            }
            for name in evaluator._best_miou_by_probe
        }
        payload = {
            "num_probes": len(raw_model.probe_names),
            "select_metric": evaluator.select_metric,
            "winner": {
                "probe_name": winner_name,
                "probe_config": dict(raw_model.probe_configs[winner_name]),
                "best_val_mIoU": _num(evaluator._best_miou_by_probe, winner_name),
                "best_val_macro_f1": _num(evaluator._best_f1_by_probe, winner_name),
                "test_mIoU": test_metrics.get("test/mIoU"),
                "test_mAcc": test_metrics.get("test/mAcc"),
                "test_allAcc": test_metrics.get("test/allAcc"),
                "test_f1_macro": test_metrics.get("test/f1_macro"),
            },
            "leaderboard": leaderboard,
        }
        out_path = os.path.join(cfg.save_path, "grid_search_results.json")
        _atomic_write_json(out_path, payload)
        self.trainer.logger.info("Wrote grid search results to: %s", out_path)

        if getattr(cfg, "enable_wandb", False) and wandb.run is not None:
            # Replay the winner's per-epoch val curve (x-axis "Epoch", like the
            # other val/* metrics); GridProbeEvaluator does not log it live.
            winner_history = sorted(
                (row for row in evaluator._history if row["probe_name"] == winner_name),
                key=lambda row: row["epoch"],
            )
            for row in winner_history:
                log_row = {
                    "Epoch": row["epoch"],
                    "val/mIoU/winner": row["mIoU"],
                    "val/mIoU_best/winner": row["mIoU_best"],
                }
                # Rows restored from a checkpoint may lack the f1_macro keys.
                if row.get("f1_macro") is not None:
                    log_row["val/f1_macro/winner"] = row["f1_macro"]
                if row.get("f1_macro_best") is not None:
                    log_row["val/f1_macro_best/winner"] = row["f1_macro_best"]
                wandb.log(log_row)

            names = list(getattr(cfg.data, "names", []) or [])

            # Same replay for the winner's train loss/mIoU (and per-class IoU
            # with InformationWriter(write_cls_iou=True)), which
            # InformationWriter does not log per probe to W&B.
            info_writer = _find_hook(self.trainer, InformationWriter)
            if info_writer is not None:
                train_history = sorted(
                    (
                        row
                        for row in info_writer._grid_probe_train_history
                        if row["task_name"] == winner_name
                    ),
                    key=lambda row: row["epoch"],
                )
                for row in train_history:
                    log_row = {"Epoch": row["epoch"]}
                    if row.get("loss") is not None:
                        log_row["train/loss/winner"] = row["loss"]
                    if row.get("mIoU") is not None:
                        log_row["train/mIoU/winner"] = row["mIoU"]
                    cls_iou = row.get("cls_iou")
                    if cls_iou:
                        for i, cls_name in enumerate(names):
                            if i == raw_model.ignore_index or i >= len(cls_iou):
                                continue
                            log_row[f"train/iou_{cls_name}/winner"] = cls_iou[i]
                    if len(log_row) > 1:
                        wandb.log(log_row)

            summary = {
                "winner/probe_name": winner_name,
                "winner/select_metric": evaluator.select_metric,
                "winner/best_val_mIoU": evaluator.best_miou(winner_name),
                "winner/best_val_macro_f1": evaluator.best_macro_f1(winner_name),
            }
            summary.update(
                _flatten_for_wandb(raw_model.probe_configs[winner_name], "winner/config")
            )
            # Per-class val IoU/accuracy/F1 at the winner's best select_metric
            # epoch. winner/best_val_{mIoU,macro_f1} above are running maxima over
            # all epochs: with select_metric="mIoU", the per-class F1 comes from the
            # best-mIoU epoch and may not match best_val_macro_f1.
            cls_iou = evaluator._best_cls_iou_by_probe.get(winner_name)
            cls_acc = evaluator._best_cls_acc_by_probe.get(winner_name)
            cls_f1 = evaluator._best_cls_f1_by_probe.get(winner_name)
            if cls_iou is not None:
                for i, cls_name in enumerate(names):
                    if i == raw_model.ignore_index or i >= len(cls_iou):
                        continue
                    summary[f"winner/val/iou_{cls_name}"] = float(cls_iou[i])
                    if cls_acc is not None:
                        summary[f"winner/val/acc_{cls_name}"] = float(cls_acc[i])
                    if cls_f1 is not None and i < len(cls_f1):
                        summary[f"winner/val/f1_{cls_name}"] = float(cls_f1[i])
            summary.update({f"winner/{k}": v for k, v in test_metrics.items()})
            for key, value in summary.items():
                wandb.run.summary[key] = value
            wandb.log(summary)


@HOOKS.register_module()
class GridProbeSeedEnsembleTester(HookBase):
    """Test every probe of a seed ensemble (same hyperparameters, different random
    inits) and report mean/std over probes.

    Reloads each probe's best-val head (saved by GridProbeCheckpointSaver), runs one
    shared-backbone test pass over all probes (GridProbeSemSegTester) and writes
    ``save_path/seed_ensemble_results.json``: mean/std/min/max of test mIoU, mAcc,
    allAcc and f1_macro, per-class IoU (mean/std/min/max) and F1 (mean, with
    ``log_test_f1=True``), and per-probe results.

    Use instead of GridProbeWinnerSelector, last in the hook list, with
    ``cfg.test = dict(type="GridProbeSemSegTester", ...)``. The JSON is written even
    if the test pass fails (test fields are then null); statistics only use the
    probes that have test metrics.
    """

    def after_train(self):
        raw_model = _raw_model(self.trainer)
        model_dir = os.path.join(self.trainer.cfg.save_path, "model")
        for name in raw_model.probe_names:
            ckpt_path = os.path.join(model_dir, f"probe_best_{name}.pth")
            if os.path.isfile(ckpt_path):
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                raw_model.heads[name].load_state_dict(state)
            else:
                self.trainer.logger.warning(
                    "No probe_best_%s.pth found; testing with the probe's "
                    "last-epoch weights instead of its best-epoch weights.",
                    name,
                )
        raw_model.active_probe = None  # all-probes mode, required by GridProbeSemSegTester

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
        metrics_by_probe = {}
        test_error = None
        try:
            test_cfg = dict(cfg=cfg, model=self.trainer.model, **cfg.test)
            tester = TESTERS.build(test_cfg)
            tester.test()
            metrics_by_probe = getattr(tester, "test_metrics_by_probe", None) or {}
        except Exception:
            # Keep the val results (best val metrics, checkpoints) if the test
            # pass fails (e.g. OOM); only the test/* fields end up null.
            test_error = traceback.format_exc()
            self.trainer.logger.error(
                "GridProbeSeedEnsembleTester: test pass failed, writing "
                "val-only results for all probes instead of crashing.\n%s",
                test_error,
            )

        if not is_main_process():
            return

        evaluator = _find_hook(self.trainer, GridProbeEvaluator)
        best_val_by_probe = evaluator._best_miou_by_probe if evaluator is not None else {}
        best_f1_by_probe = evaluator._best_f1_by_probe if evaluator is not None else {}

        missing = [name for name in raw_model.probe_names if name not in metrics_by_probe]
        if missing:
            self.trainer.logger.warning(
                "GridProbeSeedEnsembleTester: %d/%d probes have no test metrics "
                "(missing: %s) -- mean/std/max/min computed over the rest only.",
                len(missing), len(raw_model.probe_names), missing,
            )

        miou_stats = _metric_stats(metrics_by_probe, "test/mIoU")
        macc_stats = _metric_stats(metrics_by_probe, "test/mAcc")
        allacc_stats = _metric_stats(metrics_by_probe, "test/allAcc")
        f1_macro_stats = _metric_stats(metrics_by_probe, "test/f1_macro")

        # Per-class F1 (log_test_f1=True) and IoU (always written by
        # GridProbeSemSegTester as test/iou_<cls>), in cfg.data.names order.
        names = getattr(cfg.data, "names", None) or []
        f1_cls_keys = _discover_class_metric_keys(
            metrics_by_probe,
            "test/f1_",
            names=names,
            exclude=("test/f1_macro",),
        )
        f1_cls_stats = {
            key: _metric_stats(metrics_by_probe, key) for key in f1_cls_keys
        }
        f1_cls_flat = _flatten_class_stats(
            f1_cls_keys, f1_cls_stats, "test/f1_", with_spread=False
        )
        iou_cls_keys = _discover_class_metric_keys(
            metrics_by_probe, "test/iou_", names=names
        )
        iou_cls_stats = {
            key: _metric_stats(metrics_by_probe, key) for key in iou_cls_keys
        }
        iou_cls_flat = _flatten_class_stats(
            iou_cls_keys, iou_cls_stats, "test/iou_", with_spread=True
        )

        summary = {
            "num_probes": len(raw_model.probe_names),
            "num_probes_with_test_metrics": len(metrics_by_probe),
            "test_error": test_error,
            "select_metric": (
                evaluator.select_metric if evaluator is not None else None
            ),
            "test_mIoU_mean": miou_stats["mean"],
            "test_mIoU_std": miou_stats["std"],
            "test_mIoU_max": miou_stats["max"],
            "test_mIoU_min": miou_stats["min"],
            "test_mAcc_mean": macc_stats["mean"],
            "test_mAcc_std": macc_stats["std"],
            "test_mAcc_max": macc_stats["max"],
            "test_mAcc_min": macc_stats["min"],
            "test_allAcc_mean": allacc_stats["mean"],
            "test_allAcc_std": allacc_stats["std"],
            "test_allAcc_max": allacc_stats["max"],
            "test_allAcc_min": allacc_stats["min"],
            "test_f1_macro_mean": f1_macro_stats["mean"],
            "test_f1_macro_std": f1_macro_stats["std"],
            "test_f1_macro_max": f1_macro_stats["max"],
            "test_f1_macro_min": f1_macro_stats["min"],
            **f1_cls_flat,
            **iou_cls_flat,
            "per_probe": {
                name: {
                    "probe_config": dict(raw_model.probe_configs[name]),
                    "best_val_mIoU": (
                        float(best_val_by_probe[name])
                        if name in best_val_by_probe
                        else None
                    ),
                    "best_val_macro_f1": (
                        float(best_f1_by_probe[name])
                        if name in best_f1_by_probe
                        else None
                    ),
                    **metrics_by_probe.get(name, {}),
                }
                for name in raw_model.probe_names
            },
        }
        out_path = os.path.join(cfg.save_path, "seed_ensemble_results.json")
        _atomic_write_json(out_path, summary)
        self.trainer.logger.info(
            "Seed-ensemble test result (%d/%d probes with test metrics): "
            "mIoU=%.4f+/-%.4f [%.4f, %.4f], mAcc=%.4f+/-%.4f, "
            "allAcc=%.4f+/-%.4f, f1_macro=%.4f+/-%.4f [%.4f, %.4f]. Wrote %s",
            summary["num_probes_with_test_metrics"], summary["num_probes"],
            summary["test_mIoU_mean"] or float("nan"),
            summary["test_mIoU_std"] or float("nan"),
            summary["test_mIoU_min"] or float("nan"),
            summary["test_mIoU_max"] or float("nan"),
            summary["test_mAcc_mean"] or float("nan"),
            summary["test_mAcc_std"] or float("nan"),
            summary["test_allAcc_mean"] or float("nan"),
            summary["test_allAcc_std"] or float("nan"),
            summary["test_f1_macro_mean"] or float("nan"),
            summary["test_f1_macro_std"] or float("nan"),
            summary["test_f1_macro_min"] or float("nan"),
            summary["test_f1_macro_max"] or float("nan"),
            out_path,
        )
        for key in iou_cls_keys:
            st = iou_cls_stats[key]
            if st["mean"] is None:
                continue
            self.trainer.logger.info(
                "  IoU %-20s %.4f+/-%.4f [%.4f, %.4f]",
                key[len("test/iou_") :],
                st["mean"],
                st["std"],
                st["min"],
                st["max"],
            )

        if getattr(cfg, "enable_wandb", False) and wandb.run is not None:
            wandb_summary = {
                "seed_ensemble/num_probes": summary["num_probes"],
                "seed_ensemble/num_probes_with_test_metrics": summary["num_probes_with_test_metrics"],
                "seed_ensemble/test_mIoU_mean": summary["test_mIoU_mean"],
                "seed_ensemble/test_mIoU_std": summary["test_mIoU_std"],
                "seed_ensemble/test_mIoU_max": summary["test_mIoU_max"],
                "seed_ensemble/test_mIoU_min": summary["test_mIoU_min"],
                "seed_ensemble/test_mAcc_mean": summary["test_mAcc_mean"],
                "seed_ensemble/test_mAcc_std": summary["test_mAcc_std"],
                "seed_ensemble/test_mAcc_max": summary["test_mAcc_max"],
                "seed_ensemble/test_mAcc_min": summary["test_mAcc_min"],
                "seed_ensemble/test_allAcc_mean": summary["test_allAcc_mean"],
                "seed_ensemble/test_allAcc_std": summary["test_allAcc_std"],
                "seed_ensemble/test_allAcc_max": summary["test_allAcc_max"],
                "seed_ensemble/test_allAcc_min": summary["test_allAcc_min"],
                "seed_ensemble/test_f1_macro_mean": summary["test_f1_macro_mean"],
                "seed_ensemble/test_f1_macro_std": summary["test_f1_macro_std"],
                "seed_ensemble/test_f1_macro_max": summary["test_f1_macro_max"],
                "seed_ensemble/test_f1_macro_min": summary["test_f1_macro_min"],
            }
            for key, st in f1_cls_stats.items():
                cls_name = key[len("test/f1_"):]
                wandb_summary[f"seed_ensemble/test_f1_{cls_name}_mean"] = st["mean"]
            for key, st in iou_cls_stats.items():
                cls_name = key[len("test/iou_"):]
                wandb_summary[f"seed_ensemble/test_iou_{cls_name}_mean"] = st["mean"]
                wandb_summary[f"seed_ensemble/test_iou_{cls_name}_std"] = st["std"]
            for key, value in wandb_summary.items():
                wandb.run.summary[key] = value
            wandb.log(wandb_summary)

            # Per-probe results as a W&B table, to inspect the distribution
            # beyond the summary statistics.
            table = wandb.Table(
                columns=[
                    "probe_name", "test_mIoU", "test_mAcc", "test_allAcc",
                    "test_f1_macro", "best_val_mIoU", "best_val_macro_f1",
                ]
            )
            for name in raw_model.probe_names:
                row = summary["per_probe"][name]
                table.add_data(
                    name,
                    row.get("test/mIoU"),
                    row.get("test/mAcc"),
                    row.get("test/allAcc"),
                    row.get("test/f1_macro"),
                    row["best_val_mIoU"],
                    row["best_val_macro_f1"],
                )
            wandb.log({"seed_ensemble/per_probe_table": table})

            if f1_cls_stats:
                cls_table = wandb.Table(columns=["class", "f1_mean"])
                for key, st in f1_cls_stats.items():
                    cls_table.add_data(key[len("test/f1_"):], st["mean"])
                wandb.log({"seed_ensemble/per_class_f1_table": cls_table})

            if iou_cls_stats:
                iou_table = wandb.Table(columns=["class", "iou_mean", "iou_std"])
                for key, st in iou_cls_stats.items():
                    iou_table.add_data(
                        key[len("test/iou_"):], st["mean"], st["std"]
                    )
                wandb.log({"seed_ensemble/per_class_iou_table": iou_table})
