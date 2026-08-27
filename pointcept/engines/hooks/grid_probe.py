"""
Grid-Search Linear Probing Hooks

Per-probe validation, per-probe best-checkpoint saving, and end-of-training
winner selection + automatic test pass for GridProbeSegmentorV2 /
GridProbeTrainer (see pointcept/models/grid_probe.py,
pointcept/engines/train.py).
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


# Which validation metric drives per-probe best-checkpoint selection
# (GridProbeEvaluator.select_metric). "mIoU" keeps the historical behavior;
# "macro_f1" selects on validation macro-F1 (same estimator as test/f1_macro).
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
        # restval="" so rows restored from a pre-feature checkpoint (no f1_macro
        # keys) still write; extrasaction default "raise" guards typos.
        writer = csv.DictWriter(
            f, fieldnames=_HISTORY_CSV_FIELDNAMES, restval=""
        )
        writer.writeheader()
        writer.writerows(rows)
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
    confusion-matrix mIoU / macro-F1 computed independently per probe head,
    resume-safe running-best tracking per probe (via
    HookBase.state_dict/load_state_dict).

    ``select_metric`` ("mIoU" default, or "macro_f1") picks which validation
    metric drives per-probe best-checkpoint selection (GridProbeCheckpointSaver)
    and end-of-run winner selection (GridProbeWinnerSelector). Both metrics are
    always computed and logged regardless; select_metric is only the selection
    knob. Under select_metric="mIoU" every decision and every mIoU-named output
    is bit-identical to the pre-feature behavior.

    comm_info["current_metric_value"] is set to THIS epoch's max of the
    selected metric across probes (not a running best) so CheckpointSaver's
    own max-over-epochs bookkeeping ends up computing max_p(max_e sel(p, e))
    into trainer.best_metric_value — which equals max_e(max_p sel(p, e)), i.e.
    exactly the eventual winning probe's own best value (max commutes over
    the epoch/probe axes). MetricsJsonWriter's existing best_val_mIoU output
    is therefore already correct for grid runs (with select_metric="mIoU")
    with zero changes to that hook.

    Also writes save_path/grid_probe_miou_history.csv: one row per
    (epoch, probe) with mIoU/mIoU_best and f1_macro/f1_macro_best, rewritten
    atomically after every eval so the full per-epoch curve for every probe is
    on disk — unlike grid_search_results.json (GridProbeWinnerSelector), which
    only keeps each probe's single best value, not its trajectory.
    """

    def __init__(self, write_cls_iou=False, select_metric="mIoU"):
        assert select_metric in _SELECT_METRICS, (
            f"GridProbeEvaluator.select_metric must be one of {_SELECT_METRICS}, "
            f"got {select_metric!r}"
        )
        self.select_metric = select_metric
        self.write_cls_iou = write_cls_iou
        # Running-best of each metric per probe. BOTH are tracked every epoch
        # regardless of select_metric (F1 is a cheap numpy call on the
        # already-synced confusion hist) — select_metric only picks which one
        # drives selection. Persisted across resume.
        self._best_miou_by_probe = {}
        self._best_f1_by_probe = {}
        # Per-class IoU/accuracy/F1 at each probe's own best epoch for the
        # SELECTED metric (i.e. the same epoch GridProbeCheckpointSaver saved
        # probe_best_{name}.pth for), so GridProbeWinnerSelector can report the
        # winner's per-class val breakdown without a separate re-evaluation
        # pass. Persisted across resume (see state_dict/load_state_dict).
        self._best_cls_iou_by_probe = {}
        self._best_cls_acc_by_probe = {}
        self._best_cls_f1_by_probe = {}
        # This epoch's raw (non-running-best) values; read by
        # GridProbeCheckpointSaver to decide which probes to save. Not
        # persisted across resume — recomputed every eval() call.
        self._last_miou_by_probe = {}
        self._last_f1_by_probe = {}
        # Full (epoch, probe) history for grid_probe_miou_history.csv —
        # persisted across resume (see state_dict/load_state_dict).
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
        """Probe's running-best val mIoU, or nan if not tracked yet (e.g. a
        pre-macro-F1 checkpoint resumed under select_metric='mIoU' with no eval
        since — _best_miou_by_probe is populated, _best_f1_by_probe is not)."""
        v = self._best_miou_by_probe.get(name)
        return float(v) if v is not None else float("nan")

    def best_macro_f1(self, name):
        """Probe's running-best val macro-F1, or nan if not tracked yet."""
        v = self._best_f1_by_probe.get(name)
        return float(v) if v is not None else float("nan")

    def _update_bests(
        self, m_iou_by_probe, macro_f1_by_probe, f1_cls_by_probe, cls_hist_by_probe
    ):
        """Fold this epoch's per-probe metrics into the running-best state.

        - ``_best_miou_by_probe`` / ``_best_f1_by_probe``: running max of each
          metric, tracked unconditionally (select_metric only picks which one
          is *used*, not which one is *computed*).
        - per-class snapshots (``_best_cls_{iou,acc,f1}_by_probe``): taken on the
          epoch that (re)sets the running best of the SELECTED metric, ``>=`` so
          the latest epoch wins ties — identical rule to
          GridProbeCheckpointSaver, which reads ``_selected_{last,best}_by_probe``
          so the two can't drift.
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
        # Called only from CheckpointLoader.before_train under cfg.resume, and
        # CheckpointLoader is the first hook — raising here aborts cleanly
        # before any training step.
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
                f"Resuming a pre-macro-F1 checkpoint with "
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
        # Finding: a checkpoint whose hook_states has no "GridProbeEvaluator"
        # entry (predates this hook's state tracking) is resumed WITHOUT
        # load_state_dict ever running (CheckpointLoader guards `if state:`), so
        # the select_metric consistency check is bypassed while
        # trainer.best_metric_value has already been restored in the OLD metric's
        # scale. Reset it here so best-checkpoint selection restarts cleanly on
        # the configured metric. (A fresh run has resume=False; a resume that DID
        # restore evaluator state either populated _best_f1_by_probe or already
        # hard-failed in load_state_dict.)
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
            # Same estimator as test/f1_macro (unmasked mean over all classes,
            # from the already-synced confusion hist — no extra collective).
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
                # Per-probe curves would fan out into len(probe_names) * 4
                # wandb metrics every epoch; only the eventual winner's own
                # curve is worth a dashboard line, and GridProbeWinnerSelector
                # replays that one from _history once training ends. Console
                # (per-probe logger.info above) and TensorBoard (per-probe
                # add_scalar above) keep the full per-probe detail.
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

        # See class docstring: this is THIS epoch's max of the SELECTED metric,
        # not a running best — CheckpointSaver does the running-max itself via
        # best_metric_value.
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
    """Save each probe head's own best-val weights separately (a few KB —
    just that head's state_dict, not the whole shared model) whenever it
    improves this epoch, judged on GridProbeEvaluator.select_metric ("mIoU"
    default, or "macro_f1"). Must be registered after GridProbeEvaluator.
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
    """After training: pick the probe with the best val score over the whole
    run — judged on GridProbeEvaluator.select_metric ("mIoU" default, or
    "macro_f1") — reload its best-epoch weights (the live weights at end of
    training are that probe's LAST epoch, not its BEST, since other probes may
    have kept training after it peaked), optionally run a precise test pass
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
            # None (not nan) for JSON when a probe's value isn't tracked yet
            # (pre-macro-F1 checkpoint resumed under select_metric="mIoU").
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
            # Replay the winning probe's full per-epoch trajectory as its own
            # clean wandb curve (val/* uses Epoch as step_metric, so this
            # slots into the normal val chart, not a new x-axis) — this is
            # the one per-probe curve GridProbeEvaluator intentionally didn't
            # stream live, to avoid a wandb chart per probe during training.
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
                # .get: history rows restored from a pre-macro-F1 checkpoint
                # (only reachable under select_metric="mIoU") lack these keys.
                if row.get("f1_macro") is not None:
                    log_row["val/f1_macro/winner"] = row["f1_macro"]
                if row.get("f1_macro_best") is not None:
                    log_row["val/f1_macro_best/winner"] = row["f1_macro_best"]
                wandb.log(log_row)

            names = list(getattr(cfg.data, "names", []) or [])

            # Same replay for the winner's train loss/mIoU (+ per-class IoU
            # when InformationWriter(write_cls_iou=True)) — see
            # InformationWriter.after_epoch, which intentionally skips
            # per-probe train/* W&B logging during grid probing for the same
            # reason as val/* above.
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
            # Per-class val IoU/accuracy/F1 snapshotted at the winner's best
            # *select_metric* epoch (see _best_cls_{iou,acc,f1}_by_probe). NB
            # winner/best_val_{mIoU,macro_f1} above are all-time running maxes:
            # under select_metric="macro_f1" the F1 pair is consistent (snapshot
            # taken on the peak-F1 epoch); under "mIoU" the per-class F1 here is
            # from the peak-mIoU epoch and can differ from best_val_macro_f1.
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
    """For a "same hyperparameters, different random init" grid rather than a
    hyperparameter search: there is no single winner to pick, just N
    replicate probes to average. Reloads each probe's own best-val checkpoint
    (already saved individually by GridProbeCheckpointSaver for every probe,
    not just a winner, judged on GridProbeEvaluator.select_metric), runs ONE
    shared-backbone test pass across all of them
    (GridProbeSemSegTester -- one backbone forward per fragment, N heads),
    and writes mean/std of the per-probe test metrics (mIoU/mAcc/allAcc,
    plus f1_macro and per-class f1_mean when log_test_f1=True) to
    save_path/seed_ensemble_results.json.

    Use in place of GridProbeWinnerSelector, last in the hooks list, with
    cfg.test = dict(type="GridProbeSemSegTester", ...).

    Robustness: per_probe always has one entry per probe in raw_model.probe_names
    (probe_config + best_val_mIoU are always populated from in-memory state that
    can't be missing; test metrics are filled in whenever the tester produced
    them for that probe, null otherwise) and the JSON is written even if the
    test pass raises partway through -- a crash/OOM on the shared test forward
    must not also destroy the val-only results that already exist for every
    probe. mean/std/max/min are computed only over probes that actually have a
    numeric test/mIoU, so one missing/NaN probe can't crash the whole write.
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
            # Whatever happened (OOM on the shared test forward, a bad scene,
            # ...), every probe's val-only results (best_val_mIoU, checkpoint)
            # already exist and must still make it to disk below -- only the
            # test/* fields end up null instead of losing the whole write.
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

        def _stats(metric_key):
            values = [
                m[metric_key] for m in metrics_by_probe.values() if metric_key in m
            ]
            if not values:
                return dict(mean=None, std=None, max=None, min=None)
            arr = np.array(values, dtype=float)
            return dict(
                mean=float(arr.mean()), std=float(arr.std()),
                max=float(arr.max()), min=float(arr.min()),
            )

        miou_stats = _stats("test/mIoU")
        macc_stats = _stats("test/mAcc")
        allacc_stats = _stats("test/allAcc")
        f1_macro_stats = _stats("test/f1_macro")

        # Per-class F1 means: discover test/f1_{cls} keys (present only when
        # log_test_f1=True). Prefer cfg.data.names order; fall back to first
        # probe's key insertion order so this stays dataset-agnostic.
        f1_cls_keys = []
        names = getattr(cfg.data, "names", None) or []
        for cls_name in names:
            key = f"test/f1_{cls_name}"
            if any(key in m for m in metrics_by_probe.values()):
                f1_cls_keys.append(key)
        if not f1_cls_keys:
            seen = set()
            for m in metrics_by_probe.values():
                for key in m:
                    if (
                        key.startswith("test/f1_")
                        and key != "test/f1_macro"
                        and key not in seen
                    ):
                        seen.add(key)
                        f1_cls_keys.append(key)
        f1_cls_means = {
            key: _stats(key)["mean"] for key in f1_cls_keys
        }

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
            **{
                f"test_f1_{key[len('test/f1_'):]}_mean": mean
                for key, mean in f1_cls_means.items()
            },
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
            for key, mean in f1_cls_means.items():
                cls_name = key[len("test/f1_"):]
                wandb_summary[f"seed_ensemble/test_f1_{cls_name}_mean"] = mean
            for key, value in wandb_summary.items():
                wandb.run.summary[key] = value
            wandb.log(wandb_summary)

            # Per-seed test mIoU as a small wandb Table -- the summary above
            # only ever gives 4 numbers (mean/std/max/min); this is what lets
            # the actual 10-point distribution be inspected/plotted in the UI.
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

            if f1_cls_means:
                cls_table = wandb.Table(columns=["class", "f1_mean"])
                for key, mean in f1_cls_means.items():
                    cls_table.add_data(key[len("test/f1_"):], mean)
                wandb.log({"seed_ensemble/per_class_f1_table": cls_table})
