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


_HISTORY_CSV_FIELDNAMES = ("epoch", "probe_name", "mIoU", "mIoU_best")


def _atomic_write_history_csv(path, rows):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_HISTORY_CSV_FIELDNAMES)
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
    confusion-matrix mIoU computed independently per probe head, resume-safe
    running-best tracking per probe (via HookBase.state_dict/load_state_dict).

    comm_info["current_metric_value"] is set to THIS epoch's max mIoU across
    probes (not a running best) so CheckpointSaver's own max-over-epochs
    bookkeeping ends up computing max_p(max_e m_iou(p, e)) into
    trainer.best_metric_value — which equals max_e(max_p m_iou(p, e)), i.e.
    exactly the eventual winning probe's own best value (max commutes over
    the epoch/probe axes). MetricsJsonWriter's existing best_val_mIoU output
    is therefore already correct for grid runs with zero changes to that hook.

    Also writes save_path/grid_probe_miou_history.csv: one row per
    (epoch, probe) with mIoU/mIoU_best, rewritten atomically after every eval
    so the full per-epoch curve for every probe is on disk — unlike
    grid_search_results.json (GridProbeWinnerSelector), which only keeps each
    probe's single best value, not its trajectory.
    """

    def __init__(self, write_cls_iou=False):
        self.write_cls_iou = write_cls_iou
        self._best_miou_by_probe = {}
        # Per-class IoU/accuracy at each probe's own best epoch (i.e. the same
        # epoch GridProbeCheckpointSaver saved probe_best_{name}.pth for), so
        # GridProbeWinnerSelector can report the winner's per-class val
        # breakdown without a separate re-evaluation pass. Persisted across
        # resume (see state_dict/load_state_dict).
        self._best_cls_iou_by_probe = {}
        self._best_cls_acc_by_probe = {}
        # This epoch's raw (non-running-best) values; read by
        # GridProbeCheckpointSaver to decide which probes to save. Not
        # persisted across resume — recomputed every eval() call.
        self._last_miou_by_probe = {}
        # Full (epoch, probe) history for grid_probe_miou_history.csv —
        # persisted across resume (see state_dict/load_state_dict).
        self._history = []

    def state_dict(self):
        return {
            "best_miou_by_probe": dict(self._best_miou_by_probe),
            "best_cls_iou_by_probe": dict(self._best_cls_iou_by_probe),
            "best_cls_acc_by_probe": dict(self._best_cls_acc_by_probe),
            "history": list(self._history),
        }

    def load_state_dict(self, state):
        self._best_miou_by_probe = dict(state.get("best_miou_by_probe", {}))
        self._best_cls_iou_by_probe = dict(state.get("best_cls_iou_by_probe", {}))
        self._best_cls_acc_by_probe = dict(state.get("best_cls_acc_by_probe", {}))
        self._history = list(state.get("history", []))

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
            cls_hist_by_probe[name] = (intersection, union, target_hist)

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
            if m_iou >= prev:
                # Matches GridProbeCheckpointSaver's own tie-breaking (latest
                # epoch wins ties), so the per-class snapshot here always
                # corresponds to the epoch whose weights actually get saved
                # as probe_best_{name}.pth.
                intersection, union, target_hist = cls_hist_by_probe[name]
                self._best_cls_iou_by_probe[name] = (
                    intersection / (union + 1e-10)
                ).tolist()
                self._best_cls_acc_by_probe[name] = (
                    intersection / (target_hist + 1e-10)
                ).tolist()
            self._best_miou_by_probe[name] = max(prev, m_iou)

        if comm.is_main_process():
            for name in probe_names:
                self._history.append(
                    {
                        "epoch": current_epoch,
                        "probe_name": name,
                        "mIoU": float(m_iou_by_probe[name]),
                        "mIoU_best": float(self._best_miou_by_probe[name]),
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
            if self.trainer.cfg.enable_wandb:
                # Per-probe curves would fan out into len(probe_names) * 2
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
                wandb.log(
                    {
                        "Epoch": row["epoch"],
                        "val/mIoU/winner": row["mIoU"],
                        "val/mIoU_best/winner": row["mIoU_best"],
                    }
                )

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
                "winner/best_val_mIoU": float(evaluator._best_miou_by_probe[winner_name]),
            }
            summary.update(
                _flatten_for_wandb(raw_model.probe_configs[winner_name], "winner/config")
            )
            # Per-class val IoU/accuracy at the winner's own best epoch (see
            # GridProbeEvaluator._best_cls_iou_by_probe / _best_cls_acc_by_probe).
            cls_iou = evaluator._best_cls_iou_by_probe.get(winner_name)
            cls_acc = evaluator._best_cls_acc_by_probe.get(winner_name)
            if cls_iou is not None:
                for i, cls_name in enumerate(names):
                    if i == raw_model.ignore_index or i >= len(cls_iou):
                        continue
                    summary[f"winner/val/iou_{cls_name}"] = float(cls_iou[i])
                    if cls_acc is not None:
                        summary[f"winner/val/acc_{cls_name}"] = float(cls_acc[i])
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
    not just a winner), runs ONE shared-backbone test pass across all of them
    (GridProbeSemSegTester -- one backbone forward per fragment, N heads),
    and writes mean/std of the per-probe test metrics to
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

        summary = {
            "num_probes": len(raw_model.probe_names),
            "num_probes_with_test_metrics": len(metrics_by_probe),
            "test_error": test_error,
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
            "per_probe": {
                name: {
                    "probe_config": dict(raw_model.probe_configs[name]),
                    "best_val_mIoU": (
                        float(best_val_by_probe[name])
                        if name in best_val_by_probe
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
            "mIoU=%.4f+/-%.4f [%.4f, %.4f], mAcc=%.4f+/-%.4f, allAcc=%.4f+/-%.4f. Wrote %s",
            summary["num_probes_with_test_metrics"], summary["num_probes"],
            summary["test_mIoU_mean"] or float("nan"),
            summary["test_mIoU_std"] or float("nan"),
            summary["test_mIoU_min"] or float("nan"),
            summary["test_mIoU_max"] or float("nan"),
            summary["test_mAcc_mean"] or float("nan"),
            summary["test_mAcc_std"] or float("nan"),
            summary["test_allAcc_mean"] or float("nan"),
            summary["test_allAcc_std"] or float("nan"),
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
            }
            for key, value in wandb_summary.items():
                wandb.run.summary[key] = value
            wandb.log(wandb_summary)

            # Per-seed test mIoU as a small wandb Table -- the summary above
            # only ever gives 4 numbers (mean/std/max/min); this is what lets
            # the actual 10-point distribution be inspected/plotted in the UI.
            table = wandb.Table(columns=["probe_name", "test_mIoU", "test_mAcc", "test_allAcc", "best_val_mIoU"])
            for name in raw_model.probe_names:
                row = summary["per_probe"][name]
                table.add_data(
                    name,
                    row.get("test/mIoU"),
                    row.get("test/mAcc"),
                    row.get("test/allAcc"),
                    row["best_val_mIoU"],
                )
            wandb.log({"seed_ensemble/per_probe_table": table})
