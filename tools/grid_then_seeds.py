#!/usr/bin/env python3
"""
Chain a GridProbe hyperparameter sweep into a seed-ensemble robustness run.

Phase 1 -- grid probe: run the given GridProbeSegmentorV2 / GridProbeTrainer
config (any ``*-lin-grid*`` config: h3d / dales / eclair / malibu3d, any frozen
backbone). ``GridProbeWinnerSelector`` writes
``<grid_dir>/grid_search_results.json`` with the probe that had the best
*validation* mIoU over the whole run.

Phase 2 -- seed ensemble: take the winner's full ``probe_config`` (loss, lr,
optimizer, scheduler, input_norm, feat_norm, dropout, grad_clip -- not just the
lr), replicate it into ``--n-seeds`` probes differing only by random init, swap
``GridProbeWinnerSelector`` -> ``GridProbeSeedEnsembleTester``, and run once.
``GridProbeSeedEnsembleTester`` writes ``<seed_dir>/seed_ensemble_results.json``
with test mIoU/mAcc/allAcc/f1_macro as mean +/- std across the seeds.

This is the dynamic-winner counterpart of ``tools/gen_grid_seed_configs.py``
(which bakes each sweep's winning lr into a hardcoded table).

Both phases are ordinary ``tools/train.py`` invocations -- one W&B run each,
put in a shared ``wandb_group`` so they show up together. The driver is
idempotent: a phase whose result JSON already exists is skipped, and an
interrupted phase resumes from ``model/model_last.pth`` -- so a Slurm requeue
just re-runs this script.

Typical use (see ``README_grid_then_seed.md``)::

    python tools/grid_then_seeds.py \\
      --grid-config configs/h3d/sonata-v1m2-h3d-lin-grid.py \\
      --weight ckpt/malibu3d/sonata_outdoor/epoch_120.pth \\
      --save-root exp/grid_then_seeds/h3d_sonata --n-seeds 10

Grid already done (e.g. the 336-probe wide sweep, run separately)::

    python tools/grid_then_seeds.py --grid-config <same cfg> --weight <ckpt> \\
      --skip-grid --grid-dir logs/slurm/<gridjob> --save-root logs/slurm/<thisjob>
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GRID_RESULT = "grid_search_results.json"
SEED_RESULT = "seed_ensemble_results.json"
GEN_CONFIG_NAME = "seed_ensemble_config.py"
SUMMARY_CSV_NAME = "grid_then_seeds_summary.csv"

SUMMARY_FIELDS = [
    "timestamp",
    "grid_config",
    "weight",
    "winner_probe_name",
    "winner_select_metric",
    "winner_val_mIoU",
    "winner_val_f1_macro",
    "n_seeds",
    "val_split",
    "test_split",
    "val_eq_test_split",
    "num_probes_with_test_metrics",
    "test_mIoU_mean",
    "test_mIoU_std",
    "test_mAcc_mean",
    "test_mAcc_std",
    "test_allAcc_mean",
    "test_allAcc_std",
    "test_f1_macro_mean",
    "test_f1_macro_std",
    "grid_dir",
    "seed_dir",
]


def log(msg: str) -> None:
    print(f"[grid_then_seeds] {msg}", flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_config_path(spec: str) -> Path:
    """Mirror scripts/train.sh: accept an absolute/relative file path, or a
    name resolved under ``configs/`` (e.g. ``experiment/w110/1/.../foo``)."""
    p = Path(spec)
    if p.is_file():
        return p.resolve()
    cand = REPO_ROOT / "configs" / f"{spec}.py"
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(
        f"grid config not found: {spec!r} (tried {spec!r} and configs/{spec}.py)"
    )


def result_exists(d: Path, fname: str) -> bool:
    return (d / fname).is_file()


def needs_resume(save_path: Path) -> bool:
    return (save_path / "model" / "model_last.pth").is_file()


def _split_of(entry) -> str | None:
    try:
        return entry.get("split")
    except AttributeError:
        return None


def split_info_from_cfg(cfg) -> dict:
    """Which split each of data.val / data.test reads. On DALES there is no
    held-out val, so every DALES config points both at ``split="test"`` -- the
    seed-ensemble ``test_*`` numbers are then on the same tiles the winner was
    selected on. h3d/eclair keep val and test separate."""
    val_split = _split_of(cfg.data.val) if "val" in cfg.data else None
    test_split = _split_of(cfg.data.test) if "test" in cfg.data else None
    return {
        "val_split": val_split,
        "test_split": test_split,
        "val_eq_test_split": bool(
            val_split is not None and val_split == test_split
        ),
    }


def read_split_info(config_path: Path) -> dict:
    from pointcept.utils.config import Config

    try:
        return split_info_from_cfg(Config.fromfile(str(config_path)))
    except Exception as exc:  # never let a reporting nicety break the run
        log(f"warning: could not read split info from {config_path}: {exc}")
        return {"val_split": None, "test_split": None, "val_eq_test_split": False}


def read_winner(grid_dir: Path) -> tuple[str, dict, dict]:
    path = grid_dir / GRID_RESULT
    data = json.loads(path.read_text(encoding="utf-8"))
    winner = data.get("winner") or {}
    name = winner.get("probe_name")
    probe_config = winner.get("probe_config")
    if not name or not probe_config:
        raise ValueError(f"{path}: no usable 'winner' entry ({winner!r})")

    def _f(v):
        return float(v) if v is not None else float("nan")

    # select_metric is absent in pre-macro-F1 grid_search_results.json -> "mIoU".
    stats = {
        "select_metric": data.get("select_metric") or "mIoU",
        "mIoU": _f(winner.get("best_val_mIoU")),
        "macro_f1": _f(winner.get("best_val_macro_f1")),
    }
    return str(name), dict(probe_config), stats


def build_seed_ensemble_config(
    grid_config_path: Path,
    winner_name: str,
    winner_probe_config: dict,
    n_seeds: int,
    out_path: Path,
) -> tuple[Path, list[str], dict]:
    """Derive a standalone seed-ensemble config from a grid-probe config +
    its winning ``probe_config``.

    Same transform as ``tools/gen_grid_seed_configs.py`` but on a live
    ``Config`` object (winner comes from ``grid_search_results.json`` at
    runtime, not a hardcoded lr table), and it carries the *full* winner
    ``probe_config`` -- loss/optimizer/scheduler/norms/dropout/grad_clip --
    not only the lr.
    """
    from pointcept.utils.config import Config

    cfg = Config.fromfile(str(grid_config_path))

    probe_names = [f"seed{i}" for i in range(n_seeds)]
    cfg.model.probes = {
        name: copy.deepcopy(dict(winner_probe_config)) for name in probe_names
    }

    num_classes = cfg.get("num_classes", None)
    if num_classes is None:
        num_classes = cfg.data.num_classes
    ignore_index = cfg.get("ignore_index", None)
    if ignore_index is None:
        ignore_index = cfg.data.ignore_index
    task_common = dict(
        task_type="semantic", num_classes=num_classes, ignore_index=ignore_index
    )
    names = list(cfg.data.names) if "names" in cfg.data else None
    if names is not None:
        task_common["names"] = names
    # mirror probe names -- InformationWriter keys per-task train metrics off this
    cfg.data.task_configs = {name: dict(task_common) for name in probe_names}

    swapped = False
    new_hooks = []
    for hook in cfg.hooks:
        hook = dict(hook)
        if hook.get("type") == "GridProbeWinnerSelector":
            new_hooks.append(dict(type="GridProbeSeedEnsembleTester"))
            swapped = True
        else:
            new_hooks.append(hook)
    if not swapped:
        raise ValueError(
            f"{grid_config_path}: no GridProbeWinnerSelector hook -- not a "
            "standard grid-probe config, cannot derive a seed-ensemble config."
        )
    cfg.hooks = new_hooks

    # the seed-ensemble test pass needs the shared-backbone, all-probes tester
    cfg.test = dict(type="GridProbeSemSegTester", verbose=True)
    cfg.log_test_f1 = True

    prev_name = str(cfg.get("wandb_run_name", "") or "")
    cfg.wandb_run_name = (
        f"SeedEnsemble {n_seeds} inits - winner={winner_name}"
        + (f" | {prev_name}" if prev_name else "")
    )[:250]

    info = split_info_from_cfg(cfg)
    if info["val_eq_test_split"]:
        log(
            f"note: data.val and data.test both use split="
            f"{info['test_split']!r} -- the seed-ensemble test_* metrics are on "
            "the same tiles used for winner selection (expected on DALES: no "
            "held-out test split)."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(out_path))
    return out_path, probe_names, info


def run_train(
    config_file: Path,
    save_path: Path,
    *,
    weight: str | None,
    extra_options: str | None,
    num_gpus: int,
    wandb_group: str | None,
    resume: bool,
) -> None:
    opts = [f"save_path={save_path}"]
    if resume:
        opts.append("resume=True")
        opts.append(f"weight={save_path / 'model' / 'model_last.pth'}")
    elif weight:
        opts.append(f"weight={weight}")
    if wandb_group:
        opts.append(f"wandb_group={wandb_group}")
    if extra_options:
        opts.extend(extra_options.split())

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "train.py"),
        "--config-file",
        str(config_file),
        "--num-gpus",
        str(num_gpus),
        "--options",
        *opts,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=env)


def _fmt(x) -> str:
    return "" if x is None else f"{float(x):.6f}"


def append_summary_csv(csv_path: Path, row: dict) -> bool:
    """Append one row, flock-guarded. Idempotent: a row with the same
    ``seed_dir`` is not written twice. Returns True if a row was written."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a+", encoding="utf-8", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            existing = list(csv.DictReader(f))
            if any(r.get("seed_dir") == row["seed_dir"] for r in existing):
                return False
            write_header = os.fstat(f.fileno()).st_size == 0
            f.seek(0, os.SEEK_END)
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return True


def report(
    *,
    grid_config: Path,
    weight: str | None,
    winner_name: str,
    winner_val: dict,
    n_seeds: int,
    grid_dir: Path,
    seed_dir: Path,
    save_root: Path,
    extra_csv: str | None,
    split: dict | None = None,
) -> None:
    res = json.loads((seed_dir / SEED_RESULT).read_text(encoding="utf-8"))
    split = split or {}

    def g(k):
        return res.get(k)

    def line(label, mean, std, lo, hi):
        if mean is None:
            print(f"  {label:<20}: n/a", flush=True)
        else:
            print(
                f"  {label:<20}: {mean:.4f} +/- {std:.4f}   "
                f"[min {lo:.4f}, max {hi:.4f}]",
                flush=True,
            )

    print("\n" + "=" * 72, flush=True)
    print("  grid -> seed-ensemble summary", flush=True)
    print("=" * 72, flush=True)
    print(f"  grid config          : {grid_config}", flush=True)
    print(f"  winner probe         : {winner_name}", flush=True)
    print(
        f"  winner best val      : select_metric={winner_val['select_metric']}  "
        f"mIoU={winner_val['mIoU']:.4f}  macro_f1={winner_val['macro_f1']:.4f}  "
        f"(split {split.get('val_split')!r})",
        flush=True,
    )
    print(
        f"  seeds w/ test metrics: "
        f"{g('num_probes_with_test_metrics')}/{g('num_probes')}",
        flush=True,
    )
    if split.get("val_eq_test_split"):
        print(
            f"  note: test_* below are on split {split.get('test_split')!r} "
            "== the winner-selection split (DALES has no held-out test)",
            flush=True,
        )
    if g("test_error"):
        print(
            f"  !! test pass raised (metrics may be partial) -- see "
            f"{seed_dir / SEED_RESULT}",
            flush=True,
        )
    line("test mIoU", g("test_mIoU_mean"), g("test_mIoU_std"), g("test_mIoU_min"), g("test_mIoU_max"))
    line("test mAcc", g("test_mAcc_mean"), g("test_mAcc_std"), g("test_mAcc_min"), g("test_mAcc_max"))
    line("test allAcc", g("test_allAcc_mean"), g("test_allAcc_std"), g("test_allAcc_min"), g("test_allAcc_max"))
    line("test F1-macro", g("test_f1_macro_mean"), g("test_f1_macro_std"), g("test_f1_macro_min"), g("test_f1_macro_max"))
    print("=" * 72 + "\n", flush=True)

    row = {
        "timestamp": _utc_now(),
        "grid_config": str(grid_config),
        "weight": weight or "",
        "winner_probe_name": winner_name,
        "winner_select_metric": winner_val["select_metric"],
        "winner_val_mIoU": _fmt(winner_val["mIoU"]),
        "winner_val_f1_macro": _fmt(winner_val["macro_f1"]),
        "n_seeds": n_seeds,
        "val_split": split.get("val_split") or "",
        "test_split": split.get("test_split") or "",
        "val_eq_test_split": split.get("val_eq_test_split", False),
        "num_probes_with_test_metrics": g("num_probes_with_test_metrics"),
        "test_mIoU_mean": _fmt(g("test_mIoU_mean")),
        "test_mIoU_std": _fmt(g("test_mIoU_std")),
        "test_mAcc_mean": _fmt(g("test_mAcc_mean")),
        "test_mAcc_std": _fmt(g("test_mAcc_std")),
        "test_allAcc_mean": _fmt(g("test_allAcc_mean")),
        "test_allAcc_std": _fmt(g("test_allAcc_std")),
        "test_f1_macro_mean": _fmt(g("test_f1_macro_mean")),
        "test_f1_macro_std": _fmt(g("test_f1_macro_std")),
        "grid_dir": str(grid_dir),
        "seed_dir": str(seed_dir),
    }
    targets = [save_root / SUMMARY_CSV_NAME]
    if extra_csv:
        targets.append(Path(extra_csv).resolve())
    for target in targets:
        wrote = append_summary_csv(target, row)
        log(("appended summary row -> " if wrote else "summary row already present in ") + str(target))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--grid-config", required=True, help="grid-probe config (path or configs/ name)")
    ap.add_argument("--weight", default=None, help="frozen backbone checkpoint (overrides the config's weight=)")
    ap.add_argument("--save-root", default=None, help="output root (default: $JOB_DIR or exp/grid_then_seeds/<ts>)")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--extra-options", default=None, help='appended to --options of both phases, e.g. "epoch=2 eval_epoch=2"')
    ap.add_argument("--wandb-group", default=None, help="W&B group for both runs (default: gts-<save-root basename>)")
    ap.add_argument("--skip-grid", action="store_true", help="grid already ran; use --grid-dir and go straight to winner->seeds")
    ap.add_argument("--grid-dir", default=None, help="existing grid save dir (required with --skip-grid / --make-config-only)")
    ap.add_argument("--make-config-only", action="store_true", help="just write the seed-ensemble config from an existing grid dir, then exit")
    ap.add_argument("--summary-csv", default=None, help="extra CSV to append the final mean+/-std row to")
    args = ap.parse_args()

    grid_config = resolve_config_path(args.grid_config)
    weight = str(Path(args.weight).resolve()) if args.weight else None

    save_root = Path(
        args.save_root
        or os.environ.get("JOB_DIR")
        or (REPO_ROOT / "exp" / "grid_then_seeds" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    ).resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    grid_dir = Path(args.grid_dir).resolve() if args.grid_dir else (save_root / "grid")
    seed_dir = save_root / "seeds"
    gen_cfg = save_root / GEN_CONFIG_NAME
    wandb_group = args.wandb_group or f"gts-{save_root.name}"

    log(f"grid config : {grid_config}")
    log(f"weight      : {weight or '(from config)'}")
    log(f"save root   : {save_root}")
    log(f"grid dir    : {grid_dir}   seed dir: {seed_dir}")

    # ---------------- Phase 1: grid probe ----------------
    if args.skip_grid or args.make_config_only:
        if not result_exists(grid_dir, GRID_RESULT):
            log(f"ERROR: --skip-grid/--make-config-only need an existing {grid_dir / GRID_RESULT}")
            return 2
        log("phase 1 (grid): skipped (using existing results)")
    elif result_exists(grid_dir, GRID_RESULT):
        log(f"phase 1 (grid): already done ({grid_dir / GRID_RESULT}) -- skipping")
    else:
        grid_dir.mkdir(parents=True, exist_ok=True)
        resume = needs_resume(grid_dir)
        log(f"phase 1 (grid): {'RESUMING' if resume else 'starting'}")
        run_train(
            grid_dir / "config.py" if resume else grid_config,
            grid_dir,
            weight=weight,
            extra_options=args.extra_options,
            num_gpus=args.num_gpus,
            wandb_group=wandb_group,
            resume=resume,
        )
        if not result_exists(grid_dir, GRID_RESULT):
            log(f"ERROR: phase 1 finished but {grid_dir / GRID_RESULT} is missing")
            return 3

    winner_name, winner_cfg, winner_val = read_winner(grid_dir)
    log(
        f"winner: {winner_name!r}  select_metric={winner_val['select_metric']}  "
        f"best_val_mIoU={winner_val['mIoU']:.4f}  "
        f"best_val_macro_f1={winner_val['macro_f1']:.4f}"
    )
    log("winner probe_config:\n" + json.dumps(winner_cfg, indent=2, default=str))

    # ---------------- seed-ensemble config ----------------
    if gen_cfg.is_file():
        log(f"seed-ensemble config exists ({gen_cfg}) -- reusing")
        split = read_split_info(gen_cfg)
    else:
        out, probe_names, split = build_seed_ensemble_config(
            grid_config, winner_name, winner_cfg, args.n_seeds, gen_cfg
        )
        log(f"wrote {out}  ({len(probe_names)} probes: {probe_names[0]}..{probe_names[-1]})")

    if args.make_config_only:
        return 0

    # ---------------- Phase 2: seed ensemble ----------------
    if result_exists(seed_dir, SEED_RESULT):
        log(f"phase 2 (seeds): already done ({seed_dir / SEED_RESULT}) -- skipping")
    else:
        seed_dir.mkdir(parents=True, exist_ok=True)
        resume = needs_resume(seed_dir)
        log(f"phase 2 (seeds): {'RESUMING' if resume else 'starting'}")
        run_train(
            seed_dir / "config.py" if resume else gen_cfg,
            seed_dir,
            weight=weight,
            extra_options=args.extra_options,
            num_gpus=args.num_gpus,
            wandb_group=wandb_group,
            resume=resume,
        )
        if not result_exists(seed_dir, SEED_RESULT):
            log(f"ERROR: phase 2 finished but {seed_dir / SEED_RESULT} is missing")
            return 4

    report(
        grid_config=grid_config,
        weight=weight,
        winner_name=winner_name,
        winner_val=winner_val,
        n_seeds=args.n_seeds,
        grid_dir=grid_dir,
        seed_dir=seed_dir,
        save_root=save_root,
        extra_csv=args.summary_csv,
        split=split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
