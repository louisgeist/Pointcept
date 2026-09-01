#!/usr/bin/env python3
"""Binary-search max val/test voxel budget (VRAM probe).

Same spirit as scripts/find_max_batch_size.py, but searches
``val_voxel_budget`` / ``test_voxel_budget`` while keeping a high scene-count
cap (``batch_size_val`` / ``batch_size_test``) so packing is budget-limited.

Requires ``n_voxels`` in ``data.*.csv_manifest`` (enrich with
``analyze_malibu3d_test_point_voxel_counts.py --write_manifest``).

Typical safe order of magnitude is around 2M voxels; default search range is
``[500_000, 4_000_000]`` with step ``100_000``.

Examples::

  # Val
  python scripts/find_max_voxel_budget.py \\
    --config-file configs/experiment/w105/2/10h/litept-v1m0-malibu3d_12.py \\
    --mode val --min-budget 500000 --max-budget 4000000 --max-sample 128

  # Test
  python scripts/find_max_voxel_budget.py \\
    --config-file configs/experiment/w105/2/10h/litept-v1m0-malibu3d_12.py \\
    --mode test --min-budget 500000 --max-budget 4000000 --max-sample 128
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Reuse helpers from the batch-size probe script.
_FMB_PATH = REPO_ROOT / "scripts" / "find_max_batch_size.py"
_spec = importlib.util.spec_from_file_location("find_max_batch_size", _FMB_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load helpers from {_FMB_PATH}")
_fmb = importlib.util.module_from_spec(_spec)
sys.modules["find_max_batch_size"] = _fmb
_spec.loader.exec_module(_fmb)

log = _fmb.log
write_probe_config = _fmb.write_probe_config
run_cmd = _fmb.run_cmd
is_oom = _fmb.is_oom
python_train_cmd = _fmb.python_train_cmd
python_test_cmd = _fmb.python_test_cmd
detect_evaluator_type = _fmb.detect_evaluator_type
build_hooks = _fmb.build_hooks
_gpu_name = _fmb._gpu_name


def align_budget(budget: int, step: int) -> int:
    budget = max(int(step), int(budget))
    step = max(1, int(step))
    return (budget // step) * step


def probe_val(
    *,
    args: argparse.Namespace,
    work_dir: Path,
    trial_name: str,
    voxel_budget: int,
    max_sample_val: int | None,
    evaluator_type: str | None,
) -> tuple[bool, float | None, str]:
    save_path = work_dir / "runs" / trial_name
    cfg_path = work_dir / "configs" / f"{trial_name}.py"
    overrides = {
        # Tiny train step so VRAM is dominated by val packing.
        "batch_size": max(1, int(args.val_train_batch_size)),
        "mix_prob": 0.0,
        "total_iters": 1,
        "iter_per_epoch": 1,
        "evaluate": True,
        "eval_every": 1,
        "enable_wandb": False,
        "save_path": str(save_path),
        "hooks": build_hooks("val", evaluator_type, save_checkpoint=False),
        "empty_cache": False,
        "empty_cache_per_epoch": False,
        "val_voxel_budget": int(voxel_budget),
        "batch_size_val": int(args.max_batch_size),
        # Avoid accidental test packing during this probe.
        "test_voxel_budget": None,
    }
    if args.num_worker is not None:
        overrides["num_worker"] = int(args.num_worker)
    write_probe_config(src_config=Path(args.config_file), dest=cfg_path, overrides=overrides)

    options: list[str] = []
    if max_sample_val is not None:
        options.append(f"data.val.max_sample={int(max_sample_val)}")
    if args.extra_options:
        options.extend(args.extra_options)

    cmd = python_train_cmd(
        python=args.python,
        config_file=cfg_path,
        num_gpus=args.num_gpus,
        extra_options=options,
    )
    log_path = work_dir / "logs" / f"{trial_name}.log"
    try:
        code, out, peak = run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, timeout_s=args.timeout)
    except Exception:
        return False, None, "timeout"

    if code == 0:
        return True, peak, "ok"
    if is_oom(out, code):
        return False, peak, "oom"
    return False, peak, f"error_exit_{code}"


def probe_test(
    *,
    args: argparse.Namespace,
    work_dir: Path,
    trial_name: str,
    voxel_budget: int,
    weight: Path,
    max_sample_test: int | None,
) -> tuple[bool, float | None, str]:
    save_path = work_dir / "runs" / trial_name
    cfg_path = work_dir / "configs" / f"{trial_name}.py"
    overrides = {
        "enable_wandb": False,
        "save_path": str(save_path),
        "weight": str(weight),
        "test_voxel_budget": int(voxel_budget),
        "batch_size_test": int(args.max_batch_size),
        "val_voxel_budget": None,
    }
    if args.num_worker is not None:
        overrides["num_worker"] = int(args.num_worker)
    write_probe_config(src_config=Path(args.config_file), dest=cfg_path, overrides=overrides)

    options: list[str] = [f"weight={weight}", f"save_path={save_path}"]
    if max_sample_test is not None:
        options.append(f"data.test.max_sample={int(max_sample_test)}")
    if args.extra_options:
        options.extend(args.extra_options)

    cmd = python_test_cmd(
        python=args.python,
        config_file=cfg_path,
        num_gpus=args.num_gpus,
        extra_options=options,
    )
    log_path = work_dir / "logs" / f"{trial_name}.log"
    try:
        code, out, peak = run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, timeout_s=args.timeout)
    except Exception:
        return False, None, "timeout"

    if code == 0:
        return True, peak, "ok"
    if is_oom(out, code):
        return False, peak, "oom"
    return False, peak, f"error_exit_{code}"


def ensure_seed_checkpoint(args: argparse.Namespace, work_dir: Path) -> Path:
    """One-step train to produce model_last.pth for tools/test.py probes."""
    dest = work_dir / "seed" / "model" / "model_last.pth"
    if dest.is_file():
        log(f"Reusing seed checkpoint: {dest}")
        return dest

    log("Building seed checkpoint (1 train step) for test probes...")
    # Reuse find_max_batch_size.probe_train with a tiny Namespace.
    seed_ns = argparse.Namespace(
        config_file=args.config_file,
        python=args.python,
        num_gpus=args.num_gpus,
        num_worker=args.num_worker,
        mix_prob=0.0,
        save_checkpoint=True,
        timeout=args.timeout,
        extra_options=args.extra_options,
    )
    ok, _, status = _fmb.probe_train(
        args=seed_ns,
        work_dir=work_dir,
        trial_name="seed_train",
        batch_size=1,
        steps=1,
        evaluate=False,
        batch_size_val=None,
        max_sample_val=None,
        evaluator_type=None,
    )
    weight = work_dir / "runs" / "seed_train" / "model" / "model_last.pth"
    if not ok or not weight.is_file():
        raise RuntimeError(
            f"Failed to build seed checkpoint (status={status}). "
            f"See {work_dir / 'logs' / 'seed_train.log'}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(weight.read_bytes())
    return dest


def run_trial(
    *,
    args: argparse.Namespace,
    work_dir: Path,
    trial_name: str,
    budget: int,
    evaluator_type: str | None,
    seed_weight: Path | None,
    max_sample: int,
) -> tuple[bool, float | None, str]:
    if args.mode == "val":
        return probe_val(
            args=args,
            work_dir=work_dir,
            trial_name=trial_name,
            voxel_budget=budget,
            max_sample_val=max_sample,
            evaluator_type=evaluator_type,
        )
    assert args.mode == "test"
    assert seed_weight is not None
    return probe_test(
        args=args,
        work_dir=work_dir,
        trial_name=trial_name,
        voxel_budget=budget,
        weight=seed_weight,
        max_sample_test=max_sample,
    )


def binary_search(args: argparse.Namespace, work_dir: Path) -> dict:
    step = max(1, int(args.budget_step))
    lo = align_budget(args.min_budget, step)
    hi = align_budget(args.max_budget, step)
    if lo > hi:
        raise ValueError(f"Invalid search range after align: lo={lo} hi={hi}")

    evaluator_type = detect_evaluator_type(Path(args.config_file))
    seed_weight = None
    if args.mode == "test":
        seed_weight = ensure_seed_checkpoint(args, work_dir)

    trials: list[dict] = []
    best = None
    best_peak = None
    trial_i = 0

    log(
        f"Phase 1: binary search mode={args.mode} "
        f"budget_range=[{lo}, {hi}] step={step} max_batch_size={args.max_batch_size}"
    )
    while lo <= hi:
        mid = align_budget((lo + hi) // 2, step)
        if mid < lo:
            mid = lo
        trial_i += 1
        name = f"probe_{trial_i:02d}_vb{mid}"
        log(f"\n=== Probe {name} (lo={lo} hi={hi}) ===")
        ok, peak, status = run_trial(
            args=args,
            work_dir=work_dir,
            trial_name=name,
            budget=mid,
            evaluator_type=evaluator_type,
            seed_weight=seed_weight,
            max_sample=args.max_sample,
        )
        trials.append(
            dict(
                phase="probe",
                trial=name,
                voxel_budget=mid,
                ok=ok,
                status=status,
                peak_mem_mb=peak,
            )
        )
        if status.startswith("error_exit") or status == "timeout":
            raise RuntimeError(
                f"Non-OOM failure at voxel_budget={mid} status={status}. "
                f"Inspect {work_dir / 'logs' / (name + '.log')}"
            )
        if ok:
            best = mid
            best_peak = peak
            lo = mid + step
        else:
            hi = mid - step

    if best is None:
        log("No voxel budget in range succeeded.")
        return dict(
            candidate_budget=None,
            confirmed_budget=None,
            peak_mem_mb=None,
            trials=trials,
        )

    log(f"\nPhase 1 candidate: voxel_budget={best} peak_mem={best_peak}")

    confirmed = best
    confirmed_peak = best_peak
    if args.soak_samples > 0:
        soak_samples = max(int(args.max_sample), int(args.soak_samples))
        name = f"soak_vb{confirmed}"
        log(f"\nPhase 2: soak max_sample={soak_samples}")
        ok, peak, status = run_trial(
            args=args,
            work_dir=work_dir,
            trial_name=name,
            budget=confirmed,
            evaluator_type=evaluator_type,
            seed_weight=seed_weight,
            max_sample=soak_samples,
        )
        trials.append(
            dict(
                phase="soak",
                trial=name,
                voxel_budget=confirmed,
                ok=ok,
                status=status,
                peak_mem_mb=peak,
            )
        )
        if status.startswith("error_exit") or status == "timeout":
            raise RuntimeError(
                f"Non-OOM failure during soak budget={confirmed} status={status}"
            )
        if not ok:
            # Step down once on soak OOM.
            nxt = confirmed - step
            confirmed = nxt if nxt >= align_budget(args.min_budget, step) else None
            confirmed_peak = peak
            if confirmed is not None:
                name = f"soak_vb{confirmed}"
                log(f"\n=== Soak fallback {name} ===")
                ok2, peak2, status2 = run_trial(
                    args=args,
                    work_dir=work_dir,
                    trial_name=name,
                    budget=confirmed,
                    evaluator_type=evaluator_type,
                    seed_weight=seed_weight,
                    max_sample=soak_samples,
                )
                trials.append(
                    dict(
                        phase="soak",
                        trial=name,
                        voxel_budget=confirmed,
                        ok=ok2,
                        status=status2,
                        peak_mem_mb=peak2,
                    )
                )
                if not ok2:
                    confirmed = None
                    confirmed_peak = peak2
                else:
                    confirmed_peak = peak2 if peak2 is not None else confirmed_peak
        else:
            confirmed_peak = peak if peak is not None else confirmed_peak

    return dict(
        candidate_budget=best,
        confirmed_budget=confirmed,
        peak_mem_mb=confirmed_peak,
        trials=trials,
    )


def write_csv(path: Path, args: argparse.Namespace, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config",
        "mode",
        "gpu_name",
        "candidate_budget",
        "confirmed_budget",
        "peak_mem_mb",
        "max_batch_size",
        "budget_step",
        "max_sample",
        "soak_samples",
        "trial",
        "phase",
        "voxel_budget",
        "ok",
        "status",
        "trial_peak_mem_mb",
    ]
    gpu_name = _gpu_name()
    rows = []
    if not result["trials"]:
        rows.append(
            {
                "config": args.config_file,
                "mode": args.mode,
                "gpu_name": gpu_name,
                "candidate_budget": result["candidate_budget"],
                "confirmed_budget": result["confirmed_budget"],
                "peak_mem_mb": result["peak_mem_mb"],
                "max_batch_size": args.max_batch_size,
                "budget_step": args.budget_step,
                "max_sample": args.max_sample,
                "soak_samples": args.soak_samples,
                "trial": "",
                "phase": "",
                "voxel_budget": "",
                "ok": "",
                "status": "",
                "trial_peak_mem_mb": "",
            }
        )
    else:
        for t in result["trials"]:
            rows.append(
                {
                    "config": args.config_file,
                    "mode": args.mode,
                    "gpu_name": gpu_name,
                    "candidate_budget": result["candidate_budget"],
                    "confirmed_budget": result["confirmed_budget"],
                    "peak_mem_mb": result["peak_mem_mb"],
                    "max_batch_size": args.max_batch_size,
                    "budget_step": args.budget_step,
                    "max_sample": args.max_sample,
                    "soak_samples": args.soak_samples,
                    "trial": t["trial"],
                    "phase": t["phase"],
                    "voxel_budget": t["voxel_budget"],
                    "ok": t["ok"],
                    "status": t["status"],
                    "trial_peak_mem_mb": t["peak_mem_mb"],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"Wrote CSV: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config-file", required=True, help="Path to Pointcept config .py")
    p.add_argument(
        "--mode",
        choices=("val", "test"),
        default="val",
        help="Which *_voxel_budget to search",
    )
    p.add_argument("--min-budget", type=int, default=500_000)
    p.add_argument("--max-budget", type=int, default=4_000_000)
    p.add_argument(
        "--budget-step",
        type=int,
        default=100_000,
        help="Align / step size for dichotomie (default 100k)",
    )
    p.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Scene-count cap so packing is limited by voxel budget, not BS",
    )
    p.add_argument(
        "--max-sample",
        type=int,
        default=128,
        help="Cap val/test samples during probes",
    )
    p.add_argument(
        "--soak-samples",
        type=int,
        default=0,
        help="Optional longer soak max_sample (0 = skip). "
        "If set, uses max(max_sample, soak_samples).",
    )
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--num-worker", type=int, default=None, help="Override num_worker")
    p.add_argument(
        "--val-train-batch-size",
        type=int,
        default=1,
        help="Tiny train BS used before val probe",
    )
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument(
        "--extra-options",
        nargs="*",
        default=[],
        help="Extra Pointcept --options key=value pairs",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config_file = Path(args.config_file)
    if not config_file.is_file():
        alt = REPO_ROOT / args.config_file
        if alt.is_file():
            args.config_file = str(alt)
            config_file = alt
        else:
            log(f"Config not found: {args.config_file}")
            return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    cfg_stem = config_file.stem
    work_dir = Path(
        args.work_dir
        or (
            REPO_ROOT
            / "exp"
            / "voxel_budget_search"
            / f"{cfg_stem}_{args.mode}_{stamp}"
        )
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    log(f"Work dir: {work_dir}")
    log(f"Config: {config_file}")
    log(f"GPU: {_gpu_name()}")
    log(
        "Note: requires n_voxels in csv_manifest "
        "(analyze_malibu3d_test_point_voxel_counts.py --write_manifest)."
    )

    try:
        result = binary_search(args, work_dir)
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 1

    csv_path = Path(args.csv) if args.csv else work_dir / "results.csv"
    write_csv(csv_path, args, result)

    key = "val_voxel_budget" if args.mode == "val" else "test_voxel_budget"
    log("\n======== SUMMARY ========")
    log(f"mode={args.mode}")
    log(f"candidate_budget={result['candidate_budget']}")
    log(f"confirmed_budget={result['confirmed_budget']}")
    log(f"peak_mem_mb={result['peak_mem_mb']}")
    if result["confirmed_budget"] is not None:
        log(f"Suggested config: {key} = {result['confirmed_budget']}")
        log(
            f"(keep batch_size_{'val' if args.mode == 'val' else 'test'} as a "
            f"scene-count cap, e.g. {args.max_batch_size} or lower)"
        )
        return 0
    log("No confirmed voxel budget — lower --min-budget / check logs / enrich manifest.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
