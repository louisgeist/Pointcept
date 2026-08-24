#!/usr/bin/env python3
"""Binary-search max batch size for train / val / test (VRAM probe).

Two phases (see plan max_batch_size_probe):
  1) Short dichotomie probe
  2) Optional longer soak on the candidate (train only by default)

Train probes default to mix_prob=1 (worst-case Mix3D). Prefer even batch sizes.
Align --mix-prob with the real training config:
  - supervised / lin-probe with Mix3D: keep 0.8 or 1.0
  - Sonata SSL pretrain (no Mix3D): pass --mix-prob 0

**Always pass --num-worker** on 1-GPU probes (CLI flag → ``args.num_worker`` →
config key ``num_worker``). If omitted (default None), the overlay does *not*
override and you inherit the source config — e.g. Sonata
``num_worker = 8 * num_gpu`` with ``num_gpu=32`` → **256 DataLoader workers**
on a single GPU. PyTorch will spawn them even if Slurm only gave you 24 CPUs;
workers then die with ``signal: Killed`` (host RAM OOM), which is *not* a VRAM
verdict. Recommended: ``--num-worker 8`` (or ``0`` / ``2`` for a pure VRAM
smoke). ``sbatch_find_max_batch_size.sh`` already defaults ``NUM_WORKER=8``.

**--point-max** (alias ``--point_max``) overrides SphereCrop ``point_max`` via
overlay keys ``point_max`` + ``override_point_max``. A top-level ``point_max``
assignment alone does *not* rewrite already-baked SphereCrop dicts;
``default_config_parser`` walks the merged pipeline when
``override_point_max`` is set. Default None inherits the source config.
Does not change GridSample packing. No-op (trainer warning) if the config
has no SphereCrop (Sonata SSL ``max_size``, most val/test pipelines).

Probe overlays replace ``hooks`` entirely (see ``build_hooks``), so side-effect
hooks such as ``LinProbeSbatchHook`` from the source config are never run.

Examples (JeanZay / local GPU)::

  # Train (supervised, Mix3D worst-case)
  python scripts/find_max_batch_size.py \\
    --config-file configs/experiment/w105/2/10h/litept-v1m0-flair3d_13.py \\
    --mode train --min-bs 2 --max-bs 32 --probe-steps 64 --soak-steps 500 \\
    --num-worker 8

  # Sonata SSL pretrain (no Mix3D)
  python scripts/find_max_batch_size.py \\
    --config-file configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py \\
    --mode train --min-bs 1 --max-bs 8 --probe-steps 32 --soak-steps 200 \\
    --mix-prob 0 --num-gpus 1 --num-worker 8

  # Sonata linear probe (Mix3D as in config)
  python scripts/find_max_batch_size.py \\
    --config-file configs/flair3d_default/probe/sonata-v1m2-flair3d-lin.py \\
    --mode train --min-bs 1 --max-bs 8 --probe-steps 32 --soak-steps 200 \\
    --mix-prob 0.8 --num-gpus 1 --num-worker 8

  # Sonata linear probe, only reasonable batch sizes (4..32), not a bisection
  python scripts/find_max_batch_size.py \\
    --config-file configs/flair3d_default/probe/sonata-v1m2-flair3d-lin.py \\
    --mode train --candidates 4 8 12 16 20 24 32 \\
    --probe-steps 32 --soak-steps 200 --mix-prob 0.8 --num-gpus 1 \\
    --num-worker 8

  # Val (capped samples, no Mix3D)
  python scripts/find_max_batch_size.py \\
    --config-file configs/experiment/w105/2/10h/litept-v1m0-flair3d_13.py \\
    --mode val --min-bs 1 --max-bs 16 --max-sample 128 --soak-steps 0 \\
    --num-worker 8

  # Test (builds a 1-step seed checkpoint, then probes tools/test.py)
  python scripts/find_max_batch_size.py \\
    --config-file configs/experiment/w105/2/10h/litept-v1m0-flair3d_13.py \\
    --mode test --min-bs 1 --max-bs 16 --max-sample 128 --soak-steps 0 \\
    --num-worker 8

  # Override SphereCrop budget (does not change GridSample packing)
  python scripts/find_max_batch_size.py \\
    --config-file configs/experiment/w101/1/grid_kpconvx/kpconvx-v1m0-dales-lin-grid-enc_1.py \\
    --mode train --min-bs 2 --max-bs 32 --probe-steps 32 --soak-steps 200 \\
    --mix-prob 0.8 --num-gpus 1 --num-worker 8 --point-max 102400
"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

OOM_PATTERNS = (
    "cuda out of memory",
    "out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "hip out of memory",
)

MAX_MEM_RE = re.compile(r"max_mem:\s*([0-9]+(?:\.[0-9]+)?)M")

EVALUATOR_CANDIDATES = (
    "MultiTaskEvaluator",
    "SemSegEvaluator",
    "ClsEvaluator",
    "RegressionEvaluator",
    "InsSegEvaluator",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def align_bs(bs: int, *, even: bool) -> int:
    bs = max(1, int(bs))
    if not even:
        return bs
    if bs < 2:
        return 2
    return bs if bs % 2 == 0 else bs - 1


def detect_evaluator_type(config_file: Path) -> str | None:
    text = config_file.read_text(encoding="utf-8")
    for name in EVALUATOR_CANDIDATES:
        if f'type="{name}"' in text or f"type='{name}'" in text:
            return name
    return None


def build_hooks(
    mode: str,
    evaluator_type: str | None,
    *,
    save_checkpoint: bool = False,
) -> list[dict]:
    """Minimal hook list for VRAM probes.

    Fully replaces the source config ``hooks`` in the overlay, so side-effect
    hooks (e.g. ``LinProbeSbatchHook``) are disabled during the search.
    """
    hooks: list[dict] = [
        dict(type="CheckpointLoader"),
        dict(type="ModelHook"),
        dict(type="IterationTimer", warmup_iter=2),
        dict(type="InformationWriter"),
    ]
    if mode == "val" and evaluator_type is not None:
        # Keep the same evaluator class as the source config when possible.
        if evaluator_type == "MultiTaskEvaluator":
            hooks.append(dict(type=evaluator_type, write_cls_iou=True))
        else:
            hooks.append(dict(type=evaluator_type))
    if save_checkpoint:
        hooks.append(dict(type="CheckpointSaver", save_freq=None))
    # PreciseEvaluator / LinProbeSbatchHook intentionally omitted.
    return hooks


def source_has_lin_probe_sbatch_hook(config_file: Path) -> bool:
    text = config_file.read_text(encoding="utf-8")
    return "LinProbeSbatchHook" in text


def write_probe_config(
    *,
    src_config: Path,
    dest: Path,
    overrides: dict,
) -> None:
    """Write a thin _base_ config so we do not re-import heavy deps here."""
    src_abs = str(src_config.resolve())
    lines = [
        "# Auto-generated by scripts/find_max_batch_size.py — do not edit.",
        f"_base_ = [{src_abs!r}]",
    ]
    for key, value in overrides.items():
        lines.append(f"{key} = {repr(value)}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_point_max_overrides(overrides: dict, args: argparse.Namespace) -> None:
    """Patch SphereCrop via overlay ``override_point_max`` (see default_config_parser).

    Top-level ``point_max`` alone is not enough: parent configs bake the integer
    into ``data.*.transform`` SphereCrop dicts at import time.
    """
    if args.point_max is None:
        return
    point_max = int(args.point_max)
    overrides["point_max"] = point_max
    overrides["override_point_max"] = point_max


def parse_peak_mem_mb(text: str) -> float | None:
    matches = MAX_MEM_RE.findall(text)
    if not matches:
        return None
    return max(float(x) for x in matches)


def is_oom(text: str, returncode: int) -> bool:
    low = text.lower()
    if any(p in low for p in OOM_PATTERNS):
        return True
    # Some CUDA OOMs still exit non-zero without a clear string in captured pipes.
    return False


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_s: int | None,
) -> tuple[int, str, float | None]:
    log(f"$ {shlex.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=None,
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        log_path.write_text(out, encoding="utf-8")
        raise
    out = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(out, encoding="utf-8")
    elapsed = time.perf_counter() - t0
    peak = parse_peak_mem_mb(out)
    log(
        f"  -> exit={proc.returncode} elapsed={elapsed:.1f}s "
        f"peak_mem={peak if peak is not None else 'n/a'}M "
        f"log={log_path}"
    )
    return proc.returncode, out, peak


def python_train_cmd(
    *,
    python: str,
    config_file: Path,
    num_gpus: int,
    extra_options: list[str],
) -> list[str]:
    cmd = [
        python,
        str(REPO_ROOT / "tools" / "train.py"),
        "--config-file",
        str(config_file),
        "--num-gpus",
        str(num_gpus),
    ]
    if extra_options:
        cmd.append("--options")
        cmd.extend(extra_options)
    return cmd


def python_test_cmd(
    *,
    python: str,
    config_file: Path,
    num_gpus: int,
    extra_options: list[str],
) -> list[str]:
    cmd = [
        python,
        str(REPO_ROOT / "tools" / "test.py"),
        "--config-file",
        str(config_file),
        "--num-gpus",
        str(num_gpus),
    ]
    if extra_options:
        cmd.append("--options")
        cmd.extend(extra_options)
    return cmd


def probe_train(
    *,
    args: argparse.Namespace,
    work_dir: Path,
    trial_name: str,
    batch_size: int,
    steps: int,
    evaluate: bool,
    batch_size_val: int | None,
    max_sample_val: int | None,
    evaluator_type: str | None,
) -> tuple[bool, float | None, str]:
    """Run a short train (optional val). Returns (ok, peak_mem_mb, status)."""
    save_path = work_dir / "runs" / trial_name
    cfg_path = work_dir / "configs" / f"{trial_name}.py"

    mode_hooks = "val" if evaluate else "train"
    overrides = {
        "batch_size": int(batch_size),
        "mix_prob": float(args.mix_prob),
        "total_iters": int(steps),
        "iter_per_epoch": int(steps),
        "evaluate": bool(evaluate),
        "enable_wandb": False,
        "save_path": str(save_path),
        "hooks": build_hooks(
            mode_hooks,
            evaluator_type,
            save_checkpoint=bool(getattr(args, "save_checkpoint", False)),
        ),
        "empty_cache": False,
        "empty_cache_per_epoch": False,
    }
    if args.num_worker is not None:
        overrides["num_worker"] = int(args.num_worker)
    apply_point_max_overrides(overrides, args)
    if batch_size_val is not None:
        overrides["batch_size_val"] = int(batch_size_val)
    if evaluate:
        overrides["eval_every"] = 1
        # One trainer epoch only; val runs after it.
        overrides["total_iters"] = 1
        overrides["iter_per_epoch"] = 1
        # Keep train BS tiny so VRAM is dominated by val.
        overrides["batch_size"] = max(1, int(args.val_train_batch_size))
        overrides["mix_prob"] = 0.0

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
    except subprocess.TimeoutExpired:
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
    batch_size_test: int,
    weight: Path,
    max_sample_test: int | None,
) -> tuple[bool, float | None, str]:
    save_path = work_dir / "runs" / trial_name
    cfg_path = work_dir / "configs" / f"{trial_name}.py"
    overrides = {
        "batch_size_test": int(batch_size_test),
        "enable_wandb": False,
        "save_path": str(save_path),
        "weight": str(weight),
    }
    if args.num_worker is not None:
        overrides["num_worker"] = int(args.num_worker)
    apply_point_max_overrides(overrides, args)
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
    except subprocess.TimeoutExpired:
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

    log("Building seed checkpoint (1 train step, mix_prob=0) for test probes...")
    seed_args = argparse.Namespace(**vars(args))
    seed_args.mix_prob = 0.0
    seed_args.save_checkpoint = True
    ok, _, status = probe_train(
        args=seed_args,
        work_dir=work_dir,
        trial_name="seed_train",
        batch_size=2 if args.even_bs else 1,
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
    bs: int,
    steps: int,
    evaluator_type: str | None,
    seed_weight: Path | None,
) -> tuple[bool, float | None, str]:
    if args.mode == "train":
        return probe_train(
            args=args,
            work_dir=work_dir,
            trial_name=trial_name,
            batch_size=bs,
            steps=steps,
            evaluate=False,
            batch_size_val=None,
            max_sample_val=None,
            evaluator_type=evaluator_type,
        )
    if args.mode == "val":
        return probe_train(
            args=args,
            work_dir=work_dir,
            trial_name=trial_name,
            batch_size=args.val_train_batch_size,
            steps=1,
            evaluate=True,
            batch_size_val=bs,
            max_sample_val=args.max_sample,
            evaluator_type=evaluator_type,
        )
    assert args.mode == "test"
    assert seed_weight is not None
    return probe_test(
        args=args,
        work_dir=work_dir,
        trial_name=trial_name,
        batch_size_test=bs,
        weight=seed_weight,
        max_sample_test=args.max_sample,
    )


def run_soak(
    *,
    args: argparse.Namespace,
    work_dir: Path,
    trial_name: str,
    bs: int,
    evaluator_type: str | None,
    seed_weight: Path | None,
) -> tuple[bool, float | None, str]:
    """One soak trial at a fixed bs: more train steps, or a larger eval sample floor."""
    if args.mode == "train":
        return run_trial(
            args=args,
            work_dir=work_dir,
            trial_name=trial_name,
            bs=bs,
            steps=args.soak_steps,
            evaluator_type=evaluator_type,
            seed_weight=seed_weight,
        )
    soak_samples = max(int(args.max_sample), int(args.soak_steps))
    if args.mode == "val":
        return probe_train(
            args=args,
            work_dir=work_dir,
            trial_name=trial_name,
            batch_size=args.val_train_batch_size,
            steps=1,
            evaluate=True,
            batch_size_val=bs,
            max_sample_val=soak_samples,
            evaluator_type=evaluator_type,
        )
    assert args.mode == "test"
    return probe_test(
        args=args,
        work_dir=work_dir,
        trial_name=trial_name,
        batch_size_test=bs,
        weight=seed_weight,
        max_sample_test=soak_samples,
    )


def candidate_search(args: argparse.Namespace, work_dir: Path) -> dict:
    """Sweep a fixed, explicit list of batch sizes instead of bisecting a range.

    Ascending order, probe each candidate, stop at the first probe OOM (VRAM
    grows monotonically with batch_size, so larger candidates are assumed
    worse). Then soak-verify from the largest probe-passing candidate
    downward through the same list (not arbitrary -2 decrements) until one
    survives.
    """
    even = bool(args.even_bs and args.mode == "train")
    candidates = sorted({align_bs(c, even=even) for c in args.candidates})

    evaluator_type = detect_evaluator_type(Path(args.config_file))
    seed_weight = None
    if args.mode == "test":
        seed_weight = ensure_seed_checkpoint(args, work_dir)

    trials: list[dict] = []
    passing: list[tuple[int, float | None]] = []

    log(f"Phase 1: candidate probe sweep mode={args.mode} candidates={candidates}")
    for i, bs in enumerate(candidates, start=1):
        name = f"probe_{i:02d}_bs{bs}"
        log(f"\n=== Probe {name} ===")
        ok, peak, status = run_trial(
            args=args,
            work_dir=work_dir,
            trial_name=name,
            bs=bs,
            steps=args.probe_steps,
            evaluator_type=evaluator_type,
            seed_weight=seed_weight,
        )
        trials.append(
            dict(phase="probe", trial=name, batch_size=bs, ok=ok, status=status, peak_mem_mb=peak)
        )
        if status.startswith("error_exit") or status == "timeout":
            raise RuntimeError(
                f"Non-OOM failure at bs={bs} status={status}. "
                f"Inspect {work_dir / 'logs' / (name + '.log')}"
            )
        if not ok:
            log(f"bs={bs} OOM on probe — stopping sweep (larger candidates assumed worse).")
            break
        passing.append((bs, peak))

    if not passing:
        log("No candidate batch size succeeded.")
        return dict(candidate_bs=None, confirmed_bs=None, peak_mem_mb=None, trials=trials)

    best, best_peak = passing[-1]
    log(f"\nPhase 1 candidate: batch_size={best} peak_mem={best_peak}")

    confirmed, confirmed_peak = best, best_peak
    if args.soak_steps > 0:
        log(f"\nPhase 2: soak-verify candidates (largest first, soak_steps={args.soak_steps})")
        confirmed, confirmed_peak = None, None
        for bs, _ in reversed(passing):
            name = f"soak_bs{bs}"
            log(f"\n=== Soak {name} ===")
            ok, peak, status = run_soak(
                args=args,
                work_dir=work_dir,
                trial_name=name,
                bs=bs,
                evaluator_type=evaluator_type,
                seed_weight=seed_weight,
            )
            trials.append(
                dict(phase="soak", trial=name, batch_size=bs, ok=ok, status=status, peak_mem_mb=peak)
            )
            if status.startswith("error_exit") or status == "timeout":
                raise RuntimeError(f"Non-OOM failure during soak bs={bs} status={status}")
            if ok:
                confirmed, confirmed_peak = bs, peak
                break
            log(f"bs={bs} OOM on soak, falling back to next smaller candidate")
        if confirmed is None:
            log("Soak failed for all probed candidates.")

    return dict(candidate_bs=best, confirmed_bs=confirmed, peak_mem_mb=confirmed_peak, trials=trials)


def binary_search(args: argparse.Namespace, work_dir: Path) -> dict:
    even = bool(args.even_bs and args.mode == "train")
    lo = align_bs(args.min_bs, even=even)
    hi = align_bs(args.max_bs, even=even)
    if lo > hi:
        raise ValueError(f"Invalid search range after align: lo={lo} hi={hi}")

    evaluator_type = detect_evaluator_type(Path(args.config_file))
    seed_weight = None
    if args.mode == "test":
        seed_weight = ensure_seed_checkpoint(args, work_dir)

    trials: list[dict] = []
    best = None
    best_peak = None
    step = 2 if even else 1
    trial_i = 0

    log(f"Phase 1: binary search mode={args.mode} range=[{lo}, {hi}] even={even}")
    while lo <= hi:
        mid = align_bs((lo + hi) // 2, even=even)
        if mid < lo:
            mid = lo
        trial_i += 1
        name = f"probe_{trial_i:02d}_bs{mid}"
        log(f"\n=== Probe {name} (lo={lo} hi={hi}) ===")
        ok, peak, status = run_trial(
            args=args,
            work_dir=work_dir,
            trial_name=name,
            bs=mid,
            steps=args.probe_steps,
            evaluator_type=evaluator_type,
            seed_weight=seed_weight,
        )
        trials.append(
            dict(
                phase="probe",
                trial=name,
                batch_size=mid,
                ok=ok,
                status=status,
                peak_mem_mb=peak,
            )
        )
        if status.startswith("error_exit") or status == "timeout":
            raise RuntimeError(
                f"Non-OOM failure at bs={mid} status={status}. "
                f"Inspect {work_dir / 'logs' / (name + '.log')}"
            )
        if ok:
            best = mid
            best_peak = peak
            lo = mid + step
        else:
            hi = mid - step

    if best is None:
        log("No batch size in range succeeded.")
        return dict(
            candidate_bs=None,
            confirmed_bs=None,
            peak_mem_mb=None,
            trials=trials,
        )

    log(f"\nPhase 1 candidate: batch_size={best} peak_mem={best_peak}")

    confirmed = best
    confirmed_peak = best_peak
    if args.soak_steps > 0 and args.mode == "train":
        log(f"\nPhase 2: soak up to {args.soak_steps} steps (mix_prob={args.mix_prob})")
        while confirmed is not None and confirmed >= align_bs(args.min_bs, even=even):
            name = f"soak_bs{confirmed}"
            log(f"\n=== Soak {name} ===")
            ok, peak, status = run_trial(
                args=args,
                work_dir=work_dir,
                trial_name=name,
                bs=confirmed,
                steps=args.soak_steps,
                evaluator_type=evaluator_type,
                seed_weight=seed_weight,
            )
            trials.append(
                dict(
                    phase="soak",
                    trial=name,
                    batch_size=confirmed,
                    ok=ok,
                    status=status,
                    peak_mem_mb=peak,
                )
            )
            if status.startswith("error_exit") or status == "timeout":
                raise RuntimeError(
                    f"Non-OOM failure during soak bs={confirmed} status={status}"
                )
            if ok:
                confirmed_peak = peak if peak is not None else confirmed_peak
                break
            confirmed = confirmed - step if confirmed - step >= align_bs(args.min_bs, even=even) else None
        if confirmed is None:
            log("Soak failed for all candidates.")
    elif args.soak_steps > 0 and args.mode in ("val", "test"):
        # Longer eval soak: re-run candidate with more samples.
        soak_samples = max(int(args.max_sample), int(args.soak_steps))
        name = f"soak_bs{confirmed}"
        log(f"\nPhase 2: eval soak max_sample={soak_samples}")
        if args.mode == "val":
            ok, peak, status = probe_train(
                args=args,
                work_dir=work_dir,
                trial_name=name,
                batch_size=args.val_train_batch_size,
                steps=1,
                evaluate=True,
                batch_size_val=confirmed,
                max_sample_val=soak_samples,
                evaluator_type=evaluator_type,
            )
        else:
            ok, peak, status = probe_test(
                args=args,
                work_dir=work_dir,
                trial_name=name,
                batch_size_test=confirmed,
                weight=seed_weight,
                max_sample_test=soak_samples,
            )
        trials.append(
            dict(
                phase="soak",
                trial=name,
                batch_size=confirmed,
                ok=ok,
                status=status,
                peak_mem_mb=peak,
            )
        )
        if status.startswith("error_exit") or status == "timeout":
            raise RuntimeError(
                f"Non-OOM failure during soak bs={confirmed} status={status}"
            )
        if not ok:
            confirmed = None
            confirmed_peak = peak
        else:
            confirmed_peak = peak if peak is not None else confirmed_peak

    return dict(
        candidate_bs=best,
        confirmed_bs=confirmed,
        peak_mem_mb=confirmed_peak,
        trials=trials,
    )


def write_csv(path: Path, args: argparse.Namespace, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config",
        "mode",
        "gpu_name",
        "candidate_bs",
        "confirmed_bs",
        "peak_mem_mb",
        "mix_prob",
        "point_max",
        "probe_steps",
        "soak_steps",
        "trial",
        "phase",
        "batch_size",
        "ok",
        "status",
        "trial_peak_mem_mb",
    ]
    gpu_name = _gpu_name()
    point_max = args.point_max if args.point_max is not None else ""
    rows = []
    if not result["trials"]:
        rows.append(
            {
                "config": args.config_file,
                "mode": args.mode,
                "gpu_name": gpu_name,
                "candidate_bs": result["candidate_bs"],
                "confirmed_bs": result["confirmed_bs"],
                "peak_mem_mb": result["peak_mem_mb"],
                "mix_prob": args.mix_prob,
                "point_max": point_max,
                "probe_steps": args.probe_steps,
                "soak_steps": args.soak_steps,
                "trial": "",
                "phase": "",
                "batch_size": "",
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
                    "candidate_bs": result["candidate_bs"],
                    "confirmed_bs": result["confirmed_bs"],
                    "peak_mem_mb": result["peak_mem_mb"],
                    "mix_prob": args.mix_prob if args.mode == "train" else 0,
                    "point_max": point_max,
                    "probe_steps": args.probe_steps,
                    "soak_steps": args.soak_steps,
                    "trial": t["trial"],
                    "phase": t["phase"],
                    "batch_size": t["batch_size"],
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


def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config-file", required=True, help="Path to Pointcept config .py")
    p.add_argument(
        "--mode",
        choices=("train", "val", "test"),
        default="train",
        help="Which batch_size_* to search",
    )
    p.add_argument("--min-bs", type=int, default=2)
    p.add_argument("--max-bs", type=int, default=32)
    p.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of batch sizes to sweep instead of bisecting "
        "[--min-bs, --max-bs] (e.g. --candidates 4 8 12 16 20 24 32). "
        "Ascending sweep, stops at the first probe OOM, then soak-verifies "
        "from the largest probe-passing candidate downward through the "
        "same list. --min-bs/--max-bs are ignored when this is set.",
    )
    p.add_argument(
        "--probe-steps",
        type=int,
        default=64,
        help="Train steps per dichotomie trial (train mode)",
    )
    p.add_argument(
        "--soak-steps",
        type=int,
        default=500,
        help="Train soak steps after dichotomie (0 to skip). "
        "For val/test, interpreted as a larger max_sample floor.",
    )
    p.add_argument(
        "--max-sample",
        type=int,
        default=128,
        help="Cap val/test samples during probes",
    )
    p.add_argument(
        "--mix-prob",
        type=float,
        default=1.0,
        help="Train Mix3D probability for probes (default 1.0 = worst case)",
    )
    p.add_argument(
        "--point-max",
        "--point_max",
        type=int,
        default=None,
        dest="point_max",
        help=(
            "Override SphereCrop point_max for this probe (default: inherit the "
            "source config). Sets overlay point_max + override_point_max so "
            "default_config_parser rewrites every SphereCrop in the merged "
            "pipeline. Does not change GridSample packing. No-op (with a "
            "warning from the trainer) if the config has no SphereCrop — e.g. "
            "Sonata SSL max_size, or most val/test pipelines."
        ),
    )
    p.add_argument(
        "--even-bs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force even batch sizes in train mode (Mix3D pairs); default on",
    )
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument(
        "--num-worker",
        type=int,
        default=None,
        help=(
            "Override config num_worker (strongly recommended on 1-GPU probes). "
            "Default None = inherit source config (dangerous when the config sets "
            "num_worker = 8 * num_gpu for multi-GPU training, e.g. Sonata → 256 "
            "workers and host-RAM OOM / 'DataLoader worker ... Killed'). "
            "Use 8 normally, or 0/2 for a VRAM-only smoke."
        ),
    )
    p.add_argument(
        "--val-train-batch-size",
        type=int,
        default=1,
        help="Tiny train BS used before val probe (VRAM should be val-dominated)",
    )
    p.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Directory for temp configs / logs / CSV (default under exp/)",
    )
    p.add_argument("--csv", type=str, default=None, help="Output CSV path")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-trial timeout in seconds (default: none)",
    )
    p.add_argument(
        "--extra-options",
        nargs="*",
        default=[],
        help="Extra Pointcept --options key=value pairs",
    )
    args = p.parse_args()
    if args.point_max is not None and args.point_max <= 0:
        p.error("--point-max must be a positive integer")
    # Internal flag used by seed checkpoint creation.
    if not hasattr(args, "save_checkpoint"):
        args.save_checkpoint = False
    return args


def main() -> int:
    args = parse_args()
    config_file = Path(args.config_file)
    if not config_file.is_file():
        # Allow repo-relative paths
        alt = REPO_ROOT / args.config_file
        if alt.is_file():
            args.config_file = str(alt)
            config_file = alt
        else:
            log(f"Config not found: {args.config_file}")
            return 2

    if args.mode == "train" and args.mix_prob <= 0:
        log(
            "NOTE: mix_prob<=0 disables Mix3D in this probe. Use only if the real "
            "training config also has mix_prob=0 (e.g. Sonata SSL); otherwise VRAM "
            "will be underestimated vs Mix3D training."
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    cfg_stem = config_file.stem
    work_dir = Path(
        args.work_dir
        or (REPO_ROOT / "exp" / "batch_size_search" / f"{cfg_stem}_{args.mode}_{stamp}")
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    log(f"Work dir: {work_dir}")
    log(f"Config: {config_file}")
    log(f"GPU: {_gpu_name()}")
    if args.point_max is not None:
        log(
            f"Override point_max={args.point_max} "
            "(patches every SphereCrop via override_point_max)"
        )
    if source_has_lin_probe_sbatch_hook(config_file):
        log(
            "NOTE: source config has LinProbeSbatchHook; probe overlays replace "
            "hooks entirely, so sbatch lin-probe submits are disabled during VRAM search."
        )

    try:
        if args.candidates:
            result = candidate_search(args, work_dir)
        else:
            result = binary_search(args, work_dir)
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 1

    csv_path = Path(args.csv) if args.csv else work_dir / "results.csv"
    write_csv(csv_path, args, result)

    log("\n======== SUMMARY ========")
    log(f"mode={args.mode}")
    log(f"candidate_bs={result['candidate_bs']}")
    log(f"confirmed_bs={result['confirmed_bs']}")
    log(f"peak_mem_mb={result['peak_mem_mb']}")
    if result["confirmed_bs"] is not None and args.mode == "train":
        log(
            f"Suggested config: batch_size = {result['confirmed_bs']} * num_gpu  "
            f"(probed with mix_prob={args.mix_prob}; keep your real mix_prob in training)"
        )
    elif result["confirmed_bs"] is not None and args.mode == "val":
        log(f"Suggested config: batch_size_val = {result['confirmed_bs']} * num_gpu")
    elif result["confirmed_bs"] is not None and args.mode == "test":
        log(f"Suggested config: batch_size_test = {result['confirmed_bs']} * num_gpu")
    else:
        log("No confirmed batch size — lower --min-bs / check logs.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
