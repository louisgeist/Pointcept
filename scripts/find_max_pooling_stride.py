#!/usr/bin/env python3
"""Sweep PT-v3 pooling ``stride`` candidates for a Sonata VRAM probe.

Companion to ``scripts/find_max_view_size.py``: that script binary-searches
``max_size`` (per-view point budget) at a fixed ``stride``. This script does
the opposite — it keeps ``max_size`` and ``batch_size_per_gpu`` FIXED and
instead sweeps ``model.backbone.stride`` (the PT-v3 encoder pooling stride,
see ``configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py``), which
controls how many points survive each encoder stage and is a much more
direct VRAM lever than ``max_size`` once the base stride (2,2,2,2) already
OOMs outright.

For each stride candidate, in the order given on the CLI:
  1. Short probe (``--probe-steps``). OOM -> move to the next candidate.
     Any other failure (non-OOM exit, timeout) aborts immediately, since
     that is a bug, not a VRAM problem.
  2. If the probe passes, a longer soak (``--soak-steps``, 0 to skip)
     re-runs the SAME stride for more steps to catch VRAM creep that a
     short probe can miss.
     - Soak OK -> stop. This is the finest stride that is confirmed safe.
     - Soak OOM -> move to the next (coarser) candidate.

Unlike ``find_max_view_size.py`` this is not a binary search: stride
candidates are a short, hand-picked, monotonically-coarser list, so a linear
sweep with early stop on first confirmed success is simpler and cheap enough.

Reuses the VRAM-probe plumbing (overlay config generation, hook list, OOM /
peak-mem parsing, MultiViewGenerator index resolution) from
``find_max_view_size.py`` instead of duplicating it — only the stride sweep
loop and CLI are specific to this script.

Examples::

  # Resolve the transform index + preview the first trial, no GPU needed
  python scripts/find_max_pooling_stride.py \\
    --config-file configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py \\
    --dry-run

  # Default sweep: (2,2,2,3) -> (2,2,3,3) -> (2,3,3,3) -> (3,3,3,3),
  # max_size=65536, batch_size_per_gpu=3
  python scripts/find_max_pooling_stride.py \\
    --config-file configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py

  # Fast smoke test on the debug config, probe only (no soak)
  python scripts/find_max_pooling_stride.py \\
    --config-file configs/flair3d_default/pretrain-sonata-v1m2-flair3d-debug.py \\
    --probe-steps 4 --soak-steps 0

  # Custom candidate list / fixed values
  python scripts/find_max_pooling_stride.py \\
    --config-file configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py \\
    --strides 2,2,2,3 2,3,3,3 3,3,3,3 \\
    --max-size 49152 --batch-size-per-gpu 3
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

import find_max_view_size as fmvs  # noqa: E402  (needs SCRIPTS_DIR on sys.path)


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_stride(text: str) -> tuple[int, ...]:
    parts = text.strip().strip("()[]").split(",")
    try:
        return tuple(int(p.strip()) for p in parts if p.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid stride {text!r}: {exc}") from exc


def build_trial_options(
    args: argparse.Namespace, transform_idx: int, stride: tuple[int, ...]
) -> list[str]:
    options = [f"data.train.transform.{transform_idx}.max_size={int(args.max_size)}"]
    if args.force_worst_case_scale:
        options.append(f"data.train.transform.{transform_idx}.global_view_scale=[1.0,1.0]")
    stride_str = "(" + ",".join(str(s) for s in stride) + ")"
    options.append(f"{args.stride_key}={stride_str}")
    options.extend(args.extra_options)
    return options


def probe_stride(
    *,
    args: argparse.Namespace,
    work_dir: Path,
    trial_name: str,
    stride: tuple[int, ...],
    steps: int,
    transform_idx: int,
) -> tuple[bool, float | None, str]:
    """Run a short train probe at a given stride. Returns (ok, peak_mem_mb, status)."""
    save_path = work_dir / "runs" / trial_name
    cfg_path = work_dir / "configs" / f"{trial_name}.py"

    overrides = {
        "batch_size": int(args.batch_size_per_gpu),
        "mix_prob": float(args.mix_prob),
        "total_iters": int(steps),
        "iter_per_epoch": int(steps),
        "evaluate": False,
        "enable_wandb": False,
        "save_path": str(save_path),
        "hooks": fmvs.build_hooks(),
        "empty_cache": False,
        "empty_cache_per_epoch": False,
    }
    if args.num_worker is not None:
        overrides["num_worker"] = int(args.num_worker)

    fmvs.write_probe_config(
        src_config=Path(args.config_file),
        dest=cfg_path,
        overrides=overrides,
        generator="scripts/find_max_pooling_stride.py",
    )

    options = build_trial_options(args, transform_idx, stride)
    cmd = fmvs.python_train_cmd(
        python=args.python,
        config_file=cfg_path,
        num_gpus=args.num_gpus,
        extra_options=options,
    )
    log_path = work_dir / "logs" / f"{trial_name}.log"
    try:
        code, out, peak = fmvs.run_cmd(cmd, cwd=REPO_ROOT, log_path=log_path, timeout_s=args.timeout)
    except subprocess.TimeoutExpired:
        return False, None, "timeout"

    if code == 0:
        return True, peak, "ok"
    if fmvs.is_oom(out, code):
        return False, peak, "oom"
    return False, peak, f"error_exit_{code}"


def stride_str(stride: tuple[int, ...]) -> str:
    return "-".join(str(s) for s in stride)


def sweep(args: argparse.Namespace, work_dir: Path, transform_idx: int) -> dict:
    trials: list[dict] = []
    confirmed: tuple[int, ...] | None = None
    confirmed_peak: float | None = None

    log(
        f"Sweeping {len(args.strides)} stride candidates (in order): "
        f"{[stride_str(s) for s in args.strides]}"
    )
    for i, stride in enumerate(args.strides, start=1):
        s_str = stride_str(stride)
        probe_name = f"probe_{i:02d}_stride{s_str}"
        log(f"\n=== Probe {probe_name} (stride={stride}) ===")
        ok, peak, status = probe_stride(
            args=args,
            work_dir=work_dir,
            trial_name=probe_name,
            stride=stride,
            steps=args.probe_steps,
            transform_idx=transform_idx,
        )
        trials.append(
            dict(phase="probe", trial=probe_name, stride=s_str, ok=ok, status=status, peak_mem_mb=peak)
        )
        if not ok:
            if status != "oom":
                raise RuntimeError(
                    f"Non-OOM failure at stride={stride} status={status}. "
                    f"Inspect {work_dir / 'logs' / (probe_name + '.log')}"
                )
            log(f"stride={stride} OOM on probe, trying next candidate")
            continue

        if args.soak_steps == 0:
            log(f"\nConfirmed (no soak): stride={stride} peak_mem={peak}")
            confirmed, confirmed_peak = stride, peak
            break

        soak_name = f"soak_{i:02d}_stride{s_str}"
        log(f"\n=== Soak {soak_name} (stride={stride}, {args.soak_steps} steps) ===")
        soak_ok, soak_peak, soak_status = probe_stride(
            args=args,
            work_dir=work_dir,
            trial_name=soak_name,
            stride=stride,
            steps=args.soak_steps,
            transform_idx=transform_idx,
        )
        trials.append(
            dict(
                phase="soak",
                trial=soak_name,
                stride=s_str,
                ok=soak_ok,
                status=soak_status,
                peak_mem_mb=soak_peak,
            )
        )
        if soak_ok:
            log(f"\nConfirmed: stride={stride} peak_mem={soak_peak}")
            confirmed, confirmed_peak = stride, soak_peak
            break
        if soak_status != "oom":
            raise RuntimeError(
                f"Non-OOM failure during soak stride={stride} status={soak_status}. "
                f"Inspect {work_dir / 'logs' / (soak_name + '.log')}"
            )
        log(f"stride={stride} passed probe but OOM on soak, trying next candidate")

    return dict(confirmed_stride=confirmed, peak_mem_mb=confirmed_peak, trials=trials)


def write_csv(path: Path, args: argparse.Namespace, result: dict, transform_idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config",
        "gpu_name",
        "transform_index",
        "stride_key",
        "max_size",
        "batch_size_per_gpu",
        "mix_prob",
        "force_worst_case_scale",
        "confirmed_stride",
        "peak_mem_mb",
        "probe_steps",
        "soak_steps",
        "trial",
        "phase",
        "stride",
        "ok",
        "status",
        "trial_peak_mem_mb",
    ]
    common = {
        "config": args.config_file,
        "gpu_name": fmvs._gpu_name(),
        "transform_index": transform_idx,
        "stride_key": args.stride_key,
        "max_size": args.max_size,
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "mix_prob": args.mix_prob,
        "force_worst_case_scale": args.force_worst_case_scale,
        "confirmed_stride": stride_str(result["confirmed_stride"]) if result["confirmed_stride"] else "",
        "peak_mem_mb": result["peak_mem_mb"],
        "probe_steps": args.probe_steps,
        "soak_steps": args.soak_steps,
    }
    rows = []
    if not result["trials"]:
        rows.append(
            {**common, "trial": "", "phase": "", "stride": "", "ok": "", "status": "", "trial_peak_mem_mb": ""}
        )
    else:
        for t in result["trials"]:
            rows.append(
                {
                    **common,
                    "trial": t["trial"],
                    "phase": t["phase"],
                    "stride": t["stride"],
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


def preview_trial(args: argparse.Namespace, work_dir: Path, transform_idx: int) -> None:
    stride = args.strides[0]
    name = f"dry_run_stride{stride_str(stride)}"
    save_path = work_dir / "runs" / name
    cfg_path = work_dir / "configs" / f"{name}.py"
    overrides = {
        "batch_size": int(args.batch_size_per_gpu),
        "mix_prob": float(args.mix_prob),
        "total_iters": int(args.probe_steps),
        "iter_per_epoch": int(args.probe_steps),
        "evaluate": False,
        "enable_wandb": False,
        "save_path": str(save_path),
        "hooks": fmvs.build_hooks(),
        "empty_cache": False,
        "empty_cache_per_epoch": False,
    }
    if args.num_worker is not None:
        overrides["num_worker"] = int(args.num_worker)
    fmvs.write_probe_config(
        src_config=Path(args.config_file),
        dest=cfg_path,
        overrides=overrides,
        generator="scripts/find_max_pooling_stride.py",
    )
    options = build_trial_options(args, transform_idx, stride)
    cmd = fmvs.python_train_cmd(
        python=args.python, config_file=cfg_path, num_gpus=args.num_gpus, extra_options=options
    )

    log(f"Resolved transform index: {transform_idx}")
    log(f"Stride sweep order: {[stride_str(s) for s in args.strides]} (first trial: stride={stride})")
    log(f"\n--- Overlay config: {cfg_path} ---")
    log(cfg_path.read_text(encoding="utf-8"))
    log(f"--- Command that would run ---\n$ {shlex.join(cmd)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config-file", required=True, help="Path to Pointcept config .py")
    p.add_argument(
        "--transform-type",
        type=str,
        default="MultiViewGenerator",
        help="Transform type to locate in data.train.transform (for injecting max_size)",
    )
    p.add_argument(
        "--transform-index",
        type=int,
        default=None,
        help="Manual override if auto-detection is ambiguous/fails",
    )
    p.add_argument(
        "--strides",
        type=parse_stride,
        nargs="+",
        default=[parse_stride(s) for s in ("2,2,2,3", "2,2,3,3", "2,3,3,3", "3,3,3,3")],
        help="Stride candidates to sweep, in order, as comma-separated ints "
        "(e.g. --strides 2,2,2,3 2,2,3,3). Base stride (2,2,2,2) is not "
        "included by default since it OOMs outright.",
    )
    p.add_argument(
        "--stride-key",
        type=str,
        default="model.backbone.stride",
        help="Dotted config key for the backbone pooling stride",
    )
    p.add_argument(
        "--max-size",
        type=int,
        default=65536,
        help="Fixed MultiViewGenerator max_size for every trial (not searched)",
    )
    p.add_argument(
        "--batch-size-per-gpu",
        type=int,
        default=3,
        help="Fixed batch_size_per_gpu for every trial (not searched)",
    )
    p.add_argument(
        "--mix-prob",
        type=float,
        default=0.0,
        help="Train Mix3D probability (default 0 = Sonata SSL pretrain, no Mix3D)",
    )
    p.add_argument(
        "--force-worst-case-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force global_view_scale=[1.0,1.0] per trial so views deterministically hit "
        "max_size points (default on); disable to measure the average/stochastic case",
    )
    p.add_argument(
        "--probe-steps", type=int, default=16, help="Short train steps per candidate"
    )
    p.add_argument(
        "--soak-steps",
        type=int,
        default=300,
        help="Longer re-verification steps on a probe success before confirming a candidate "
        "(0 to skip and confirm on probe alone)",
    )
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--num-worker", type=int, default=None, help="Override num_worker")
    p.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Directory for temp configs / logs / CSV (default under exp/)",
    )
    p.add_argument("--csv", type=str, default=None, help="Output CSV path")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument(
        "--timeout", type=int, default=None, help="Per-trial timeout in seconds (default: none)"
    )
    p.add_argument(
        "--extra-options",
        nargs="*",
        default=[],
        help="Extra Pointcept --options key=value pairs",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the transform index and preview the first trial, then exit (no subprocess)",
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

    if args.mix_prob != 0:
        log(
            "NOTE: mix_prob!=0 — this script defaults to 0 for Sonata SSL pretrain "
            "(no Mix3D). Only pass a non-zero value if the real training config does too."
        )

    try:
        transform_idx = fmvs.resolve_transform_index(
            config_file, args.transform_type, args.transform_index
        )
    except Exception as exc:
        log(f"ERROR resolving transform index: {exc}")
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    cfg_stem = config_file.stem
    work_dir = Path(
        args.work_dir or (REPO_ROOT / "exp" / "pooling_stride_search" / f"{cfg_stem}_{stamp}")
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    log(f"Work dir: {work_dir}")
    log(f"Config: {config_file}")
    log(f"GPU: {fmvs._gpu_name()}")
    log(
        f"transform_index={transform_idx} max_size={args.max_size} "
        f"batch_size_per_gpu={args.batch_size_per_gpu} "
        f"force_worst_case_scale={args.force_worst_case_scale}"
    )

    if args.dry_run:
        preview_trial(args, work_dir, transform_idx)
        return 0

    try:
        result = sweep(args, work_dir, transform_idx)
    except Exception as exc:
        log(f"ERROR: {exc}")
        return 1

    csv_path = Path(args.csv) if args.csv else work_dir / "results.csv"
    write_csv(csv_path, args, result, transform_idx)

    log("\n======== SUMMARY ========")
    log(f"confirmed_stride={result['confirmed_stride']}")
    log(f"peak_mem_mb={result['peak_mem_mb']}")
    if result["confirmed_stride"] is not None:
        log(
            f"Suggested config: {args.stride_key} = {result['confirmed_stride']}  "
            f"(probed with max_size={args.max_size}, batch_size_per_gpu={args.batch_size_per_gpu})"
        )
    else:
        log("No stride candidate confirmed — widen --strides / check logs.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
