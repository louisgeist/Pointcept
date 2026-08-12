#!/usr/bin/env python3
"""Watch Sonata pretrain checkpoints and launch short linear-probe jobs.

Primary metric source after each probe: ``<probe_job_dir>/metrics.json``.
Results are appended to ``<pretrain_job_dir>/lin_probe_results.csv`` for live
monitoring (``tail -f``), independent of wandb sync.

Examples:
  # Local (subprocess train.sh)
  python scripts/sonata/periodic_lin_probe.py \\
    --pretrain_job_dir exp/flair3d_default/sonata_pretrain_flair3dplus \\
    --mode local --gpus 1

  # Jean-Zay (sbatch probe jobs)
  python scripts/sonata/periodic_lin_probe.py \\
    --pretrain_job_dir logs/slurm/<PRETRAIN_JOB_ID> \\
    --mode sbatch --lin_sbatch scripts/sonata/sbatch_lin_probe.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


EPOCH_CKPT_RE = re.compile(r"^epoch_(\d+)\.pth$")
BEST_MIOU_LOG_RE = re.compile(r"Best\s+mIoU:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
CSV_FIELDS = [
    "pretrain_epoch",
    "pretrain_iters",
    "ckpt",
    "probe_job_dir",
    "best_val_mIoU",
    "status",
    "timestamp",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"completed": {}, "in_flight": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("completed", {})
    data.setdefault("in_flight", {})
    return data


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _discover_epoch_ckpts(model_dir: Path) -> List[Path]:
    ckpts = []
    if not model_dir.is_dir():
        return ckpts
    for p in sorted(model_dir.iterdir()):
        m = EPOCH_CKPT_RE.match(p.name)
        if m and p.is_file():
            ckpts.append(p)
    return ckpts


def _parse_epoch(ckpt: Path) -> int:
    m = EPOCH_CKPT_RE.match(ckpt.name)
    if not m:
        raise ValueError(f"Not an epoch checkpoint: {ckpt}")
    return int(m.group(1))


def _read_metrics_json(probe_dir: Path) -> Optional[float]:
    path = probe_dir / "metrics.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "best_val_mIoU" in data:
        return float(data["best_val_mIoU"])
    if data.get("metric_name") == "mIoU" and "best_metric_value" in data:
        return float(data["best_metric_value"])
    return None


def _parse_miou_from_logs(probe_dir: Path) -> Optional[float]:
    """Fallback: last 'Best mIoU:' line in common log locations."""
    candidates: List[Path] = []
    for pattern in ("*.log", "train.log", "log.txt"):
        candidates.extend(probe_dir.glob(pattern))
    # Jean-Zay: sibling slurm.out under JOB_DIR
    out = probe_dir / "slurm.out"
    if out.is_file():
        candidates.append(out)
    best: Optional[float] = None
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in BEST_MIOU_LOG_RE.finditer(text):
            best = float(m.group(1))
    return best


def _extract_miou(probe_dir: Path) -> Optional[float]:
    miou = _read_metrics_json(probe_dir)
    if miou is not None:
        return miou
    return _parse_miou_from_logs(probe_dir)


def _append_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    write_header = not csv_path.is_file()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _launch_local(
    ckpt: Path,
    exp_name: str,
    gpus: int,
    dataset: str,
    config: str,
) -> Path:
    env = os.environ.copy()
    # Keep probe under pretrain sibling unless JOB_DIR is forced by caller.
    cmd = [
        "sh",
        str(_repo_root() / "scripts" / "train.sh"),
        "-g",
        str(gpus),
        "-d",
        dataset,
        "-c",
        config,
        "-n",
        exp_name,
        "-w",
        str(ckpt),
    ]
    print(f"[local] launching: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(_repo_root()), env=env, check=True)
    return _repo_root() / "exp" / dataset / exp_name


def _launch_sbatch(
    ckpt: Path,
    exp_name: str,
    sbatch_script: Path,
) -> str:
    env = os.environ.copy()
    env["WEIGHT"] = str(ckpt.resolve())
    env["EXP_NAME"] = exp_name
    # IMAGINE sbatch wrapper accepts only the script path (no --comment/--export).
    # Tags: `#SBATCH --comment=...` in the script; WEIGHT/EXP_NAME via env below.
    cmd = ["sbatch", str(sbatch_script)]
    print(f"[sbatch] launching: WEIGHT={env['WEIGHT']} EXP_NAME={exp_name}", flush=True)
    out = subprocess.check_output(cmd, cwd=str(_repo_root()), env=env, text=True)
    # "Submitted batch job 123456"
    m = re.search(r"(\d+)", out.strip())
    if not m:
        raise RuntimeError(f"Could not parse sbatch job id from: {out!r}")
    job_id = m.group(1)
    print(f"[sbatch] submitted job {job_id}", flush=True)
    return job_id


def _sbatch_job_done(job_id: str) -> bool:
    try:
        out = subprocess.check_output(
            ["sacct", "-j", job_id, "-n", "-X", "-o", "State"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: squeue empty means done/not found
        try:
            q = subprocess.check_output(
                ["squeue", "-j", job_id, "-h", "-o", "%i"],
                text=True,
            ).strip()
            return q == ""
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    states = {line.strip().split()[0] for line in out.splitlines() if line.strip()}
    if not states:
        return True
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED"}
    return bool(states) and states.issubset(terminal)


def _sbatch_job_ok(job_id: str) -> bool:
    try:
        out = subprocess.check_output(
            ["sacct", "-j", job_id, "-n", "-X", "-o", "State"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    states = [line.strip().split()[0] for line in out.splitlines() if line.strip()]
    return bool(states) and all(s == "COMPLETED" for s in states)


def _probe_job_dir_from_slurm(job_id: str, logs_root: Path) -> Path:
    return logs_root / job_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrain_job_dir",
        type=Path,
        required=True,
        help="Pretrain experiment / JOB_DIR (contains model/epoch_*.pth)",
    )
    parser.add_argument(
        "--mode",
        choices=("local", "sbatch"),
        default="sbatch",
    )
    parser.add_argument(
        "--lin_sbatch",
        type=Path,
        default=Path("scripts/sonata/sbatch_lin_probe.sh"),
        help="Sbatch script for linear probes (mode=sbatch)",
    )
    parser.add_argument("--gpus", type=int, default=1, help="GPUs for mode=local")
    parser.add_argument(
        "--dataset",
        default="flair3d_default",
        help="train.sh -d value",
    )
    parser.add_argument(
        "--config",
        default="probe/sonata-v1m2-flair3d-lin",
        help="train.sh -c value",
    )
    parser.add_argument(
        "--poll_seconds",
        type=float,
        default=120.0,
        help="Sleep between directory polls",
    )
    parser.add_argument(
        "--iter_per_epoch",
        type=int,
        default=1000,
        help="Pretrain iter_per_epoch (for CSV pretrain_iters column)",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=0,
        help="Max in-flight probe jobs (0 = unlimited; submit as soon as ckpts appear)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process currently available checkpoints once, then exit",
    )
    parser.add_argument(
        "--slurm_logs_root",
        type=Path,
        default=None,
        help="Root for logs/slurm/<jobid> (default: <repo>/logs/slurm)",
    )
    args = parser.parse_args()

    pretrain_dir = args.pretrain_job_dir.resolve()
    model_dir = pretrain_dir / "model"
    state_path = pretrain_dir / "lin_probe_state.json"
    csv_path = pretrain_dir / "lin_probe_results.csv"
    logs_root = (
        args.slurm_logs_root.resolve()
        if args.slurm_logs_root is not None
        else (_repo_root() / "logs" / "slurm")
    )
    sbatch_script = args.lin_sbatch
    if not sbatch_script.is_absolute():
        sbatch_script = (_repo_root() / sbatch_script).resolve()

    if not pretrain_dir.is_dir():
        print(f"ERROR: pretrain_job_dir not found: {pretrain_dir}", file=sys.stderr)
        return 1

    state = _load_state(state_path)
    print(f"Watching {model_dir} (mode={args.mode})", flush=True)
    print(f"Results CSV: {csv_path}", flush=True)

    while True:
        # Reap finished in-flight sbatch jobs
        finished_keys: List[str] = []
        for ckpt_name, meta in list(state.get("in_flight", {}).items()):
            if args.mode != "sbatch":
                continue
            job_id = str(meta.get("job_id", ""))
            if not job_id or not _sbatch_job_done(job_id):
                continue
            probe_dir = _probe_job_dir_from_slurm(job_id, logs_root)
            ok = _sbatch_job_ok(job_id)
            miou = _extract_miou(probe_dir) if ok else None
            epoch = int(meta.get("pretrain_epoch", 0))
            status = "ok" if (ok and miou is not None) else "failed"
            row = {
                "pretrain_epoch": epoch,
                "pretrain_iters": epoch * args.iter_per_epoch,
                "ckpt": meta.get("ckpt", ckpt_name),
                "probe_job_dir": str(probe_dir),
                "best_val_mIoU": "" if miou is None else f"{miou:.6f}",
                "status": status,
                "timestamp": _utc_now(),
            }
            _append_csv(csv_path, row)
            state["completed"][ckpt_name] = row
            finished_keys.append(ckpt_name)
            print(
                f"[done] {ckpt_name} status={status} mIoU={row['best_val_mIoU']} "
                f"dir={probe_dir}",
                flush=True,
            )
        for k in finished_keys:
            state["in_flight"].pop(k, None)
        if finished_keys:
            _save_state(state_path, state)

        in_flight_n = len(state.get("in_flight", {}))
        # 0 = unlimited: submit every pending checkpoint as soon as it appears.
        if args.max_concurrent <= 0:
            slots = None  # unlimited
        else:
            slots = max(0, args.max_concurrent - in_flight_n)
        launched = 0
        for ckpt in _discover_epoch_ckpts(model_dir):
            if slots is not None and slots <= 0:
                break
            name = ckpt.name
            if name in state.get("completed", {}) or name in state.get("in_flight", {}):
                continue
            epoch = _parse_epoch(ckpt)
            exp_name = f"sonata_lin_ep{epoch}"
            try:
                if args.mode == "local":
                    probe_dir = _launch_local(
                        ckpt=ckpt,
                        exp_name=exp_name,
                        gpus=args.gpus,
                        dataset=args.dataset,
                        config=args.config,
                    )
                    miou = _extract_miou(probe_dir)
                    status = "ok" if miou is not None else "failed"
                    row = {
                        "pretrain_epoch": epoch,
                        "pretrain_iters": epoch * args.iter_per_epoch,
                        "ckpt": str(ckpt),
                        "probe_job_dir": str(probe_dir),
                        "best_val_mIoU": "" if miou is None else f"{miou:.6f}",
                        "status": status,
                        "timestamp": _utc_now(),
                    }
                    _append_csv(csv_path, row)
                    state["completed"][name] = row
                    _save_state(state_path, state)
                    print(
                        f"[done] {name} status={status} mIoU={row['best_val_mIoU']}",
                        flush=True,
                    )
                else:
                    job_id = _launch_sbatch(ckpt, exp_name, sbatch_script)
                    state["in_flight"][name] = {
                        "job_id": job_id,
                        "ckpt": str(ckpt),
                        "pretrain_epoch": epoch,
                        "submitted_at": _utc_now(),
                    }
                    _save_state(state_path, state)
                    if slots is not None:
                        slots -= 1
                    launched += 1
            except Exception as exc:  # noqa: BLE001 — keep watcher alive
                row = {
                    "pretrain_epoch": epoch,
                    "pretrain_iters": epoch * args.iter_per_epoch,
                    "ckpt": str(ckpt),
                    "probe_job_dir": "",
                    "best_val_mIoU": "",
                    "status": f"launch_error:{exc}",
                    "timestamp": _utc_now(),
                }
                _append_csv(csv_path, row)
                state["completed"][name] = row
                _save_state(state_path, state)
                print(f"[error] launch failed for {name}: {exc}", flush=True)

        if args.once and launched == 0 and not state.get("in_flight"):
            print("No pending checkpoints; --once exit.", flush=True)
            return 0

        if args.once and state.get("in_flight"):
            # Wait for in-flight before exit
            time.sleep(args.poll_seconds)
            continue

        if args.once:
            return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
