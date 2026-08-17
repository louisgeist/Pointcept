#!/usr/bin/env python3
"""Append a Sonata grid-probe winner row to the pretrain CSV.

Called at the end of ``scripts/sonata/sbatch_lin_grid_probe_mini_h100.sh``
when PRETRAIN_JOB_DIR is set. Concurrent array tasks flock the CSV.

Example:
  python scripts/sonata/append_grid_probe_result.py \\
    --pretrain_job_dir logs/slurm/862680 \\
    --probe_job_dir logs/slurm/456 \\
    --ckpt /path/to/epoch_10.pth \\
    --pretrain_epoch 10
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


CSV_FIELDS = [
    "pretrain_epoch",
    "best_val_mIoU",
    "best_config",
    "probe_job_dir",
    "status",
    "timestamp",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_grid_search_results(probe_dir: Path) -> Tuple[Optional[float], str]:
    path = probe_dir / "grid_search_results.json"
    if not path.is_file():
        return None, ""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    winner = data.get("winner") or {}
    miou = winner.get("best_val_mIoU")
    name = winner.get("probe_name") or ""
    if miou is None:
        return None, str(name)
    return float(miou), str(name)


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


def _append_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            write_header = os.fstat(f.fileno()).st_size == 0
            f.seek(0, os.SEEK_END)
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain_job_dir", type=Path, required=True)
    parser.add_argument("--probe_job_dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--pretrain_epoch", type=int, default=0)
    parser.add_argument(
        "--train_exit_code",
        type=int,
        default=0,
        help="Exit code of the probe train.sh (non-zero => failed unless metrics exist)",
    )
    args = parser.parse_args()

    pretrain_dir = args.pretrain_job_dir.resolve()
    probe_dir = args.probe_job_dir.resolve()
    if not pretrain_dir.is_dir():
        print(f"ERROR: pretrain_job_dir not found: {pretrain_dir}", file=sys.stderr)
        return 1

    miou, best_config = _read_grid_search_results(probe_dir)
    if miou is None:
        miou = _read_metrics_json(probe_dir)

    if args.train_exit_code == 0 and miou is not None:
        status = "ok"
    elif miou is not None:
        status = "ok"
    else:
        status = "failed"

    row = {
        "pretrain_epoch": args.pretrain_epoch,
        "best_val_mIoU": "" if miou is None else f"{miou:.6f}",
        "best_config": best_config,
        "probe_job_dir": str(probe_dir),
        "status": status,
        "timestamp": _utc_now(),
    }
    _append_csv(pretrain_dir / "grid_probe_results.csv", row)

    ckpt_name = Path(args.ckpt).name if args.ckpt else f"epoch_{args.pretrain_epoch}.pth"
    print(
        f"Appended grid-probe result: {ckpt_name} status={status} "
        f"mIoU={row['best_val_mIoU']} config={best_config or '<none>'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
