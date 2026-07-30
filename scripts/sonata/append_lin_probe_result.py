#!/usr/bin/env python3
"""Append a Sonata linear-probe result row to the pretrain CSV / state files.

Called at the end of ``scripts/sonata/sbatch_lin_probe.sh`` when PRETRAIN_JOB_DIR is set.

Example:
  python scripts/sonata/append_lin_probe_result.py \\
    --pretrain_job_dir logs/slurm/123 \\
    --probe_job_dir logs/slurm/456 \\
    --ckpt /path/to/epoch_10.pth \\
    --pretrain_epoch 10 \\
    --pretrain_iters 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


CSV_FIELDS = [
    "pretrain_epoch",
    "pretrain_iters",
    "ckpt",
    "probe_job_dir",
    "best_val_mIoU",
    "status",
    "timestamp",
]
BEST_MIOU_LOG_RE = re.compile(r"Best\s+mIoU:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    candidates = list(probe_dir.glob("*.log"))
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain_job_dir", type=Path, required=True)
    parser.add_argument("--probe_job_dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--pretrain_epoch", type=int, default=0)
    parser.add_argument("--pretrain_iters", type=int, default=0)
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

    miou = _extract_miou(probe_dir)
    if args.train_exit_code == 0 and miou is not None:
        status = "ok"
    elif miou is not None:
        status = "ok"
    else:
        status = "failed"

    ckpt = args.ckpt
    ckpt_name = Path(ckpt).name if ckpt else ""
    row = {
        "pretrain_epoch": args.pretrain_epoch,
        "pretrain_iters": args.pretrain_iters,
        "ckpt": ckpt,
        "probe_job_dir": str(probe_dir),
        "best_val_mIoU": "" if miou is None else f"{miou:.6f}",
        "status": status,
        "timestamp": _utc_now(),
    }
    _append_csv(pretrain_dir / "lin_probe_results.csv", row)

    state_path = pretrain_dir / "lin_probe_state.json"
    state = _load_state(state_path)
    key = ckpt_name or f"epoch_{args.pretrain_epoch}.pth"
    state["in_flight"].pop(key, None)
    state["completed"][key] = {
        **row,
        "job_id": probe_dir.name,
    }
    _save_state(state_path, state)

    print(
        f"Appended probe result: {key} status={status} mIoU={row['best_val_mIoU']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
