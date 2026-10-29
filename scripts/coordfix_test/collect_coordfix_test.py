"""Collect every test metric of the coordfix_test jobs (and optional reference jobs) into CSVs.

Parses ``logs/slurm/<job>/slurm.err`` (MultiTaskTester lines) and
``logs/slurm/<job>/result/network_apls_metrics.json`` (NetworkAPLSEvaluator). Writes
``<out>/metrics_long.csv`` (job, model, data, task, metric, value) and ``<out>/metrics_wide.csv``
(one row per task/metric, one column per job).

Usage (repo root on Jean Zay):
    python scripts/coordfix_test/collect_coordfix_test.py              # auto-find coordfix_test jobs
    python scripts/coordfix_test/collect_coordfix_test.py --jobs 1700001 1700002 \
        --ref-jobs 873542 2126589                                     # + old quantized-data runs
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

LOGS = Path("logs/slurm")
FLOAT = r"(nan|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
PATTERNS = [
    (re.compile(rf"\[task=(\S+)\] Test result: mIoU/mAcc/allAcc {FLOAT}/{FLOAT}/{FLOAT}"),
     ["mIoU", "mAcc", "allAcc"]),
    (re.compile(rf"\[task=(\S+)\] Class_\d+-(.+?) Result: iou/accuracy {FLOAT}/{FLOAT}"),
     ["IoU/{c}", "Acc/{c}"]),
    (re.compile(rf"\[task=(\S+)\] Channel (\S+) Test result: precision/recall/f1 {FLOAT}/{FLOAT}/{FLOAT}"
                rf"(?: \(.*?\))?(?: \| dilated\(r=(\d+)px\) precision/recall/f1 {FLOAT}/{FLOAT}/{FLOAT})?"),
     None),  # handled below
    (re.compile(rf"\[task=(\S+)\] Test regression: MAE {FLOAT} RMSE {FLOAT}"), ["MAE", "RMSE"]),
    (re.compile(rf"\[task=(\S+)\] Test tile-distribution: weighted KL {FLOAT} TV {FLOAT}"), ["KL", "TV"]),
    (re.compile(rf"\[task=(\S+)\] Class_\d+-(.+?) Result: mae {FLOAT}"), ["MAE/{c}"]),
]


def job_info(job):
    cfg = LOGS / str(job) / "config.py"
    text = cfg.read_text() if cfg.exists() else ""
    # Last assignment wins (coordfix_test configs override both at the end of the file).
    names = re.findall(r'^wandb_run_name\s*=\s*\(?\s*f?"([^"]+)"', text, flags=re.M)
    roots = re.findall(r'^data_root\s*=\s*"([^"]+)"', text, flags=re.M)
    return (names[-1] if names else str(job)), (roots[-1] if roots else "?")


def parse_job(job):
    rows = []
    err = LOGS / str(job) / "slurm.err"
    if not err.exists():
        print(f"[warn] {err} missing")
        return rows
    lines = err.read_text(errors="replace").splitlines()
    # Keep only the last evaluation block (a requeued job may hold several).
    starts = [i for i, l in enumerate(lines) if "Start Multi-task Evaluation" in l]
    if not starts:
        print(f"[warn] job {job}: no test block (still running or crashed?)")
        return rows
    block = lines[starts[-1]:]
    if not any("End Evaluation" in l for l in block):
        print(f"[warn] job {job}: test block not finished")
    for line in block:
        for pat, names in PATTERNS:
            m = pat.search(line)
            if not m:
                continue
            g = m.groups()
            task = g[0]
            if names is None:  # channel P/R/F1 (+ optional dilated)
                ch = g[1]
                for k, v in zip(["P", "R", "F1"], g[2:5]):
                    rows.append((task, f"{ch}/{k}", float(v)))
                if g[5] is not None:
                    for k, v in zip(["P", "R", "F1"], g[6:9]):
                        rows.append((task, f"{ch}/buf{g[5]}px_{k}", float(v)))
            elif "{c}" in names[0]:
                cls = g[1]
                for n, v in zip(names, g[2:]):
                    rows.append((task, n.format(c=cls), float(v)))
            else:
                for n, v in zip(names, g[1:]):
                    rows.append((task, n, float(v)))
            break
    apls = LOGS / str(job) / "result" / "network_apls_metrics.json"
    if apls.exists():
        d = json.loads(apls.read_text().replace("NaN", "null"))
        rows.append(("network", "APLS_macro", d.get("macro_apls")))
        for key in ["per_channel", "per_channel_gt_to_pred", "per_channel_pred_to_gt"]:
            for ch, v in (d.get(key) or {}).items():
                if v is not None:
                    rows.append(("network", f"{ch}/APLS{key.removeprefix('per_channel')}", v))
        rows.append(("network", "APLS_n_rois", d.get("n_rois_processed")))
    else:
        print(f"[warn] job {job}: no network_apls_metrics.json")
    return rows


def find_coordfix_jobs():
    jobs = []
    for cfg in LOGS.glob("*/config.py"):
        if "w118/4 coordfix_test" in cfg.read_text(errors="replace"):
            jobs.append(int(cfg.parent.name))
    return sorted(jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="*", type=int)
    ap.add_argument("--ref-jobs", nargs="*", type=int, default=[])
    ap.add_argument("--out", default="summaries/coordfix_test")
    args = ap.parse_args()
    jobs = (args.jobs or find_coordfix_jobs()) + args.ref_jobs
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    long_rows, cols, wide = [], [], {}
    for job in jobs:
        name, data = job_info(job)
        col = f"{job} {name}"
        cols.append(col)
        for task, metric, value in parse_job(job):
            long_rows.append((job, name, data, task, metric, value))
            wide.setdefault((task, metric), {})[col] = value
    with open(out / "metrics_long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job", "model", "data_root", "task", "metric", "value"])
        w.writerows(long_rows)
    with open(out / "metrics_wide.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "metric"] + cols)
        for (task, metric), vals in wide.items():
            w.writerow([task, metric] + [
                "" if vals.get(c) is None or (isinstance(vals[c], float) and math.isnan(vals[c]))
                else f"{vals[c]:.4f}" if isinstance(vals[c], float) else vals[c]
                for c in cols
            ])
    print(f"{len(jobs)} jobs, {len(wide)} metrics -> {out}/metrics_long.csv, {out}/metrics_wide.csv")


if __name__ == "__main__":
    main()
