#!/usr/bin/env python3
"""Patch seed_ensemble_results.json with F1-macro (mean/std/min/max) and
per-class F1 means, computed from the per-probe test/f1_* keys already
written by GridProbeSemSegTester (log_test_f1=True). CPU-only; does not
re-run the test pass.

Matches GridProbeSeedEnsembleTester aggregation (np.std ddof=0).

Example (Jean Zay, after cd to the Pointcept checkout):
  python scripts/aggregate_seed_ensemble_f1.py --inplace \\
    /lustre/fsn1/projects/rech/unv/usi32yh/logs/pointcept_logs/slurm/*/seed_ensemble_results.json
  python scripts/aggregate_seed_ensemble_f1.py --inplace \\
    $WORK/Pointcept/logs/slurm/*/seed_ensemble_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

F1_MACRO_KEY = "test/f1_macro"
F1_CLS_PREFIX = "test/f1_"

# Display / JSON key order when these classes are present (H3D). Other
# datasets keep first-seen key order.
H3D_CLASS_ORDER = [
    "Low Vegetation",
    "Impervious Surface",
    "Vehicle",
    "Urban Furniture",
    "Roof",
    "Façade",
    "Shrub",
    "Tree",
    "Soil or Gravel",
    "Vertical Surface",
    "Chimney",
]


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return dict(mean=None, std=None, max=None, min=None)
    arr = np.array(values, dtype=float)
    return dict(
        mean=float(arr.mean()),
        std=float(arr.std()),
        max=float(arr.max()),
        min=float(arr.min()),
    )


def _probe_metrics(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    per_probe = data.get("per_probe") or {}
    return [m for m in per_probe.values() if isinstance(m, dict)]


def _collect(metrics: Iterable[Dict[str, Any]], key: str) -> List[float]:
    out = []
    for m in metrics:
        if key in m and m[key] is not None:
            out.append(float(m[key]))
    return out


def _f1_class_keys(metrics: Sequence[Dict[str, Any]]) -> List[str]:
    present = set()
    for m in metrics:
        for key in m:
            if key.startswith(F1_CLS_PREFIX) and key != F1_MACRO_KEY:
                present.add(key)
    ordered = []
    for cls_name in H3D_CLASS_ORDER:
        key = f"{F1_CLS_PREFIX}{cls_name}"
        if key in present:
            ordered.append(key)
            present.remove(key)
    ordered.extend(sorted(present))
    return ordered


def aggregate_f1(
    data: Dict[str, Any],
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    metrics = _probe_metrics(data)
    f1_macro = _stats(_collect(metrics, F1_MACRO_KEY))
    f1_cls_means = {
        key: _stats(_collect(metrics, key))["mean"] for key in _f1_class_keys(metrics)
    }
    return f1_macro, f1_cls_means


def apply_aggregates(
    data: Dict[str, Any],
    f1_macro: Dict[str, Optional[float]],
    f1_cls_means: Dict[str, Optional[float]],
) -> None:
    data["test_f1_macro_mean"] = f1_macro["mean"]
    data["test_f1_macro_std"] = f1_macro["std"]
    data["test_f1_macro_max"] = f1_macro["max"]
    data["test_f1_macro_min"] = f1_macro["min"]
    for key, mean in f1_cls_means.items():
        cls_name = key[len(F1_CLS_PREFIX) :]
        data[f"test_f1_{cls_name}_mean"] = mean


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp_path, path)


def _fmt(value: Optional[float]) -> str:
    return "nan" if value is None else f"{value:.4f}"


def print_report(
    path: Path,
    data: Dict[str, Any],
    f1_macro: Dict[str, Optional[float]],
    f1_cls_means: Dict[str, Optional[float]],
) -> None:
    n_probes = data.get("num_probes")
    n_with = sum(1 for m in _probe_metrics(data) if F1_MACRO_KEY in m)
    print(f"=== {path} ===")
    print(f"probes with f1_macro: {n_with}/{n_probes}")
    print(
        "f1_macro  mean={mean}  std={std}  min={min}  max={max}".format(
            mean=_fmt(f1_macro["mean"]),
            std=_fmt(f1_macro["std"]),
            min=_fmt(f1_macro["min"]),
            max=_fmt(f1_macro["max"]),
        )
    )
    if f1_cls_means:
        print("f1_mean by class:")
        width = max(len(k[len(F1_CLS_PREFIX) :]) for k in f1_cls_means)
        for key, mean in f1_cls_means.items():
            print(f"  {key[len(F1_CLS_PREFIX):]:<{width}}  {_fmt(mean)}")
    else:
        print("f1_mean by class: (no test/f1_<cls> keys — skip, log_test_f1 was likely False)")
    print()


def resolve_json_paths(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            found = sorted(p.rglob("seed_ensemble_results.json"))
            if not found:
                print(f"WARNING: no seed_ensemble_results.json under {p}", file=sys.stderr)
            paths.extend(found)
        elif p.is_file():
            paths.append(p)
        else:
            print(f"WARNING: not found: {p}", file=sys.stderr)
    # unique, keep order
    seen = set()
    unique = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="seed_ensemble_results.json files, or job dirs to search recursively",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="write F1 aggregates back into each JSON",
    )
    args = parser.parse_args()

    json_paths = resolve_json_paths(args.paths)
    if not json_paths:
        print("ERROR: no seed_ensemble_results.json found", file=sys.stderr)
        return 1

    n_ok = 0
    n_skip = 0
    for path in json_paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        f1_macro, f1_cls_means = aggregate_f1(data)
        if f1_macro["mean"] is None:
            print(f"=== {path} ===")
            print("skip: no per-probe test/f1_macro (not an H3D log_test_f1 run, or test failed)")
            print()
            n_skip += 1
            continue
        print_report(path, data, f1_macro, f1_cls_means)
        if args.inplace:
            apply_aggregates(data, f1_macro, f1_cls_means)
            _atomic_write_json(path, data)
            print(f"wrote {path}")
            print()
        n_ok += 1

    print(f"done: {n_ok} aggregated, {n_skip} skipped, {len(json_paths)} files")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
