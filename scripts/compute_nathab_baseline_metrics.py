#!/usr/bin/env python3
"""
Compute the two naive baselines (uniform, dataset-wide static) for the nathab
tile-distribution axes, per the spec:

    m_a(pi_hat) = (1 / N_T^a) * sum_t N_t^a * KL(q_t^a || pi_hat^a)

for pi_hat = uniform (m_unif) and pi_hat = q_bar, the global marginal (m_static == H_a,
the intra-dataset heterogeneity term). Ln is natural log (nats), matching
torch.nn.functional.kl_div.

Per-tile, per-axis class counts (points with non-void natural_habitat annotation,
fanned out to each axis exactly like Flair3DLabelRemap / count_flair3d_train_label_distribution.py)
are the required input; Void points are excluded before building the count vectors,
which happens automatically here since Void sits at ignore_index == num_classes and only
in-range ids are binned.

A tile with zero valid points for an axis is dropped for that axis (N_t^a = 0 case).

Reuses scene listing / exclusion logic from count_flair3d_train_label_distribution.py
(same manifest, same excluded-tile rules) so the tile population matches the existing
dataset-wide stats in stats/flair3d/label_distribution*/. Locally (Hecate), only a subset
of the manifest's tiles are actually mirrored on disk; by default this script skips
manifest rows with no local scene directory (reported explicitly in the output) rather
than treating them as errors -- run on Jean Zay with the full manifest for the true
national test-set numbers.

Example:
python scripts/compute_nathab_baseline_metrics.py \
    --data_root data/flair3d_plus \
    --csv_manifest data/flair3d_plus/raw/scene_split_manifest.csv \
    --split test \
    --num_workers 8 \
    --output_dir stats/flair3d/nathab_baseline_metrics
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(module_name: str, rel_path: str):
    path = os.path.join(REPO_ROOT, *rel_path.split("/"))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_count_script = _load_module(
    "flair3d_count_label_distribution",
    "scripts/count_flair3d_train_label_distribution.py",
)
_label_remap = _load_module(
    "flair3d_label_remap_baseline_script",
    "pointcept/datasets/preprocessing/flair3d_plus/flair3d_label_remap.py",
)

NATHAB_AXIS_TASKS: Tuple[str, ...] = _count_script.NATHAB_AXIS_TASKS
NATHAB_AXIS_SOURCE_TASK: str = _count_script.NATHAB_AXIS_SOURCE_TASK
SceneRecord = _count_script.SceneRecord

AXIS_DISPLAY_NAMES: Dict[str, str] = {
    "nathab_habitat_type": "HabitatType",
    "nathab_moisture_regime": "MoistureRegime",
    "nathab_soil_chemistry": "SoilChemistry",
    "nathab_bioclimatic_zone": "BioclimaticZone",
}

# Set once in main() before worker processes are forked; child processes inherit it
# via COW fork (mirrors the pattern in count_flair3d_train_label_distribution.py).
_AXIS_STATE: Dict[str, "AxisState"] = {}


@dataclass(frozen=True)
class AxisState:
    num_classes: int
    class_names: Tuple[str, ...]
    lut: np.ndarray  # storage id -> target id (or missing_fill, treated as void/out-of-range)


def build_axis_states(storage_definition: str) -> Dict[str, AxisState]:
    get_definition = _label_remap.get_definition
    build_stored_to_train_lut = _label_remap.build_stored_to_train_lut
    definition_to_task_config = _label_remap.definition_to_task_config

    storage_def = get_definition(NATHAB_AXIS_SOURCE_TASK, storage_definition)
    states: Dict[str, AxisState] = {}
    for axis, target_name in _count_script.FLAIR3D_TILE_DISTRIBUTION_TASKS.items():
        target_def = get_definition(NATHAB_AXIS_SOURCE_TASK, target_name)
        cfg = definition_to_task_config(target_def)
        lut = build_stored_to_train_lut(storage_def, target_def)
        states[axis] = AxisState(
            num_classes=int(cfg["num_classes"]),
            class_names=tuple(cfg["names"][: int(cfg["num_classes"])]),
            lut=lut,
        )
    return states


def _process_scene(scene: SceneRecord) -> Tuple[str, str, Dict[str, np.ndarray] | None, str | None]:
    """Returns (split, patch_id, {axis: counts[C_a]} or None, error)."""
    nh_path = os.path.join(scene.scene_path, f"{NATHAB_AXIS_SOURCE_TASK}.npy")
    if not os.path.isfile(nh_path):
        return scene.split, scene.patch_id, None, f"missing {NATHAB_AXIS_SOURCE_TASK}.npy: {scene.scene_path}"

    stored = np.load(nh_path).reshape(-1).astype(np.int64, copy=False)
    counts: Dict[str, np.ndarray] = {}
    for axis, state in _AXIS_STATE.items():
        lut = state.lut
        valid = (stored >= 0) & (stored < lut.shape[0])
        mapped = lut[stored[valid]]
        in_range = (mapped >= 0) & (mapped < state.num_classes)
        counts[axis] = np.bincount(
            mapped[in_range], minlength=state.num_classes
        ).astype(np.int64)
    return scene.split, scene.patch_id, counts, None


def kl(q: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """q, p sum to 1. Convention 0*log(0/p) = 0."""
    mask = q > 0
    return float(np.sum(q[mask] * np.log(q[mask] / np.clip(p[mask], eps, None))))


def compute_axis_metrics(tile_counts: Dict[str, np.ndarray], class_names: Tuple[str, ...]) -> dict:
    C_a = len(class_names)
    tile_ids = [t for t, c in tile_counts.items() if c.sum() > 0]
    if not tile_ids:
        raise ValueError("no tiles with valid (non-void) points for this axis")

    N_t = {t: int(tile_counts[t].sum()) for t in tile_ids}
    q_t = {t: tile_counts[t] / N_t[t] for t in tile_ids}
    N_T = sum(N_t.values())
    q_bar = sum(tile_counts[t] for t in tile_ids) / N_T
    U = np.ones(C_a) / C_a

    zero_qbar_but_present = [
        (t, c) for t in tile_ids for c in np.nonzero(tile_counts[t])[0] if q_bar[c] == 0
    ]
    assert not zero_qbar_but_present, (
        f"class present in a tile but q_bar==0 (should be impossible): {zero_qbar_but_present[:5]}"
    )

    def m_a(pi_hat: np.ndarray) -> float:
        return sum(N_t[t] * kl(q_t[t], pi_hat) for t in tile_ids) / N_T

    m_static = m_a(q_bar)
    m_unif = m_a(U)

    H_qbar = -float(np.sum(q_bar[q_bar > 0] * np.log(q_bar[q_bar > 0])))
    lnC = float(np.log(C_a))

    gap_direct = m_unif - m_static
    gap_analytic = lnC - H_qbar
    sanity_gap_ok = abs(gap_direct - gap_analytic) < 1e-6
    sanity_bounds_ok = (
        -1e-9 <= m_static <= H_qbar + 1e-9
        and gap_analytic - 1e-9 <= m_unif <= lnC + 1e-9
    )

    return dict(
        H_a=m_static,
        m_static=m_static,
        m_unif=m_unif,
        H_qbar=H_qbar,
        lnC=lnC,
        gap_direct=gap_direct,
        gap_analytic=gap_analytic,
        sanity_gap_ok=sanity_gap_ok,
        sanity_bounds_ok=sanity_bounds_ok,
        q_bar=q_bar.tolist(),
        class_names=list(class_names),
        N_T=N_T,
        n_tiles=len(tile_ids),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default="data/flair3d_plus")
    parser.add_argument("--csv_manifest", default="data/flair3d_plus/raw/scene_split_manifest.csv")
    parser.add_argument("--split", default="test", help="Comma-separated splits (default: test)")
    parser.add_argument("--missing_tiles_manifest", default="data/flair3d_plus/missing_ply_preflight.txt")
    parser.add_argument("--too_small_tiles_manifest", default="data/flair3d_plus/too_small_tiles.csv")
    parser.add_argument("--no_exclude_hardcoded", action="store_true")
    parser.add_argument("--no_exclude_missing_manifest", action="store_true")
    parser.add_argument("--no_exclude_too_small", action="store_true")
    parser.add_argument(
        "--storage_definition",
        default="default",
        help="On-disk natural_habitat.npy definition (default matches preprocessing default).",
    )
    parser.add_argument(
        "--require_local_dir",
        action="store_true",
        default=True,
        help="Skip manifest rows with no local scene directory instead of erroring "
        "(default: on; needed on Hecate where only a subset of the national manifest "
        "is mirrored on disk).",
    )
    parser.add_argument("--no_require_local_dir", dest="require_local_dir", action="store_false")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_scenes", type=int, default=0, help="Debug: limit number of scenes (0=all)")
    parser.add_argument("--output_dir", default="stats/flair3d/nathab_baseline_metrics")
    args = parser.parse_args()

    global _AXIS_STATE
    _AXIS_STATE = build_axis_states(args.storage_definition)

    data_root = _count_script.resolve_repo_path(args.data_root)
    csv_manifest = _count_script.resolve_repo_path(args.csv_manifest)
    target_splits = _count_script.parse_splits(args.split)

    excluded: set[tuple[str, str]] = set()
    if not args.no_exclude_hardcoded:
        excluded |= _count_script.load_hardcoded_excluded_tiles()
    if not args.no_exclude_missing_manifest:
        excluded |= _count_script.load_missing_tiles_manifest(
            _count_script.resolve_repo_path(args.missing_tiles_manifest)
        )
    if not args.no_exclude_too_small:
        excluded |= _count_script.load_too_small_tiles_manifest(
            _count_script.resolve_repo_path(args.too_small_tiles_manifest),
            train_only=(target_splits == {"train"}),
        )

    scenes = _count_script.load_scene_records(data_root, csv_manifest, target_splits, excluded)
    n_manifest = len(scenes)

    n_no_local_dir = 0
    if args.require_local_dir:
        kept = []
        for s in scenes:
            if os.path.isdir(s.scene_path):
                kept.append(s)
            else:
                n_no_local_dir += 1
        scenes = kept

    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]

    print(f"data_root={data_root}")
    print(f"csv_manifest={csv_manifest}")
    print(f"splits={sorted(target_splits)}")
    print(f"excluded tiles (manifest-level): {len(excluded)}")
    print(f"manifest rows matching split(s): {n_manifest}")
    if args.require_local_dir:
        print(f"  of which no local scene dir on this machine: {n_no_local_dir}")
    print(f"scenes to scan: {len(scenes)}")
    for axis in NATHAB_AXIS_TASKS:
        state = _AXIS_STATE[axis]
        print(f"  {axis} ({AXIS_DISPLAY_NAMES[axis]}): C_a={state.num_classes} names={state.class_names}")

    tile_counts_by_axis: Dict[str, Dict[str, np.ndarray]] = {axis: {} for axis in NATHAB_AXIS_TASKS}
    errors: List[str] = []

    def _consume(split, patch_id, counts, err):
        if err is not None:
            errors.append(err)
            return
        for axis, arr in counts.items():
            tile_counts_by_axis[axis][patch_id] = arr

    if args.num_workers <= 1:
        it = (_process_scene(s) for s in scenes)
        for split, patch_id, counts, err in tqdm(it, total=len(scenes), desc="Scenes", unit="scene"):
            _consume(split, patch_id, counts, err)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            mapped = pool.map(_process_scene, scenes, chunksize=16)
            for split, patch_id, counts, err in tqdm(mapped, total=len(scenes), desc="Scenes", unit="scene"):
                _consume(split, patch_id, counts, err)

    print(f"\nScanned OK: {len(scenes) - len(errors)}/{len(scenes)} (errors: {len(errors)})")
    if errors:
        for line in errors[:10]:
            print(f"  - {line}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    results = {}
    for axis in NATHAB_AXIS_TASKS:
        state = _AXIS_STATE[axis]
        results[axis] = compute_axis_metrics(tile_counts_by_axis[axis], state.class_names)

    m_static_total = sum(r["m_static"] for r in results.values())
    m_unif_total = sum(r["m_unif"] for r in results.values())

    print("\n" + "=" * 88)
    print(f"{'axis':20s} {'C_a':>4s} {'n_tiles':>8s} {'H_a=m_static':>14s} {'m_unif':>10s} "
          f"{'gap_ok':>7s} {'bounds_ok':>10s}")
    print("-" * 88)
    for axis in NATHAB_AXIS_TASKS:
        r = results[axis]
        print(
            f"{AXIS_DISPLAY_NAMES[axis]:20s} {len(r['class_names']):>4d} {r['n_tiles']:>8d} "
            f"{r['H_a']:>14.4f} {r['m_unif']:>10.4f} "
            f"{str(r['sanity_gap_ok']):>7s} {str(r['sanity_bounds_ok']):>10s}"
        )
    print("-" * 88)
    print(f"{'TOTAL':20s} {'':>4s} {'':>8s} {m_static_total:>14.4f} {m_unif_total:>10.4f}")
    print("=" * 88)

    for axis in NATHAB_AXIS_TASKS:
        r = results[axis]
        if not (r["sanity_gap_ok"] and r["sanity_bounds_ok"]):
            print(
                f"WARNING: sanity check failed for {axis}: "
                f"gap_direct={r['gap_direct']:.6f} gap_analytic={r['gap_analytic']:.6f} "
                f"H_qbar={r['H_qbar']:.6f} lnC={r['lnC']:.6f}"
            )

    output_dir = _count_script.resolve_repo_path(args.output_dir)
    split_tag = "_".join(sorted(target_splits))
    out_dir = os.path.join(output_dir, split_tag)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_root": data_root,
                "csv_manifest": csv_manifest,
                "splits": sorted(target_splits),
                "storage_definition": args.storage_definition,
                "require_local_dir": args.require_local_dir,
                "manifest_rows": n_manifest,
                "manifest_rows_no_local_dir": n_no_local_dir,
                "scenes_scanned": len(scenes),
                "scenes_errors": len(errors),
                "axes": results,
                "m_static_total": m_static_total,
                "m_unif_total": m_unif_total,
            },
            f,
            indent=2,
        )

    with open(os.path.join(out_dir, "summary.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["axis", "C_a", "n_tiles", "N_T", "H_a_m_static", "m_unif", "H_qbar", "lnC"])
        for axis in NATHAB_AXIS_TASKS:
            r = results[axis]
            writer.writerow(
                [AXIS_DISPLAY_NAMES[axis], len(r["class_names"]), r["n_tiles"], r["N_T"],
                 f"{r['H_a']:.6f}", f"{r['m_unif']:.6f}", f"{r['H_qbar']:.6f}", f"{r['lnC']:.6f}"]
            )
        writer.writerow(["TOTAL", "", "", "", f"{m_static_total:.6f}", f"{m_unif_total:.6f}", "", ""])

    print(f"\nSaved: {out_dir}/results.json, {out_dir}/summary.csv")
    if args.require_local_dir and n_no_local_dir > 0:
        print(
            f"\nNOTE: {n_no_local_dir}/{n_manifest} manifest tiles for split(s) "
            f"{sorted(target_splits)} have no local directory on this machine -- "
            "these numbers are computed on a PARTIAL test set, not the full national one. "
            "Re-run on Jean Zay with the full manifest for the paper-reportable numbers."
        )


if __name__ == "__main__":
    main()
