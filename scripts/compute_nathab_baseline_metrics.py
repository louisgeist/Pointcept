#!/usr/bin/env python3
"""
Compute the two naive baselines (uniform, dataset-wide static) for the nathab
tile-distribution axes, for both evaluation divergences used in training/testing:

    m_a^KL (pi_hat) = (1 / N_T^a) * sum_t N_t^a * KL(q_t^a || pi_hat^a)
    m_a^TV (pi_hat) = (1 / N_T^a) * sum_t N_t^a * TV(q_t^a,  pi_hat^a)

for pi_hat = uniform (m_unif) and pi_hat = q_bar, the global marginal
(m_static_KL == H_a, the intra-dataset heterogeneity term; m_static_TV == H_a^TV).

KL uses natural log (nats), matching torch.nn.functional.kl_div.
TV matches the evaluator convention in pointcept.utils.misc.tv_from_abs_errors:
    TV(q, p) = sum_c |q_c - p_c|   (= L1 = 2 * classical total-variation distance).

Unlike KL, TV has no exact Pythagorean identity m(pi_hat) = H_a + D(q_bar, pi_hat);
only the triangle inequality m_TV(pi_hat) <= H_a^TV + TV(q_bar, pi_hat) holds.
Aggregate TV(q_bar, *) is still reported separately (same layout as KL(q_bar, *)).

Per-tile, per-axis class counts (points with non-void natural_habitat annotation,
fanned out to each axis exactly like Malibu3DLabelRemap / count_malibu3d_train_label_distribution.py)
are the required input; Void points are excluded before building the count vectors,
which happens automatically here since Void sits at ignore_index == num_classes and only
in-range ids are binned.

A tile with zero valid points for an axis is dropped for that axis (N_t^a = 0 case).

Reuses scene listing / exclusion logic from count_malibu3d_train_label_distribution.py
(same manifest, same excluded-tile rules) so the tile population matches the existing
dataset-wide stats in stats/malibu3d/label_distribution*/. Locally (local machine), only a subset
of the manifest's tiles are actually mirrored on disk; by default this script skips
manifest rows with no local scene directory (reported explicitly in the output) rather
than treating them as errors -- run on cluster with the full manifest for the true
national test-set numbers.

Example (two steps -- step 1 provides the "train" pi_hat for the KL/TV(qbar_test, pi_hat_train)
columns in step 2's output; skip it and drop --extra_pi_hat_csv_dir/--extra_pi_hat_name from
step 2 if you only want H_a / H_a^TV and D(qbar_test, U)):

# 1) train's global per-axis marginal (pi_hat_train) -- cheap, only aggregate counts needed.
python scripts/count_malibu3d_train_label_distribution.py \
    --data_root data/malibu3d_plus \
    --csv_manifest data/malibu3d_plus/raw/scene_split_manifest.csv \
    --split train \
    --num_workers 24 \
    --output_dir stats/malibu3d/label_distribution_national/train

# 2) main computation on the test split (KL + TV tables).
python scripts/compute_nathab_baseline_metrics.py \
    --data_root data/malibu3d_plus \
    --csv_manifest data/malibu3d_plus/raw/scene_split_manifest.csv \
    --split test \
    --num_workers 24 \
    --extra_pi_hat_csv_dir stats/malibu3d/label_distribution_national/train \
    --extra_pi_hat_name train \
    --output_dir stats/malibu3d/nathab_baseline_metrics
# (--output_dir gets the split name appended automatically -> results land in
#  stats/malibu3d/nathab_baseline_metrics/test/, no need to add "/test" yourself)
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
    "malibu3d_count_label_distribution",
    "scripts/count_malibu3d_train_label_distribution.py",
)
_label_remap = _load_module(
    "malibu3d_label_remap_baseline_script",
    "pointcept/datasets/preprocessing/malibu3d_plus/malibu3d_label_remap.py",
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
# via COW fork (mirrors the pattern in count_malibu3d_train_label_distribution.py).
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
    for axis, target_name in _count_script.MALIBU3D_TILE_DISTRIBUTION_TASKS.items():
        target_def = get_definition(NATHAB_AXIS_SOURCE_TASK, target_name)
        cfg = definition_to_task_config(target_def)
        lut = build_stored_to_train_lut(storage_def, target_def)
        states[axis] = AxisState(
            num_classes=int(cfg["num_classes"]),
            class_names=tuple(cfg["names"][: int(cfg["num_classes"])]),
            lut=lut,
        )
    return states


def load_natural_habitat_manifest_flags(
    csv_manifest: str, target_splits: set[str]
) -> Dict[Tuple[str, str], bool]:
    """(split, patch_id) -> manifest's NATURAL_HABITAT boolean.

    Used only as a cross-check against actual on-disk file presence (never to filter
    scenes: file presence is the source of truth for what gets counted). A tile flagged
    True with a missing file is a real anomaly worth investigating; a tile flagged False
    with the file actually present just means the manifest is stale for that row (observed
    once already, locally, for department D068) -- harmless for the metric itself since
    file presence already governs inclusion, but worth surfacing either way.
    """
    flags: Dict[Tuple[str, str], bool] = {}
    with open(csv_manifest, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = str(row["split"]).strip()
            if split not in target_splits:
                continue
            patch_id = str(row["patch_id"]).strip()
            flags[(split, patch_id)] = _count_script.parse_manifest_bool(row.get("NATURAL_HABITAT"))
    return flags


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


def tv(q: np.ndarray, p: np.ndarray) -> float:
    """L1 total variation matching evaluator TV: sum_c |q - p| (in [0, 2])."""
    return float(np.sum(np.abs(q - p)))


def load_axis_marginal_from_csv(csv_path: str, num_classes: int) -> np.ndarray:
    """Read the per-class 'count' column (bucket=='class', in class_id order) from a
    {axis}_label_distribution.csv produced by count_malibu3d_train_label_distribution.py.
    Returns raw (un-normalized) counts, length num_classes."""
    counts = [None] * num_classes
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["bucket"] != "class":
                continue
            cid = int(row["class_id"])
            if cid < num_classes:
                counts[cid] = float(row["count"])
    if any(c is None for c in counts):
        raise ValueError(f"{csv_path}: missing class rows for 0..{num_classes - 1}")
    return np.array(counts, dtype=np.float64)


def compute_axis_metrics(
    tile_counts: Dict[str, np.ndarray],
    class_names: Tuple[str, ...],
    extra_pi_hats: Dict[str, np.ndarray] | None = None,
) -> dict:
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

    def m_a_kl(pi_hat: np.ndarray) -> float:
        return sum(N_t[t] * kl(q_t[t], pi_hat) for t in tile_ids) / N_T

    def m_a_tv(pi_hat: np.ndarray) -> float:
        return sum(N_t[t] * tv(q_t[t], pi_hat) for t in tile_ids) / N_T

    m_static = m_a_kl(q_bar)
    m_unif = m_a_kl(U)
    m_static_tv = m_a_tv(q_bar)
    m_unif_tv = m_a_tv(U)
    tv_qbar_unif = tv(q_bar, U)

    H_qbar = -float(np.sum(q_bar[q_bar > 0] * np.log(q_bar[q_bar > 0])))
    lnC = float(np.log(C_a))

    gap_direct = m_unif - m_static
    gap_analytic = lnC - H_qbar
    sanity_gap_ok = abs(gap_direct - gap_analytic) < 1e-6
    sanity_bounds_ok = (
        -1e-9 <= m_static <= H_qbar + 1e-9
        and gap_analytic - 1e-9 <= m_unif <= lnC + 1e-9
    )
    # TV analogue of the KL gap: no exact identity, but triangle inequality must hold.
    sanity_tv_triangle_unif_ok = m_unif_tv <= m_static_tv + tv_qbar_unif + 1e-6
    sanity_tv_bounds_ok = (
        -1e-9 <= m_static_tv <= 2.0 + 1e-9
        and -1e-9 <= m_unif_tv <= 2.0 + 1e-9
        and -1e-9 <= tv_qbar_unif <= 2.0 + 1e-9
    )

    # For any fixed (non-tile-dependent) pi_hat: m_a_KL(pi_hat) = H_a + KL(q_bar || pi_hat).
    # Same identity as the uniform gap check above, generalized; kept as a sanity check.
    # For TV only the triangle inequality is checked (no exact decomposition).
    extra: Dict[str, dict] = {}
    for name, raw in (extra_pi_hats or {}).items():
        pi_hat_extra = raw / raw.sum()
        m_extra = m_a_kl(pi_hat_extra)
        kl_qbar_extra = kl(q_bar, pi_hat_extra)
        m_extra_tv = m_a_tv(pi_hat_extra)
        tv_qbar_extra = tv(q_bar, pi_hat_extra)
        extra[name] = dict(
            m_a=m_extra,
            kl_qbar=kl_qbar_extra,
            sanity_decomp_ok=abs((m_extra - m_static) - kl_qbar_extra) < 1e-6,
            m_a_tv=m_extra_tv,
            tv_qbar=tv_qbar_extra,
            sanity_tv_triangle_ok=m_extra_tv <= m_static_tv + tv_qbar_extra + 1e-6,
        )

    return dict(
        H_a=m_static,
        m_static=m_static,
        m_unif=m_unif,
        H_a_tv=m_static_tv,
        m_static_tv=m_static_tv,
        m_unif_tv=m_unif_tv,
        tv_qbar_U=tv_qbar_unif,
        H_qbar=H_qbar,
        lnC=lnC,
        gap_direct=gap_direct,
        gap_analytic=gap_analytic,
        sanity_gap_ok=sanity_gap_ok,
        sanity_bounds_ok=sanity_bounds_ok,
        sanity_tv_triangle_unif_ok=sanity_tv_triangle_unif_ok,
        sanity_tv_bounds_ok=sanity_tv_bounds_ok,
        q_bar=q_bar.tolist(),
        class_names=list(class_names),
        N_T=N_T,
        n_tiles=len(tile_ids),
        extra=extra,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default="data/malibu3d_plus")
    parser.add_argument("--csv_manifest", default="data/malibu3d_plus/raw/scene_split_manifest.csv")
    parser.add_argument("--split", default="test", help="Comma-separated splits (default: test)")
    parser.add_argument("--missing_tiles_manifest", default="data/malibu3d_plus/missing_ply_preflight.txt")
    parser.add_argument("--too_small_tiles_manifest", default="data/malibu3d_plus/too_small_tiles.csv")
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
        "(default: on; needed on local machine where only a subset of the national manifest "
        "is mirrored on disk).",
    )
    parser.add_argument("--no_require_local_dir", dest="require_local_dir", action="store_false")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_scenes", type=int, default=0, help="Debug: limit number of scenes (0=all)")
    parser.add_argument("--output_dir", default="stats/malibu3d/nathab_baseline_metrics")
    parser.add_argument(
        "--extra_pi_hat_csv_dir",
        default="",
        help="Directory with {axis}_label_distribution.csv files (e.g. from "
        "count_malibu3d_train_label_distribution.py run on --split train) providing an "
        "additional, fixed pi_hat to evaluate against the SAME per-tile q_t/N_t as "
        "--split (e.g. tiles are test, but pi_hat is train's global marginal -- measures "
        "train/test distribution shift on top of intra-split heterogeneity). Optional; "
        "H_a/m_static/m_unif (evaluation split's own q_bar and uniform) are unaffected.",
    )
    parser.add_argument(
        "--extra_pi_hat_name",
        default="train",
        help="Label for the --extra_pi_hat_csv_dir distribution in the output (default: train).",
    )
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

    nathab_flags = load_natural_habitat_manifest_flags(csv_manifest, target_splits)

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
    dept_totals: Dict[str, int] = {}
    dept_missing: Dict[str, int] = {}
    # manifest=True (covered) but file actually missing -- the real anomaly to watch for.
    flagged_true_but_missing: Dict[str, List[str]] = {}
    # manifest=False (not covered) but file actually present -- stale manifest row, harmless
    # for the metric (file presence already governs inclusion) but worth surfacing.
    flagged_false_but_present: Dict[str, List[str]] = {}

    def _dept_of(patch_id: str) -> str:
        return patch_id.split("_", 1)[0]

    def _consume(split, patch_id, counts, err):
        dept = _dept_of(patch_id)
        dept_totals[dept] = dept_totals.get(dept, 0) + 1
        manifest_covered = nathab_flags.get((split, patch_id), False)
        file_present = err is None
        if manifest_covered and not file_present:
            flagged_true_but_missing.setdefault(dept, []).append(patch_id)
        elif not manifest_covered and file_present:
            flagged_false_but_present.setdefault(dept, []).append(patch_id)
        if err is not None:
            errors.append(err)
            dept_missing[dept] = dept_missing.get(dept, 0) + 1
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
        # Diagnostic only (not filtered on): the manifest's NATURAL_HABITAT column has been
        # observed to be stale for at least one department (D068, verified against on-disk
        # data), so department-level exclusion is NOT applied here -- file presence is the
        # sole source of truth for whether a tile contributes to the metric (matches the
        # spec's N_t^a=0 exclusion rule automatically). This breakdown just helps distinguish
        # "whole department has no natural_habitat.npy" (expected per README_malibu3dplus.md's
        # documented 55/74 dept-year NATURAL_HABITAT coverage) from isolated missing files
        # within an otherwise-covered department (worth investigating).
        print("\nMissing natural_habitat.npy by department (diagnostic, not filtered on):")
        for dept in sorted(dept_missing, key=lambda d: -dept_missing[d]):
            total = dept_totals[dept]
            missing = dept_missing[dept]
            tag = "whole dept" if missing == total else "PARTIAL -- check"
            manifest_flag = (
                "manifest=covered" if dept in flagged_true_but_missing else "manifest=not-covered"
            )
            print(f"  {dept:20s} {missing:>6d}/{total:<6d} missing  [{tag}, {manifest_flag}]")

    n_real_anomalies = sum(len(v) for v in flagged_true_but_missing.values())
    if n_real_anomalies:
        print(
            f"\n*** {n_real_anomalies} tile(s) flagged NATURAL_HABITAT=True in the manifest "
            "but natural_habitat.npy is actually MISSING on disk -- likely a real dataset "
            "problem (unfinished/failed preprocessing sync), not the documented coverage gap: ***"
        )
        for dept in sorted(flagged_true_but_missing, key=lambda d: -len(flagged_true_but_missing[d])):
            ids = flagged_true_but_missing[dept]
            print(f"  {dept:20s} {len(ids):>6d} tile(s), e.g. {ids[:3]}")
    else:
        print("\nNo manifest=True/file-missing anomalies found (every tile the manifest claims is covered has a file).")

    n_stale_manifest = sum(len(v) for v in flagged_false_but_present.values())
    if n_stale_manifest:
        print(
            f"\n{n_stale_manifest} tile(s) flagged NATURAL_HABITAT=False in the manifest but "
            "the file IS present on disk (manifest under-reports coverage; harmless for the "
            "metric since file presence already governs inclusion, but worth a look):"
        )
        for dept in sorted(flagged_false_but_present, key=lambda d: -len(flagged_false_but_present[d])):
            ids = flagged_false_but_present[dept]
            print(f"  {dept:20s} {len(ids):>6d} tile(s), e.g. {ids[:3]}")

    results = {}
    for axis in NATHAB_AXIS_TASKS:
        state = _AXIS_STATE[axis]
        extra_pi_hats = {}
        if args.extra_pi_hat_csv_dir:
            csv_path = os.path.join(
                _count_script.resolve_repo_path(args.extra_pi_hat_csv_dir),
                f"{axis}_label_distribution.csv",
            )
            extra_pi_hats[args.extra_pi_hat_name] = load_axis_marginal_from_csv(
                csv_path, state.num_classes
            )
        results[axis] = compute_axis_metrics(
            tile_counts_by_axis[axis], state.class_names, extra_pi_hats=extra_pi_hats
        )

    extra_name = args.extra_pi_hat_name if args.extra_pi_hat_csv_dir else None

    m_static_total = sum(r["m_static"] for r in results.values())
    m_unif_total = sum(r["m_unif"] for r in results.values())
    kl_unif_total = sum(r["gap_analytic"] for r in results.values())  # = sum KL(q_bar||U)
    m_static_tv_total = sum(r["m_static_tv"] for r in results.values())
    m_unif_tv_total = sum(r["m_unif_tv"] for r in results.values())
    tv_unif_total = sum(r["tv_qbar_U"] for r in results.values())  # = sum TV(q_bar, U)
    extra_totals = {}
    if extra_name:
        extra_totals["m_a"] = sum(r["extra"][extra_name]["m_a"] for r in results.values())
        extra_totals["kl_qbar"] = sum(r["extra"][extra_name]["kl_qbar"] for r in results.values())
        extra_totals["m_a_tv"] = sum(r["extra"][extra_name]["m_a_tv"] for r in results.values())
        extra_totals["tv_qbar"] = sum(r["extra"][extra_name]["tv_qbar"] for r in results.values())

    # Main table, per the requested layout: H_a (heterogeneity term) alongside the two
    # AGGREGATE-level divergences KL(q_bar_test, pi_hat_train) and KL(q_bar_test, U) --
    # these are single numbers comparing the two *global* distributions directly, distinct
    # from m_unif/m_train (tile-weighted averages of per-tile KL, kept in JSON/CSV only).
    # KL(q_bar_test, pi_hat_test) is omitted: it's 0 by construction (pi_hat_test := q_bar_test).
    kl_train_header = f" {'KL(qbar,train)':>15s}" if extra_name else ""
    width = 96 + (16 if extra_name else 0)
    print("\n" + "=" * width)
    print("KL baselines (nats)")
    print(
        f"{'axis':20s} {'C_a':>4s} {'n_tiles':>8s} {'N_T':>10s} {'H_a':>10s}"
        f"{kl_train_header} {'KL(qbar,U)':>11s} {'lnC':>7s} {'H_a/lnC':>8s}"
    )
    print("-" * width)
    for axis in NATHAB_AXIS_TASKS:
        r = results[axis]
        kl_train_col = f" {r['extra'][extra_name]['kl_qbar']:>15.4f}" if extra_name else ""
        print(
            f"{AXIS_DISPLAY_NAMES[axis]:20s} {len(r['class_names']):>4d} {r['n_tiles']:>8d} "
            f"{r['N_T']:>10d} {r['H_a']:>10.4f}{kl_train_col} {r['gap_analytic']:>11.4f} "
            f"{r['lnC']:>7.4f} {r['H_a'] / r['lnC']:>8.2%}"
        )
    print("-" * width)
    kl_train_total_col = f" {extra_totals['kl_qbar']:>15.4f}" if extra_name else ""
    print(
        f"{'TOTAL':20s} {'':>4s} {'':>8s} {'':>10s} {m_static_total:>10.4f}"
        f"{kl_train_total_col} {kl_unif_total:>11.4f}"
    )
    print("=" * width)
    print(
        "H_a = m_static = intra-test heterogeneity (tile-weighted avg KL(q_t||q_bar_test)); "
        "KL(qbar,*) = single divergence between the AGGREGATE test distribution and the "
        "reference (not tile-weighted). KL(qbar,pi_hat_test) omitted: 0 by construction."
    )
    if extra_name:
        print(
            f"m_{extra_name} (tile-weighted avg KL(q_t||{extra_name} marginal), the 'total' a "
            f"static-{extra_name}-prior predictor would incur on test) = {extra_totals['m_a']:.4f} "
            f"nats total; identity check: m_{extra_name} - H_a == KL(qbar,{extra_name}) per axis "
            f"(see sanity_decomp_ok below)."
        )

    # TV table (same layout; TV = L1 in [0, 2], matching evaluator logging).
    tv_train_header = f" {'TV(qbar,train)':>15s}" if extra_name else ""
    width_tv = 88 + (16 if extra_name else 0)
    print("\n" + "=" * width_tv)
    print("TV baselines (L1 = sum_c |pi - q|, matching val/test TV)")
    print(
        f"{'axis':20s} {'C_a':>4s} {'n_tiles':>8s} {'N_T':>10s} {'H_a^TV':>10s}"
        f"{tv_train_header} {'TV(qbar,U)':>11s} {'m_unif^TV':>10s}"
    )
    print("-" * width_tv)
    for axis in NATHAB_AXIS_TASKS:
        r = results[axis]
        tv_train_col = f" {r['extra'][extra_name]['tv_qbar']:>15.4f}" if extra_name else ""
        print(
            f"{AXIS_DISPLAY_NAMES[axis]:20s} {len(r['class_names']):>4d} {r['n_tiles']:>8d} "
            f"{r['N_T']:>10d} {r['H_a_tv']:>10.4f}{tv_train_col} {r['tv_qbar_U']:>11.4f} "
            f"{r['m_unif_tv']:>10.4f}"
        )
    print("-" * width_tv)
    tv_train_total_col = f" {extra_totals['tv_qbar']:>15.4f}" if extra_name else ""
    print(
        f"{'TOTAL':20s} {'':>4s} {'':>8s} {'':>10s} {m_static_tv_total:>10.4f}"
        f"{tv_train_total_col} {tv_unif_total:>11.4f} {m_unif_tv_total:>10.4f}"
    )
    print("=" * width_tv)
    print(
        "H_a^TV = m_static_TV = intra-test heterogeneity (tile-weighted avg TV(q_t, q_bar_test)); "
        "TV(qbar,*) = single L1 between the AGGREGATE test distribution and the reference "
        "(not tile-weighted). TV(qbar,pi_hat_test) omitted: 0 by construction. "
        "No exact KL-style decomposition; triangle: m(pi) <= H_a^TV + TV(qbar, pi)."
    )
    if extra_name:
        print(
            f"m_{extra_name}^TV (tile-weighted avg TV(q_t, {extra_name} marginal)) = "
            f"{extra_totals['m_a_tv']:.4f} total; this is the TV a static-{extra_name}-prior "
            f"predictor would incur on test."
        )

    for axis in NATHAB_AXIS_TASKS:
        r = results[axis]
        if not (r["sanity_gap_ok"] and r["sanity_bounds_ok"]):
            print(
                f"WARNING: uniform KL sanity check failed for {axis}: "
                f"gap_direct={r['gap_direct']:.6f} gap_analytic={r['gap_analytic']:.6f} "
                f"H_qbar={r['H_qbar']:.6f} lnC={r['lnC']:.6f}"
            )
        if not (r["sanity_tv_triangle_unif_ok"] and r["sanity_tv_bounds_ok"]):
            print(
                f"WARNING: uniform TV sanity check failed for {axis}: "
                f"m_unif_tv={r['m_unif_tv']:.6f} H_a_tv={r['H_a_tv']:.6f} "
                f"TV(qbar,U)={r['tv_qbar_U']:.6f}"
            )
        if extra_name and not r["extra"][extra_name]["sanity_decomp_ok"]:
            e = r["extra"][extra_name]
            print(
                f"WARNING: {extra_name} KL decomposition check failed for {axis}: "
                f"m_a - H_a = {e['m_a'] - r['H_a']:.6f}, KL(qbar,{extra_name}) = {e['kl_qbar']:.6f}"
            )
        if extra_name and not r["extra"][extra_name]["sanity_tv_triangle_ok"]:
            e = r["extra"][extra_name]
            print(
                f"WARNING: {extra_name} TV triangle check failed for {axis}: "
                f"m_a_tv={e['m_a_tv']:.6f} H_a_tv={r['H_a_tv']:.6f} "
                f"TV(qbar,{extra_name})={e['tv_qbar']:.6f}"
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
                "dept_missing_natural_habitat": dept_missing,
                "dept_totals": dept_totals,
                "manifest_true_but_file_missing": flagged_true_but_missing,
                "manifest_false_but_file_present": flagged_false_but_present,
                "axes": results,
                "m_static_total": m_static_total,
                "m_unif_total": m_unif_total,
                "m_static_tv_total": m_static_tv_total,
                "m_unif_tv_total": m_unif_tv_total,
                "tv_qbar_U_total": tv_unif_total,
                "extra_pi_hat_name": args.extra_pi_hat_name if args.extra_pi_hat_csv_dir else None,
                "extra_pi_hat_csv_dir": args.extra_pi_hat_csv_dir or None,
                "extra_totals": extra_totals,
            },
            f,
            indent=2,
        )

    with open(os.path.join(out_dir, "summary.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = [
            "axis", "C_a", "n_tiles", "N_T",
            "H_a", "KL_qbar_U", "lnC", "H_a_over_lnC",
            "m_unif",
            "H_a_TV", "TV_qbar_U", "m_unif_TV",
        ]
        if extra_name:
            header += [
                f"KL_qbar_{extra_name}", f"m_{extra_name}", f"sanity_decomp_ok_{extra_name}",
                f"TV_qbar_{extra_name}", f"m_{extra_name}_TV", f"sanity_tv_triangle_ok_{extra_name}",
            ]
        writer.writerow(header)
        for axis in NATHAB_AXIS_TASKS:
            r = results[axis]
            row = [
                AXIS_DISPLAY_NAMES[axis], len(r["class_names"]), r["n_tiles"], r["N_T"],
                f"{r['H_a']:.6f}", f"{r['gap_analytic']:.6f}", f"{r['lnC']:.6f}",
                f"{r['H_a'] / r['lnC']:.6f}", f"{r['m_unif']:.6f}",
                f"{r['H_a_tv']:.6f}", f"{r['tv_qbar_U']:.6f}", f"{r['m_unif_tv']:.6f}",
            ]
            if extra_name:
                e = r["extra"][extra_name]
                row += [
                    f"{e['kl_qbar']:.6f}", f"{e['m_a']:.6f}", str(e["sanity_decomp_ok"]),
                    f"{e['tv_qbar']:.6f}", f"{e['m_a_tv']:.6f}", str(e["sanity_tv_triangle_ok"]),
                ]
            writer.writerow(row)
        total_row = [
            "TOTAL", "", "", "",
            f"{m_static_total:.6f}", f"{kl_unif_total:.6f}", "", "", f"{m_unif_total:.6f}",
            f"{m_static_tv_total:.6f}", f"{tv_unif_total:.6f}", f"{m_unif_tv_total:.6f}",
        ]
        if extra_name:
            total_row += [
                f"{extra_totals['kl_qbar']:.6f}", f"{extra_totals['m_a']:.6f}", "",
                f"{extra_totals['tv_qbar']:.6f}", f"{extra_totals['m_a_tv']:.6f}", "",
            ]
        writer.writerow(total_row)

    print(f"\nSaved: {out_dir}/results.json, {out_dir}/summary.csv")
    if args.require_local_dir and n_no_local_dir > 0:
        print(
            f"\nNOTE: {n_no_local_dir}/{n_manifest} manifest tiles for split(s) "
            f"{sorted(target_splits)} have no local directory on this machine -- "
            "these numbers are computed on a PARTIAL test set, not the full national one. "
            "Re-run on cluster with the full manifest for the paper-reportable numbers."
        )


if __name__ == "__main__":
    main()
