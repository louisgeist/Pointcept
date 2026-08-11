"""Evaluate predicted road/rail/transmission-line network graphs against GT with APLS.

Standalone script (runs in the same `pointcept` conda env as training -- geopandas/shapely/
scipy are already dependencies, see environment.yml). Consumes the per-subtile dense
probability rasters (``{patch_id}_logits_network.npy``) already written to ``save_path`` by
``MultiTaskTester``/``PreciseEvaluator`` (see pointcept/engines/test.py), stitches them into
per-ROI rasters, builds a predicted graph per network channel by reusing the same
mask -> graph post-processing pipeline used to export the ground-truth graphs, and computes
APLS (Average Path Length Similarity) against the GT graphs exported by Flair3D-build.

Example::

    python tools/eval_network_apls.py \\
        --data_root data/flair3d_plus \\
        --save_path exp/flair3d/network_run/result \\
        --network_graphs_root /data/geist/Flair3D-build/data/network_graphs \\
        --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest_D067.csv \\
        --split val --threshold 0.5 \\
        --out_dir exp/flair3d/network_run/result
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PREPROC_DIR = (
    _HERE.parent
    / "pointcept"
    / "datasets"
    / "preprocessing"
    / "flair3d_plus"
)
if str(_PREPROC_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROC_DIR))

import apls_metric as apls  # type: ignore
import network_graph_pipeline as ngp  # type: ignore
import network_label_utils as nlu  # type: ignore
import network_prediction_stitch as nps  # type: ignore

NETWORK_TYPES = nlu.NETWORK_TYPES


class _ProfileAggregator:
    """Accumulate wall-clock samples per stage name across ROI/channel calls."""

    def __init__(self) -> None:
        self._totals: Dict[str, float] = defaultdict(float)
        self._maxes: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)

    def add(self, name: str, seconds: float) -> None:
        s = float(seconds)
        self._totals[name] += s
        if s > self._maxes[name]:
            self._maxes[name] = s
        self._counts[name] += 1

    def add_dict(self, timings: Optional[Dict[str, float]]) -> None:
        if not timings:
            return
        for name, seconds in timings.items():
            self.add(name, seconds)

    def summary(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for name in self._totals:
            total = self._totals[name]
            count = self._counts[name]
            out[name] = {
                "count": count,
                "total_s": total,
                "mean_s": total / count if count else 0.0,
                "max_s": self._maxes[name],
            }
        return out

    def print_report(self) -> None:
        summary = self.summary()
        if not summary:
            print("profiling: (no stages recorded)")
            return
        rows = sorted(summary.items(), key=lambda kv: kv[1]["total_s"], reverse=True)
        print("profiling (stages sorted by total_s desc):")
        print(
            f"{'stage':28s} {'count':>7s} {'total_s':>10s} {'mean_s':>10s} {'max_s':>10s}"
        )
        for name, stats in rows:
            print(
                f"{name:28s} {stats['count']:7d} {stats['total_s']:10.4f} "
                f"{stats['mean_s']:10.4f} {stats['max_s']:10.4f}"
            )


def _merge_timings(
    dest: Dict[str, float], src: Optional[Dict[str, float]]
) -> None:
    if not src:
        return
    for name, seconds in src.items():
        dest[name] = dest.get(name, 0.0) + float(seconds)


def _format_timings(timings: Dict[str, float], *, top_n: Optional[int] = None) -> str:
    """Compact ``name=0.123s`` list, sorted by time desc."""
    if not timings:
        return "(none)"
    rows = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        rows = rows[: max(0, int(top_n))]
    return " ".join(f"{name}={seconds:.3f}s" for name, seconds in rows)


@contextmanager
def _timed(
    aggregator: Optional[_ProfileAggregator],
    name: str,
    local: Optional[Dict[str, float]] = None,
) -> Iterator[None]:
    if aggregator is None and local is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        if aggregator is not None:
            aggregator.add(name, elapsed)
        if local is not None:
            local[name] = local.get(name, 0.0) + elapsed


def _parse_optional_float(s: str) -> Optional[float]:
    """CLI helper: ``none``/``null``/``false`` -> None, else float meters."""
    if str(s).strip().lower() in ("none", "null", "false", ""):
        return None
    return float(s)


def _parse_optional_int(s: str) -> Optional[int]:
    """CLI helper: ``none``/``null``/``false`` -> None, else int."""
    if str(s).strip().lower() in ("none", "null", "false", ""):
        return None
    return int(s)


def _coerce_optional_meters(value, *, default_when_true: float, name: str) -> Optional[float]:
    """Accept ``None``, a positive float (meters), or legacy bool (True->default, False->None)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return default_when_true if value else None
    v = float(value)
    if v <= 0:
        raise ValueError(f"{name} must be > 0 or None, got {value!r}")
    return v


def _coerce_densify(value) -> Optional[float]:
    return _coerce_optional_meters(value, default_when_true=50.0, name="apls_densify")


def _coerce_snap_to_edge(value) -> Optional[float]:
    return _coerce_optional_meters(value, default_when_true=4.0, name="apls_snap_to_edge")


def _parse_bool(s: str) -> bool:
    """CLI helper: truthy/falsy strings -> bool."""
    v = str(s).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {s!r}")


def _build_predicted_graph(
    roi_probs: np.ndarray,
    roi_grid,
    channel_idx: int,
    *,
    threshold: float,
    connectivity: int,
    rdp_epsilon_m: float,
    endpoint_fix_enabled: bool,
    endpoint_fix_stage: str,
    merge_hop_threshold: float,
    radius_fix_radius_m: Optional[float] = None,
    open_iterations: int = 0,
    close_iterations: int = 0,
    morph_connectivity: int = 4,
    remove_small_objects_enabled: bool = True,
    remove_small_objects_min_size_px: int = 8,
    skeletonize_enabled: bool = True,
    min_component_nodes: int = 5,
    collect_timings: bool = False,
) -> "ngp.ProcessedNetworkGraph":
    prob = roi_probs[channel_idx]
    # Unobserved (NaN) cells are treated as background -- see plan's design-decision note:
    # avoids inventing positive network pixels in unscanned area.
    mask = np.isfinite(prob) & (prob >= threshold)
    extra: dict = {}
    if radius_fix_radius_m is not None:
        extra["radius_fix_enabled"] = True
        extra["radius_fix_radius_m"] = float(radius_fix_radius_m)
    return ngp.build_processed_network_graph_from_mask(
        mask,
        roi_grid,
        connectivity=connectivity,
        open_iterations=open_iterations,
        close_iterations=close_iterations,
        morph_connectivity=morph_connectivity,
        remove_small_objects_enabled=remove_small_objects_enabled,
        remove_small_objects_min_size_px=remove_small_objects_min_size_px,
        skeletonize_enabled=skeletonize_enabled,
        rdp_epsilon_m=rdp_epsilon_m,
        endpoint_fix_enabled=endpoint_fix_enabled,
        endpoint_fix_stage=endpoint_fix_stage,
        merge_enabled=True,
        merge_hop_threshold=merge_hop_threshold,
        min_component_nodes=min_component_nodes,
        collect_timings=collect_timings,
        **extra,
    )


def run(
    data_root: Path,
    save_path: Path,
    network_graphs_root: Path,
    split_manifest_csv: Path,
    out_dir: Path,
    *,
    split: str = "val",
    threshold: float = 0.5,
    overlap_combine: str = "nanmean",
    connectivity: int = 4,
    rdp_epsilon_m: float = 2.0,
    endpoint_fix_enabled: bool = True,
    endpoint_fix_stage: str = "pre_rdp",
    merge_hop_threshold: float = 2.5,
    apls_max_nodes_exact: Optional[int] = None,
    max_rois: Optional[int] = None,
    apls_densify: Optional[float] = 50.0,
    apls_snap_to_edge: Optional[float] = 4.0,
    apls_symmetric: bool = True,
    radius_fix_radius_m: Optional[float] = None,
    apls_min_path_length_m: Optional[float] = None,
    open_iterations: int = 0,
    close_iterations: int = 0,
    morph_connectivity: int = 4,
    remove_small_objects_enabled: bool = True,
    remove_small_objects_min_size_px: int = 8,
    skeletonize_enabled: bool = True,
    min_component_nodes: int = 5,
    network_types: Optional[List[str]] = None,
    missing_tiles_file: Optional[Path] = None,
    show_progress: bool = True,
    profile: bool = False,
) -> dict:
    apls_densify = _coerce_densify(apls_densify)
    apls_snap_to_edge = _coerce_snap_to_edge(apls_snap_to_edge)
    types = tuple(network_types) if network_types else NETWORK_TYPES
    profiler = _ProfileAggregator() if profile else None

    # Same known-missing exclusions as rasterize_network / Flair3DDataset hardcoded set.
    if missing_tiles_file is None:
        missing_tiles_file = nps.default_missing_coord_details_csv()
    else:
        missing_tiles_file = Path(missing_tiles_file)
    known_missing = nps.load_known_missing_tiles(
        missing_tiles_file if missing_tiles_file.is_file() else None
    )
    if known_missing:
        print(
            f"Known missing tiles: {len(known_missing)} from {missing_tiles_file}"
        )
    elif missing_tiles_file is not None and not missing_tiles_file.is_file():
        print(
            f"Warning: missing-tiles file not found ({missing_tiles_file}); "
            "unexpected absences will be reported as [partial]."
        )

    patches, n_skipped_known_missing = nps.load_manifest_patches(
        split_manifest_csv,
        splits=[split],
        network_types=types,
        known_missing=known_missing,
    )
    if n_skipped_known_missing:
        print(
            f"Skipped {n_skipped_known_missing} known-missing tile(s) from "
            f"{missing_tiles_file}"
        )
    roi_items, excluded_rois = nps.group_by_roi_complete_only(patches, data_root)
    if excluded_rois:
        print(
            f"[excluded] {len(excluded_rois)} ROI(s) with no usable local subtiles "
            "(see 'excluded_rois' in the output JSON)."
        )
        for info in excluded_rois:
            dept = info.get("department") or "?"
            print(
                f"  - {dept}/{info['roi']}: reason={info.get('reason')} "
                f"missing {info.get('n_subtiles_missing', '?')}/"
                f"{info.get('n_subtiles_total', '?')} subtiles"
            )
    if max_rois is not None:
        roi_items = roi_items[: max(0, int(max_rois))]

    results: List[apls.ApsSymmetricResult] = []
    n_rois_processed = 0
    n_rois_skipped = 0
    partial_rois: List[dict] = []
    department_by_roi: dict = {}

    use_pbar = bool(show_progress) and len(roi_items) > 0
    pbar = tqdm(total=len(roi_items), desc="ROIs", unit="roi", disable=not use_pbar)
    log = pbar.write if use_pbar else print

    try:
        for roi_dir, flags, patch_dirs, coverage in roi_items:
            # roi_dir is <data_root>/<split>/<dept_year>_LIDARHD/<roi> -- recover the
            # department/year for the per-ROI CSV (roi names alone don't identify the zone).
            department = roi_dir.parent.name.removesuffix("_LIDARHD")
            department_by_roi[roi_dir.name] = department
            roi_label = f"{department}/{roi_dir.name}"
            n_total = int(coverage.get("n_subtiles_total", len(patch_dirs)))
            n_disk_missing = int(coverage.get("n_subtiles_missing", 0))
            missing_pred = [
                p.name
                for p in patch_dirs
                if not (save_path / f"{p.name}_logits_network.npy").is_file()
            ]
            n_pred_missing = len(missing_pred)
            n_pred_present = len(patch_dirs) - n_pred_missing
            if n_disk_missing or n_pred_missing:
                partial = {
                    "roi": roi_dir.name,
                    "department": department,
                    "n_subtiles_total": n_total,
                    "n_subtiles_on_disk": int(
                        coverage.get("n_subtiles_present", len(patch_dirs))
                    ),
                    "n_subtiles_missing_disk": n_disk_missing,
                    "n_predictions_present": n_pred_present,
                    "n_predictions_missing": n_pred_missing,
                    "missing_patch_ids_disk": coverage.get("missing_patch_ids", []),
                    "missing_prediction_ids": missing_pred,
                }
                partial_rois.append(partial)
                log(
                    f"[partial] {roi_label}: scoring with incomplete coverage -- "
                    f"disk missing {n_disk_missing}/{n_total} subtiles, "
                    f"predictions missing {n_pred_missing}/{len(patch_dirs)} "
                    f"(APLS vs full-ROI GT can be pessimistic; see partial_rois in JSON)."
                )

            try:
                roi_timings: Optional[Dict[str, float]] = {} if profile else None
                with _timed(profiler, "stitch", local=roi_timings):
                    roi_probs, roi_grid = nps.stitch_roi_predictions(
                        patch_dirs,
                        save_path,
                        combine=overlap_combine,
                        allow_missing_predictions=True,
                    )
            except FileNotFoundError as exc:
                log(f"[excluded] {roi_label}: no prediction file(s) -- {exc}")
                n_rois_skipped += 1
                excluded_rois.append(
                    {
                        "roi": roi_dir.name,
                        "department": department,
                        "reason": "missing_prediction_files",
                        "detail": str(exc),
                    }
                )
                pbar.update(1)
                continue
            n_rois_processed += 1

            if roi_probs.shape[0] < len(types):
                raise ValueError(
                    f"{roi_label}: stitched probs have {roi_probs.shape[0]} channels "
                    f"but network_types has {len(types)}: {list(types)}"
                )

            last_score: Optional[float] = None
            last_network_type: Optional[str] = None
            for channel_idx, network_type in enumerate(types):
                if not flags.get(network_type, False):
                    continue
                channel_timings: Optional[Dict[str, float]] = {} if profile else None
                with _timed(profiler, "build_pred_graph", local=channel_timings):
                    processed = _build_predicted_graph(
                        roi_probs,
                        roi_grid,
                        channel_idx,
                        threshold=threshold,
                        connectivity=connectivity,
                        rdp_epsilon_m=rdp_epsilon_m,
                        endpoint_fix_enabled=endpoint_fix_enabled,
                        endpoint_fix_stage=endpoint_fix_stage,
                        merge_hop_threshold=merge_hop_threshold,
                        radius_fix_radius_m=radius_fix_radius_m,
                        open_iterations=open_iterations,
                        close_iterations=close_iterations,
                        morph_connectivity=morph_connectivity,
                        remove_small_objects_enabled=remove_small_objects_enabled,
                        remove_small_objects_min_size_px=remove_small_objects_min_size_px,
                        skeletonize_enabled=skeletonize_enabled,
                        min_component_nodes=min_component_nodes,
                        collect_timings=profile,
                    )
                if profiler is not None:
                    profiler.add_dict(processed.stage_timings)
                if channel_timings is not None:
                    _merge_timings(channel_timings, processed.stage_timings)
                pred_graph = apls.apls_graph_from_pixel_graph(processed.graph_final)

                gt_path = nlu.expected_exported_graph_path(
                    network_graphs_root, roi_dir, network_type
                )
                with _timed(profiler, "load_gt", local=channel_timings):
                    loaded_gt = nlu.load_roi_exported_network_graph(gt_path)
                gt_graph = apls.apls_graph_from_loaded_graph(loaded_gt)

                with _timed(profiler, "apls", local=channel_timings):
                    result = apls.apls_symmetric_score(
                        gt_graph,
                        pred_graph,
                        roi=roi_dir.name,
                        network_type=network_type,
                        densify=apls_densify,
                        snap_to_edge=apls_snap_to_edge,
                        symmetric=apls_symmetric,
                        max_nodes_exact=apls_max_nodes_exact,
                        min_path_length_m=apls_min_path_length_m,
                    )
                results.append(result)
                last_score = float(result.score)
                last_network_type = network_type
                if channel_timings is not None and roi_timings is not None:
                    # High-level buckets only for ROI totals (pipeline stages nest under
                    # build_pred_graph and must not be summed twice).
                    for key in ("build_pred_graph", "load_gt", "apls"):
                        if key in channel_timings:
                            roi_timings[key] = (
                                roi_timings.get(key, 0.0) + channel_timings[key]
                            )
                    channel_total = (
                        channel_timings.get("build_pred_graph", 0.0)
                        + channel_timings.get("load_gt", 0.0)
                        + channel_timings.get("apls", 0.0)
                    )
                    detail = {
                        k: v
                        for k, v in channel_timings.items()
                        if k != "build_pred_graph"
                    }
                    log(
                        f"[profile] {roi_label} {network_type} "
                        f"total={channel_total:.3f}s | {_format_timings(detail)}"
                    )

            if profile and roi_timings is not None:
                roi_total = sum(roi_timings.values())
                log(
                    f"[profile] {roi_label} ROI total={roi_total:.3f}s | "
                    f"{_format_timings(roi_timings)}"
                )

            postfix = {"roi": roi_label}
            if last_network_type is not None and last_score is not None:
                postfix["type"] = last_network_type
                postfix["score"] = f"{last_score:.3f}"
            pbar.set_postfix(postfix, refresh=False)
            pbar.update(1)
    finally:
        pbar.close()

    summary = apls.aggregate_dataset_apls(results)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "split": split,
            "threshold": threshold,
            "overlap_combine": overlap_combine,
            "connectivity": connectivity,
            "rdp_epsilon_m": rdp_epsilon_m,
            "endpoint_fix_enabled": endpoint_fix_enabled,
            "endpoint_fix_stage": endpoint_fix_stage,
            "merge_hop_threshold": merge_hop_threshold,
            "apls_max_nodes_exact": apls_max_nodes_exact,
            "apls_densify": apls_densify,
            "apls_snap_to_edge": apls_snap_to_edge,
            "apls_symmetric": apls_symmetric,
            "radius_fix_radius_m": radius_fix_radius_m,
            "apls_min_path_length_m": apls_min_path_length_m,
            "open_iterations": open_iterations,
            "close_iterations": close_iterations,
            "morph_connectivity": morph_connectivity,
            "remove_small_objects_enabled": remove_small_objects_enabled,
            "remove_small_objects_min_size_px": remove_small_objects_min_size_px,
            "skeletonize_enabled": skeletonize_enabled,
            "min_component_nodes": min_component_nodes,
            "save_path": str(save_path),
            "network_graphs_root": str(network_graphs_root),
            "split_manifest_csv": str(split_manifest_csv),
            "network_types": list(types),
            "max_rois": max_rois,
            "missing_tiles_file": str(missing_tiles_file),
            "n_known_missing": len(known_missing),
            "n_skipped_known_missing": int(n_skipped_known_missing),
            "profile": bool(profile),
        },
        "n_rois_processed": n_rois_processed,
        "n_rois_skipped": n_rois_skipped,
        "n_rois_excluded_total": len(excluded_rois),
        "excluded_rois": excluded_rois,
        "partial_rois": partial_rois,
        "per_channel": summary["per_channel"],
        "per_channel_gt_to_pred": summary["per_channel_gt_to_pred"],
        "per_channel_pred_to_gt": summary["per_channel_pred_to_gt"],
        "macro_apls": summary["macro_apls"],
        "per_roi": [
            {
                "roi": r.roi,
                "department": department_by_roi.get(r.roi, ""),
                "network_type": r.network_type,
                "score": r.score,
                "score_gt_to_pred": r.score_gt_to_pred,
                "score_pred_to_gt": r.score_pred_to_gt,
                "numerator": r.numerator,
                "denom": r.denom,
                "numerator_pred_to_gt": r.numerator_pred_to_gt,
                "denom_pred_to_gt": r.denom_pred_to_gt,
                "n_nodes_gt": r.n_nodes_gt,
                "n_nodes_pred": r.n_nodes_pred,
                "n_edges_gt": r.n_edges_gt,
                "n_edges_pred": r.n_edges_pred,
            }
            for r in results
        ],
    }
    if profiler is not None:
        payload["profiling"] = profiler.summary()

    json_path = out_dir / "network_apls_metrics.json"
    tmp_path = str(json_path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, json_path)
    print(f"Wrote metrics to: {json_path}")

    csv_path = out_dir / "network_apls_per_roi.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "roi",
                "department",
                "network_type",
                "score",
                "score_gt_to_pred",
                "score_pred_to_gt",
                "numerator",
                "denom",
                "numerator_pred_to_gt",
                "denom_pred_to_gt",
                "n_nodes_gt",
                "n_nodes_pred",
                "n_edges_gt",
                "n_edges_pred",
            ],
        )
        writer.writeheader()
        for row in payload["per_roi"]:
            writer.writerow(row)
    print(f"Wrote per-ROI table to: {csv_path}")

    print(f"per_channel: {summary['per_channel']}")
    print(f"per_channel_gt_to_pred: {summary['per_channel_gt_to_pred']}")
    print(f"per_channel_pred_to_gt: {summary['per_channel_pred_to_gt']}")
    print(f"macro_apls: {summary['macro_apls']}")
    if profiler is not None:
        profiler.print_report()
    return payload


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate predicted network graphs (built from stitched per-ROI "
            "probability rasters) against Flair3D-build-exported GT graphs via APLS."
        )
    )
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Test/PreciseEvaluator output dir containing {patch_id}_logits_network.npy",
    )
    p.add_argument("--network_graphs_root", type=str, required=True)
    p.add_argument("--split_manifest_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--overlap_combine",
        type=str,
        default="nanmean",
        choices=["nanmean", "max", "first"],
    )
    p.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    p.add_argument("--rdp_epsilon_m", type=float, default=2.0)
    p.add_argument(
        "--endpoint_fix_enabled",
        type=_parse_bool,
        default=True,
        help="Run degree-1 diagonal endpoint repair in the predicted-graph pipeline "
        "(default: true). Pass false/0/off to skip.",
    )
    p.add_argument(
        "--endpoint_fix_stage",
        type=str,
        default="pre_rdp",
        choices=["pre_rdp", "post_rdp"],
        help="When to run endpoint-fix relative to RDP (ignored if "
        "--endpoint_fix_enabled=false).",
    )
    p.add_argument("--merge_hop_threshold", type=float, default=2.5)
    p.add_argument(
        "--min_component_nodes",
        type=int,
        default=5,
        help=(
            "Drop predicted-graph connected components with fewer than this many nodes, "
            "applied last (after merge/radius-fix). Pass 0 or 1 to disable."
        ),
    )
    p.add_argument(
        "--apls_max_nodes_exact",
        type=_parse_optional_int,
        default=None,
        help=(
            "Hard cap on exact O(V^2) APLS after densification (default: none = no cap). "
            "Pass a positive int to enable."
        ),
    )
    p.add_argument(
        "--max_rois", type=int, default=None, help="Optional limit on number of ROIs (debug)"
    )
    p.add_argument(
        "--apls_densify",
        type=_parse_optional_float,
        default=50.0,
        help=(
            "Max edge length in meters for densification (default: 50). "
            "Pass none/null to disable."
        ),
    )
    p.add_argument(
        "--apls_snap_to_edge",
        type=_parse_optional_float,
        default=4.0,
        help=(
            "Snap-to-edge radius in meters (default: 4). "
            "Pass none/null for unrestricted nearest-node matching."
        ),
    )
    p.add_argument(
        "--apls_symmetric",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=True,
        help="Score both directions and take harmonic mean (default: true)",
    )
    p.add_argument(
        "--no_apls_symmetric",
        action="store_true",
        help="Score only GT->pred (legacy unidirectional APLS)",
    )
    p.add_argument(
        "--radius_fix_radius_m",
        type=_parse_optional_float,
        default=None,
        help=(
            "Radius (meters) to connect every predicted-graph endpoint/isolated node to every "
            "other one within that radius, applied after merge (extension of endpoint-fix). "
            "Pass none/null (default) to disable -- opt in explicitly, it changes the predicted "
            "graph and therefore reported APLS numbers."
        ),
    )
    p.add_argument(
        "--apls_min_path_length_m",
        type=_parse_optional_float,
        default=None,
        help=(
            "SpaceNet-style short-path filter: GT/pred pairs whose shortest path is under this "
            "many meters are excluded from APLS scoring (default: none/disabled -- e.g. pass 5 "
            "for roads). Pass none/null to disable."
        ),
    )
    p.add_argument(
        "--open_iterations",
        type=int,
        default=0,
        help="Erode-then-dilate pass count before remove_small/skeletonize, applied before "
        "closing; 0 disables opening (default: 0, i.e. no morphology by default here, "
        "unlike the visualization scripts -- kept off so the official metric stays "
        "comparable to the baseline unless morphology is explicitly requested).",
    )
    p.add_argument(
        "--close_iterations",
        type=int,
        default=0,
        help="Dilate-then-erode pass count before remove_small/skeletonize, applied after "
        "opening; 0 disables closing (default: 0).",
    )
    p.add_argument(
        "--morph_connectivity",
        type=int,
        default=4,
        choices=[4, 8],
        help="Morphology structuring-element connectivity (default: 4)",
    )
    p.add_argument(
        "--remove_small_objects_enabled",
        type=_parse_bool,
        default=True,
        help="Drop small connected components before skeletonize (default: true)",
    )
    p.add_argument(
        "--remove_small_objects_min_size_px",
        type=int,
        default=8,
        help="Min connected-component size in pixels (default: 8)",
    )
    p.add_argument(
        "--skeletonize_enabled",
        type=_parse_bool,
        default=True,
        help="Zhang-Suen skeletonize wide masks to 1px centerlines (default: true)",
    )
    p.add_argument(
        "--network_types",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Channel names / order matching logits_network.npy channels "
            f"(default: {' '.join(NETWORK_TYPES)}). Pass e.g. ROADS RAILROADS "
            "when training dropped TRANSMISSION_LINES."
        ),
    )
    p.add_argument(
        "--missing_tiles_file",
        type=str,
        default=None,
        help=(
            "Exclusion list of known-missing (split, patch_id), same role as "
            "Flair3DDataset / rasterize_network. Default: "
            "data/flair3d_plus/missing_coord_tiles.details.csv"
        ),
    )
    p.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable the tqdm ROI progress bar (useful for CI / captured logs).",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Time pipeline stages (open/close/remove_small/skeletonize/pixel_graph/"
            "endpoint_fix/rdp/merge/radius_fix/min_component_nodes) and high-level "
            "phases (stitch/build_pred_graph/load_gt/apls); print a summary and write "
            "it under 'profiling' in the output JSON."
        ),
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    apls_symmetric = False if args.no_apls_symmetric else bool(args.apls_symmetric)
    missing = (
        Path(args.missing_tiles_file).resolve()
        if args.missing_tiles_file
        else None
    )
    run(
        Path(args.data_root).resolve(),
        Path(args.save_path).resolve(),
        Path(args.network_graphs_root).resolve(),
        Path(args.split_manifest_csv).resolve(),
        Path(args.out_dir).resolve(),
        split=args.split,
        threshold=args.threshold,
        overlap_combine=args.overlap_combine,
        connectivity=args.connectivity,
        rdp_epsilon_m=args.rdp_epsilon_m,
        endpoint_fix_enabled=args.endpoint_fix_enabled,
        endpoint_fix_stage=args.endpoint_fix_stage,
        merge_hop_threshold=args.merge_hop_threshold,
        apls_max_nodes_exact=args.apls_max_nodes_exact,
        max_rois=args.max_rois,
        apls_densify=args.apls_densify,
        apls_snap_to_edge=args.apls_snap_to_edge,
        apls_symmetric=apls_symmetric,
        radius_fix_radius_m=args.radius_fix_radius_m,
        apls_min_path_length_m=args.apls_min_path_length_m,
        open_iterations=args.open_iterations,
        close_iterations=args.close_iterations,
        morph_connectivity=args.morph_connectivity,
        remove_small_objects_enabled=args.remove_small_objects_enabled,
        remove_small_objects_min_size_px=args.remove_small_objects_min_size_px,
        skeletonize_enabled=args.skeletonize_enabled,
        min_component_nodes=args.min_component_nodes,
        network_types=args.network_types,
        missing_tiles_file=missing,
        show_progress=not args.no_progress,
        profile=bool(args.profile),
    )


if __name__ == "__main__":
    main()
