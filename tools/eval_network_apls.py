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
from pathlib import Path
from typing import List, Optional

import numpy as np

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
    return _coerce_optional_meters(value, default_when_true=50.0, name="densify")


def _coerce_snap_to_edge(value) -> Optional[float]:
    return _coerce_optional_meters(value, default_when_true=4.0, name="snap_to_edge")


def _build_predicted_graph(
    roi_probs: np.ndarray,
    roi_grid,
    channel_idx: int,
    *,
    threshold: float,
    connectivity: int,
    rdp_epsilon_m: float,
    endpoint_fix_stage: str,
    merge_weight_threshold: float,
    radius_fix_radius_m: Optional[float] = None,
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
        rdp_epsilon_m=rdp_epsilon_m,
        endpoint_fix_enabled=True,
        endpoint_fix_stage=endpoint_fix_stage,
        merge_enabled=True,
        merge_weight_threshold=merge_weight_threshold,
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
    endpoint_fix_stage: str = "pre_rdp",
    merge_weight_threshold: float = 2.5,
    max_nodes_exact: Optional[int] = 4000,
    max_rois: Optional[int] = None,
    densify: Optional[float] = 50.0,
    snap_to_edge: Optional[float] = 4.0,
    symmetric: bool = True,
    radius_fix_radius_m: Optional[float] = None,
    min_path_length_m: Optional[float] = None,
    network_types: Optional[List[str]] = None,
) -> dict:
    densify = _coerce_densify(densify)
    snap_to_edge = _coerce_snap_to_edge(snap_to_edge)
    types = tuple(network_types) if network_types else NETWORK_TYPES
    patches, _ = nps.load_manifest_patches(split_manifest_csv, splits=[split])
    roi_items, excluded_rois = nps.group_by_roi_complete_only(patches, data_root)
    if excluded_rois:
        print(
            f"[excluded] {len(excluded_rois)} ROI(s) with no usable local subtiles "
            "(see 'excluded_rois' in the output JSON)."
        )
        for info in excluded_rois:
            print(
                f"  - {info['roi']}: reason={info.get('reason')} "
                f"missing {info.get('n_subtiles_missing', '?')}/"
                f"{info.get('n_subtiles_total', '?')} subtiles"
            )
    if max_rois is not None:
        roi_items = roi_items[: max(0, int(max_rois))]

    results: List[apls.ApsSymmetricResult] = []
    n_rois_processed = 0
    n_rois_skipped = 0
    partial_rois: List[dict] = []

    for roi_dir, flags, patch_dirs, coverage in roi_items:
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
                "n_subtiles_total": n_total,
                "n_subtiles_on_disk": int(coverage.get("n_subtiles_present", len(patch_dirs))),
                "n_subtiles_missing_disk": n_disk_missing,
                "n_predictions_present": n_pred_present,
                "n_predictions_missing": n_pred_missing,
                "missing_patch_ids_disk": coverage.get("missing_patch_ids", []),
                "missing_prediction_ids": missing_pred,
            }
            partial_rois.append(partial)
            print(
                f"[partial] {roi_dir.name}: scoring with incomplete coverage -- "
                f"disk missing {n_disk_missing}/{n_total} subtiles, "
                f"predictions missing {n_pred_missing}/{len(patch_dirs)} "
                f"(APLS vs full-ROI GT can be pessimistic; see partial_rois in JSON)."
            )

        try:
            roi_probs, roi_grid = nps.stitch_roi_predictions(
                patch_dirs,
                save_path,
                combine=overlap_combine,
                allow_missing_predictions=True,
            )
        except FileNotFoundError as exc:
            print(f"[excluded] {roi_dir.name}: no prediction file(s) -- {exc}")
            n_rois_skipped += 1
            excluded_rois.append(
                {
                    "roi": roi_dir.name,
                    "reason": "missing_prediction_files",
                    "detail": str(exc),
                }
            )
            continue
        n_rois_processed += 1

        if roi_probs.shape[0] < len(types):
            raise ValueError(
                f"{roi_dir.name}: stitched probs have {roi_probs.shape[0]} channels "
                f"but network_types has {len(types)}: {list(types)}"
            )

        for channel_idx, network_type in enumerate(types):
            if not flags.get(network_type, False):
                continue
            processed = _build_predicted_graph(
                roi_probs,
                roi_grid,
                channel_idx,
                threshold=threshold,
                connectivity=connectivity,
                rdp_epsilon_m=rdp_epsilon_m,
                endpoint_fix_stage=endpoint_fix_stage,
                merge_weight_threshold=merge_weight_threshold,
                radius_fix_radius_m=radius_fix_radius_m,
            )
            pred_graph = apls.apls_graph_from_pixel_graph(processed.graph_final)

            gt_path = nlu.expected_exported_graph_path(
                network_graphs_root, roi_dir, network_type
            )
            loaded_gt = nlu.load_roi_exported_network_graph(gt_path)
            gt_graph = apls.apls_graph_from_loaded_graph(loaded_gt)

            result = apls.apls_symmetric_score(
                gt_graph,
                pred_graph,
                roi=roi_dir.name,
                network_type=network_type,
                densify=densify,
                snap_to_edge=snap_to_edge,
                symmetric=symmetric,
                max_nodes_exact=max_nodes_exact,
                min_path_length_m=min_path_length_m,
            )
            results.append(result)
            print(
                f"{roi_dir.name:24s} {network_type:20s} score={result.score:.4f} "
                f"gt->pred={result.score_gt_to_pred:.4f} "
                f"pred->gt={result.score_pred_to_gt:.4f} "
                f"n_gt={result.n_nodes_gt} n_pred={result.n_nodes_pred}"
            )

    summary = apls.aggregate_dataset_apls(results)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "split": split,
            "threshold": threshold,
            "overlap_combine": overlap_combine,
            "connectivity": connectivity,
            "rdp_epsilon_m": rdp_epsilon_m,
            "endpoint_fix_stage": endpoint_fix_stage,
            "merge_weight_threshold": merge_weight_threshold,
            "max_nodes_exact": max_nodes_exact,
            "densify": densify,
            "snap_to_edge": snap_to_edge,
            "symmetric": symmetric,
            "radius_fix_radius_m": radius_fix_radius_m,
            "min_path_length_m": min_path_length_m,
            "save_path": str(save_path),
            "network_graphs_root": str(network_graphs_root),
            "split_manifest_csv": str(split_manifest_csv),
            "network_types": list(types),
            "max_rois": max_rois,
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
        "--endpoint_fix_stage",
        type=str,
        default="pre_rdp",
        choices=["pre_rdp", "post_rdp"],
    )
    p.add_argument("--merge_weight_threshold", type=float, default=2.5)
    p.add_argument(
        "--max_nodes_exact",
        type=_parse_optional_int,
        default=4000,
        help=(
            "Hard cap on exact O(V^2) APLS after densification (default: 4000). "
            "Pass none/null to disable the cap."
        ),
    )
    p.add_argument(
        "--max_rois", type=int, default=None, help="Optional limit on number of ROIs (debug)"
    )
    p.add_argument(
        "--densify",
        type=_parse_optional_float,
        default=50.0,
        help=(
            "Max edge length in meters for densification (default: 50). "
            "Pass none/null to disable."
        ),
    )
    p.add_argument(
        "--snap_to_edge",
        type=_parse_optional_float,
        default=4.0,
        help=(
            "Snap-to-edge radius in meters (default: 4). "
            "Pass none/null for unrestricted nearest-node matching."
        ),
    )
    p.add_argument(
        "--symmetric",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=True,
        help="Score both directions and take harmonic mean (default: true)",
    )
    p.add_argument(
        "--no_symmetric",
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
        "--min_path_length_m",
        type=_parse_optional_float,
        default=None,
        help=(
            "SpaceNet-style short-path filter: GT/pred pairs whose shortest path is under this "
            "many meters are excluded from APLS scoring (default: none/disabled -- e.g. pass 5 "
            "for roads). Pass none/null to disable."
        ),
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
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    symmetric = False if args.no_symmetric else bool(args.symmetric)
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
        endpoint_fix_stage=args.endpoint_fix_stage,
        merge_weight_threshold=args.merge_weight_threshold,
        max_nodes_exact=args.max_nodes_exact,
        max_rois=args.max_rois,
        densify=args.densify,
        snap_to_edge=args.snap_to_edge,
        symmetric=symmetric,
        radius_fix_radius_m=args.radius_fix_radius_m,
        min_path_length_m=args.min_path_length_m,
        network_types=args.network_types,
    )


if __name__ == "__main__":
    main()
