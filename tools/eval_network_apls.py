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
) -> "ngp.ProcessedNetworkGraph":
    prob = roi_probs[channel_idx]
    # Unobserved (NaN) cells are treated as background -- see plan's design-decision note:
    # avoids inventing positive network pixels in unscanned area.
    mask = np.isfinite(prob) & (prob >= threshold)
    return ngp.build_processed_network_graph_from_mask(
        mask,
        roi_grid,
        connectivity=connectivity,
        rdp_epsilon_m=rdp_epsilon_m,
        endpoint_fix_enabled=True,
        endpoint_fix_stage=endpoint_fix_stage,
        merge_enabled=True,
        merge_weight_threshold=merge_weight_threshold,
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
    max_nodes_exact: int = 4000,
    max_rois: Optional[int] = None,
) -> dict:
    patches, _ = nps.load_manifest_patches(split_manifest_csv, splits=[split])
    roi_items, excluded_rois = nps.group_by_roi_complete_only(patches, data_root)
    if excluded_rois:
        print(
            f"[excluded] {len(excluded_rois)} ROI(s) dropped entirely: incomplete local "
            "subtile mirror (would bias APLS -- run on a fully-mirrored manifest, e.g. "
            "Jean Zay, for complete/official numbers). See 'excluded_rois' in the output "
            "JSON for the full list."
        )
        for info in excluded_rois:
            print(
                f"  - {info['roi']}: missing {info['n_subtiles_missing']}/"
                f"{info['n_subtiles_total']} subtiles"
            )
    if max_rois is not None:
        roi_items = roi_items[: max(0, int(max_rois))]

    results: List[apls.ApsPairResult] = []
    n_rois_processed = 0
    n_rois_skipped = 0

    for roi_dir, flags, patch_dirs in roi_items:
        try:
            roi_probs, roi_grid = nps.stitch_roi_predictions(
                patch_dirs, save_path, combine=overlap_combine
            )
        except FileNotFoundError as exc:
            print(f"[excluded] {roi_dir.name}: missing prediction file(s) -- {exc}")
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

        for channel_idx, network_type in enumerate(NETWORK_TYPES):
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
            )
            pred_graph = apls.apls_graph_from_pixel_graph(processed.graph_final)

            gt_path = nlu.expected_exported_graph_path(
                network_graphs_root, roi_dir, network_type
            )
            loaded_gt = nlu.load_roi_exported_network_graph(gt_path)
            gt_graph = apls.apls_graph_from_loaded_graph(loaded_gt)

            result = apls.apls_pair_score(
                gt_graph,
                pred_graph,
                roi=roi_dir.name,
                network_type=network_type,
                max_nodes_exact=max_nodes_exact,
            )
            results.append(result)
            print(
                f"{roi_dir.name:24s} {network_type:20s} score={result.score:.4f} "
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
            "save_path": str(save_path),
            "network_graphs_root": str(network_graphs_root),
            "split_manifest_csv": str(split_manifest_csv),
        },
        "n_rois_processed": n_rois_processed,
        "n_rois_skipped": n_rois_skipped,
        "n_rois_excluded_total": len(excluded_rois),
        "excluded_rois": excluded_rois,
        "per_channel": summary["per_channel"],
        "macro_apls": summary["macro_apls"],
        "per_roi": [
            {
                "roi": r.roi,
                "network_type": r.network_type,
                "score": r.score,
                "numerator": r.numerator,
                "denom": r.denom,
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
                "numerator",
                "denom",
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
    p.add_argument("--max_nodes_exact", type=int, default=4000)
    p.add_argument(
        "--max_rois", type=int, default=None, help="Optional limit on number of ROIs (debug)"
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
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
    )


if __name__ == "__main__":
    main()
