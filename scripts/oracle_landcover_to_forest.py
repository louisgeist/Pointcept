#!/usr/bin/env python3
"""
Oracle: map land-cover (segment.npy) labels to binary forest predictions.

Evaluates how well vegetation-type segment classes predict the FOREST modality
(forest.npy), without running a model.

Settings (segment class names -> Forest; everything else including Void -> Not Forest):
  - trees_brushwood: Brushwood, Deciduous, Coniferous
  - all_vegetation:  Herbaceous, Vineyard, Brushwood, Deciduous, Coniferous

Oracle never predicts Void. Forest GT Void (ignore_index) is excluded from metrics.

Example:
python scripts/oracle_landcover_to_forest.py \\
  --data_root data/flair3d_plus \\
  --csv_manifest data/flair3d_plus/raw/scene_split_manifest.csv \\
  --splits test \\
  --settings trees_brushwood,all_vegetation \\
  --num_workers 16 \\
  --output_dir stats/flair3d/oracle_landcover_to_forest
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
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Oracle settings: segment class names that map to Forest (1)
# ---------------------------------------------------------------------------
ORACLE_SETTING_CLASS_NAMES: Dict[str, Tuple[str, ...]] = {
    "trees_brushwood": ("Brushwood", "Deciduous", "Coniferous"),
    "all_vegetation": (
        "Herbaceous",
        "Vineyard",
        "Brushwood",
        "Deciduous",
        "Coniferous",
    ),
}

FOREST_NUM_CLASSES = 2  # Not Forest, Forest (Void is ignore)
FOREST_CLASS_NAMES = ("Not Forest", "Forest")

# Populated in main() before workers start; read by child processes.
_WORKER_STATE: Dict[str, object] = {}


def _load_label_remap_module():
    module_name = "flair3d_label_remap_oracle_script"
    path = os.path.join(
        REPO_ROOT,
        "pointcept",
        "datasets",
        "preprocessing",
        "flair3d_plus",
        "flair3d_label_remap.py",
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load flair3d_label_remap from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def intersection_and_union(
    output: np.ndarray, target: np.ndarray, k: int, ignore_index: int = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU intersection/union/target hist (same logic as pointcept.utils.misc)."""
    assert output.ndim in (1, 2, 3)
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    mask = target != ignore_index
    output, target = output[mask], target[mask]
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(k + 1))
    area_output, _ = np.histogram(output, bins=np.arange(k + 1))
    area_target, _ = np.histogram(target, bins=np.arange(k + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def f1_scores_from_hist(
    intersection: np.ndarray, union: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, float]:
    intersection = np.asarray(intersection, dtype=np.float64)
    union = np.asarray(union, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pred_count = union + intersection - target
    precision = np.divide(
        intersection,
        pred_count,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=pred_count > 0,
    )
    recall = np.divide(
        intersection,
        target,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=target > 0,
    )
    pr_sum = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        pr_sum,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=pr_sum > 0,
    )
    return f1, float(np.mean(f1))


def mean_iou_from_hist(intersection: np.ndarray, union: np.ndarray) -> float:
    intersection = np.asarray(intersection, dtype=np.float64)
    union = np.asarray(union, dtype=np.float64)
    iou_class = intersection / (union + 1e-10)
    mask = union != 0
    if mask.any():
        return float(np.mean(iou_class[mask]))
    return 0.0


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def parse_csv_list(arg: str) -> List[str]:
    return [token.strip() for token in arg.split(",") if token.strip()]


def parse_manifest_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def build_scene_path(
    output_root: str,
    split: str,
    patch_id: str,
    dept_year: str,
    roi: str,
) -> str:
    return os.path.join(output_root, split, f"{dept_year}_LIDARHD", roi, patch_id)


@dataclass(frozen=True)
class SceneRecord:
    split: str
    patch_id: str
    scene_path: str


@dataclass
class SceneOracleResult:
    split: str
    patch_id: str
    scene_path: str
    n_points: int
    # setting -> (intersection, union, target, confusion_2x2, n_eval)
    metrics: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]
    error: str = ""


def load_hardcoded_excluded_tiles() -> set[tuple[str, str]]:
    excluded: set[tuple[str, str]] = set()
    details_csv = os.path.join(
        REPO_ROOT, "data", "flair3d_plus", "missing_coord_tiles.details.csv"
    )
    if not os.path.isfile(details_csv):
        return excluded
    with open(details_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("reason") != "missing_coord_file":
                continue
            split = (row.get("split") or "").strip()
            patch_id = (row.get("patch_id") or "").strip()
            if split and patch_id:
                excluded.add((split, patch_id))
    return excluded


def load_missing_tiles_manifest(path: str | None) -> set[tuple[str, str]]:
    missing_tiles: set[tuple[str, str]] = set()
    if not path or not os.path.isfile(path):
        if path:
            print(f"Warning: missing tiles manifest not found: {path}")
        return missing_tiles
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [part.strip() for part in stripped.split(",", 2)]
            if len(parts) < 2:
                continue
            split, patch_id = parts[0], parts[1]
            if split and patch_id:
                missing_tiles.add((split, patch_id))
    return missing_tiles


def load_too_small_tiles_manifest(path: str | None) -> set[tuple[str, str]]:
    too_small_tiles: set[tuple[str, str]] = set()
    if not path or not os.path.isfile(path):
        if path:
            print(f"Warning: too-small tiles manifest not found: {path}")
        return too_small_tiles
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_split = (row.get("split") or "").strip()
            patch_id = (row.get("patch_id") or "").strip()
            if row_split and patch_id:
                too_small_tiles.add((row_split, patch_id))
    return too_small_tiles


def load_scene_records(
    data_root: str,
    csv_manifest: str,
    target_splits: set[str],
    excluded_tiles: set[tuple[str, str]],
) -> list[SceneRecord]:
    scene_records: list[SceneRecord] = []
    with open(csv_manifest, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "patch_id", "LIDARHD"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise KeyError(
                f"Missing required columns in manifest: {sorted(missing_cols)}"
            )

        for row in reader:
            split = str(row["split"]).strip()
            patch_id = str(row["patch_id"]).strip()
            if not split or not patch_id:
                continue
            if split not in target_splits:
                continue
            if not parse_manifest_bool(row.get("LIDARHD")):
                continue
            if (split, patch_id) in excluded_tiles:
                continue

            dept_year = (row.get("dept_year") or "").strip() or patch_id.split("_", 2)[0]
            roi = (row.get("roi") or "").strip() or patch_id.split("_", 2)[1]
            scene_path = build_scene_path(data_root, split, patch_id, dept_year, roi)
            scene_records.append(
                SceneRecord(split=split, patch_id=patch_id, scene_path=scene_path)
            )
    return scene_records


def class_ids_from_names(
    names: Sequence[str], class_names: Sequence[str], *, setting: str
) -> np.ndarray:
    name_to_id = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in class_names if name not in name_to_id]
    if missing:
        known = ", ".join(names)
        raise KeyError(
            f"Setting '{setting}': unknown segment class names {missing}. "
            f"Available: {known}"
        )
    return np.asarray([name_to_id[name] for name in class_names], dtype=np.int32)


def build_setting_vegetation_ids(
    segment_names: Sequence[str], settings: Sequence[str]
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for setting in settings:
        if setting not in ORACLE_SETTING_CLASS_NAMES:
            known = ", ".join(sorted(ORACLE_SETTING_CLASS_NAMES))
            raise KeyError(f"Unknown setting '{setting}'. Supported: {known}")
        out[setting] = class_ids_from_names(
            segment_names,
            ORACLE_SETTING_CLASS_NAMES[setting],
            setting=setting,
        )
    return out


def oracle_predict_forest(
    segment: np.ndarray, vegetation_ids: np.ndarray
) -> np.ndarray:
    """Map segment labels to forest pred in {0, 1}. Never predicts Void."""
    pred = np.zeros(segment.shape, dtype=np.int32)
    pred[np.isin(segment, vegetation_ids)] = 1
    return pred


def confusion_2x2(
    pred: np.ndarray, target: np.ndarray, ignore_index: int
) -> np.ndarray:
    """Rows = GT (0/1), cols = pred (0/1); ignore GT void."""
    mask = target != ignore_index
    p = pred[mask]
    t = target[mask]
    conf = np.zeros((2, 2), dtype=np.int64)
    for gt_c in (0, 1):
        for pr_c in (0, 1):
            conf[gt_c, pr_c] = int(np.sum((t == gt_c) & (p == pr_c)))
    return conf


def analyze_scene(
    scene: SceneRecord,
    settings: Sequence[str],
    vegetation_ids_by_setting: Dict[str, np.ndarray],
    forest_ignore_index: int,
) -> SceneOracleResult:
    segment_path = os.path.join(scene.scene_path, "segment.npy")
    forest_path = os.path.join(scene.scene_path, "forest.npy")

    if not os.path.isfile(segment_path):
        return SceneOracleResult(
            split=scene.split,
            patch_id=scene.patch_id,
            scene_path=scene.scene_path,
            n_points=0,
            metrics={},
            error="missing_segment",
        )
    if not os.path.isfile(forest_path):
        return SceneOracleResult(
            split=scene.split,
            patch_id=scene.patch_id,
            scene_path=scene.scene_path,
            n_points=0,
            metrics={},
            error="missing_forest",
        )

    segment = np.asarray(np.load(segment_path, mmap_mode="r")).reshape(-1)
    forest = np.asarray(np.load(forest_path, mmap_mode="r")).reshape(-1)
    if segment.shape != forest.shape:
        return SceneOracleResult(
            split=scene.split,
            patch_id=scene.patch_id,
            scene_path=scene.scene_path,
            n_points=int(segment.shape[0]),
            metrics={},
            error=f"shape_mismatch:segment{segment.shape}_forest{forest.shape}",
        )

    n_points = int(segment.shape[0])
    metrics: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for setting in settings:
        pred = oracle_predict_forest(segment, vegetation_ids_by_setting[setting])
        inter, union, target_hist = intersection_and_union(
            pred, forest, FOREST_NUM_CLASSES, ignore_index=forest_ignore_index
        )
        conf = confusion_2x2(pred, forest, forest_ignore_index)
        n_eval = int(np.sum(forest != forest_ignore_index))
        metrics[setting] = (inter, union, target_hist, conf, n_eval)

    return SceneOracleResult(
        split=scene.split,
        patch_id=scene.patch_id,
        scene_path=scene.scene_path,
        n_points=n_points,
        metrics=metrics,
    )


def _process_scene(scene: SceneRecord) -> SceneOracleResult:
    try:
        return analyze_scene(
            scene,
            settings=_WORKER_STATE["settings"],  # type: ignore[arg-type]
            vegetation_ids_by_setting=_WORKER_STATE["vegetation_ids"],  # type: ignore[arg-type]
            forest_ignore_index=int(_WORKER_STATE["forest_ignore_index"]),
        )
    except Exception as exc:  # noqa: BLE001 — keep worker alive, record error
        return SceneOracleResult(
            split=scene.split,
            patch_id=scene.patch_id,
            scene_path=scene.scene_path,
            n_points=0,
            metrics={},
            error=str(exc),
        )


def _init_worker(state: Dict[str, object]) -> None:
    global _WORKER_STATE
    _WORKER_STATE = state


def aggregate_metrics(
    results: Sequence[SceneOracleResult], settings: Sequence[str]
) -> Dict[str, dict]:
    aggregates: Dict[str, dict] = {}
    for setting in settings:
        inter = np.zeros(FOREST_NUM_CLASSES, dtype=np.float64)
        union = np.zeros(FOREST_NUM_CLASSES, dtype=np.float64)
        target = np.zeros(FOREST_NUM_CLASSES, dtype=np.float64)
        conf = np.zeros((2, 2), dtype=np.int64)
        n_eval = 0
        n_points = 0
        n_scenes_ok = 0

        for row in results:
            if row.error or setting not in row.metrics:
                continue
            i, u, t, c, ne = row.metrics[setting]
            inter += i.astype(np.float64)
            union += u.astype(np.float64)
            target += t.astype(np.float64)
            conf += c
            n_eval += ne
            n_points += row.n_points
            n_scenes_ok += 1

        iou_class = inter / (union + 1e-10)
        f1_class, macro_f1 = f1_scores_from_hist(inter, union, target)
        miou = mean_iou_from_hist(inter, union)

        per_class = {}
        for c_idx, name in enumerate(FOREST_CLASS_NAMES):
            per_class[name] = {
                "iou": float(iou_class[c_idx]),
                "f1": float(f1_class[c_idx]),
                "intersection": float(inter[c_idx]),
                "union": float(union[c_idx]),
                "target": float(target[c_idx]),
            }

        aggregates[setting] = {
            "miou": miou,
            "macro_f1": macro_f1,
            "per_class": per_class,
            "confusion": {
                "rows_gt": list(FOREST_CLASS_NAMES),
                "cols_pred": list(FOREST_CLASS_NAMES),
                "matrix": conf.tolist(),
            },
            "n_points_total": int(n_points),
            "n_points_eval": int(n_eval),
            "n_scenes_ok": int(n_scenes_ok),
        }
    return aggregates


def _print_summary(summary: dict) -> None:
    print("\n=== Oracle land-cover -> forest ===")
    print(
        f"splits={summary['splits']}  scenes_ok={summary['n_scenes_ok']}  "
        f"errors={summary['n_scenes_error']}"
    )
    print(f"segment_definition={summary['segment_definition']}")
    for setting, block in summary["settings"].items():
        veg = summary["vegetation_ids"][setting]
        print(f"\n--- setting: {setting} ---")
        print(f"vegetation class ids: {veg}")
        print(f"mIoU={block['miou']:.4f}  macro_F1={block['macro_f1']:.4f}")
        print(
            f"n_points_eval={block['n_points_eval']}  "
            f"n_points_total={block['n_points_total']}  "
            f"n_scenes_ok={block['n_scenes_ok']}"
        )
        for name, stats in block["per_class"].items():
            print(
                f"  {name}: IoU={stats['iou']:.4f} F1={stats['f1']:.4f} "
                f"(target={stats['target']:.0f})"
            )
        conf = block["confusion"]["matrix"]
        print(
            "  confusion [rows=GT, cols=pred]:\n"
            f"    [[{conf[0][0]:8d}, {conf[0][1]:8d}],\n"
            f"     [{conf[1][0]:8d}, {conf[1][1]:8d}]]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_root", default="data/flair3d_plus")
    parser.add_argument(
        "--csv_manifest",
        default="data/flair3d_plus/raw/scene_split_manifest.csv",
    )
    parser.add_argument(
        "--splits",
        default="test",
        help="Comma-separated splits (default: test)",
    )
    parser.add_argument(
        "--settings",
        default="trees_brushwood,all_vegetation",
        help=(
            "Comma-separated oracle settings "
            f"(default: trees_brushwood,all_vegetation). "
            f"Supported: {', '.join(sorted(ORACLE_SETTING_CLASS_NAMES))}"
        ),
    )
    parser.add_argument(
        "--segment_definition",
        default="v19",
        help="Segment label definition name (default: v19)",
    )
    parser.add_argument(
        "--forest_definition",
        default="default",
        help="Forest label definition name (default: default)",
    )
    parser.add_argument(
        "--missing_tiles_manifest",
        default="data/flair3d_plus/missing_ply_preflight.txt",
    )
    parser.add_argument(
        "--too_small_tiles_manifest",
        default="data/flair3d_plus/too_small_tiles.csv",
    )
    parser.add_argument("--no_exclude_hardcoded", action="store_true")
    parser.add_argument("--no_exclude_missing_manifest", action="store_true")
    parser.add_argument("--no_exclude_too_small", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument(
        "--max_scenes", type=int, default=0, help="Debug: limit scenes (0=all)"
    )
    parser.add_argument(
        "--existing_only",
        action="store_true",
        help="Keep only scenes that already have segment.npy and forest.npy",
    )
    parser.add_argument(
        "--output_dir",
        default="stats/flair3d/oracle_landcover_to_forest",
    )
    args = parser.parse_args()

    data_root = resolve_repo_path(args.data_root)
    csv_manifest = resolve_repo_path(args.csv_manifest)
    target_splits = set(parse_csv_list(args.splits))
    settings = parse_csv_list(args.settings)
    output_dir = resolve_repo_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not settings:
        raise ValueError("--settings must list at least one setting")
    unknown = [s for s in settings if s not in ORACLE_SETTING_CLASS_NAMES]
    if unknown:
        known = ", ".join(sorted(ORACLE_SETTING_CLASS_NAMES))
        raise KeyError(f"Unknown settings {unknown}. Supported: {known}")

    label_remap = _load_label_remap_module()
    segment_def = label_remap.get_definition("segment", args.segment_definition)
    forest_def = label_remap.get_definition("forest", args.forest_definition)
    # Forest trains on 2 classes; ignore is Void id in names.
    forest_ignore_index = int(forest_def.ignore_index)
    if forest_ignore_index != 2:
        print(
            f"Warning: forest ignore_index={forest_ignore_index} "
            f"(expected 2 for default forest definition)"
        )

    vegetation_ids = build_setting_vegetation_ids(segment_def.names, settings)
    vegetation_ids_serializable = {
        k: v.tolist() for k, v in vegetation_ids.items()
    }

    excluded: set[tuple[str, str]] = set()
    if not args.no_exclude_hardcoded:
        excluded |= load_hardcoded_excluded_tiles()
    if not args.no_exclude_missing_manifest:
        excluded |= load_missing_tiles_manifest(
            resolve_repo_path(args.missing_tiles_manifest)
        )
    if not args.no_exclude_too_small:
        excluded |= load_too_small_tiles_manifest(
            resolve_repo_path(args.too_small_tiles_manifest)
        )

    if not os.path.isfile(csv_manifest):
        raise FileNotFoundError(f"CSV manifest not found: {csv_manifest}")

    scenes = load_scene_records(data_root, csv_manifest, target_splits, excluded)
    if args.existing_only:
        before = len(scenes)
        scenes = [
            s
            for s in scenes
            if os.path.isfile(os.path.join(s.scene_path, "segment.npy"))
            and os.path.isfile(os.path.join(s.scene_path, "forest.npy"))
        ]
        print(
            f"existing_only: kept {len(scenes)}/{before} scenes with "
            "segment.npy + forest.npy"
        )
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]

    worker_state: Dict[str, object] = {
        "settings": settings,
        "vegetation_ids": vegetation_ids,
        "forest_ignore_index": forest_ignore_index,
    }
    global _WORKER_STATE
    _WORKER_STATE = worker_state

    print(f"data_root={data_root}")
    print(f"csv_manifest={csv_manifest}")
    print(f"splits={sorted(target_splits)}")
    print(f"settings={settings}")
    print(f"segment_definition={args.segment_definition}")
    print(f"forest_definition={args.forest_definition}")
    for setting, ids in vegetation_ids.items():
        names = [segment_def.names[i] for i in ids]
        print(f"  {setting}: ids={ids.tolist()} names={names}")
    print(f"excluded tiles: {len(excluded)}")
    print(f"tiles to scan: {len(scenes)}")

    results: List[SceneOracleResult] = []
    show_progress = not args.no_progress and len(scenes) > 0

    if args.num_workers <= 1:
        iterator = (_process_scene(scene) for scene in scenes)
        if show_progress:
            iterator = tqdm(iterator, total=len(scenes), desc="Scenes", unit="scene")
        results = list(iterator)
    else:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=(worker_state,),
        ) as pool:
            mapped = pool.map(_process_scene, scenes, chunksize=4)
            if show_progress:
                mapped = tqdm(mapped, total=len(scenes), desc="Scenes", unit="scene")
            results = list(mapped)

    ok = [r for r in results if not r.error]
    errors = [r for r in results if r.error]
    setting_metrics = aggregate_metrics(results, settings)

    split_tag = "_".join(sorted(target_splits))
    summary = {
        "splits": sorted(target_splits),
        "segment_definition": args.segment_definition,
        "forest_definition": args.forest_definition,
        "forest_ignore_index": forest_ignore_index,
        "forest_class_names": list(FOREST_CLASS_NAMES),
        "vegetation_ids": vegetation_ids_serializable,
        "vegetation_class_names": {
            s: list(ORACLE_SETTING_CLASS_NAMES[s]) for s in settings
        },
        "n_scenes_ok": len(ok),
        "n_scenes_error": len(errors),
        "error_examples": [
            {
                "patch_id": r.patch_id,
                "error": r.error,
                "scene_path": r.scene_path,
            }
            for r in errors[:50]
        ],
        "settings": setting_metrics,
    }

    summary_json = os.path.join(output_dir, f"oracle_landcover_to_forest_{split_tag}.json")
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Per-setting CSV with headline metrics
    metrics_csv = os.path.join(
        output_dir, f"oracle_landcover_to_forest_{split_tag}.csv"
    )
    with open(metrics_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "setting",
                "miou",
                "macro_f1",
                "iou_not_forest",
                "iou_forest",
                "f1_not_forest",
                "f1_forest",
                "n_points_eval",
                "n_points_total",
                "n_scenes_ok",
            ],
        )
        writer.writeheader()
        for setting, block in setting_metrics.items():
            writer.writerow(
                {
                    "setting": setting,
                    "miou": block["miou"],
                    "macro_f1": block["macro_f1"],
                    "iou_not_forest": block["per_class"]["Not Forest"]["iou"],
                    "iou_forest": block["per_class"]["Forest"]["iou"],
                    "f1_not_forest": block["per_class"]["Not Forest"]["f1"],
                    "f1_forest": block["per_class"]["Forest"]["f1"],
                    "n_points_eval": block["n_points_eval"],
                    "n_points_total": block["n_points_total"],
                    "n_scenes_ok": block["n_scenes_ok"],
                }
            )

    _print_summary(summary)
    print(f"\nWrote {summary_json}")
    print(f"Wrote {metrics_csv}")


if __name__ == "__main__":
    main()
