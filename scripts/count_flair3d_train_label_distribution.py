#!/usr/bin/env python3
"""
Compute per-class label distributions on Flair3D+ train (or other splits).

Uses the same scene list as Flair3DDataset (CSV manifest, LIDARHD=True, excluded tiles).
Aggregates segment, forest, land_use, and natural_habitat in one pass per scene.

Example:
python scripts/count_flair3d_train_label_distribution.py \
--data_root data/flair3d_plus \
--csv_manifest data/flair3d_plus/raw/scene_split_manifest.csv \
--split train \
--missing_tiles_manifest data/flair3d_plus/missing_ply_preflight.txt \
--too_small_tiles_manifest data/flair3d_plus/too_small_tiles.csv \
--num_workers 8
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_flair3d_config_utils():
    path = os.path.join(REPO_ROOT, "pointcept", "datasets", "flair3d_config_utils.py")
    spec = importlib.util.spec_from_file_location("flair3d_config_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load flair3d_config_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_flair3d_cfg = _load_flair3d_config_utils()
FLAIR3D_SEMANTIC_TARGET_KEYS = _flair3d_cfg.FLAIR3D_SEMANTIC_TARGET_KEYS
get_missing_target_fill_value = _flair3d_cfg.get_missing_target_fill_value
get_semantic_config = _flair3d_cfg.get_semantic_config

SEMANTIC_TASKS = tuple(FLAIR3D_SEMANTIC_TARGET_KEYS)


@dataclass(frozen=True)
class SceneRecord:
    split: str
    patch_id: str
    scene_path: str


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def parse_splits(splits_arg: str) -> set[str]:
    return {token.strip() for token in splits_arg.split(",") if token.strip()}


def parse_manifest_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def build_scene_path(data_root: str, split: str, patch_id: str, dept_year: str, roi: str) -> str:
    return os.path.join(data_root, split, f"{dept_year}_LIDARHD", roi, patch_id)


def load_hardcoded_excluded_tiles() -> set[tuple[str, str]]:
    """Flair3DDataset.get_hardcoded_excluded_tiles() (CORRUPTED_TILES + missing coord CSV)."""
    excluded = set()
    details_csv = os.path.join(REPO_ROOT, "data", "flair3d_plus", "missing_coord_tiles.details.csv")
    if not os.path.exists(details_csv):
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
    if not path:
        return missing_tiles
    if not os.path.isfile(path):
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


def load_too_small_tiles_manifest(path: str | None, *, train_only: bool) -> set[tuple[str, str]]:
    too_small_tiles: set[tuple[str, str]] = set()
    if not path:
        return too_small_tiles
    if not os.path.isfile(path):
        print(f"Warning: too-small tiles manifest not found: {path}")
        return too_small_tiles
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_split = (row.get("split") or "").strip()
            patch_id = (row.get("patch_id") or "").strip()
            if not row_split or not patch_id:
                continue
            if train_only and row_split != "train":
                continue
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
            raise KeyError(f"Missing required columns in manifest: {sorted(missing_cols)}")

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
            scene_records.append(SceneRecord(split=split, patch_id=patch_id, scene_path=scene_path))
    return scene_records


def _count_array_length() -> Dict[str, int]:
    lengths: Dict[str, int] = {}
    for task in SEMANTIC_TASKS:
        cfg = get_semantic_config(task)
        num_classes = int(cfg["num_classes"])
        ignore_index = cfg["ignore_index"]
        void_bin = num_classes
        if ignore_index is not None and ignore_index >= 0:
            void_bin = max(void_bin, int(ignore_index))
        lengths[task] = void_bin + 2
    return lengths


def _init_task_counts() -> Dict[str, np.ndarray]:
    return {task: np.zeros(_count_array_length()[task], dtype=np.int64) for task in SEMANTIC_TASKS}


def _accumulate_labels(
    counts: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    ignore_index: int | None,
) -> None:
    labels = labels.reshape(-1).astype(np.int64)
    void_bin = num_classes
    if ignore_index is not None and ignore_index >= 0:
        void_bin = max(void_bin, int(ignore_index))

    mapped = labels.copy()
    if ignore_index is not None:
        mapped[labels == ignore_index] = void_bin

    in_range = (labels >= 0) & (labels < num_classes)
    if in_range.any():
        bc = np.bincount(labels[in_range], minlength=num_classes)
        counts[:num_classes] += bc

    if ignore_index is not None:
        counts[void_bin] += int((labels == ignore_index).sum())

    other = ~(in_range | ((ignore_index is not None) & (labels == ignore_index)))
    if other.any():
        counts[void_bin + 1] += int(other.sum())


def _load_task_labels(scene_path: str, task: str, n_points: int) -> np.ndarray | None:
    path = os.path.join(scene_path, f"{task}.npy")
    if os.path.isfile(path):
        labels = np.load(path)
        return labels.reshape(-1)
    if task == "segment":
        return None
    fill = get_missing_target_fill_value(task)
    return np.full(n_points, fill, dtype=np.int32)


def _process_scene(task: tuple[SceneRecord, bool]) -> tuple[Dict[str, np.ndarray], Dict[str, int], str | None]:
    scene, use_fill_for_missing = task
    partial = _init_task_counts()
    meta = {f"{t}_from_file": 0 for t in SEMANTIC_TASKS}

    coord_path = os.path.join(scene.scene_path, "coord.npy")
    if not os.path.isfile(coord_path):
        return partial, meta, f"missing coord.npy: {scene.scene_path}"

    n_points = int(np.load(coord_path, mmap_mode="r").shape[0])

    for task in SEMANTIC_TASKS:
        file_path = os.path.join(scene.scene_path, f"{task}.npy")
        has_file = os.path.isfile(file_path)
        if not has_file:
            if task == "segment":
                return partial, meta, f"missing {task}.npy: {scene.scene_path}"
            if not use_fill_for_missing:
                continue

        labels = _load_task_labels(scene.scene_path, task, n_points)
        if labels is None:
            return partial, meta, f"missing {task}.npy: {scene.scene_path}"
        if labels.shape[0] != n_points:
            return (
                partial,
                meta,
                f"{task} length {labels.shape[0]} != coord {n_points}: {scene.scene_path}",
            )

        if has_file:
            meta[f"{task}_from_file"] = 1

        cfg = get_semantic_config(task)
        _accumulate_labels(
            partial[task],
            labels,
            num_classes=int(cfg["num_classes"]),
            ignore_index=cfg["ignore_index"],
        )
    return partial, meta, None


def _merge_counts(total: Dict[str, np.ndarray], partial: Dict[str, np.ndarray]) -> None:
    for task in SEMANTIC_TASKS:
        total[task] += partial[task]


def _format_count(n: int) -> str:
    if n < 5000:
        return f"{n:,}"
    return f"{n / 1e6:.2f} M"


def _print_task_distribution(task: str, counts: np.ndarray, scenes_with_file: int, scenes_total: int) -> None:
    cfg = get_semantic_config(task)
    class_names = cfg["names"]
    num_classes = int(cfg["num_classes"])
    ignore_index = cfg["ignore_index"]

    void_bin = num_classes
    if ignore_index is not None and ignore_index >= 0:
        void_bin = max(void_bin, int(ignore_index))

    counted = int(counts.sum())
    print(f"\n=== {task} ===")
    print(f"Scenes with on-disk {task}.npy: {scenes_with_file}/{scenes_total}")
    print(f"Total labeled points (incl. void/ignore buckets): {_format_count(counted)}")
    print("-" * 72)

    denom = max(counted, 1)
    for i in range(num_classes):
        n = int(counts[i])
        pct = 100.0 * n / denom
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        print(f"  {i:2d} {name[:45]:45s}: {_format_count(n):>10} ({pct:5.2f}%)")

    void_count = int(counts[void_bin])
    if void_count > 0:
        void_name = "ignore/void"
        if ignore_index is not None and ignore_index < len(class_names):
            void_name = class_names[ignore_index]
        elif ignore_index is not None:
            void_name = f"ignore_index={ignore_index}"
        pct = 100.0 * void_count / denom
        print(f"  -- {void_name[:45]:45s}: {_format_count(void_count):>10} ({pct:5.2f}%)")

    other = int(counts[void_bin + 1]) if counts.size > void_bin + 1 else 0
    if other > 0:
        pct = 100.0 * other / denom
        print(f"  -- out-of-range ids                         : {_format_count(other):>10} ({pct:5.2f}%)")
    print("-" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data/flair3d_plus")
    parser.add_argument(
        "--csv_manifest",
        default="data/flair3d_plus/raw/scene_split_manifest.csv",
    )
    parser.add_argument("--split", default="train", help="Comma-separated splits (default: train)")
    parser.add_argument(
        "--missing_tiles_manifest",
        default="data/flair3d_plus/missing_ply_preflight.txt",
    )
    parser.add_argument(
        "--too_small_tiles_manifest",
        default="data/flair3d_plus/too_small_tiles.csv",
    )
    parser.add_argument(
        "--no_exclude_hardcoded",
        action="store_true",
        help="Do not apply Flair3DDataset hardcoded/missing-coord exclusions.",
    )
    parser.add_argument(
        "--no_exclude_missing_manifest",
        action="store_true",
        help="Do not apply missing_tiles_manifest exclusions.",
    )
    parser.add_argument(
        "--no_exclude_too_small",
        action="store_true",
        help="Do not apply too_small_tiles_manifest exclusions.",
    )
    parser.add_argument(
        "--skip_missing_optional",
        action="store_true",
        help="Skip optional tasks (forest/land_use/natural_habitat) when .npy is absent "
        "(default: use ignore fill like Flair3DDataset).",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_scenes", type=int, default=0, help="Debug: limit number of scenes (0=all)")
    args = parser.parse_args()

    data_root = resolve_repo_path(args.data_root)
    csv_manifest = resolve_repo_path(args.csv_manifest)
    target_splits = parse_splits(args.split)

    excluded: set[tuple[str, str]] = set()
    if not args.no_exclude_hardcoded:
        excluded |= load_hardcoded_excluded_tiles()
    if not args.no_exclude_missing_manifest:
        excluded |= load_missing_tiles_manifest(resolve_repo_path(args.missing_tiles_manifest))
    if not args.no_exclude_too_small:
        train_only = target_splits == {"train"}
        excluded |= load_too_small_tiles_manifest(
            resolve_repo_path(args.too_small_tiles_manifest),
            train_only=train_only,
        )

    if not os.path.isfile(csv_manifest):
        raise FileNotFoundError(f"CSV manifest not found: {csv_manifest}")

    scenes = load_scene_records(data_root, csv_manifest, target_splits, excluded)
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]

    print(f"data_root={data_root}")
    print(f"csv_manifest={csv_manifest}")
    print(f"splits={sorted(target_splits)}")
    print(f"excluded tiles: {len(excluded)}")
    print(f"scenes to scan: {len(scenes)}")
    print(f"use_fill_for_missing={not args.skip_missing_optional}")

    total_counts = _init_task_counts()
    scenes_with_file = {task: 0 for task in SEMANTIC_TASKS}
    errors: list[str] = []

    use_fill = not args.skip_missing_optional
    tasks = [(scene, use_fill) for scene in scenes]

    if args.num_workers <= 1:
        results = (_process_scene(t) for t in tasks)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            results = pool.map(_process_scene, tasks, chunksize=8)

    processed = 0
    for partial, meta, err in results:
        if err:
            errors.append(err)
            continue
        _merge_counts(total_counts, partial)
        for task in SEMANTIC_TASKS:
            scenes_with_file[task] += meta[f"{task}_from_file"]
        processed += 1

    print(f"\nProcessed scenes: {processed}/{len(scenes)}")
    if errors:
        print(f"Skipped scenes with errors: {len(errors)}")
        for line in errors[:10]:
            print(f"  - {line}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    for task in SEMANTIC_TASKS:
        _print_task_distribution(
            task,
            total_counts[task],
            scenes_with_file[task],
            processed,
        )


if __name__ == "__main__":
    main()
