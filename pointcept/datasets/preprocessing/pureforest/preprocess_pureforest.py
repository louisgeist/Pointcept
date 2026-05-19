"""
Preprocess PureForest LAZ tiles into Pointcept scene folders (manifest-driven).

Reads ``PureForest-patches.csv`` to determine which tiles to process, resolves
expected LAZ paths as ``lidar/{split}/{TRAIN|VAL|TEST}-{patch_id}.laz``, and
logs missing files before conversion.

Input layout (``--dataset_root``)::

    dataset_root/
    ├── lidar/train|val|test/*.laz
    └── metadata/PureForest-patches.csv

Output layout (``--output_root``)::

    output_root/
    ├── train/<patch_id>/coord.npy, color.npy, category.npy
    └── ...

Requires: ``pip install laspy lazrs``

Usage:

python pointcept/datasets/preprocessing/pureforest/preprocess_pureforest.py \
  --dataset_root data/pureforest/extracted \
  --output_root data/pureforest \
  --num_workers 8

Toy subset (2 tiles per class per split, on-the-fly from the full manifest)::

python pointcept/datasets/preprocessing/pureforest/preprocess_pureforest.py \
  --toy \
  --dataset_root data/pureforest/extracted \
  --output_root data/pureforest_toy
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    from pointcept.datasets.preprocessing.pureforest.pureforest_classes import (
        NUM_CLASSES,
        SPLITS,
        build_laz_path,
        parse_class_id_from_patch_stem,
    )
except ModuleNotFoundError:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    if THIS_DIR not in sys.path:
        sys.path.insert(0, THIS_DIR)
    from pureforest_classes import (
        NUM_CLASSES,
        SPLITS,
        build_laz_path,
        parse_class_id_from_patch_stem,
    )

REQUIRED_MANIFEST_COLUMNS = frozenset({"patch_id", "split", "class_index"})
COORD_SCALE_M = 25.0
TOY_TILES_PER_CLASS = 2


def read_pureforest_laz(filepath: str) -> Dict[str, np.ndarray]:
    try:
        import laspy
    except ImportError as err:
        raise ImportError(
            "Install: `pip install laspy lazrs` or "
            "`mamba install -c conda-forge laspy lazrs-python`"
        ) from err

    las = laspy.read(filepath)
    coord = np.column_stack(
        (
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64),
        )
    ).astype(np.float32)

    if not (hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue")):
        raise ValueError(f"LAZ file has no RGB colorization: {filepath}")

    rgb = np.stack(
        [
            np.asarray(las.red, dtype=np.float64),
            np.asarray(las.green, dtype=np.float64),
            np.asarray(las.blue, dtype=np.float64),
        ],
        axis=1,
    ).astype(np.float32)
    mx = float(np.max(rgb))
    if mx > 255.5:
        rgb = rgb / max(mx, 1e-6) * 255.0
    color = np.clip(rgb, 0.0, 255.0).astype(np.float32)

    return {"coord": coord, "color": color}


def normalize_tile_coord(coord: np.ndarray) -> np.ndarray:
    """Center XY on mean, Z on minimum; scale to ~[-1, 1] (paper baseline)."""
    if coord.shape[0] == 0:
        return coord.astype(np.float32, copy=False)
    out = coord.astype(np.float32, copy=True)
    out[:, 0] -= float(np.mean(out[:, 0]))
    out[:, 1] -= float(np.mean(out[:, 1]))
    out[:, 2] -= float(np.min(out[:, 2]))
    out /= COORD_SCALE_M
    return out


def save_tile_scene(
    output_scene_dir: str,
    coord: np.ndarray,
    color: np.ndarray,
    category: int,
) -> None:
    os.makedirs(output_scene_dir, exist_ok=True)
    np.save(os.path.join(output_scene_dir, "color.npy"), color.astype(np.float32))
    np.save(
        os.path.join(output_scene_dir, "category.npy"),
        np.int32(category),
    )
    np.save(os.path.join(output_scene_dir, "coord.npy"), coord.astype(np.float32))


@dataclass(frozen=True)
class TileTask:
    patch_id: str
    split: str
    laz_path: str
    class_id: int

    @property
    def patch_stem(self) -> str:
        return self.patch_id


def _open_manifest_reader(manifest_csv: str) -> Tuple[csv.DictReader, object]:
    if not os.path.isfile(manifest_csv):
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
    f = open(manifest_csv, newline="", encoding="utf-8")
    reader = csv.DictReader(f)
    if reader.fieldnames is None:
        f.close()
        raise ValueError(f"Empty manifest CSV: {manifest_csv}")
    missing_cols = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames)
    if missing_cols:
        f.close()
        raise ValueError(
            f"Manifest {manifest_csv} missing columns: {sorted(missing_cols)}"
        )
    return reader, f


def _parse_manifest_row_fields(
    row: Dict[str, str],
    splits_set: set,
) -> Optional[Tuple[str, str, int]]:
    patch_id = (row.get("patch_id") or "").strip()
    split = (row.get("split") or "").strip().lower()
    class_index_raw = (row.get("class_index") or "").strip()
    if not patch_id or not split or not class_index_raw:
        return None
    if split not in splits_set:
        return None
    return patch_id, split, int(class_index_raw)


def _tile_task_from_manifest_fields(
    patch_id: str,
    split: str,
    class_id: int,
    dataset_root: str,
    manifest_csv: str,
    class_mismatches: List[str],
) -> TileTask:
    if class_id < 0 or class_id >= NUM_CLASSES:
        raise ValueError(
            f"Invalid class_index={class_id} for patch_id={patch_id!r} "
            f"in {manifest_csv}."
        )
    try:
        parsed_id = parse_class_id_from_patch_stem(patch_id)
        if parsed_id != class_id:
            class_mismatches.append(
                f"{patch_id}: manifest class_index={class_id}, "
                f"filename -C{parsed_id}-"
            )
    except ValueError:
        pass
    return TileTask(
        patch_id=patch_id,
        split=split,
        laz_path=build_laz_path(dataset_root, split, patch_id),
        class_id=class_id,
    )


def _print_class_mismatch_warnings(class_mismatches: List[str]) -> None:
    if not class_mismatches:
        return
    print(
        f"WARNING: {len(class_mismatches)} patch(es) with class_index != "
        f"filename -C{{id}}- (showing up to 5):"
    )
    for line in class_mismatches[:5]:
        print(f"  {line}")


def _register_tile_task(
    tasks_by_patch: Dict[str, TileTask],
    task: TileTask,
    manifest_csv: str,
) -> None:
    if task.patch_id in tasks_by_patch:
        prev = tasks_by_patch[task.patch_id]
        if prev.split != task.split or prev.class_id != task.class_id:
            raise ValueError(
                f"Duplicate patch_id {task.patch_id!r} with conflicting rows "
                f"in {manifest_csv}."
            )
    else:
        tasks_by_patch[task.patch_id] = task


def load_manifest_tasks(
    manifest_csv: str,
    dataset_root: str,
    splits: Tuple[str, ...],
) -> List[TileTask]:
    """Load one TileTask per manifest row for the requested splits."""
    splits_set = set(splits)
    tasks_by_patch: Dict[str, TileTask] = {}
    class_mismatches: List[str] = []

    reader, manifest_file = _open_manifest_reader(manifest_csv)
    try:
        for row in reader:
            parsed = _parse_manifest_row_fields(row, splits_set)
            if parsed is None:
                continue
            patch_id, split, class_id = parsed
            task = _tile_task_from_manifest_fields(
                patch_id,
                split,
                class_id,
                dataset_root,
                manifest_csv,
                class_mismatches,
            )
            _register_tile_task(tasks_by_patch, task, manifest_csv)
    finally:
        manifest_file.close()

    _print_class_mismatch_warnings(class_mismatches)
    return list(tasks_by_patch.values())


def select_toy_tasks(
    manifest_csv: str,
    dataset_root: str,
    splits: Tuple[str, ...],
    tiles_per_class: int = TOY_TILES_PER_CLASS,
) -> List[TileTask]:
    """Select a small balanced subset: ``tiles_per_class`` tiles per class per split."""
    splits_set = set(splits)
    grouped: Dict[Tuple[str, int], List[str]] = {}
    class_mismatches: List[str] = []

    reader, manifest_file = _open_manifest_reader(manifest_csv)
    try:
        for row in reader:
            parsed = _parse_manifest_row_fields(row, splits_set)
            if parsed is None:
                continue
            patch_id, split, class_id = parsed
            grouped.setdefault((split, class_id), []).append(patch_id)
    finally:
        manifest_file.close()

    tasks_by_patch: Dict[str, TileTask] = {}
    for split in splits:
        for class_id in range(NUM_CLASSES):
            patch_ids = sorted(grouped.get((split, class_id), []))
            if not patch_ids:
                print(
                    f"WARNING: toy subset has no manifest row for "
                    f"split={split!r} class_index={class_id}."
                )
                continue
            if len(patch_ids) < tiles_per_class:
                print(
                    f"WARNING: split={split!r} class_index={class_id} has only "
                    f"{len(patch_ids)} tile(s), expected {tiles_per_class}."
                )
            for patch_id in patch_ids[:tiles_per_class]:
                task = _tile_task_from_manifest_fields(
                    patch_id,
                    split,
                    class_id,
                    dataset_root,
                    manifest_csv,
                    class_mismatches,
                )
                _register_tile_task(tasks_by_patch, task, manifest_csv)

    _print_class_mismatch_warnings(class_mismatches)
    return list(tasks_by_patch.values())


def partition_tasks_for_processing(
    tasks: List[TileTask],
    output_root: str,
    overwrite: bool,
) -> Tuple[List[TileTask], List[TileTask], List[Dict[str, str]]]:
    """Split manifest tasks into process, already-done, and missing LAZ."""
    tasks_to_process: List[TileTask] = []
    already_done: List[TileTask] = []
    missing_laz: List[Dict[str, str]] = []

    for task in tasks:
        if not os.path.isfile(task.laz_path):
            missing_laz.append(
                {
                    "split": task.split,
                    "patch_id": task.patch_id,
                    "laz_path": task.laz_path,
                }
            )
            continue

        coord_path = os.path.join(output_root, task.split, task.patch_id, "coord.npy")
        if not overwrite and os.path.isfile(coord_path):
            already_done.append(task)
            continue

        tasks_to_process.append(task)

    return tasks_to_process, already_done, missing_laz


def write_missing_laz_report(
    output_path: str,
    missing_laz: List[Dict[str, str]],
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Missing LAZ pre-flight report\n\n")
        f.write(f"missing_laz={len(missing_laz)}\n\n")
        f.write("## Manifest rows with no LAZ file on disk\n")
        f.write("# split,patch_id,laz_path\n")
        if not missing_laz:
            f.write("none\n")
        else:
            for item in missing_laz:
                f.write(
                    f"{item['split']},{item['patch_id']},{item['laz_path']}\n"
                )


def process_tile_task(
    task: TileTask,
    output_root: str,
) -> Tuple[str, str]:
    """Returns (status, message) where status is ok|error."""
    output_scene_dir = os.path.join(output_root, task.split, task.patch_id)
    try:
        scene = read_pureforest_laz(task.laz_path)
        coord = normalize_tile_coord(scene["coord"])
        save_tile_scene(
            output_scene_dir,
            coord=coord,
            color=scene["color"],
            category=task.class_id,
        )
        return "ok", task.patch_id
    except Exception as exc:
        return "error", f"{task.patch_id}: {exc}\n{traceback.format_exc()}"


def _worker(args: Tuple[TileTask, str]) -> Tuple[str, str]:
    task, output_root = args
    return process_tile_task(task, output_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess PureForest LAZ to Pointcept .npy (manifest-driven)."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/pureforest/extracted",
        help="Root with lidar/{train,val,test} and metadata/.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/pureforest",
        help="Pointcept root; writes train|val|test/<patch_id>/.",
    )
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default=None,
        help="Defaults to {dataset_root}/metadata/PureForest-patches.csv",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=",".join(SPLITS),
        help="Comma-separated splits to process.",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help=(
            "Process a small balanced subset only: "
            f"{TOY_TILES_PER_CLASS} tiles per class per split (on-the-fly)."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess even when coord.npy exists.",
    )
    args = parser.parse_args()

    splits = tuple(s.strip().lower() for s in args.splits.split(",") if s.strip())
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")

    manifest_csv = args.manifest_csv or os.path.join(
        args.dataset_root, "metadata", "PureForest-patches.csv"
    )

    if args.toy:
        tasks = select_toy_tasks(manifest_csv, args.dataset_root, splits)
        print(
            f"Toy mode: {len(tasks)} tile(s) selected "
            f"({TOY_TILES_PER_CLASS} per class per split)."
        )
    else:
        tasks = load_manifest_tasks(manifest_csv, args.dataset_root, splits)
        print(f"Loaded {len(tasks)} manifest row(s) for splits {splits}.")
    split_counts = Counter(t.split for t in tasks)
    for split in splits:
        print(f"  split '{split}': {split_counts.get(split, 0)} row(s)")

    os.makedirs(args.output_root, exist_ok=True)
    for split in splits:
        os.makedirs(os.path.join(args.output_root, split), exist_ok=True)

    tasks_to_process, already_done, missing_laz = partition_tasks_for_processing(
        tasks, args.output_root, args.overwrite
    )

    missing_report = os.path.join(args.output_root, "missing_laz_preflight.txt")
    write_missing_laz_report(missing_report, missing_laz)
    if missing_laz:
        missing_by_split = Counter(m["split"] for m in missing_laz)
        print(
            f"WARNING: {len(missing_laz)} manifest tile(s) missing LAZ on disk "
            f"(by split: {dict(missing_by_split)})."
        )
        print(f"  Examples: {[m['patch_id'] for m in missing_laz[:5]]}")
    print(f"Wrote missing-LAZ pre-flight report ({len(missing_laz)} entries) to: {missing_report}")

    if already_done:
        print(
            f"Resume: skipping {len(already_done)} tile(s) with existing coord.npy; "
            f"{len(tasks_to_process)} remaining."
        )
    if args.overwrite:
        print(f"--overwrite: reprocessing {len(tasks_to_process)} tile(s) with LAZ present.")

    if not tasks_to_process:
        print("No tiles to process.")
        return

    counts = {"ok": 0, "error": 0}
    errors: List[str] = []
    worker_args = [(t, args.output_root) for t in tasks_to_process]

    if args.num_workers <= 1:
        pbar = tqdm(
            worker_args,
            total=len(worker_args),
            desc="tiles",
            unit="tile",
        )
        for wa in pbar:
            status, msg = _worker(wa)
            counts[status] = counts.get(status, 0) + 1
            if status == "error":
                errors.append(msg)
            pbar.set_postfix(ok=counts["ok"], err=counts["error"], refresh=False)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [pool.submit(_worker, wa) for wa in worker_args]
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="tiles",
                unit="tile",
            )
            for fut in pbar:
                status, msg = fut.result()
                counts[status] = counts.get(status, 0) + 1
                if status == "error":
                    errors.append(msg)
                pbar.set_postfix(ok=counts["ok"], err=counts["error"], refresh=False)

    print(
        f"Done. ok={counts['ok']} error={counts['error']} "
        f"missing_laz={len(missing_laz)} skipped_done={len(already_done)} "
        f"(num_classes={NUM_CLASSES})."
    )
    if errors:
        err_log = os.path.join(args.output_root, "preprocess_errors.log")
        with open(err_log, "w", encoding="utf-8") as f:
            f.write("\n\n".join(errors))
        print(f"Wrote {len(errors)} error(s) to {err_log}")


if __name__ == "__main__":
    main()
