"""
Preprocess H3D (Hessigheim 3D) LAS/LAZ point clouds into Pointcept scene folders.

March 2018 release: three input files live side by side under ``<dataset_root>/LiDAR/``.
The script maps each file to a split and writes Pointcept layouts under ``--output_root``.

**Input** (``--dataset_root`` = epoch root, default: ``/data/h3d/Epoch_March2018``)::

    dataset_root/
    └── LiDAR/
        ├── Mar18_train.laz              → split train
        ├── Mar18_val.laz                → split val
        └── Mar18_test_GroundTruth.las   → split test

**Output** (``--output_root``, default: ``/data/h3d``)::

    output_root/
    ├── train/Mar18_train/coord.npy, segment.npy, color.npy
    ├── val/Mar18_val/...
    └── test/Mar18_test_GroundTruth/...

With ``--chunk-size S`` (horizontal step in coordinate units), each file is split
into an XY grid of roughly ``S``-wide tiles; scene folders are named ``<basename>_<row>-<col>``
(e.g. ``Mar18_train_0-1``). Default chunk size is infinity (one folder per source file).

Semantic class IDs are 0–10 (except 11 for Void).

Requires: ``pip install laspy lazrs`` or ``mamba install -c conda-forge laspy lazrs-python``

Usage:
    python pointcept/datasets/preprocessing/h3d/preprocess_h3d.py --chunk-size [1]
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np

try:
    from pointcept.datasets.preprocessing.h3d.h3d_class_map import CLASS_NAMES, NUM_CLASSES
    from pointcept.datasets.preprocessing.xy_grid_chunking import split_scene_xy_by_chunk_size
except ModuleNotFoundError:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    if THIS_DIR not in sys.path:
        sys.path.insert(0, THIS_DIR)
    from h3d_class_map import CLASS_NAMES, NUM_CLASSES

    _PREPROCESSING_DIR = os.path.dirname(THIS_DIR)
    if _PREPROCESSING_DIR not in sys.path:
        sys.path.insert(0, _PREPROCESSING_DIR)
    from xy_grid_chunking import split_scene_xy_by_chunk_size

# Fixed basenames under dataset_root/LiDAR/.
MAR18_FILES_PER_SPLIT: Dict[str, Tuple[str, ...]] = {
    "train": ("Mar18_train.laz",),
    "val": ("Mar18_val.laz",),
    "test": ("Mar18_test_GroundTruth.las",),
}


def read_las_laz(filepath: str) -> Dict[str, np.ndarray]:
    try:
        import laspy
    except ImportError as err:
        raise ImportError(
            "Install: `pip install laspy lazrs` or "
            "`mamba install -c conda-forge laspy lazrs-python`"
        ) from err

    las = laspy.read(filepath)
    n = len(las.points)
    coord = np.column_stack(
        (
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64),
        )
    ).astype(np.float32)
    
    coord = coord - coord.min(axis=0)

    segment = np.asarray(las.classification, dtype=np.int32)

    if not (hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue")):
        raise ValueError(f"LAS/LAZ file has no RGB; H3D preprocessing requires color: {filepath}")

    r = np.asarray(las.red, dtype=np.float64)
    g = np.asarray(las.green, dtype=np.float64)
    b = np.asarray(las.blue, dtype=np.float64)
    rgb = np.stack([r, g, b], axis=1).astype(np.float32)
    mx = float(np.max(rgb))
    if mx > 255.5:
        rgb = rgb / max(mx, 1e-6) * 255.0
    color = np.clip(rgb, 0.0, 255.0).astype(np.float32)

    return {"coord": coord, "segment": segment, "color": color}


def save_scene(output_scene_dir: str, scene: Dict[str, np.ndarray]) -> None:
    os.makedirs(output_scene_dir, exist_ok=True)
    np.save(os.path.join(output_scene_dir, "coord.npy"), scene["coord"].astype(np.float32))
    np.save(os.path.join(output_scene_dir, "segment.npy"), scene["segment"].astype(np.int32))
    np.save(os.path.join(output_scene_dir, "color.npy"), scene["color"].astype(np.float32))


def process_one_file(args_tuple: Tuple[str, str, str, float]) -> Tuple[str, int]:
    filepath, output_root, split, chunk_size = args_tuple

    basename = os.path.splitext(os.path.basename(filepath))[0]
    scene_id = basename.replace(" ", "_")

    scene = read_las_laz(filepath)
    sub_scenes = split_scene_xy_by_chunk_size(scene=scene, chunk_size=chunk_size)
    written = 0
    for suffix, sub_scene in sub_scenes:
        if sub_scene["coord"].shape[0] == 0:
            continue
        tiled_scene_id = scene_id if len(sub_scenes) == 1 else f"{scene_id}_{suffix}"
        output_scene_dir = os.path.join(output_root, split, tiled_scene_id)
        save_scene(output_scene_dir, sub_scene)
        written += 1
    return scene_id, written


def _resolve_paths(lidar_dir: str, split: str) -> List[str]:
    basenames = MAR18_FILES_PER_SPLIT.get(split)
    if not basenames:
        raise ValueError(f"Unknown split '{split}'. Expected one of: {list(MAR18_FILES_PER_SPLIT)}.")

    out: List[str] = []
    for name in basenames:
        path = os.path.join(lidar_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Expected file not found: {path}")
        out.append(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "H3D March 2018: read three LAS/LAZ files from dataset_root/LiDAR/, "
            "write output_root/train|val|test/."
        )
    )
    parser.add_argument(
        "--dataset_root",
        default="data/h3d/Epoch_March2018",
        help="Epoch directory that contains a LiDAR/ subdirectory with the three Mar18 files.",
    )
    parser.add_argument(
        "--output_root",
        default="data/h3d",
        help="Pointcept data root (creates train/, val/, test/).",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to process.",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=float("inf"),
        dest="chunk_size",
        help=(
            "Horizontal XY tile size in the same units as point X/Y (e.g. meters). "
            "Each scene is tiled into ~chunk_size-wide cells; infinity (default) writes one folder per file."
        ),
    )
    args = parser.parse_args()
    if args.chunk_size <= 0 or np.isnan(args.chunk_size):
        raise ValueError("--chunk-size must be > 0 or positive infinity.")

    lidar_root = os.path.join(args.dataset_root, "LiDAR")
    if not os.path.isdir(lidar_root):
        raise FileNotFoundError(
            f"Missing input directory: {lidar_root}\n"
            "Expect Mar18_train.laz, Mar18_val.laz, Mar18_test_GroundTruth.las inside."
        )

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(f"H3D classes ({NUM_CLASSES}): {CLASS_NAMES}")
    print(f"Reading from:\n  {lidar_root}")
    print(f"Fixed file mapping:\n  {MAR18_FILES_PER_SPLIT}")

    tasks: List[Tuple[str, str, str, float]] = []
    for split in splits:
        paths = _resolve_paths(lidar_root, split)
        print(f"\n--- Split '{split}' ({len(paths)} file(s)) ---")
        for p in paths:
            print(f"    {os.path.basename(p)} → {{output_root}}/{split}/<scene_id>/")
        for p in paths:
            tasks.append((p, args.output_root, split, args.chunk_size))

    os.makedirs(args.output_root, exist_ok=True)
    total_sources = len(tasks)
    print(f"\nProcessing {total_sources} source file(s).")

    scenes_after_chunking = 0
    if args.num_workers <= 1:
        for i, task in enumerate(tasks, start=1):
            sid, n_written = process_one_file(task)
            scenes_after_chunking += n_written
            print(f"  [{i}/{total_sources}] done → {sid} ({n_written} scene folder(s))")
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [pool.submit(process_one_file, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), start=1):
                sid, n_written = fut.result()
                scenes_after_chunking += n_written
                print(f"\r  Progress: {i}/{total_sources} ({sid}, +{n_written})", end="", flush=True)
        print()

    print(
        f"\nTotal scene folders after chunking: {scenes_after_chunking} "
        f"(from {total_sources} source file(s))."
    )
    print(f"Wrote splits under:\n  {args.output_root}/{{train,val,test}}/…")


if __name__ == "__main__":
    main()
