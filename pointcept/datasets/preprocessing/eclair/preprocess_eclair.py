"""
Preprocessing script for the ECLAIR aerial LiDAR dataset.

Raw layout (official release):
  <dataset_root>/
    labels.json
    pointclouds/pointcloud_*.laz

Usage:
ln -sfn data/datasets/ECLAIR data/eclair/raw OU ln -sfn $SCRATCH/data/eclair-dataset data/eclair/raw
python pointcept/datasets/preprocessing/eclair/preprocess_eclair.py \
    --dataset_root data/eclair/raw \
    --output_root data/eclair \
    --num_workers 8

Writes per-scene folders under output_root/{train,val,test}/<tile_id>/:
  coord.npy, color.npy, strength.npy, return_number.npy,
  number_of_returns.npy, segment.npy, meta.json

Tiles listed in labels.json only (118 LAZ files outside the manifest are skipped).
By default train includes both ground-truth (approved) and pseudo (rejected) tiles;
pass --no-include_pseudo to write approved train tiles only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np

# Load chunking helper without importing pointcept.datasets (avoids torch deps).
_chunking_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "xy_grid_chunking.py"
)
_spec = importlib.util.spec_from_file_location("xy_grid_chunking", _chunking_path)
_chunking_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_chunking_mod)
split_scene_xy_regular = _chunking_mod.split_scene_xy_regular

try:
    import laspy
except ImportError as error:
    raise ImportError("Please install 'laspy' to preprocess ECLAIR LAZ files.") from error

# Raw LAS classification IDs 1..11 → contiguous train IDs 0..10.
RAW_MIN, RAW_MAX = 1, 11


def build_scene(laz_path: str) -> Dict[str, np.ndarray]:
    las = laspy.read(laz_path)
    dim_names = set(las.point_format.dimension_names)
    gt_key = "classification" if "classification" in dim_names else "raw_classification"
    if gt_key not in dim_names:
        raise KeyError(f"Missing classification field in {laz_path}")

    coord = np.stack(
        [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)],
        axis=1,
    ).astype(np.float32)

    segment_raw = np.asarray(las[gt_key]).astype(np.int32, copy=False)
    if np.any(segment_raw < RAW_MIN) or np.any(segment_raw > RAW_MAX):
        raise ValueError(
            f"Label out of range in {laz_path}. Expected labels in [{RAW_MIN}, {RAW_MAX}]."
        )
    segment = (segment_raw - RAW_MIN).astype(np.int32)

    if "intensity" not in dim_names:
        raise KeyError(f"Missing 'intensity' field in {laz_path}")
    strength = np.asarray(las.intensity).astype(np.float32)

    # Store RGB in Malibu3D-compatible [0, 255] float range (16-bit LAZ → 8-bit scale).
    # Config pipelines then apply NormalizeColor (/255 → [0, 1]).
    if not {"red", "green", "blue"}.issubset(dim_names):
        raise KeyError(f"Missing RGB fields in {laz_path}")
    color = (
        np.stack(
            [
                np.asarray(las.red),
                np.asarray(las.green),
                np.asarray(las.blue),
            ],
            axis=1,
        ).astype(np.float32)
        / 65535.0
        * 255.0
    )

    if "return_number" not in dim_names or "number_of_returns" not in dim_names:
        raise KeyError(f"Missing return fields in {laz_path}")
    return_number = np.asarray(las.return_number).astype(np.float32)
    number_of_returns = np.asarray(las.number_of_returns).astype(np.float32)

    return {
        "coord": coord,
        "color": color,
        "strength": strength,
        "return_number": return_number,
        "number_of_returns": number_of_returns,
        "segment": segment,
    }


def save_scene(
    output_scene_dir: str,
    scene: Dict[str, np.ndarray],
    meta: Dict[str, Any],
) -> None:
    os.makedirs(output_scene_dir, exist_ok=True)
    np.save(os.path.join(output_scene_dir, "coord.npy"), scene["coord"].astype(np.float32))
    np.save(os.path.join(output_scene_dir, "color.npy"), scene["color"].astype(np.float32))
    np.save(
        os.path.join(output_scene_dir, "strength.npy"), scene["strength"].astype(np.float32)
    )
    np.save(
        os.path.join(output_scene_dir, "return_number.npy"),
        scene["return_number"].astype(np.float32),
    )
    np.save(
        os.path.join(output_scene_dir, "number_of_returns.npy"),
        scene["number_of_returns"].astype(np.float32),
    )
    np.save(
        os.path.join(output_scene_dir, "segment.npy"), scene["segment"].astype(np.int32)
    )
    with open(os.path.join(output_scene_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def process_one_tile(
    laz_path: str,
    output_root: str,
    split: str,
    review_category: str,
    tile_name: str,
    chunking: int,
) -> Tuple[str, int]:
    scene = build_scene(laz_path=laz_path)
    scene_id = os.path.splitext(os.path.basename(laz_path))[0]
    meta = {
        "tile_name": tile_name,
        "split": split,
        "review_category": review_category,
    }
    sub_scenes = split_scene_xy_regular(scene=scene, chunking=chunking)
    written = 0
    for suffix, sub_scene in sub_scenes:
        tiled_scene_id = scene_id if chunking <= 1 else f"{scene_id}_{suffix}"
        output_scene_dir = os.path.join(output_root, split, tiled_scene_id)
        save_scene(output_scene_dir, sub_scene, meta)
        written += 1
    return scene_id, written


def load_manifest(
    labels_path: str,
    include_pseudo: bool,
) -> List[Dict[str, Any]]:
    with open(labels_path, encoding="utf-8") as f:
        entries = json.load(f)
    selected: List[Dict[str, Any]] = []
    for entry in entries:
        split = entry["split"]
        review = entry["review_category"]
        if split == "train" and (not include_pseudo) and review != "approved":
            continue
        selected.append(entry)
    return selected


def main_process():
    parser = argparse.ArgumentParser(description="Preprocess ECLAIR LAZ tiles for Pointcept.")
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Directory containing labels.json and pointclouds/.",
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument(
        "--chunking",
        default=1,
        type=int,
        help=(
            "Regular XY chunking factor. 1 keeps original 100x100 m tiles; "
            "N splits each cloud into N x N subtiles."
        ),
    )
    parser.add_argument(
        "--include_pseudo",
        dest="include_pseudo",
        action="store_true",
        default=True,
        help="Include rejected (pseudo-label) tiles in the train split (default: True).",
    )
    parser.add_argument(
        "--no-include_pseudo",
        dest="include_pseudo",
        action="store_false",
        help="Write only approved (ground-truth) tiles for train.",
    )
    parser.add_argument(
        "--max_tiles",
        default=None,
        type=int,
        help="Optional cap on number of tiles (smoke tests).",
    )
    args = parser.parse_args()
    if args.chunking < 1:
        raise ValueError("--chunking must be >= 1.")

    labels_path = os.path.join(args.dataset_root, "labels.json")
    pointclouds_dir = os.path.join(args.dataset_root, "pointclouds")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Missing labels.json at {labels_path}")
    if not os.path.isdir(pointclouds_dir):
        raise FileNotFoundError(f"Missing pointclouds/ at {pointclouds_dir}")

    entries = load_manifest(labels_path, include_pseudo=args.include_pseudo)
    if args.max_tiles is not None:
        entries = entries[: args.max_tiles]

    os.makedirs(args.output_root, exist_ok=True)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(args.output_root, split), exist_ok=True)

    jobs: List[Tuple[str, str, str, str]] = []
    missing = 0
    for entry in entries:
        tile_name = entry["tile_name"]
        laz_path = os.path.join(pointclouds_dir, tile_name)
        if not os.path.isfile(laz_path):
            print(f"[WARN] Missing LAZ, skipped: {laz_path}")
            missing += 1
            continue
        jobs.append((laz_path, entry["split"], entry["review_category"], tile_name))

    total = len(jobs)
    print(f"Manifest tiles selected: {len(entries)} (missing LAZ: {missing})")
    print(f"Processing {total} tiles → {args.output_root}")
    print(
        f"include_pseudo={args.include_pseudo}, chunking={args.chunking}, "
        f"num_workers={args.num_workers}"
    )
    if total == 0:
        return

    scenes_written = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [
            pool.submit(
                process_one_tile,
                laz_path,
                args.output_root,
                split,
                review_category,
                tile_name,
                args.chunking,
            )
            for laz_path, split, review_category, tile_name in jobs
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            _, n_written = future.result()
            scenes_written += n_written
            print(f"\rProgress: {idx}/{total}", end="", flush=True)
    print()
    print(
        f"Done. {scenes_written} scene folder(s) after chunking "
        f"({total} source LAZ file(s)) → {args.output_root}"
    )


if __name__ == "__main__":
    main_process()
