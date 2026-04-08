"""
Preprocessing script for DALES dataset.

Usage:
1) unzip the DALESObjects.zip file

2) python pointcept/datasets/preprocessing/dales/preprocess_dales.py \
    --dataset_root data/dales/raw/DALESObjects \
    --output_root data/dales/ \
    --num_workers 1

This script converts raw DALES PLY files into Pointcept scene folders containing:
- coord.npy
- segment.npy
- strength.npy
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict

import numpy as np
try:
    from plyfile import PlyData
except ImportError as error:
    raise ImportError("Please install 'plyfile' to preprocess DALES PLY files.") from error

ID2TRAINID = np.asarray([8, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)


def read_ply_fast(filepath: str) -> Dict[str, np.ndarray]:
    """Read a PLY file (ASCII or binary) and return vertex attributes."""
    ply_data = PlyData.read(filepath)
    if len(ply_data.elements) == 0:
        raise ValueError(f"No element found in PLY file: {filepath}")
    element_data = ply_data.elements[0].data
    return {name: np.asarray(element_data[name]) for name in element_data.dtype.names}


def build_scene(ply_path: str) -> Dict[str, np.ndarray]:
    attributes = read_ply_fast(ply_path)

    for axis in ("x", "y", "z"):
        if axis not in attributes:
            raise KeyError(f"Missing '{axis}' in {ply_path}")
    coord = np.stack([attributes["x"], attributes["y"], attributes["z"]], axis=1).astype(
        np.float32
    )
    if "sem_class" not in attributes:
        raise KeyError(f"Missing 'sem_class' label field in {ply_path}")
    segment_raw = attributes["sem_class"].astype(np.int32, copy=False)
    if np.any(segment_raw < 0) or np.any(segment_raw >= len(ID2TRAINID)):
        raise ValueError(
            f"Label out of range in {ply_path}. Expected labels in [0, {len(ID2TRAINID) - 1}]."
        )
    segment = ID2TRAINID[segment_raw]
    if "intensity" not in attributes:
        raise KeyError(f"Missing 'intensity' field in {ply_path}")
    strength = attributes["intensity"].astype(np.float32, copy=False)

    return {"coord": coord, "segment": segment, "strength": strength}


def save_scene(output_scene_dir: str, scene: Dict[str, np.ndarray]) -> None:
    os.makedirs(output_scene_dir, exist_ok=True)
    np.save(os.path.join(output_scene_dir, "coord.npy"), scene["coord"].astype(np.float32))
    np.save(os.path.join(output_scene_dir, "segment.npy"), scene["segment"].astype(np.int32))
    np.save(os.path.join(output_scene_dir, "strength.npy"), scene["strength"].astype(np.float32))


def process_one_file(
    ply_path: str,
    output_root: str,
    split: str,
) -> str:
    scene = build_scene(ply_path=ply_path)
    scene_id = os.path.splitext(os.path.basename(ply_path))[0]
    output_scene_dir = os.path.join(output_root, split, scene_id)
    save_scene(output_scene_dir, scene)
    return scene_id


def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--num_workers", default=1, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    splits = ["train", "test"]
    for split in splits:
        split_input_dir = os.path.join(args.dataset_root, split)
        split_output_dir = os.path.join(args.output_root, split)
        os.makedirs(split_output_dir, exist_ok=True)

        if not os.path.isdir(split_input_dir):
            print(f"[WARN] Split directory not found, skipped: {split_input_dir}")
            continue

        file_list = sorted(glob.glob(os.path.join(split_input_dir, "*.ply")))
        total = len(file_list)
        print(f"\n--- Split '{split}' ---")
        print(f"Found {total} files")
        if total == 0:
            continue

        scene_ids = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [
                pool.submit(
                    process_one_file,
                    ply_path,
                    args.output_root,
                    split,
                )
                for ply_path in file_list
            ]
            for idx, future in enumerate(as_completed(futures), start=1):
                scene_ids.append(future.result())
                print(f"\rProgress: {idx}/{total}", end="", flush=True)
        print()
        print(f"Done. Processed {len(scene_ids)} scenes into {split_output_dir}")


if __name__ == "__main__":
    main_process()
