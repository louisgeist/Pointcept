"""
Preprocessing script for Flair3D / LidarHD-like ASCII PLY data.

This script converts raw tiles/subtiles into Pointcept scene folders containing:
- coord.npy
- color.npy
- segment.npy
- strength.npy (optional, from intensity)
"""

import argparse
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from pointcept.datasets.preprocessing.flair3d.flair3d_label_remap import (
        SUPPORTED_LABEL_REMAPS,
        build_segment,
    )
except ModuleNotFoundError:
    # Allow running this file directly without setting PYTHONPATH.
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    if THIS_DIR not in sys.path:
        sys.path.insert(0, THIS_DIR)
    from flair3d_label_remap import SUPPORTED_LABEL_REMAPS, build_segment

RAW_FOLDER_NAMES = {
    'train': [
        'D005-2018_LIDARHD/AA-S1-14', # 19/02/2026: move to train from test
        'D006-2020_LIDARHD/AU-S2-13',
        'D009-2019_LIDARHD/AA-S1-14',
        'D013-2020_LIDARHD/AA-S1-14',
        'D015-2020_LIDARHD/AA-S1-11',
        # 'D023-2020_LIDARHD/AA-S1-23', # 20/02/2026: move from train to test
        # 'D033-2021_LIDARHD/AA-S1-1',
        'D034-2021_LIDARHD/AA-S1-27',
        'D035-2020_LIDARHD/AA-S1-37',
        'D036-2020_LIDARHD/AA-S1-14',
        'D038-2021_LIDARHD/FA-S1-18',
        'D041-2021_LIDARHD/AA-S1-24',
        'D052-2019_LIDARHD/AA-S1-28',
        'D058-2020_LIDARHD/AA-S1-17',
        # 'D060-2021_LIDARHD/AA-S1-40',
        'D064-2021_LIDARHD/AA-S1-26',
        'D069-2020_LIDARHD/AA-S1-28',
        'D070-2020_LIDARHD/FA-S1-18',
        'D072-2019_LIDARHD/AA-S1-15', # 19/02/2026: move from val to train
        'D074-2020_LIDARHD/AA-S1-47',
        'D080-2017_LIDARHD/UU-S1-25',
        # 'D081-2020_LIDARHD/AA-S1-13', # 19/02/2026: move from train to val
        # 'D084-2021_LIDARHD/AA-S1-21',
        ],
    'val': [
        'D060-2021_LIDARHD/AA-S1-40',
        # 'D072-2019_LIDARHD/AA-S1-15', # 19/02/2026: move from val to train
        'D081-2020_LIDARHD/AA-S1-13', # 19/02/2026: move from train to val
        ],
    'test': [
        # 'D005-2018_LIDARHD/AA-S1-14', # removed on 19/02/2026 because wrong "impervious surface" label
        'D023-2020_LIDARHD/AA-S1-23', # 20/02/2026: move from train to test
        'D033-2021_LIDARHD/AA-S1-1',
        'D084-2021_LIDARHD/AA-S1-21',
        ]
} 

PLY_TO_NUMPY_DTYPE = {
    "char": np.int8,
    "int8": np.int8,
    "uchar": np.uint8,
    "uint8": np.uint8,
    "short": np.int16,
    "int16": np.int16,
    "ushort": np.uint16,
    "uint16": np.uint16,
    "int": np.int32,
    "int32": np.int32,
    "uint": np.uint32,
    "uint32": np.uint32,
    "float": np.float32,
    "float32": np.float32,
    "double": np.float64,
    "float64": np.float64,
}


def folder_to_id(file_dir: str, row_id: int = None, col_id: int = None) -> str:
    """Generate a Flair3D-style id from the last 2 folder names."""
    assert type(row_id) == type(
        col_id
    ), "row_id and col_id must be either both None or both int."
    has_tiling_ids = row_id is not None

    files = file_dir.split(os.sep)[-2:]
    scene_id = "_".join(files)

    if has_tiling_ids:
        scene_id += f"_{row_id}-{col_id}"

    return scene_id


def id_to_folder(scene_id: str) -> str:
    """Convert a scene id to the corresponding folder format."""
    return scene_id.replace("LIDARHD_", f"LIDARHD{os.sep}")


def _parse_subtile_indices(filename: str) -> Optional[Tuple[int, int]]:
    """Try extracting trailing row/col indices from filename stem."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"(\d+)[-_](\d+)$", stem)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def read_ply_ascii_fast(filepath: str) -> Dict[str, np.ndarray]:
    """
    Fast reading of an ASCII PLY file.
    Returns a dict mapping attribute names to numpy arrays.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    properties: List[str] = []
    dtypes: List[np.dtype] = []
    header_end = 0
    format_ascii = False

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line.startswith("format"):
            if "ascii" in line:
                format_ascii = True
        elif line.startswith("property"):
            parts = line.split()
            dtype_str = parts[1].lower()
            prop_name = parts[2]
            if dtype_str not in PLY_TO_NUMPY_DTYPE:
                raise ValueError(f"Unsupported PLY dtype '{dtype_str}' in {filepath}")
            properties.append(prop_name)
            dtypes.append(PLY_TO_NUMPY_DTYPE[dtype_str])
        elif line == "end_header":
            header_end = i
            break

    if not format_ascii:
        raise ValueError(
            f"{filepath} is not an ASCII PLY file. Use an ASCII export first."
        )

    data_lines = lines[header_end + 1 :]
    data = np.loadtxt(data_lines, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]

    result = {}
    for i, prop_name in enumerate(properties):
        result[prop_name] = data[:, i].astype(dtypes[i], copy=False)
    return result


def _build_scene_from_subtile(
    filepath: str,
    label_definition: str,
    save_strength: bool,
    ignore_index: int,
) -> Dict[str, np.ndarray]:
    """Build a Pointcept scene dict from one ASCII PLY subtile."""
    attributes = read_ply_ascii_fast(filepath)

    for axis in ("x", "y", "z"):
        if axis not in attributes:
            raise KeyError(f"Missing '{axis}' in {filepath}")
    coord = np.stack([attributes["x"], attributes["y"], attributes["z"]], axis=1).astype(
        np.float32
    )

    if all(k in attributes for k in ("red", "green", "blue")):
        color = np.stack(
            [attributes["red"], attributes["green"], attributes["blue"]], axis=1
        ).astype(np.uint8)
    else:
        color = np.zeros((coord.shape[0], 3), dtype=np.uint8)

    segment = build_segment(
        attributes=attributes,
        ignore_index=ignore_index,
        label_definition=label_definition,
    )

    out = {
        "coord": coord,
        "color": color,
        "segment": segment,
    }

    if save_strength and "intensity" in attributes:
        strength = np.clip(attributes["intensity"].astype(np.float32), 0, 60000) / 60000
        out["strength"] = strength.astype(np.float32)

    return out


def _merge_scene_parts(
    parts: List[Dict[str, np.ndarray]], center_coord: bool
) -> Dict[str, np.ndarray]:
    """Concatenate subtile parts into one full-tile scene."""
    out = {}
    keys = set()
    for part in parts:
        keys.update(part.keys())

    for key in keys:
        out[key] = np.concatenate([part[key] for part in parts if key in part], axis=0)

    if center_coord and out["coord"].shape[0] > 0:
        out["coord"] = out["coord"] - out["coord"][0]

    return out


def _save_scene(output_scene_dir: str, scene: Dict[str, np.ndarray]) -> None:
    os.makedirs(output_scene_dir, exist_ok=True)
    np.save(os.path.join(output_scene_dir, "coord.npy"), scene["coord"].astype(np.float32))
    np.save(os.path.join(output_scene_dir, "color.npy"), scene["color"].astype(np.uint8))
    np.save(os.path.join(output_scene_dir, "segment.npy"), scene["segment"].astype(np.int32))
    if "strength" in scene:
        np.save(
            os.path.join(output_scene_dir, "strength.npy"),
            scene["strength"].astype(np.float32),
        )


def _process_subtile_task(
    ply_path: str,
    output_root: str,
    split: str,
    label_definition: str,
    save_strength: bool,
    ignore_index: int,
) -> str:
    part = _build_scene_from_subtile(
        filepath=ply_path,
        label_definition=label_definition,
        save_strength=save_strength,
        ignore_index=ignore_index,
    )
    scene_parent = os.path.dirname(ply_path)
    rc = _parse_subtile_indices(ply_path)
    if rc is None:
        stem = os.path.splitext(os.path.basename(ply_path))[0]
        scene_id = f"{folder_to_id(scene_parent)}_{stem}"
    else:
        scene_id = folder_to_id(scene_parent, rc[0], rc[1])
    output_scene_dir = os.path.join(output_root, split, scene_id)
    _save_scene(output_scene_dir, part)
    return scene_id


def _process_fulltile_task(
    tile_dir: str,
    output_root: str,
    split: str,
    label_definition: str,
    save_strength: bool,
    ignore_index: int,
    center_coord: bool,
) -> str:
    ply_files = sorted(glob.glob(os.path.join(tile_dir, "*.ply")))
    if len(ply_files) == 0:
        raise FileNotFoundError(f"No .ply subtile found in {tile_dir}")

    parts = [
        _build_scene_from_subtile(
            filepath=ply_path,
            label_definition=label_definition,
            save_strength=save_strength,
            ignore_index=ignore_index,
        )
        for ply_path in ply_files
    ]
    scene = _merge_scene_parts(parts, center_coord=center_coord)
    scene_id = folder_to_id(tile_dir)
    output_scene_dir = os.path.join(output_root, split, scene_id)
    _save_scene(output_scene_dir, scene)
    return scene_id


def _run_task(args_tuple):
    """Pool helper to avoid lambda pickling issues."""
    process_fn, fn_args = args_tuple
    return process_fn(*fn_args)


def _get_split_tile_dirs(dataset_root: str, split: str) -> List[str]:
    """Resolve configured raw tile directories for a split."""
    if split not in RAW_FOLDER_NAMES:
        return []
    tile_dirs = []
    for rel_path in RAW_FOLDER_NAMES[split]:
        # Keep split definitions OS-independent.
        rel_path_os = rel_path.replace("/", os.sep)
        tile_dir = os.path.join(dataset_root, rel_path_os)
        if os.path.isdir(tile_dir):
            tile_dirs.append(tile_dir)
        else:
            print(f"[WARN] Missing tile directory, skipped: {tile_dir}")
    return tile_dirs


def main_process():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset_root",
        type=str,
        # required=True,
        default="/data/geist/datasets/sample_flairhub_3d",
        help="Root directory containing Flair3D/LidarHD tiles",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        # required=True,
        default="/data/geist/Pointcept/data/flair3d",
        help="Output root directory for Pointcept-formatted scenes",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["train"],
        choices=["train", "val", "test"],
        metavar="SPLIT",
        help="One or more output split folder names under output_root (e.g. train val test)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="subtile",
        choices=["subtile", "fulltile"],
        help="subtile: one scene per .ply, fulltile: merge all .ply inside each tile folder",
    )
    parser.add_argument(
        "--label_definition",
        type=str,
        required=True,
        choices=SUPPORTED_LABEL_REMAPS,
        help="Label definition (simple or fusion) from Flair3D mappings",
    )
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=-1,
        help="Ignore label id when label is missing/out-of-map-range",
    )
    parser.add_argument(
        "--save_strength",
        action="store_true",
        help="Save intensity as strength.npy when intensity exists",
    )
    parser.add_argument(
        "--center_coord",
        action="store_true",
        help="Center merged full-tile coordinates around first point",
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Number of workers for preprocessing"
    )
    args = parser.parse_args()

    # Normalize to list (argparse with nargs='+' returns list; default was single element)
    splits = args.split if isinstance(args.split, list) else [args.split]

    print(f"Using label definition '{args.label_definition}'")
    print(f"Splits to process: {splits}")

    for split in splits:
        os.makedirs(os.path.join(args.output_root, split), exist_ok=True)

        split_tile_dirs = _get_split_tile_dirs(args.dataset_root, split)
        if len(split_tile_dirs) > 0:
            print(
                f"\n--- Split '{split}' ---\n"
                f"Using predefined Flair3D split with {len(split_tile_dirs)} tile folders"
            )
        else:
            print(
                f"\n--- Split '{split}' ---\n"
                f"[WARN] No predefined split folders found. Falling back to full dataset scan."
            )

        if args.mode == "subtile":
            if len(split_tile_dirs) > 0:
                task_list = []
                for tile_dir in split_tile_dirs:
                    task_list.extend(sorted(glob.glob(os.path.join(tile_dir, "*.ply"))))
            else:
                task_list = sorted(
                    glob.glob(os.path.join(args.dataset_root, "**", "*.ply"), recursive=True)
                )
            process_fn = _process_subtile_task
            process_args = [
                (
                    path,
                    args.output_root,
                    split,
                    args.label_definition,
                    args.save_strength,
                    args.ignore_index,
                )
                for path in task_list
            ]
        else:
            # A full tile folder is expected at depth 2:
            # <dataset_root>/<departement-year>_LIDARHD/<zone_id>/
            if len(split_tile_dirs) > 0:
                task_list = split_tile_dirs
            else:
                candidate_dirs = sorted(glob.glob(os.path.join(args.dataset_root, "*", "*")))
                task_list = [d for d in candidate_dirs if os.path.isdir(d)]
            process_fn = _process_fulltile_task
            process_args = [
                (
                    path,
                    args.output_root,
                    split,
                    args.label_definition,
                    args.save_strength,
                    args.ignore_index,
                    args.center_coord,
                )
                for path in task_list
            ]

        if len(task_list) == 0:
            print(f"[WARN] No task found for split '{split}', skipping.")
            continue

        total = len(task_list)
        print(f"Found {total} items to process in mode='{args.mode}'")
        scene_ids = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [
                pool.submit(_run_task, (process_fn, process_arg))
                for process_arg in process_args
            ]
            for idx, future in enumerate(as_completed(futures), start=1):
                scene_ids.append(future.result())
                print(f"\rProgress: {idx}/{total}", end="", flush=True)
        print()
        out_split_dir = os.path.join(args.output_root, split)
        print(f"Done. Processed {len(scene_ids)} scenes into {out_split_dir}")


if __name__ == "__main__":
    main_process()
