"""
Preprocessing script for SensatUrban dataset.

Usage:
python pointcept/datasets/preprocessing/sensaturban/preprocess_sensaturban.py \
    --dataset_root data/sensaturban/SensatUrban_Dataset/ply \
    --output_root data/sensaturban \
    --split_config pointcept/datasets/preprocessing/sensaturban/splits.py \
    --num_workers 8
    

Directory layout (--dataset_root)
---------------------------------
Typical SensatUrban release (PLY under split folders):

    dataset_root/
    ├── train/
    │   ├── birmingham_block_0.ply
    │   ├── cambridge_block_1.ply
    │   └── ...
    └── test/
        └── ...

Without --split_config, the script reads *.ply only from dataset_root/train/, dataset_root/val/,
and dataset_root/test/ (each optional).

With --split_config (SPLITS lists scene stems such as birmingham_block_0), each name is resolved
to a unique *.ply anywhere under dataset_root (recursive search).

Expected output (--output_root, per scene folder)
- coord.npy (required)
- color.npy (optional)
- segment.npy (optional, absent for unlabeled test split)


num_classes = 13
label_names = [Ground,
  "Vegetation",
  "Building",
  "Wall",
  "Bridge",
  "Parking",
  "Rail",
  "Traffic Road",
  "Street Furniture",
  "Car",
  "Footpath",
  "Bike",
  "Water",
  "Void",]
"""

import argparse
import glob
import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

try:
    from plyfile import PlyData, PlyElementParseError
except ImportError as error:
    raise ImportError(
        "Please install 'plyfile' to preprocess SensatUrban PLY files."
    ) from error


RAW_LABEL_MIN = 1
RAW_LABEL_MAX = 13
VOID_LABEL = 13


def read_ply(filepath: str) -> Dict[str, np.ndarray]:
    """Read a PLY file (ASCII or binary) and return vertex attributes."""
    try:
        ply_data = PlyData.read(filepath)
    except PlyElementParseError as exc:
        raise RuntimeError(
            f"PLY parse failed (often truncated/corrupt download): {filepath}"
        ) from exc
    if len(ply_data.elements) == 0:
        raise ValueError(f"No element found in PLY file: {filepath}")
    element_data = ply_data.elements[0].data
    return {name: np.asarray(element_data[name]) for name in element_data.dtype.names}


def build_scene(
    ply_path: str,
) -> Dict[str, np.ndarray]:
    """Build scene dictionary from SensatUrban PLY."""
    attributes = read_ply(ply_path)

    for axis in ("x", "y", "z"):
        if axis not in attributes:
            raise KeyError(f"Missing '{axis}' in {ply_path}")
    coord = np.stack(
        [attributes["x"], attributes["y"], attributes["z"]],
        axis=1,
    ).astype(np.float32)
    scene = {"coord": coord}

    if all(k in attributes for k in ("red", "green", "blue")):
        scene["color"] = np.stack(
            [attributes["red"], attributes["green"], attributes["blue"]],
            axis=1,
        ).astype(np.float32)

    # In SensatUrban, labels are provided under the "class" field.
    # We map the legal labels to [0, 12] and set the void label to 13.
    if "class" in attributes:
        segment_raw = attributes["class"].astype(np.int32, copy=False)
        segment = np.full_like(segment_raw, fill_value=VOID_LABEL, dtype=np.int32)
        valid_mask = (segment_raw >= RAW_LABEL_MIN) & (segment_raw <= RAW_LABEL_MAX)
        segment[valid_mask] = segment_raw[valid_mask] - 1
        scene["segment"] = segment

    return scene


def save_scene(output_scene_dir: str, scene: Dict[str, np.ndarray]) -> None:
    """Save scene arrays to output directory."""
    os.makedirs(output_scene_dir, exist_ok=True)
    np.save(os.path.join(output_scene_dir, "coord.npy"), scene["coord"].astype(np.float32))
    if "color" in scene:
        np.save(os.path.join(output_scene_dir, "color.npy"), scene["color"].astype(np.float32))
    if "segment" in scene:
        np.save(os.path.join(output_scene_dir, "segment.npy"), scene["segment"].astype(np.int32))


def process_one_file(
    ply_path: str,
    output_root: str,
    split: str,
) -> Tuple[str, List[str]]:
    """Process one PLY file and save arrays."""
    scene = build_scene(ply_path=ply_path)
    scene_id = os.path.splitext(os.path.basename(ply_path))[0]
    output_scene_dir = os.path.join(output_root, split, scene_id)
    save_scene(output_scene_dir, scene)
    return scene_id, sorted(scene.keys())


def inspect_ply_features(dataset_root: str, max_files: int = 10) -> None:
    """Print available vertex attributes in sample PLY files."""
    all_ply = sorted(glob.glob(os.path.join(dataset_root, "**", "*.ply"), recursive=True))
    if not all_ply:
        print(f"No PLY files found under: {dataset_root}")
        return
    sampled = all_ply[: max(1, min(max_files, len(all_ply)))]
    print(f"Inspecting {len(sampled)} / {len(all_ply)} PLY files...\n")
    for path in sampled:
        attrs = read_ply(path)
        print(f"- {path}")
        print(f"  attributes: {', '.join(sorted(attrs.keys()))}")


def collect_split_files(dataset_root: str, split: str) -> List[str]:
    """Collect split files from standard split directory."""
    split_dir = os.path.join(dataset_root, split)
    if not os.path.isdir(split_dir):
        return []
    return sorted(glob.glob(os.path.join(split_dir, "*.ply")))


def _normalize_split_entries(values) -> List[str]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise TypeError("Each split config field must be a list/tuple of strings.")
    entries = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError("Split entries must be strings.")
        value = value.strip()
        if value:
            entries.append(value)
    return entries


def load_split_config(split_config_path: str) -> Dict[str, List[str]]:
    """
    Load split config from a Python file.

    Supported format only:
    - SPLITS = {"train": [...], "val": [...], "test": [...]}
    """
    spec = importlib.util.spec_from_file_location("sensaturban_splits", split_config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load split config: {split_config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "SPLITS"):
        raise ValueError(
            f"Split config '{split_config_path}' must define a SPLITS dict."
        )
    splits = getattr(module, "SPLITS")
    if not isinstance(splits, dict):
        raise TypeError("SPLITS must be a dict with split names as keys.")
    return {
        "train": _normalize_split_entries(splits.get("train", [])),
        "val": _normalize_split_entries(splits.get("val", [])),
        "test": _normalize_split_entries(splits.get("test", [])),
    }


def resolve_split_entries_to_files(dataset_root: str, entries: List[str]) -> List[str]:
    """
    Resolve split entries to PLY files under dataset_root.

    Supported entry formats:
    - relative path (with or without ".ply"), e.g. "train/scene_0001"
    - file stem or filename, e.g. "scene_0001" or "scene_0001.ply"
    """
    all_ply = sorted(glob.glob(os.path.join(dataset_root, "**", "*.ply"), recursive=True))
    rel_to_abs = {os.path.relpath(path, dataset_root): path for path in all_ply}
    stem_to_paths: Dict[str, List[str]] = {}
    name_to_paths: Dict[str, List[str]] = {}
    for path in all_ply:
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        name_to_paths.setdefault(name, []).append(path)
        stem_to_paths.setdefault(stem, []).append(path)

    resolved: List[str] = []
    seen = set()
    for entry in entries:
        direct = os.path.join(dataset_root, entry)
        direct_ply = direct if direct.endswith(".ply") else direct + ".ply"
        if os.path.isfile(direct_ply):
            if direct_ply not in seen:
                resolved.append(direct_ply)
                seen.add(direct_ply)
            continue

        rel_entry = entry if entry.endswith(".ply") else entry + ".ply"
        if rel_entry in rel_to_abs:
            path = rel_to_abs[rel_entry]
            if path not in seen:
                resolved.append(path)
                seen.add(path)
            continue

        candidates = name_to_paths.get(entry, [])
        if not candidates:
            stem = entry[:-4] if entry.endswith(".ply") else entry
            candidates = stem_to_paths.get(stem, [])

        if len(candidates) == 1:
            path = candidates[0]
            if path not in seen:
                resolved.append(path)
                seen.add(path)
        elif len(candidates) == 0:
            raise FileNotFoundError(
                f"Cannot resolve split entry '{entry}' under '{dataset_root}'."
            )
        else:
            example_paths = [os.path.relpath(p, dataset_root) for p in candidates[:5]]
            raise ValueError(
                f"Ambiguous split entry '{entry}' matched multiple files: {example_paths}. "
                "Use relative paths in split config to disambiguate."
            )
    return resolved


def print_split_summary(
    dataset_root: str,
    output_root: str,
    split_to_files: Dict[str, List[str]],
    split_config_path: str,
) -> None:
    """Print resolved train/val/test file lists in a readable layout."""
    bar = "=" * 72
    print()
    print(bar)
    print("SensatUrban preprocessing — split summary")
    print(bar)
    print(f"dataset_root : {os.path.abspath(dataset_root)}")
    print(f"output_root  : {os.path.abspath(output_root)}")
    if split_config_path:
        print(f"split_config : {os.path.abspath(split_config_path)}")
    else:
        print("split_config : (none — using train/, val/, test/ under dataset_root)")
    for split_name in ("train", "val", "test"):
        paths = split_to_files.get(split_name, [])
        paths_sorted = sorted(paths, key=lambda p: os.path.basename(p))
        print()
        print(f"{split_name} — {len(paths_sorted)} scene(s)")
        print("-" * 72)
        if not paths_sorted:
            print("  (empty)")
            continue
        stems = [os.path.splitext(os.path.basename(p))[0] for p in paths_sorted]
        col_width = max(len(s) for s in stems)
        for stem, ply_path in zip(stems, paths_sorted):
            rel = os.path.relpath(ply_path, dataset_root)
            print(f"  {stem:<{col_width}}  ← {rel}")
    print(bar)
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Root directory containing PLY files.")
    parser.add_argument("--output_root", required=True, help="Output directory for processed npy files.")
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument(
        "--split_config",
        default="",
        type=str,
        help=(
            "Optional Python file with split lists. "
            "Supports only SPLITS dict."
        ),
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Only print available PLY attributes, no conversion.",
    )
    parser.add_argument(
        "--inspect_max_files",
        default=10,
        type=int,
        help="Maximum number of files to inspect with --inspect_only.",
    )
    args = parser.parse_args()

    if args.inspect_only:
        inspect_ply_features(dataset_root=args.dataset_root, max_files=args.inspect_max_files)
        return

    os.makedirs(args.output_root, exist_ok=True)
    split_names = ("train", "val", "test")
    if args.split_config:
        split_entries = load_split_config(args.split_config)
        split_to_files = {
            split: resolve_split_entries_to_files(args.dataset_root, split_entries.get(split, []))
            for split in split_names
        }
    else:
        split_to_files = {split: collect_split_files(args.dataset_root, split) for split in split_names}

    total_found = sum(len(files) for files in split_to_files.values())
    if total_found == 0:
        raise FileNotFoundError(
            f"No files found in split directories under {args.dataset_root}. "
            "Expected at least one *.ply in train/val/test."
        )

    print_split_summary(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        split_to_files=split_to_files,
        split_config_path=args.split_config,
    )

    for split, file_list in split_to_files.items():
        split_output_dir = os.path.join(args.output_root, split)
        os.makedirs(split_output_dir, exist_ok=True)
        total = len(file_list)
        print(f"\n>>> Split '{split}' — {total} scene(s) → {os.path.abspath(split_output_dir)}")
        if total == 0:
            continue

        scene_ids = []
        key_counter: Dict[str, int] = {}
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
            for future in tqdm(
                as_completed(futures),
                total=total,
                desc=f"{split}",
                unit="scene",
                leave=True,
            ):
                scene_id, keys = future.result()
                scene_ids.append(scene_id)
                for key in keys:
                    key_counter[key] = key_counter.get(key, 0) + 1
        print(f"Done. Processed {len(scene_ids)} scenes into {split_output_dir}")
        print(f"Saved keys frequency: {key_counter}")


if __name__ == "__main__":
    main()


