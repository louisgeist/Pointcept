#!/usr/bin/env python3
"""Estimate on-disk size of preprocessed Malibu3D+ scenes (train/val/test).

Walks scene folders listed in the split manifest (LIDARHD=True), sums file
sizes by basename, and peeks ``.npy`` headers for dtype/shape. Does not apply
training exclusions (too_small / missing-tiles / hardcoded).

Example (cluster):
python scripts/estimate_malibu3d_disk_usage.py \
  --data_root $WORK/Pointcept/data/malibu3d_plus \
  --csv_manifest data/malibu3d_plus/raw/scene_split_manifest.csv \
  --splits train,val,test \
  --num_workers 16 \
  --output_csv stats/malibu3d/disk_usage.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from numpy.lib import format as npy_format
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_EXCLUDE = (
    "forest.npy",
    "natural_habitat_multilabel.npy",
    "climatic_domain.npy",
    "land_use.npy",
)

# Fallback when a .npy header cannot be read. Shapes use N / H / W placeholders.
KNOWN_NPY_META: dict[str, tuple[str, str]] = {
    "coord.npy": ("float32", "(N, 3)"),
    "color.npy": ("uint8", "(N, 3)"),
    "segment.npy": ("int32", "(N,)"),
    "strength.npy": ("float32", "(N,)"),
    "forest.npy": ("int16", "(N,)"),
    "natural_habitat.npy": ("int16", "(N,)"),
    "land_use.npy": ("int16", "(N,)"),
    "elevation.npy": ("float32", "(N,)"),
    "coord_translation.npy": ("float64", "(3,)"),
    "climatic_domain.npy": ("int32", "()"),
    "natural_habitat_multilabel.npy": ("int8", "(15,)"),
    "forest_2d.npy": ("uint8", "(1, H, W)"),
    "network.npy": ("uint8", "(3, H, W)"),
}

_PEEK_CACHE: dict[str, tuple[str, str]] = {}


@dataclass(frozen=True)
class SceneRecord:
    split: str
    patch_id: str
    scene_path: str


@dataclass
class SceneScanResult:
    split: str
    present: bool
    sizes: dict[str, int] = field(default_factory=dict)
    meta: dict[str, tuple[str, str]] = field(default_factory=dict)


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def parse_splits(splits_arg: str) -> list[str]:
    seen: list[str] = []
    for token in splits_arg.split(","):
        name = token.strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def parse_exclude(exclude_arg: str) -> set[str]:
    return {token.strip() for token in exclude_arg.split(",") if token.strip()}


def parse_manifest_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def build_scene_path(
    data_root: str, split: str, patch_id: str, dept_year: str, roi: str
) -> str:
    return os.path.join(data_root, split, f"{dept_year}_LIDARHD", roi, patch_id)


def load_scene_records(
    data_root: str,
    csv_manifest: str,
    target_splits: set[str],
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

            dept_year = (row.get("dept_year") or "").strip() or patch_id.split("_", 2)[0]
            roi = (row.get("roi") or "").strip() or patch_id.split("_", 2)[1]
            scene_path = build_scene_path(data_root, split, patch_id, dept_year, roi)
            scene_records.append(
                SceneRecord(split=split, patch_id=patch_id, scene_path=scene_path)
            )
    return scene_records


def peek_npy_header(path: str) -> tuple[str, tuple[int, ...]]:
    """Read dtype and shape from an .npy header without mapping the payload."""
    with open(path, "rb") as handle:
        version = npy_format.read_magic(handle)
        if version == (1, 0):
            shape, _fortran, dtype = npy_format.read_array_header_1_0(handle)
        elif version in ((2, 0), (3, 0)):
            shape, _fortran, dtype = npy_format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported npy version {version}")
    return np.dtype(dtype).name, tuple(int(dim) for dim in shape)


def format_shape(filename: str, shape: tuple[int, ...]) -> str:
    known = KNOWN_NPY_META.get(filename)
    if known is not None:
        return known[1]
    if len(shape) == 0:
        return "()"
    if len(shape) == 1:
        return "(N,)" if shape[0] > 32 else f"({shape[0]},)"
    if len(shape) == 2 and shape[0] > 32:
        return f"(N, {shape[1]})"
    if len(shape) == 3:
        return f"({shape[0]}, H, W)"
    return "(" + ", ".join(str(dim) for dim in shape) + ")"


def _peek_npy_meta(path: str, filename: str) -> tuple[str, str] | None:
    try:
        dtype_name, shape = peek_npy_header(path)
    except (OSError, ValueError, TypeError):
        return None
    return dtype_name, format_shape(filename, shape)


def _scan_one_scene(scene: SceneRecord) -> SceneScanResult:
    if not os.path.isdir(scene.scene_path):
        return SceneScanResult(split=scene.split, present=False)

    sizes: dict[str, int] = {}
    meta: dict[str, tuple[str, str]] = {}
    try:
        iterator = os.scandir(scene.scene_path)
    except OSError:
        return SceneScanResult(split=scene.split, present=False)

    with iterator:
        for entry in iterator:
            try:
                is_file = entry.is_file(follow_symlinks=False)
            except OSError:
                continue
            if not is_file:
                continue
            name = entry.name
            try:
                sizes[name] = int(entry.stat(follow_symlinks=False).st_size)
            except OSError:
                continue
            if not name.endswith(".npy"):
                continue
            cached = _PEEK_CACHE.get(name)
            if cached is None:
                peeked = _peek_npy_meta(entry.path, name)
                if peeked is not None:
                    _PEEK_CACHE[name] = peeked
                    cached = peeked
            if cached is not None:
                meta[name] = cached

    return SceneScanResult(split=scene.split, present=True, sizes=sizes, meta=meta)


def _imap_chunksize(n_tasks: int, num_workers: int) -> int:
    if n_tasks <= 0:
        return 1
    return max(1, min(128, n_tasks // max(1, num_workers * 8)))


def scan_scenes(
    scene_records: list[SceneRecord],
    num_workers: int,
    show_progress: bool,
) -> list[SceneScanResult]:
    if num_workers <= 1:
        iterator: Iterable[SceneRecord] = scene_records
        if show_progress:
            iterator = tqdm(iterator, total=len(scene_records), desc="Scenes", unit="scene")
        return [_scan_one_scene(scene) for scene in iterator]

    chunksize = _imap_chunksize(len(scene_records), num_workers)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        mapped = pool.map(_scan_one_scene, scene_records, chunksize=chunksize)
        if show_progress:
            mapped = tqdm(
                mapped, total=len(scene_records), desc="Scenes", unit="scene"
            )
        return list(mapped)


def bytes_to_gib(n_bytes: int) -> float:
    return n_bytes / (1024.0 ** 3)


def resolve_file_meta(
    filename: str,
    peeked: tuple[str, str] | None,
) -> tuple[str, str]:
    if filename.endswith(".npy"):
        if peeked is not None:
            return peeked
        fallback = KNOWN_NPY_META.get(filename)
        if fallback is not None:
            return fallback
        return ("unknown", "?")
    if filename == "meta.json":
        return ("n/a", "n/a")
    return ("n/a", "n/a")


def aggregate_results(
    scene_records: list[SceneRecord],
    scans: list[SceneScanResult],
    split_order: list[str],
    exclude: set[str],
) -> dict:
    listed_by_split: dict[str, int] = defaultdict(int)
    for scene in scene_records:
        listed_by_split[scene.split] += 1

    present_by_split: dict[str, int] = defaultdict(int)
    missing_by_split: dict[str, int] = defaultdict(int)
    n_files: dict[str, int] = defaultdict(int)
    bytes_total: dict[str, int] = defaultdict(int)
    bytes_by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    peeked_meta: dict[str, tuple[str, str]] = {}

    for scan in scans:
        if not scan.present:
            missing_by_split[scan.split] += 1
            continue
        present_by_split[scan.split] += 1
        for filename, size in scan.sizes.items():
            n_files[filename] += 1
            bytes_total[filename] += size
            bytes_by_split[scan.split][filename] += size
        for filename, meta in scan.meta.items():
            peeked_meta.setdefault(filename, meta)

    filenames = sorted(n_files, key=lambda name: (-bytes_total[name], name))
    file_rows = []
    all_bytes = 0
    kept_bytes = 0
    excluded_bytes = 0
    for filename in filenames:
        dtype_name, shape_str = resolve_file_meta(filename, peeked_meta.get(filename))
        n_bytes = bytes_total[filename]
        is_excluded = filename in exclude
        all_bytes += n_bytes
        if is_excluded:
            excluded_bytes += n_bytes
        else:
            kept_bytes += n_bytes
        per_split = {
            split: {
                "bytes": int(bytes_by_split[split].get(filename, 0)),
            }
            for split in split_order
        }
        file_rows.append(
            {
                "filename": filename,
                "dtype": dtype_name,
                "shape": shape_str,
                "n_files": int(n_files[filename]),
                "bytes": int(n_bytes),
                "gib": bytes_to_gib(n_bytes),
                "excluded": is_excluded,
                "by_split": per_split,
            }
        )

    split_totals = []
    for split in split_order:
        listed = int(listed_by_split[split])
        present = int(present_by_split[split])
        missing = int(missing_by_split[split])
        split_all = sum(bytes_by_split[split].values())
        split_kept = sum(
            size
            for name, size in bytes_by_split[split].items()
            if name not in exclude
        )
        split_totals.append(
            {
                "split": split,
                "tiles_listed": listed,
                "tiles_present": present,
                "tiles_missing": missing,
                "bytes": int(split_all),
                "gib": bytes_to_gib(split_all),
                "bytes_kept": int(split_kept),
                "gib_kept": bytes_to_gib(split_kept),
            }
        )

    return {
        "exclude": sorted(exclude),
        "files": file_rows,
        "splits": split_totals,
        "tiles_listed": int(sum(listed_by_split.values())),
        "tiles_present": int(sum(present_by_split.values())),
        "tiles_missing": int(sum(missing_by_split.values())),
        "bytes": int(all_bytes),
        "gib": bytes_to_gib(all_bytes),
        "bytes_excluded": int(excluded_bytes),
        "gib_excluded": bytes_to_gib(excluded_bytes),
        "bytes_kept": int(kept_bytes),
        "gib_kept": bytes_to_gib(kept_bytes),
    }


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_gib(value: float) -> str:
    return f"{value:,.3f}"


def format_report(summary: dict) -> str:
    exclude = summary["exclude"]
    lines = [
        "Malibu3D+ preprocessed disk usage",
        f"Tiles listed (manifest LIDARHD=True): {_fmt_int(summary['tiles_listed'])}",
        f"Tiles present on disk:                {_fmt_int(summary['tiles_present'])}",
        f"Tiles missing on disk:                {_fmt_int(summary['tiles_missing'])}",
        "",
        "Per split:",
    ]
    split_header = (
        f"{'split':<8} {'listed':>10} {'present':>10} {'missing':>10} "
        f"{'GiB_all':>12} {'GiB_kept':>12}"
    )
    lines.append(split_header)
    lines.append("-" * len(split_header))
    for row in summary["splits"]:
        lines.append(
            f"{row['split']:<8} {_fmt_int(row['tiles_listed']):>10} "
            f"{_fmt_int(row['tiles_present']):>10} {_fmt_int(row['tiles_missing']):>10} "
            f"{_fmt_gib(row['gib']):>12} {_fmt_gib(row['gib_kept']):>12}"
        )

    lines += ["", "By filename (all splits):"]
    file_header = (
        f"{'filename':<32} {'dtype':<10} {'shape':<14} {'n_files':>10} "
        f"{'bytes':>18} {'GiB':>10} {'excl':>5}"
    )
    lines.append(file_header)
    lines.append("-" * len(file_header))
    for row in summary["files"]:
        excl = "yes" if row["excluded"] else ""
        lines.append(
            f"{row['filename']:<32} {row['dtype']:<10} {row['shape']:<14} "
            f"{_fmt_int(row['n_files']):>10} {_fmt_int(row['bytes']):>18} "
            f"{_fmt_gib(row['gib']):>10} {excl:>5}"
        )

    lines += [
        "",
        f"Total (all files):     {_fmt_int(summary['bytes']):>18} B   "
        f"{_fmt_gib(summary['gib']):>10} GiB",
        f"Excluded ({', '.join(exclude) if exclude else 'none'}):",
        f"                       {_fmt_int(summary['bytes_excluded']):>18} B   "
        f"{_fmt_gib(summary['gib_excluded']):>10} GiB",
        f"Kept (minus exclude):  {_fmt_int(summary['bytes_kept']):>18} B   "
        f"{_fmt_gib(summary['gib_kept']):>10} GiB",
    ]
    return "\n".join(lines)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_csv(path: str, summary: dict, split_order: list[str]) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "filename",
        "dtype",
        "shape",
        "excluded",
        "n_files",
        "bytes",
        "gib",
    ]
    for split in split_order:
        fieldnames.append(f"bytes_{split}")
        fieldnames.append(f"gib_{split}")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["files"]:
            out = {
                "filename": row["filename"],
                "dtype": row["dtype"],
                "shape": row["shape"],
                "excluded": int(row["excluded"]),
                "n_files": row["n_files"],
                "bytes": row["bytes"],
                "gib": f"{row['gib']:.6f}",
            }
            for split in split_order:
                split_bytes = int(row["by_split"][split]["bytes"])
                out[f"bytes_{split}"] = split_bytes
                out[f"gib_{split}"] = f"{bytes_to_gib(split_bytes):.6f}"
            writer.writerow(out)


def write_json(path: str, summary: dict) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.join("data", "malibu3d_plus"),
        help="Preprocessed Malibu3D root (relative to repo root if not absolute).",
    )
    parser.add_argument(
        "--csv_manifest",
        type=str,
        default=os.path.join("data", "malibu3d_plus", "raw", "scene_split_manifest.csv"),
        help="Path to scene_split_manifest.csv (relative to repo root if not absolute).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split list to scan.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=",".join(DEFAULT_EXCLUDE),
        help="Comma-separated basenames excluded from the kept total.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of worker processes (1 = sequential).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional CSV path for the per-filename breakdown.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional JSON path for the full summary.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()
    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1.")

    data_root = resolve_repo_path(args.data_root)
    csv_manifest = resolve_repo_path(args.csv_manifest)
    split_order = parse_splits(args.splits)
    if not split_order:
        raise ValueError("--splits must contain at least one split name.")
    exclude = parse_exclude(args.exclude)

    if not os.path.isfile(csv_manifest):
        raise FileNotFoundError(f"Manifest not found: {csv_manifest}")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root not found: {data_root}")

    scene_records = load_scene_records(
        data_root=data_root,
        csv_manifest=csv_manifest,
        target_splits=set(split_order),
    )
    scans = scan_scenes(
        scene_records,
        num_workers=args.num_workers,
        show_progress=not args.no_progress,
    )
    summary = aggregate_results(
        scene_records=scene_records,
        scans=scans,
        split_order=split_order,
        exclude=exclude,
    )
    print(format_report(summary))

    if args.output_csv:
        csv_path = resolve_repo_path(args.output_csv)
        write_csv(csv_path, summary, split_order)
        print(f"Wrote CSV: {csv_path}")
    if args.output_json:
        json_path = resolve_repo_path(args.output_json)
        write_json(json_path, summary)
        print(f"Wrote JSON: {json_path}")


if __name__ == "__main__":
    main()
