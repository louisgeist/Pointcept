#!/usr/bin/env python3
"""
Export per-tile climatic-domain fractions for Flair3D+ (Temperate / Mediterranean / Alpine / Void).

Expects on-disk natural_habitat.npy from preprocessing with --natural_habitat_definition
default (stored ids 0-43). Maps points via by_climatic_domain (ids 36-43 -> void).

Writes one CSV row per tile with raw counts and fractions over all points (frac_* sum to 1).

Example (Jean-Zay):
python scripts/analyze_flair3d_tile_climatic_domain.py \
  --data_root $WORK/Pointcept/data/flair3d_plus \
  --csv_manifest data/flair3d_plus/raw/scene_split_manifest.csv \
  --splits train,val,test \
  --missing_tiles_manifest data/flair3d_plus/missing_ply_preflight.txt \
  --too_small_tiles_manifest data/flair3d_plus/too_small_tiles.csv \
  --num_workers 24 \
  --output_dir stats/flair3d/tile_climatic_domain
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VOID_TRAIN_ID = 3

_STORED_TO_DOMAIN_LUT: Optional[np.ndarray] = None


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def _load_label_remap_module():
    path = os.path.join(
        REPO_ROOT,
        "pointcept",
        "datasets",
        "preprocessing",
        "flair3d_plus",
        "flair3d_label_remap.py",
    )
    spec = importlib.util.spec_from_file_location("flair3d_label_remap", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load flair3d_label_remap from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["flair3d_label_remap"] = module
    spec.loader.exec_module(module)
    return module


def get_stored_to_domain_lut() -> np.ndarray:
    global _STORED_TO_DOMAIN_LUT
    if _STORED_TO_DOMAIN_LUT is None:
        remap = _load_label_remap_module()
        storage = remap.get_definition("natural_habitat", "default")
        target = remap.get_definition("natural_habitat", "by_climatic_domain")
        _STORED_TO_DOMAIN_LUT = remap.build_stored_to_train_lut(storage, target)
    return _STORED_TO_DOMAIN_LUT


@dataclass(frozen=True)
class SceneRecord:
    split: str
    patch_id: str
    scene_path: str


@dataclass
class TileResult:
    split: str
    patch_id: str
    n_points: int
    n_temperate: int
    n_mediterranean: int
    n_alpine: int
    n_void: int
    frac_temperate: float
    frac_mediterranean: float
    frac_alpine: float
    frac_void: float
    error: str = ""


def parse_splits(splits_arg: str) -> set[str]:
    return {token.strip() for token in splits_arg.split(",") if token.strip()}


def parse_manifest_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def build_scene_path(data_root: str, split: str, patch_id: str, dept_year: str, roi: str) -> str:
    return os.path.join(data_root, split, f"{dept_year}_LIDARHD", roi, patch_id)


def load_hardcoded_excluded_tiles() -> set[tuple[str, str]]:
    excluded: set[tuple[str, str]] = set()
    details_csv = os.path.join(REPO_ROOT, "data", "flair3d_plus", "missing_coord_tiles.details.csv")
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


def load_too_small_tiles_manifest(path: str | None, *, train_only: bool) -> set[tuple[str, str]]:
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


def _remap_stored_labels(stored: np.ndarray, lut: np.ndarray) -> np.ndarray:
    idx = stored.astype(np.int64, copy=False)
    fallback = VOID_TRAIN_ID
    remapped = np.full(idx.shape, fallback, dtype=np.int32)
    valid = (idx >= 0) & (idx < lut.shape[0])
    if np.any(valid):
        remapped[valid] = lut[idx[valid]]
    return remapped


def _empty_tile_result(scene: SceneRecord, error: str = "") -> TileResult:
    return TileResult(
        split=scene.split,
        patch_id=scene.patch_id,
        n_points=0,
        n_temperate=0,
        n_mediterranean=0,
        n_alpine=0,
        n_void=0,
        frac_temperate=0.0,
        frac_mediterranean=0.0,
        frac_alpine=0.0,
        frac_void=0.0,
        error=error,
    )


def analyze_tile(scene: SceneRecord, lut: np.ndarray) -> TileResult:
    nh_path = os.path.join(scene.scene_path, "natural_habitat.npy")
    if not os.path.isfile(nh_path):
        return _empty_tile_result(scene, error="missing_nh")

    stored = np.load(nh_path).reshape(-1)
    n_points = int(stored.shape[0])
    if n_points == 0:
        return _empty_tile_result(scene)

    mapped = _remap_stored_labels(stored, lut)
    n_temperate = int(np.count_nonzero(mapped == 0))
    n_mediterranean = int(np.count_nonzero(mapped == 1))
    n_alpine = int(np.count_nonzero(mapped == 2))
    n_void = int(np.count_nonzero(mapped == VOID_TRAIN_ID))

    inv = 1.0 / n_points
    return TileResult(
        split=scene.split,
        patch_id=scene.patch_id,
        n_points=n_points,
        n_temperate=n_temperate,
        n_mediterranean=n_mediterranean,
        n_alpine=n_alpine,
        n_void=n_void,
        frac_temperate=n_temperate * inv,
        frac_mediterranean=n_mediterranean * inv,
        frac_alpine=n_alpine * inv,
        frac_void=n_void * inv,
    )


def _process_scene(args: Tuple[SceneRecord, np.ndarray]) -> TileResult:
    scene, lut = args
    try:
        return analyze_tile(scene, lut)
    except Exception as exc:
        return _empty_tile_result(scene, error=str(exc))


def _write_fractions_csv(path: str, rows: List[TileResult]) -> None:
    fieldnames = [
        "split",
        "patch_id",
        "n_points",
        "n_temperate",
        "n_mediterranean",
        "n_alpine",
        "n_void",
        "frac_temperate",
        "frac_mediterranean",
        "frac_alpine",
        "frac_void",
        "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data/flair3d_plus")
    parser.add_argument(
        "--csv_manifest",
        default="data/flair3d_plus/raw/scene_split_manifest.csv",
    )
    parser.add_argument(
        "--splits",
        default="train",
        help="Comma-separated splits (default: train)",
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
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--max_scenes", type=int, default=0, help="Debug: limit scenes (0=all)")
    parser.add_argument(
        "--output_dir",
        default="stats/flair3d/tile_climatic_domain",
    )
    args = parser.parse_args()

    data_root = resolve_repo_path(args.data_root)
    csv_manifest = resolve_repo_path(args.csv_manifest)
    target_splits = parse_splits(args.splits)
    output_dir = resolve_repo_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

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

    lut = get_stored_to_domain_lut()
    tasks = [(scene, lut) for scene in scenes]

    print(f"data_root={data_root}")
    print(f"csv_manifest={csv_manifest}")
    print(f"splits={sorted(target_splits)}")
    print(f"excluded tiles: {len(excluded)}")
    print(f"tiles to scan: {len(scenes)}")

    results: List[TileResult] = []
    show_progress = not args.no_progress and len(tasks) > 0

    if args.num_workers <= 1:
        iterator = (_process_scene(task) for task in tasks)
        if show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc="Tiles", unit="tile")
        results = list(iterator)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            mapped = pool.map(_process_scene, tasks, chunksize=8)
            if show_progress:
                mapped = tqdm(mapped, total=len(tasks), desc="Tiles", unit="tile")
            results = list(mapped)

    output_csv = os.path.join(output_dir, "tile_domain_fractions.csv")
    _write_fractions_csv(output_csv, results)
    print(f"Wrote {output_csv} ({len(results)} tiles)")


if __name__ == "__main__":
    main()
