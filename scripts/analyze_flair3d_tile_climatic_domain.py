#!/usr/bin/env python3
"""
Audit Flair3D+ tiles for climatic-domain purity (Temperate / Mediterranean / Alpine).

Expects on-disk natural_habitat.npy from preprocessing with --natural_habitat_definition
default (stored ids 0-43). Maps points via by_climatic_domain (ids 36-43 -> void).

Strict tile rule: a tile gets a domain label only when all eligible points (ids 0-35
after remap -> train ids 0-2) belong to a single domain.

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
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VOID_TRAIN_ID = 3
SPECIAL_STORED_MIN = 36
SPECIAL_STORED_MAX = 41
MAX_STORED_ID = 43

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
    status: str
    tile_domain: int
    n_points: int
    n_eligible: int
    n_void: int
    n_special_stored: int
    pct_temperate: float
    pct_mediterranean: float
    pct_alpine: float
    pct_void: float
    pct_special: float
    domains_present: str
    majority_domain: int
    max_stored_id: int
    suspicious_stored_ids: bool
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


def analyze_tile(scene: SceneRecord, lut: np.ndarray) -> TileResult:
    nh_path = os.path.join(scene.scene_path, "natural_habitat.npy")
    base = dict(
        split=scene.split,
        patch_id=scene.patch_id,
        status="missing_nh",
        tile_domain=-1,
        n_points=0,
        n_eligible=0,
        n_void=0,
        n_special_stored=0,
        pct_temperate=0.0,
        pct_mediterranean=0.0,
        pct_alpine=0.0,
        pct_void=0.0,
        pct_special=0.0,
        domains_present="",
        majority_domain=-1,
        max_stored_id=-1,
        suspicious_stored_ids=False,
    )
    if not os.path.isfile(nh_path):
        return TileResult(**base)

    stored = np.load(nh_path).reshape(-1)
    n_points = int(stored.shape[0])
    max_stored = int(stored.max()) if n_points else -1
    min_stored = int(stored.min()) if n_points else -1
    suspicious = max_stored > MAX_STORED_ID or min_stored < 0

    n_special_stored = int(
        np.count_nonzero(
            (stored >= SPECIAL_STORED_MIN) & (stored <= SPECIAL_STORED_MAX)
        )
    )
    mapped = _remap_stored_labels(stored, lut)
    void_mask = mapped == VOID_TRAIN_ID
    n_void = int(void_mask.sum())
    eligible = mapped[~void_mask]
    n_eligible = int(eligible.shape[0])

    pct_void = (100.0 * n_void / n_points) if n_points else 0.0
    pct_special = (100.0 * n_special_stored / n_points) if n_points else 0.0

    if n_eligible == 0:
        return TileResult(
            **base,
            status="no_eligible",
            n_points=n_points,
            n_void=n_void,
            n_special_stored=n_special_stored,
            pct_void=pct_void,
            pct_special=pct_special,
            max_stored_id=max_stored,
            suspicious_stored_ids=suspicious,
        )

    unique, counts = np.unique(eligible, return_counts=True)
    order = np.argsort(unique)
    unique = unique[order]
    counts = counts[order]
    domains_present = ",".join(str(int(v)) for v in unique)
    majority_idx = int(np.argmax(counts))
    majority_domain = int(unique[majority_idx])

    pct_temp = 100.0 * int(np.count_nonzero(eligible == 0)) / n_eligible
    pct_med = 100.0 * int(np.count_nonzero(eligible == 1)) / n_eligible
    pct_alp = 100.0 * int(np.count_nonzero(eligible == 2)) / n_eligible

    if len(unique) == 1:
        status = "pure"
        tile_domain = int(unique[0])
    else:
        status = "mixed"
        tile_domain = -1

    return TileResult(
        split=scene.split,
        patch_id=scene.patch_id,
        status=status,
        tile_domain=tile_domain,
        n_points=n_points,
        n_eligible=n_eligible,
        n_void=n_void,
        n_special_stored=n_special_stored,
        pct_temperate=pct_temp,
        pct_mediterranean=pct_med,
        pct_alpine=pct_alp,
        pct_void=pct_void,
        pct_special=pct_special,
        domains_present=domains_present,
        majority_domain=majority_domain,
        max_stored_id=max_stored,
        suspicious_stored_ids=suspicious,
    )


def _process_scene(args: Tuple[SceneRecord, np.ndarray]) -> TileResult:
    scene, lut = args
    try:
        return analyze_tile(scene, lut)
    except Exception as exc:
        return TileResult(
            split=scene.split,
            patch_id=scene.patch_id,
            status="error",
            tile_domain=-1,
            n_points=0,
            n_eligible=0,
            n_void=0,
            n_special_stored=0,
            pct_temperate=0.0,
            pct_mediterranean=0.0,
            pct_alpine=0.0,
            pct_void=0.0,
            pct_special=0.0,
            domains_present="",
            majority_domain=-1,
            max_stored_id=-1,
            suspicious_stored_ids=False,
            error=str(exc),
        )


def _write_tiles_csv(path: str, rows: List[TileResult]) -> None:
    fieldnames = [
        "split",
        "patch_id",
        "status",
        "tile_domain",
        "n_points",
        "n_eligible",
        "n_void",
        "n_special_stored",
        "pct_temperate",
        "pct_mediterranean",
        "pct_alpine",
        "pct_void",
        "pct_special",
        "domains_present",
        "majority_domain",
        "max_stored_id",
        "suspicious_stored_ids",
        "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _build_summary(
    rows: List[TileResult],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    status_by_split: Dict[str, Dict[str, int]] = {}
    domain_counts_pure: Dict[str, int] = {"0": 0, "1": 0, "2": 0}
    suspicious_count = 0

    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
        split_bucket = status_by_split.setdefault(row.split, {})
        split_bucket[row.status] = split_bucket.get(row.status, 0) + 1
        if row.status == "pure":
            domain_counts_pure[str(row.tile_domain)] += 1
        if row.suspicious_stored_ids:
            suspicious_count += 1

    total = len(rows)
    pure = status_counts.get("pure", 0)
    mixed = status_counts.get("mixed", 0)
    no_eligible = status_counts.get("no_eligible", 0)
    missing_nh = status_counts.get("missing_nh", 0)
    errors = status_counts.get("error", 0)

    return {
        "meta": meta,
        "total_tiles": total,
        "status_counts": status_counts,
        "status_by_split": status_by_split,
        "pure_domain_counts": domain_counts_pure,
        "pct_pure": round(100.0 * pure / total, 2) if total else 0.0,
        "pct_mixed": round(100.0 * mixed / total, 2) if total else 0.0,
        "pct_no_eligible": round(100.0 * no_eligible / total, 2) if total else 0.0,
        "pct_missing_nh": round(100.0 * missing_nh / total, 2) if total else 0.0,
        "pct_error": round(100.0 * errors / total, 2) if total else 0.0,
        "suspicious_stored_ids_tiles": suspicious_count,
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Tile climatic domain audit ===")
    print(f"Total tiles: {summary['total_tiles']}")
    print(f"Pure: {summary['status_counts'].get('pure', 0)} ({summary['pct_pure']}%)")
    print(f"Mixed: {summary['status_counts'].get('mixed', 0)} ({summary['pct_mixed']}%)")
    print(f"No eligible: {summary['status_counts'].get('no_eligible', 0)} ({summary['pct_no_eligible']}%)")
    print(f"Missing NH: {summary['status_counts'].get('missing_nh', 0)} ({summary['pct_missing_nh']}%)")
    if summary["status_counts"].get("error", 0):
        print(f"Errors: {summary['status_counts']['error']} ({summary['pct_error']}%)")
    print(f"Pure domain counts (0=Temp, 1=Med, 2=Alp): {summary['pure_domain_counts']}")
    if summary["suspicious_stored_ids_tiles"]:
        print(
            f"Warning: {summary['suspicious_stored_ids_tiles']} tiles have stored ids "
            f"outside [0, {MAX_STORED_ID}] (check NH preprocessing definition)."
        )
    print("\nBy split:")
    for split, counts in sorted(summary["status_by_split"].items()):
        print(f"  {split}: {counts}")


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

    tiles_csv = os.path.join(output_dir, "tiles.csv")
    mixed_csv = os.path.join(output_dir, "mixed_tiles.csv")
    summary_json = os.path.join(output_dir, "summary.json")

    _write_tiles_csv(tiles_csv, results)
    mixed_rows = [row for row in results if row.status == "mixed"]
    _write_tiles_csv(mixed_csv, mixed_rows)

    summary = _build_summary(
        results,
        meta={
            "data_root": data_root,
            "csv_manifest": csv_manifest,
            "splits": sorted(target_splits),
            "excluded_tiles": len(excluded),
            "tiles_scanned": len(scenes),
            "storage_definition": "default",
            "target_definition": "by_climatic_domain",
            "strict_pure_rule": True,
        },
    )
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    _print_summary(summary)
    print(f"\nWrote {tiles_csv}")
    print(f"Wrote {mixed_csv} ({len(mixed_rows)} mixed tiles)")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
