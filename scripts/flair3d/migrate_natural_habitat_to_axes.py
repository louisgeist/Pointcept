#!/usr/bin/env python3
"""Migrate on-disk natural_habitat.npy CarHab ``(N,)`` → ecological axes ``(N, 4)``.

For each scene with legacy CarHab ids:
1. Optionally regenerate ``natural_habitat_multilabel.npy`` from CarHab (required
   before overwrite — cultivated/built/road are lost after bake).
2. Bake axes via ``carhab_to_nathab_axes``.
3. Overwrite ``natural_habitat.npy`` as ``uint8 (N, 4)``.
4. Update ``meta.json`` ``label_definitions.natural_habitat`` → ``ecological_axes``
   and write ``natural_habitat_layout``.

Scenes that already store ``(N, 4)`` are validated only.

Example:
  PYTHONPATH=./ python scripts/flair3d/migrate_natural_habitat_to_axes.py \\
    --data_root data/flair3d_plus \\
    --csv_manifest data/flair3d_plus/raw/scene_split_manifest_D067.csv \\
    --splits train,val,test \\
    --write-multilabel \\
    --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pointcept.datasets.preprocessing.flair3d_plus.nathab_axes import (  # noqa: E402
    NATHAB_LAYOUT_META,
    NATHAB_ONDISK_DEFINITION,
    carhab_to_nathab_axes,
    is_nathab_axes_array,
    is_nathab_carhab_array,
    validate_nathab_axes,
)
from pointcept.datasets.preprocessing.flair3d_plus.natural_habitat_multilabel_tile_labels import (  # noqa: E402
    MULTILABEL_FILENAME,
    compute_multilabel_vector,
)


@dataclass(frozen=True)
class SceneRef:
    split: str
    patch_id: str
    scene_path: str


@dataclass
class MigrateStats:
    n_seen: int = 0
    n_migrated: int = 0
    n_already_axes: int = 0
    n_missing_nh: int = 0
    n_missing_coord: int = 0
    n_errors: int = 0


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def parse_splits(splits_arg: str) -> List[str]:
    out: List[str] = []
    for token in splits_arg.split(","):
        name = token.strip()
        if name and name not in out:
            out.append(name)
    return out


def build_scene_path(
    data_root: str, split: str, patch_id: str, dept_year: str, roi: str
) -> str:
    return os.path.join(data_root, split, f"{dept_year}_LIDARHD", roi, patch_id)


def load_scene_refs(
    data_root: str,
    csv_manifest: str,
    target_splits: Sequence[str],
) -> List[SceneRef]:
    refs: List[SceneRef] = []
    split_set = set(target_splits)
    with open(csv_manifest, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("LIDARHD", "").strip().lower() != "true":
                continue
            split = row["split"].strip()
            if split not in split_set:
                continue
            if row.get("NATURAL_HABITAT", "").strip().lower() not in (
                "true",
                "1",
                "yes",
            ):
                continue
            patch_id = row["patch_id"].strip()
            dept_year = (row.get("dept_year") or patch_id.split("_")[0]).strip()
            roi = (row.get("roi") or patch_id.split("_")[1]).strip()
            refs.append(
                SceneRef(
                    split=split,
                    patch_id=patch_id,
                    scene_path=build_scene_path(
                        data_root, split, patch_id, dept_year, roi
                    ),
                )
            )
    return refs


def _update_meta(scene_path: str) -> None:
    meta_path = os.path.join(scene_path, "meta.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    label_defs = dict(meta.get("label_definitions") or {})
    label_defs["natural_habitat"] = NATHAB_ONDISK_DEFINITION
    meta["label_definitions"] = label_defs
    meta["natural_habitat_layout"] = dict(NATHAB_LAYOUT_META)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)


def migrate_one_scene(
    scene_path: str,
    *,
    write_multilabel: bool,
    dry_run: bool,
) -> str:
    """Return status token: migrated | already_axes | missing_nh | missing_coord."""
    coord_path = os.path.join(scene_path, "coord.npy")
    nh_path = os.path.join(scene_path, "natural_habitat.npy")
    if not os.path.isfile(coord_path):
        return "missing_coord"
    if not os.path.isfile(nh_path):
        return "missing_nh"

    stored = np.load(nh_path)
    if is_nathab_axes_array(stored):
        validate_nathab_axes(stored)
        if not dry_run:
            _update_meta(scene_path)
        return "already_axes"
    if not is_nathab_carhab_array(stored):
        raise ValueError(f"{nh_path}: unexpected shape {stored.shape}")

    carhab = np.asarray(stored).reshape(-1)
    if write_multilabel:
        vector = compute_multilabel_vector(carhab, int(carhab.shape[0]))
        if not dry_run:
            np.save(os.path.join(scene_path, MULTILABEL_FILENAME), vector)

    axes = carhab_to_nathab_axes(carhab)
    if not dry_run:
        np.save(nh_path, axes)
        _update_meta(scene_path)
    return "migrated"


def _worker(args: Tuple[str, bool, bool]) -> Tuple[str, str]:
    scene_path, write_multilabel, dry_run = args
    try:
        status = migrate_one_scene(
            scene_path,
            write_multilabel=write_multilabel,
            dry_run=dry_run,
        )
        return scene_path, status
    except Exception as exc:  # noqa: BLE001 — surface per-scene failures
        return scene_path, f"error:{exc}"


def merge_stats(parts: Iterable[str]) -> MigrateStats:
    stats = MigrateStats()
    for status in parts:
        stats.n_seen += 1
        if status == "migrated":
            stats.n_migrated += 1
        elif status == "already_axes":
            stats.n_already_axes += 1
        elif status == "missing_nh":
            stats.n_missing_nh += 1
        elif status == "missing_coord":
            stats.n_missing_coord += 1
        elif status.startswith("error:"):
            stats.n_errors += 1
        else:
            stats.n_errors += 1
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--csv_manifest", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument(
        "--write-multilabel",
        action="store_true",
        help="Regenerate natural_habitat_multilabel.npy from CarHab before bake.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_repo_path(args.data_root)
    csv_manifest = resolve_repo_path(args.csv_manifest)
    splits = parse_splits(args.splits)
    refs = load_scene_refs(data_root, csv_manifest, splits)
    print(f"Found {len(refs)} NATURAL_HABITAT scenes under splits={splits}")
    if args.dry_run:
        print("Dry run: no files will be written.")

    worker_args = [
        (ref.scene_path, bool(args.write_multilabel), bool(args.dry_run))
        for ref in refs
    ]
    statuses: List[str] = []
    errors: List[Tuple[str, str]] = []
    if args.num_workers <= 1:
        for item in tqdm(worker_args, desc="migrate", unit="scene"):
            path, status = _worker(item)
            statuses.append(status)
            if status.startswith("error:"):
                errors.append((path, status))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [pool.submit(_worker, item) for item in worker_args]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="migrate", unit="scene"
            ):
                path, status = future.result()
                statuses.append(status)
                if status.startswith("error:"):
                    errors.append((path, status))

    stats = merge_stats(statuses)
    print(
        "Done: "
        f"seen={stats.n_seen} migrated={stats.n_migrated} "
        f"already_axes={stats.n_already_axes} missing_nh={stats.n_missing_nh} "
        f"missing_coord={stats.n_missing_coord} errors={stats.n_errors}"
    )
    for path, status in errors[:20]:
        print(f"  ERROR {path}: {status}")
    if stats.n_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
