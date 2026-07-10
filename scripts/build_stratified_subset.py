#!/usr/bin/env python3
"""
Build a fixed stratified val/test subset sidecar CSV for fast dev evaluation.

Example (val ~34k -> 2k):
Hecate :
python scripts/build_stratified_subset.py \
  --data_root data/flair3d_plus \
  --csv_manifest data/flair3d_plus/raw/scene_split_manifest_D067.csv \
  --split val \
  --max_sample 300 \
  --warm-random 150 \
  --seed 0 \
  --keys segment natural_habitat_multilabel \
  --output data/flair3d_plus/manifests/val_dev_subset_D067_300.csv

Jean-Zay : 
VAL:
python scripts/build_stratified_subset.py \
  --data_root data/flair3d_plus \
  --csv_manifest data/flair3d_plus/raw/scene_split_manifest.csv \
  --split val \
  --max_sample 2000 \
  --warm-random 1000 \
  --seed 0 \
  --keys segment natural_habitat_multilabel \
  --output data/flair3d_plus/manifests/val_dev_subset_2000.csv
  
TEST:
python scripts/build_stratified_subset.py \
  --data_root data/flair3d_plus \
  --csv_manifest data/flair3d_plus/raw/scene_split_manifest.csv \
  --split test \
  --max_sample 10000 \
  --seed 0 \
  --keys segment natural_habitat_multilabel \
  --output data/flair3d_plus/manifests/test_dev_subset_10k.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load_subset_utils():
    path = os.path.join(REPO_ROOT, "pointcept", "datasets", "subset_utils.py")
    spec = importlib.util.spec_from_file_location("subset_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load subset_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["subset_utils"] = module
    spec.loader.exec_module(module)
    return module


_subset_utils = _load_subset_utils()
NH_MULTILABEL_CLASS_NAMES = _subset_utils.NH_MULTILABEL_CLASS_NAMES
build_scene_features = _subset_utils.build_scene_features
select_stratified_subset = _subset_utils.select_stratified_subset
write_distribution_csvs = _subset_utils.write_distribution_csvs
write_sidecar_csv = _subset_utils.write_sidecar_csv
write_sidecar_meta = _subset_utils.write_sidecar_meta


@dataclass(frozen=True)
class SceneRecord:
    split: str
    patch_id: str
    scene_path: str


def parse_manifest_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def build_scene_path(data_root: str, split: str, patch_id: str, dept_year: str, roi: str) -> str:
    return os.path.join(data_root, split, f"{dept_year}_LIDARHD", roi, patch_id)


def load_hardcoded_excluded_tiles() -> Set[Tuple[str, str]]:
    excluded: Set[Tuple[str, str]] = set()
    details_csv = os.path.join(
        REPO_ROOT, "data", "flair3d_plus", "missing_coord_tiles.details.csv"
    )
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


def load_missing_tiles_manifest(path: str | None) -> Set[Tuple[str, str]]:
    missing_tiles: Set[Tuple[str, str]] = set()
    if not path or not os.path.isfile(path):
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


def load_too_small_tiles_manifest(path: str | None) -> Set[Tuple[str, str]]:
    too_small_tiles: Set[Tuple[str, str]] = set()
    if not path or not os.path.isfile(path):
        return too_small_tiles
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_split = (row.get("split") or "").strip()
            patch_id = (row.get("patch_id") or "").strip()
            if row_split == "train" and row_split and patch_id:
                too_small_tiles.add((row_split, patch_id))
    return too_small_tiles


def load_scene_records(
    data_root: str,
    csv_manifest: str,
    split: str,
    excluded_tiles: Set[Tuple[str, str]],
) -> List[SceneRecord]:
    records: List[SceneRecord] = []
    with open(csv_manifest, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "patch_id", "LIDARHD"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise KeyError(f"Missing required columns in manifest: {sorted(missing_cols)}")

        for row in reader:
            row_split = str(row["split"]).strip()
            patch_id = str(row["patch_id"]).strip()
            if row_split != split or not patch_id:
                continue
            if not parse_manifest_bool(row.get("LIDARHD")):
                continue
            if (row_split, patch_id) in excluded_tiles:
                continue

            dept_year = (row.get("dept_year") or "").strip() or patch_id.split("_", 2)[0]
            roi = (row.get("roi") or "").strip() or patch_id.split("_", 2)[1]
            scene_path = build_scene_path(data_root, row_split, patch_id, dept_year, roi)
            records.append(
                SceneRecord(split=row_split, patch_id=patch_id, scene_path=scene_path)
            )
    return records


def _worker_build_features(
    record: SceneRecord,
    keys: Sequence[str],
    ignore_index: int,
):
    return build_scene_features(
        record.split,
        record.patch_id,
        record.scene_path,
        keys,
        ignore_index=ignore_index,
    )


def _ensure_scenes_on_disk(records: Sequence[SceneRecord]) -> None:
    missing = [record for record in records if not os.path.isdir(record.scene_path)]
    if not missing:
        return

    lines = [
        f"{len(missing)} / {len(records)} manifest scenes are missing on disk.",
        "Refusing to build subset with incomplete data (check data_root sync/preprocessing).",
        "Missing examples:",
    ]
    preview_limit = 20
    for record in missing[:preview_limit]:
        lines.append(f"  {record.split},{record.patch_id} -> {record.scene_path}")
    if len(missing) > preview_limit:
        lines.append(f"  ... and {len(missing) - preview_limit} more")
    raise SystemExit("\n".join(lines))


def _log_nh_distribution(meta: dict) -> None:
    target_u = meta.get("target_u", {})
    subset_u = meta.get("subset_u", {})
    focus = {"mineral", "aquatic", "built", "road", "cultivated"}
    print("NH label presence (subset vs full split):")
    for name in NH_MULTILABEL_CLASS_NAMES:
        full = float(target_u.get(name, 0.0))
        sub = float(subset_u.get(name, 0.0))
        marker = " *" if name in focus else ""
        print(f"  {name:14s} full={full:6.3%} subset={sub:6.3%}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a fixed stratified val/test subset sidecar CSV."
    )
    parser.add_argument("--data_root", required=True, help="Flair3D+ preprocessed root")
    parser.add_argument("--csv_manifest", required=True, help="Scene split manifest CSV")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--max_sample", type=int, required=True)
    parser.add_argument(
        "--warm-random",
        type=int,
        default=None,
        help="Warm-start random tiles (default: max_sample // 2)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["segment", "natural_habitat_multilabel"],
        help="Label keys used for stratification",
    )
    parser.add_argument("--output", required=True, help="Output sidecar CSV path")
    parser.add_argument(
        "--missing_tiles_manifest",
        default="data/flair3d_plus/missing_ply_preflight.txt",
    )
    parser.add_argument(
        "--too_small_tiles_manifest",
        default="data/flair3d_plus/too_small_tiles.csv",
    )
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument(
        "--nh-unweighted",
        action="store_true",
        help="Disable inverse-frequency weighting for NH multilabel L1",
    )
    parser.add_argument(
        "--nh-beta-scale",
        type=float,
        default=0.05,
        help="Scale for NH term in combined greedy score",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(REPO_ROOT, data_root)

    csv_manifest = args.csv_manifest
    if not os.path.isabs(csv_manifest):
        csv_manifest = os.path.join(REPO_ROOT, csv_manifest)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(REPO_ROOT, output_path)

    missing_manifest = args.missing_tiles_manifest
    if missing_manifest and not os.path.isabs(missing_manifest):
        missing_manifest = os.path.join(REPO_ROOT, missing_manifest)

    too_small_manifest = args.too_small_tiles_manifest
    if too_small_manifest and not os.path.isabs(too_small_manifest):
        too_small_manifest = os.path.join(REPO_ROOT, too_small_manifest)

    excluded = (
        load_hardcoded_excluded_tiles()
        | load_missing_tiles_manifest(missing_manifest)
        | load_too_small_tiles_manifest(too_small_manifest)
    )
    records = load_scene_records(data_root, csv_manifest, args.split, excluded)
    if not records:
        raise SystemExit(f"No scenes found for split={args.split!r}")

    _ensure_scenes_on_disk(records)

    keys = tuple(args.keys)
    use_segment = "segment" in keys
    use_nh = "natural_habitat_multilabel" in keys

    features = []
    if args.num_workers <= 1:
        for record in tqdm(records, desc="Loading scene labels"):
            features.append(
                build_scene_features(
                    record.split,
                    record.patch_id,
                    record.scene_path,
                    keys,
                    ignore_index=args.ignore_index,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {
                pool.submit(
                    _worker_build_features,
                    record,
                    keys,
                    args.ignore_index,
                ): record
                for record in records
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading scene labels",
            ):
                features.append(future.result())

    if len(features) != len(records):
        raise SystemExit(
            f"Feature loading incomplete: got {len(features)} / {len(records)} scenes."
        )

    warm_random = args.warm_random
    if warm_random is None:
        warm_random = args.max_sample // 2

    result = select_stratified_subset(
        features,
        args.max_sample,
        warm_random=warm_random,
        shuffle_seed=args.seed,
        use_segment=use_segment,
        use_nh=use_nh,
        nh_weighted=not args.nh_unweighted,
        nh_beta_scale=args.nh_beta_scale,
    )

    meta = dict(result.meta)
    meta.update(
        {
            "split": args.split,
            "data_root": data_root,
            "csv_manifest": csv_manifest,
            "keys": list(keys),
            "output": output_path,
        }
    )

    write_sidecar_csv(output_path, result.selected)
    dist_info = write_distribution_csvs(
        output_path, args.split, features, result.selected
    )
    meta.update(dist_info)
    write_sidecar_meta(output_path, meta)

    pct = 100.0 * len(result.selected) / max(len(features), 1)
    print(
        f"Selected {len(result.selected)} / {len(features)} tiles ({pct:.1f}%) "
        f"for split={args.split!r}"
    )
    print(
        f"L1 segment: warm={meta['l1_seg_after_warm']:.6f} final={meta['l1_seg_final']:.6f}"
    )
    print(
        f"L1 NH:      warm={meta['l1_nh_after_warm']:.6f} final={meta['l1_nh_final']:.6f}"
    )
    _log_nh_distribution(meta)
    print(f"Wrote sidecar CSV:           {output_path}")
    print(f"Wrote segment distribution:  {meta['distribution_segment_csv']}")
    print(f"Wrote NH distribution:       {meta['distribution_nh_csv']}")
    print(f"Wrote meta JSON:             {output_path}.meta.json")


if __name__ == "__main__":
    main()
