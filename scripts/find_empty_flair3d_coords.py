"""
Scan Flair3D scenes listed in the manifest and detect invalid coord files.

The script reads scene definitions from a CSV manifest (same logic as
Flair3DDataset), skips HARDCODED_EXCLUDED_TILES, and reports scenes where:
- coord.npy is empty
- coord.npy is missing (optional)
- coord.npy cannot be loaded

Primary output format is compatible with Flair3DDataset.missing_tiles_manifest:
  split,patch_id

Example:
  python scripts/find_empty_flair3d_coords.py \
    --data_root data/flair3d_plus \
    --csv_manifest data/flair3d_plus/raw/scene_split_manifest.csv \
    --splits train,val,test \
    --num_workers 8 \
    --check_missing_file
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAIR3D_DATASET_FILE = os.path.join(REPO_ROOT, "pointcept", "datasets", "flair3d.py")


@dataclass(frozen=True)
class SceneRecord:
    split: str
    patch_id: str
    scene_path: str


@dataclass(frozen=True)
class ScanResult:
    split: str
    patch_id: str
    scene_path: str
    reason: str
    excluded_hardcoded: bool


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def parse_splits(splits_arg: str) -> set[str]:
    return {token.strip() for token in splits_arg.split(",") if token.strip()}


def parse_manifest_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def build_scene_path(data_root: str, split: str, patch_id: str, dept_year: str, roi: str) -> str:
    return os.path.join(data_root, split, f"{dept_year}_LIDARHD", roi, patch_id)


def load_scene_records(
    data_root: str,
    csv_manifest: str,
    target_splits: set[str],
    hardcoded_excluded: set[tuple[str, str]],
) -> tuple[list[SceneRecord], list[ScanResult]]:
    scene_records: list[SceneRecord] = []
    skipped_hardcoded: list[ScanResult] = []

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

            if (split, patch_id) in hardcoded_excluded:
                skipped_hardcoded.append(
                    ScanResult(
                        split=split,
                        patch_id=patch_id,
                        scene_path="",
                        reason="skipped_hardcoded_exclusion",
                        excluded_hardcoded=True,
                    )
                )
                continue

            dept_year = (row.get("dept_year") or "").strip() or patch_id.split("_", 2)[0]
            roi = (row.get("roi") or "").strip() or patch_id.split("_", 2)[1]
            scene_path = build_scene_path(data_root, split, patch_id, dept_year, roi)
            scene_records.append(SceneRecord(split=split, patch_id=patch_id, scene_path=scene_path))

    return scene_records, skipped_hardcoded


def _scan_one_scene(task: tuple[SceneRecord, bool]) -> ScanResult | None:
    scene, check_missing_file = task
    coord_path = os.path.join(scene.scene_path, "coord.npy")

    if not os.path.isfile(coord_path):
        if check_missing_file:
            return ScanResult(
                split=scene.split,
                patch_id=scene.patch_id,
                scene_path=scene.scene_path,
                reason="missing_coord_file",
                excluded_hardcoded=False,
            )
        return None

    try:
        coord = np.load(coord_path, mmap_mode="r")
    except Exception as exc:  # noqa: BLE001
        return ScanResult(
            split=scene.split,
            patch_id=scene.patch_id,
            scene_path=scene.scene_path,
            reason=f"coord_load_error:{exc.__class__.__name__}",
            excluded_hardcoded=False,
        )

    if coord.size == 0 or coord.shape[0] == 0:
        return ScanResult(
            split=scene.split,
            patch_id=scene.patch_id,
            scene_path=scene.scene_path,
            reason="empty_coord_array",
            excluded_hardcoded=False,
        )
    return None


def scan_scenes(
    scene_records: list[SceneRecord],
    check_missing_file: bool,
    num_workers: int,
) -> list[ScanResult]:
    tasks = [(scene, check_missing_file) for scene in scene_records]
    if num_workers <= 1:
        out = [_scan_one_scene(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            out = list(pool.map(_scan_one_scene, tasks))

    results = [entry for entry in out if entry is not None]
    results.sort(key=lambda item: (item.split, item.patch_id, item.reason))
    return results


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_manifest_output(path: str, rows: Iterable[ScanResult]) -> int:
    uniq = sorted({(row.split, row.patch_id) for row in rows})
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        for split, patch_id in uniq:
            handle.write(f"{split},{patch_id}\n")
    return len(uniq)


def write_detailed_csv(path: str, rows: Iterable[ScanResult]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "patch_id", "reason", "scene_path", "excluded_hardcoded"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": row.split,
                    "patch_id": row.patch_id,
                    "reason": row.reason,
                    "scene_path": row.scene_path,
                    "excluded_hardcoded": row.excluded_hardcoded,
                }
            )


def summarize_results(
    scene_records: list[SceneRecord],
    anomalies: list[ScanResult],
    skipped_hardcoded: list[ScanResult],
) -> str:
    scanned_by_split = Counter(scene.split for scene in scene_records)
    issue_by_split = Counter(row.split for row in anomalies)
    skipped_by_split = Counter(row.split for row in skipped_hardcoded)
    reason_count = Counter(row.reason for row in anomalies)

    ordered_splits = sorted(set(scanned_by_split) | set(issue_by_split) | set(skipped_by_split))
    lines = [
        f"Scanned scenes: {len(scene_records)}",
        f"Detected anomalies: {len(anomalies)}",
        f"Skipped hardcoded exclusions: {len(skipped_hardcoded)}",
        "Per split:",
    ]
    for split in ordered_splits:
        lines.append(
            f"  - {split}: scanned={scanned_by_split[split]}, "
            f"anomalies={issue_by_split[split]}, skipped_hardcoded={skipped_by_split[split]}"
        )
    if reason_count:
        lines.append("By reason:")
        for reason, count in sorted(reason_count.items()):
            lines.append(f"  - {reason}: {count}")
    return "\n".join(lines)


def default_report_path(output_manifest_path: str) -> str:
    root, ext = os.path.splitext(output_manifest_path)
    if ext:
        return f"{root}.details.csv"
    return f"{output_manifest_path}.details.csv"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find Flair3D scenes with empty coord.npy.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.join("data", "flair3d_plus"),
        help="Preprocessed Flair3D root (relative to repo root if not absolute).",
    )
    parser.add_argument(
        "--csv_manifest",
        type=str,
        default=os.path.join("data", "flair3d_plus", "raw", "scene_split_manifest.csv"),
        help="Path to scene_split_manifest.csv (relative to repo root if not absolute).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split list to scan.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for coord checks.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "flair3d_plus", "missing_coord_tiles.txt"),
        help="Output file with one line per issue: split,patch_id.",
    )
    parser.add_argument(
        "--report_csv",
        type=str,
        default="",
        help="Optional detailed CSV path. Default: <output>.details.csv.",
    )
    parser.add_argument(
        "--check_missing_file",
        action="store_true",
        help="Also report scenes where coord.npy is missing.",
    )
    return parser


def load_hardcoded_excluded_tiles() -> set[tuple[str, str]]:
    """
    Read HARDCODED_EXCLUDED_TILES from pointcept/datasets/flair3d.py using AST.

    This avoids importing the full project package and its optional dependencies.
    """
    with open(FLAIR3D_DATASET_FILE, "r", encoding="utf-8") as handle:
        module_ast = ast.parse(handle.read(), filename=FLAIR3D_DATASET_FILE)

    for node in module_ast.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Flair3DDataset":
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "HARDCODED_EXCLUDED_TILES":
                    value = ast.literal_eval(stmt.value)
                    return {(str(split), str(patch_id)) for split, patch_id in value}
    raise RuntimeError(
        "Could not locate Flair3DDataset.HARDCODED_EXCLUDED_TILES in flair3d.py."
    )


def main() -> None:
    args = get_parser().parse_args()
    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1.")

    data_root = resolve_repo_path(args.data_root)
    csv_manifest = resolve_repo_path(args.csv_manifest)
    output_manifest = resolve_repo_path(args.output)
    output_report = resolve_repo_path(args.report_csv) if args.report_csv else default_report_path(output_manifest)
    target_splits = parse_splits(args.splits)

    hardcoded_excluded = load_hardcoded_excluded_tiles()
    scene_records, skipped_hardcoded = load_scene_records(
        data_root=data_root,
        csv_manifest=csv_manifest,
        target_splits=target_splits,
        hardcoded_excluded=hardcoded_excluded,
    )

    anomalies = scan_scenes(
        scene_records=scene_records,
        check_missing_file=args.check_missing_file,
        num_workers=args.num_workers,
    )
    manifest_count = write_manifest_output(output_manifest, anomalies)
    write_detailed_csv(output_report, [*anomalies, *skipped_hardcoded])

    print(summarize_results(scene_records, anomalies, skipped_hardcoded))
    print(f"Wrote missing-tiles manifest: {output_manifest} ({manifest_count} unique split,patch_id)")
    print(f"Wrote detailed report: {output_report}")
    print("You can set this file as missing_tiles_manifest in your Flair3D config.")


if __name__ == "__main__":
    main()
