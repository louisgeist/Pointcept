"""
# Dry-run (inchangé)
python scripts/delete_corrupted_files_from_unzip.py --workers 24

# Suppression réelle avec 8 workers
python scripts/delete_corrupted_files_from_unzip.py --apply --workers 24

# Suppression + prune des dossiers vides
python scripts/delete_corrupted_files_from_unzip.py --apply --workers 24 --prune_empty_dirs

"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


DEFAULT_MODALITIES = ("LIDARHD", "FOREST", "LAND_USE", "NATURAL_HABITAT", "DEM_ELEV")
MODALITY_EXT = {
    "LIDARHD": ".ply",
    "FOREST": ".tif",
    "LAND_USE": ".tif",
    "NATURAL_HABITAT": ".tif",
    "DEM_ELEV": ".tif",
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Delete files for Flair3D CORRUPTED_TILES from raw unzip tree. "
            "Expected layout: raw/<MODALITY>/<DEPT_YEAR>_<MODALITY>/<ROI>/<PATCH>.(ply|tif)."
        )
    )
    parser.add_argument(
        "--dataset_file",
        type=Path,
        default=Path("pointcept/datasets/flair3d.py"),
        help="Path to flair3d.py containing Flair3DDataset.CORRUPTED_TILES.",
    )
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=Path("data/flair3d_plus/raw"),
        help="Root of unzipped raw dataset tree.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=",".join(DEFAULT_MODALITIES),
        help=(
            "Comma-separated modalities to delete. "
            f"Available: {', '.join(DEFAULT_MODALITIES)}."
        ),
    )
    parser.add_argument(
        "--expected_count",
        type=int,
        default=22,
        help="Safety check: expected number of CORRUPTED_TILES entries.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. If omitted, performs dry-run only.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of worker threads used for file deletion when --apply is set "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--prune_empty_dirs",
        action="store_true",
        help="After deletion, remove empty ROI and modality subfolders.",
    )
    return parser


def parse_modalities(raw_modalities: str) -> list[str]:
    modalities = [token.strip().upper() for token in raw_modalities.split(",") if token.strip()]
    unknown = [name for name in modalities if name not in MODALITY_EXT]
    if unknown:
        raise ValueError(
            f"Unknown modality value(s): {unknown}. "
            f"Allowed: {list(MODALITY_EXT.keys())}"
        )
    return modalities


def load_corrupted_tiles(dataset_file: Path, expected_count: int) -> set[tuple[str, str]]:
    module_ast = ast.parse(dataset_file.read_text(encoding="utf-8"), filename=str(dataset_file))

    for node in module_ast.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Flair3DDataset":
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "CORRUPTED_TILES":
                    value = ast.literal_eval(stmt.value)
                    tiles = {(str(split), str(patch_id)) for split, patch_id in value}
                    if len(tiles) != expected_count:
                        raise RuntimeError(
                            f"Expected {expected_count} CORRUPTED_TILES, found {len(tiles)}."
                        )
                    return tiles

    raise RuntimeError("Could not locate Flair3DDataset.CORRUPTED_TILES in dataset_file.")


def parse_patch_id(patch_id: str) -> tuple[str, str, str]:
    parts = patch_id.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Unexpected patch_id format: '{patch_id}'. Expected '<dept_year>_<roi>_<scene_i_j>'."
        )
    return parts[0], parts[1], parts[2]


def build_paths(raw_root: Path, patch_id: str, modalities: list[str]) -> list[Path]:
    dept_year, roi, scene_i_j = parse_patch_id(patch_id)
    paths: list[Path] = []
    for modality in modalities:
        ext = MODALITY_EXT[modality]
        stem = f"{dept_year}_{modality}_{roi}_{scene_i_j}"
        path = raw_root / modality / f"{dept_year}_{modality}" / roi / f"{stem}{ext}"
        paths.append(path)
    return paths


def prune_empty_ancestors(path: Path, stop_at: Path) -> int:
    removed = 0
    current = path.parent
    stop_at = stop_at.resolve()
    while True:
        current_resolved = current.resolve()
        if current_resolved == stop_at or stop_at not in current_resolved.parents:
            break
        try:
            current.rmdir()
            removed += 1
        except OSError:
            break
        current = current.parent
    return removed


def delete_one_file(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def main() -> int:
    args = get_parser().parse_args()
    dataset_file = args.dataset_file.resolve()
    raw_root = args.raw_root.resolve()
    modalities = parse_modalities(args.modalities)
    workers = args.workers

    if not dataset_file.is_file():
        raise FileNotFoundError(f"dataset_file not found: {dataset_file}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")
    if workers < 1:
        raise ValueError("--workers must be >= 1.")

    tiles = load_corrupted_tiles(dataset_file=dataset_file, expected_count=args.expected_count)
    targets = []
    missing = []
    for _split, patch_id in sorted(tiles):
        for candidate in build_paths(raw_root=raw_root, patch_id=patch_id, modalities=modalities):
            if candidate.exists():
                targets.append(candidate)
            else:
                missing.append(candidate)

    print(f"Loaded {len(tiles)} corrupted tile IDs from: {dataset_file}")
    print(f"Modalities: {', '.join(modalities)}")
    print(f"Existing files targeted: {len(targets)}")
    for path in targets:
        print(f" - {path}")

    if missing:
        print(f"\nMissing candidate files (not deleted): {len(missing)}")
        for path in missing:
            print(f" - {path}")

    if not args.apply:
        print("\nDry-run mode. Use --apply to delete targeted files.")
        return 0

    if workers == 1:
        deleted_count = sum(1 for path in targets if delete_one_file(path))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            deleted_count = sum(executor.map(delete_one_file, targets))

    print(f"\nDeleted {deleted_count} file(s) using {workers} worker(s).")

    if args.prune_empty_dirs:
        removed_dirs = 0
        for path in targets:
            removed_dirs += prune_empty_ancestors(path=path, stop_at=raw_root)
        print(f"Pruned {removed_dirs} empty directorie(s).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
