"""
Unzip Flair3D+ archives into a standardized raw directory tree.

Example:
python pointcept/datasets/preprocessing/flair3d_plus/unzip_flair3d.py \
    --source_root /data/geist/Pointcept/data/flair3d_plus/flair3d_plus_067 \
    --target_root /data/geist/Pointcept/data/flair3d_plus/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from zipfile import ZipFile


EXPECTED_MODALITIES = (
    "DEM_ELEV",
    "FOREST",
    "LAND_USE",
    "LIDARHD",
    "NATURAL_HABITAT",
)


def resolve_extract_dir(dst_dir: Path, modality: str, zip_path: Path, archive: ZipFile) -> Path:
    """
    Resolve the destination folder for one archive extraction.

    DEM_ELEV archives are inconsistent:
    - some contain files rooted at <zip_stem>/...
    - some contain files rooted directly at <roi>/...

    We choose a destination that always yields:
    <raw>/DEM_ELEV/<zip_stem>/<roi>/<tile>.tif
    """
    if modality != "DEM_ELEV":
        return dst_dir

    expected_subdir = zip_path.stem
    prefixed_root = f"{expected_subdir}/"
    has_prefixed_root = any(
        member.filename.startswith(prefixed_root) for member in archive.infolist()
    )

    if has_prefixed_root:
        return dst_dir
    return dst_dir / expected_subdir


def extract_modality_archives(
    source_root: Path,
    target_root: Path,
    modality: str,
    overwrite: bool = False,
) -> int:
    """
    Extract all .zip files from one modality folder.

    Returns:
        Number of extracted archives.
    """
    src_dir = source_root / modality
    dst_dir = target_root / modality

    if not src_dir.is_dir():
        print(f"[WARN] Missing source folder: {src_dir}")
        return 0

    zip_files = sorted(src_dir.glob("*.zip"))
    if not zip_files:
        print(f"[WARN] No zip files found in: {src_dir}")
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n== {modality} ==")

    extracted_count = 0
    for zip_path in zip_files:
        with ZipFile(zip_path, "r") as archive:
            extract_dir = resolve_extract_dir(
                dst_dir=dst_dir, modality=modality, zip_path=zip_path, archive=archive
            )
            extract_dir.mkdir(parents=True, exist_ok=True)
            print(f"Extracting {zip_path.name} -> {extract_dir}")
            if overwrite:
                archive.extractall(extract_dir)
            else:
                for member in archive.infolist():
                    output_path = extract_dir / member.filename
                    if not output_path.exists():
                        archive.extract(member, extract_dir)

        if modality == "DEM_ELEV":
            normalize_dem_elev_layout(dst_dir=dst_dir, zip_path=zip_path)
        extracted_count += 1

    return extracted_count


def normalize_dem_elev_layout(dst_dir: Path, zip_path: Path) -> None:
    """
    Normalize DEM_ELEV extraction to match other modalities:
    <raw>/DEM_ELEV/<dept>_DEM_ELEV/<roi>/<tile>.tif

    Some archives extract directly as:
    <raw>/DEM_ELEV/<roi>/<tile>.tif
    """
    expected_subdir = zip_path.stem
    expected_dir = dst_dir / expected_subdir
    if expected_dir.exists():
        return

    # If no expected directory exists, move direct ROI folders under it.
    roi_dirs = [
        p for p in dst_dir.iterdir()
        if p.is_dir() and p.name != expected_subdir and "-" in p.name
    ]
    if not roi_dirs:
        return

    expected_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for roi_dir in roi_dirs:
        target = expected_dir / roi_dir.name
        if target.exists():
            continue
        roi_dir.rename(target)
        moved += 1
    if moved > 0:
        print(
            f"[INFO] Normalized DEM_ELEV layout: moved {moved} ROI folder(s) "
            f"under {expected_dir.name}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Flair3D+ zip archives into a raw folder layout."
    )
    parser.add_argument(
        "--source_root",
        type=Path,
        required=True,
        help="Root folder containing modality subfolders with zip files.",
    )
    parser.add_argument(
        "--target_root",
        type=Path,
        required=True,
        help="Destination raw folder (modality subfolders are created automatically).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already existing files in target_root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()

    if not source_root.is_dir():
        print(f"[ERROR] source_root does not exist: {source_root}")
        return 1

    target_root.mkdir(parents=True, exist_ok=True)

    total_extracted = 0
    for modality in EXPECTED_MODALITIES:
        total_extracted += extract_modality_archives(
            source_root=source_root,
            target_root=target_root,
            modality=modality,
            overwrite=args.overwrite,
        )

    print(f"\nDone. Extracted {total_extracted} archive(s) into {target_root}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
