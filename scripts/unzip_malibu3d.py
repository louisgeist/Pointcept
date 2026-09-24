#!/usr/bin/env python3
"""Extract MALiBU3D Hub ROI zips into the Pointcept on-disk layout.

Each ``data/{split}/{dept}_LIDARHD/{roi}.zip`` is flat (``{tile_id}/coord.npy``,
optional GPKG at zip root). Extract into
``{output_root}/data/{split}/{dept}_LIDARHD/{roi}/``.

Resume: skip a zip when every member already exists with the same uncompressed
size. If a path that should be a directory exists as a file (interrupted unzip),
replace it and continue.

Example (Jean Zay Scratch)::

    python scripts/hf_release/unzip_hf_dataset.py \\
      --zip-root /lustre/fsn1/projects/rech/unv/usi32yh/MALiBU3D_zip \\
      --output-root /lustre/fsn1/projects/rech/unv/usi32yh/MALiBU3D \\
      --workers 8
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

META_NAMES = (
    "labels.json",
    "palettes.json",
    "tiles.csv",
    "tiles.parquet",
    "scene_split_manifest.csv",
    "README.md",
    "LICENSE",
    "SHA256SUMS",
)


def _normalize_member(name: str) -> str:
    return name.replace("\\", "/").lstrip("/")


def zip_dest_dir(zip_path: Path, zip_root: Path, output_root: Path) -> Path:
    rel = zip_path.resolve().relative_to(zip_root.resolve())
    return output_root.resolve() / rel.with_suffix("")


def file_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    out: list[zipfile.ZipInfo] = []
    for info in zf.infolist():
        name = _normalize_member(info.filename)
        if not name or name.endswith("/"):
            continue
        out.append(info)
    return out


def member_dest(dest_root: Path, info: zipfile.ZipInfo) -> Path:
    return dest_root / _normalize_member(info.filename)


def is_complete(zf: zipfile.ZipFile, dest_root: Path) -> bool:
    if not dest_root.is_dir():
        return False
    for info in file_members(zf):
        path = member_dest(dest_root, info)
        if not path.is_file():
            return False
        if path.stat().st_size != int(info.file_size):
            return False
    return True


def _ensure_dir(path: Path) -> None:
    """Make ``path`` a directory, replacing a leftover file of the same name."""
    if path.exists() and not path.is_dir():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _ensure_file_parent(path: Path) -> None:
    """Create parent dirs; unlink any ancestor that exists as a file."""
    for ancestor in reversed(path.parents):
        if ancestor.exists() and not ancestor.is_dir():
            ancestor.unlink()
    _ensure_dir(path.parent)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)


def extract_zip(
    zip_path: Path,
    dest_root: Path,
    *,
    overwrite: bool = False,
) -> str:
    """Extract one ROI zip. Returns ``skipped``, ``extracted``, or raises."""
    with zipfile.ZipFile(zip_path) as zf:
        members = file_members(zf)
        if not members:
            raise ValueError(f"empty zip: {zip_path}")
        if not overwrite and is_complete(zf, dest_root):
            return "skipped"
        _ensure_dir(dest_root)
        for info in members:
            dest = member_dest(dest_root, info)
            if (
                not overwrite
                and dest.is_file()
                and dest.stat().st_size == int(info.file_size)
            ):
                continue
            _ensure_file_parent(dest)
            with zf.open(info) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            written = dest.stat().st_size
            if written != int(info.file_size):
                dest.unlink(missing_ok=True)
                raise OSError(
                    f"short write {dest}: got {written}, expected {info.file_size}"
                )
    return "extracted"


def _worker(payload: tuple[str, str, bool]) -> tuple[str, str, str]:
    zip_s, dest_s, overwrite = payload
    zip_path = Path(zip_s)
    dest_root = Path(dest_s)
    try:
        status = extract_zip(zip_path, dest_root, overwrite=overwrite)
        return (zip_s, status, "")
    except Exception as exc:  # noqa: BLE001 — report per-zip, keep other workers going
        return (zip_s, "error", f"{type(exc).__name__}: {exc}")


def list_zips(zip_root: Path, splits: list[str] | None) -> list[Path]:
    data = zip_root / "data"
    if not data.is_dir():
        raise FileNotFoundError(f"no data/ under {zip_root}")
    zips = sorted(data.rglob("*.zip"))
    if splits:
        keep = set(splits)
        zips = [path for path in zips if path.parts[-3] in keep]
    return zips


def copy_meta(zip_root: Path, output_root: Path) -> list[Path]:
    copied: list[Path] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for name in META_NAMES:
        src = zip_root / name
        if src.is_file():
            shutil.copy2(src, output_root / name)
            copied.append(output_root / name)
    toy = zip_root / "toy"
    dest_toy = output_root / "toy"
    if toy.is_dir() and not dest_toy.exists():
        shutil.copytree(toy, dest_toy)
        copied.append(dest_toy)
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-root",
        type=Path,
        required=True,
        help="Hub snapshot that still contains data/**/*.zip",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Extracted tree (Pointcept layout)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val", "test"),
        help="Repeatable; default is all splits under data/",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite members even when size already matches",
    )
    parser.add_argument(
        "--skip-meta",
        action="store_true",
        help="Do not copy labels.json / tiles.csv / toy/ …",
    )
    args = parser.parse_args(argv)

    zip_root = args.zip_root.resolve()
    output_root = args.output_root.resolve()
    if zip_root == output_root:
        raise SystemExit("--zip-root and --output-root must be different directories")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    zips = list_zips(zip_root, args.split)
    if not zips:
        raise SystemExit(f"no .zip under {zip_root / 'data'}")

    if not args.skip_meta:
        copied = copy_meta(zip_root, output_root)
        for path in copied:
            print(f"copied {path.relative_to(output_root)}")

    jobs = [
        (str(path), str(zip_dest_dir(path, zip_root, output_root)), args.overwrite)
        for path in zips
    ]

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **_kwargs):  # type: ignore[misc]
            return iterable

    n_skip = n_ok = n_err = 0
    errors: list[tuple[str, str]] = []
    workers = min(args.workers, len(jobs))
    print(f"unzip {len(jobs)} zips -> {output_root}  workers={workers}")

    if workers == 1:
        iterator = (_worker(job) for job in jobs)
        progress = tqdm(iterator, total=len(jobs), unit="zip", desc="unzip")
        results = progress
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = [pool.submit(_worker, job) for job in jobs]
        results = tqdm(
            as_completed(futures),
            total=len(futures),
            unit="zip",
            desc="unzip",
        )

    try:
        for result in results:
            if workers == 1:
                zip_s, status, err = result
            else:
                zip_s, status, err = result.result()
            if status == "skipped":
                n_skip += 1
            elif status == "extracted":
                n_ok += 1
            else:
                n_err += 1
                errors.append((zip_s, err))
                print(f"ERROR {zip_s}: {err}", file=sys.stderr)
    finally:
        if workers > 1:
            pool.shutdown(wait=True, cancel_futures=True)

    print(f"done: extracted={n_ok} skipped={n_skip} errors={n_err}")
    if errors:
        fail_log = output_root / "unzip_errors.txt"
        fail_log.write_text(
            "\n".join(f"{path}\t{msg}" for path, msg in errors) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {fail_log}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
