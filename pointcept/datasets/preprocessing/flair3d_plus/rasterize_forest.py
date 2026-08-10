"""Standalone backfill: write forest_2d.npy (1, H, W) masks per Flair3D+ tile.

Driven by the split manifest CSV (same contract as ``preprocess_flair3d_v2``):
each ``LIDARHD=True`` row must already exist under ``data_root`` with
``coord.npy``. Missing patches are hard errors (manifest is the source of
truth; disk is only checked). Known-missing tiles listed in
``missing_coord_tiles.details.csv`` are skipped.

Unlike ``rasterize_network.py`` (which rasterizes a vector graph), FOREST is
already a raster: this script reads the window of the source FOREST GeoTIFF
covering each tile's own point-cloud bounding box, resamples it (majority
vote) directly to the target ``pixel_m`` grid, and writes it out in the same
south-up ``(1, H, W)`` layout used by ``network.npy`` / ``NetworkRasterToPoint
Labels``. FOREST coverage is complete for every (dept_year, roi) couple, so
(unlike network) there is no "expected but absent" case -- every manifest
patch gets a ``forest_2d.npy``.

Example (Hecate, D067)::

python pointcept/datasets/preprocessing/flair3d_plus/rasterize_forest.py \
    --data_root data/flair3d_plus \
    --source_dataset_root data/flair3d_plus/raw \
    --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest_D067.csv \
    --pixel_m 0.5

Example (Jean Zay, full manifest)::

python pointcept/datasets/preprocessing/flair3d_plus/rasterize_forest.py \
    --data_root data/flair3d_plus \
    --source_dataset_root /lustre/fswork/projects/rech/unv/usi32yh/Pointcept/data/flair3d_plus/raw \
    --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest.csv \
    --pixel_m 0.5 \
    --num_workers 24
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    from network_label_utils import parse_bool_flag  # type: ignore
    from network_xy_raster_utils import (  # type: ignore
        abs_xy_bounds_from_coord,
        default_missing_coord_details_csv,
        grid_from_xy_bounds,
        load_known_missing_tiles,
    )
    from preprocess_flair3d_v2 import build_modality_patch_path  # type: ignore
except ImportError:  # pragma: no cover
    from pointcept.datasets.preprocessing.flair3d_plus.network_label_utils import (
        parse_bool_flag,
    )
    from pointcept.datasets.preprocessing.flair3d_plus.network_xy_raster_utils import (
        abs_xy_bounds_from_coord,
        default_missing_coord_details_csv,
        grid_from_xy_bounds,
        load_known_missing_tiles,
    )
    from pointcept.datasets.preprocessing.flair3d_plus.preprocess_flair3d_v2 import (
        build_modality_patch_path,
    )


REQUIRED_MANIFEST_COLUMNS = frozenset(
    {"split", "dept_year", "roi", "scene_i_j", "patch_id", "LIDARHD"}
)


@dataclass(frozen=True)
class ManifestPatch:
    """One LIDARHD patch listed in the split manifest."""

    split: str
    dept_year: str
    roi: str
    scene_i_j: str
    patch_id: str

    def patch_dir(self, data_root: Path) -> Path:
        return (
            data_root / self.split / f"{self.dept_year}_LIDARHD" / self.roi / self.patch_id
        )

    def lidar_patch_stem(self) -> str:
        return f"{self.dept_year}_LIDARHD_{self.roi}_{self.scene_i_j}"


def load_manifest_patches(
    split_manifest_csv: Path,
    *,
    splits: Optional[Sequence[str]] = None,
    known_missing: Optional[set] = None,
) -> Tuple[List[ManifestPatch], int]:
    """Load LIDARHD=True rows from the manifest (optionally filtered by split)."""
    if not split_manifest_csv.is_file():
        raise FileNotFoundError(f"split_manifest_csv not found: {split_manifest_csv}")

    splits_set = {s.strip().lower() for s in splits} if splits else None
    skip = known_missing or set()
    patches: List[ManifestPatch] = []
    n_skipped = 0

    with split_manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {split_manifest_csv}")
        missing_cols = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"split_manifest_csv missing columns {missing_cols}.")
        for row in reader:
            split = (row.get("split") or "").strip().lower()
            dept_year = (row.get("dept_year") or "").strip()
            roi = (row.get("roi") or "").strip()
            scene_i_j = (row.get("scene_i_j") or "").strip()
            patch_id = (row.get("patch_id") or "").strip()
            if not split or not dept_year or not roi or not scene_i_j or not patch_id:
                continue
            if splits_set is not None and split not in splits_set:
                continue
            if not parse_bool_flag(row.get("LIDARHD")):
                continue
            if (split, patch_id) in skip:
                n_skipped += 1
                continue
            patches.append(ManifestPatch(split, dept_year, roi, scene_i_j, patch_id))
    return patches, n_skipped


def _read_meta(patch_dir: Path) -> dict:
    meta_path = patch_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(patch_dir: Path, meta: dict) -> None:
    with open(patch_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _load_cached_bounds(meta: dict) -> Optional[Tuple[float, float, float, float]]:
    fr = meta.get("forest_2d")
    if not isinstance(fr, dict):
        return None
    bounds = fr.get("abs_xy_bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    try:
        xmin, ymin, xmax, ymax = (float(v) for v in bounds)
    except (TypeError, ValueError):
        return None
    if not np.isfinite([xmin, ymin, xmax, ymax]).all() or xmax < xmin or ymax < ymin:
        return None
    return xmin, ymin, xmax, ymax


def process_patch(
    patch_dir,
    forest_tiff_path,
    *,
    pixel_m: float = 0.5,
    ignore_index: int = 2,
    force_reload_bounds: bool = False,
) -> dict:
    """Write forest_2d.npy and update meta.json. Returns a stats dict."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    patch_dir = Path(patch_dir)
    meta = _read_meta(patch_dir)
    bounds = None if force_reload_bounds else _load_cached_bounds(meta)
    if bounds is None:
        bounds = abs_xy_bounds_from_coord(patch_dir)
    xmin, ymin, xmax, ymax = bounds
    grid = grid_from_xy_bounds(xmin, ymin, xmax, ymax, pixel_m=pixel_m)

    with rasterio.open(str(forest_tiff_path)) as src:
        nodata = src.nodata
        window = from_bounds(
            grid.origin_x,
            grid.origin_y,
            grid.origin_x + grid.width * grid.pixel_m,
            grid.origin_y + grid.height * grid.pixel_m,
            transform=src.transform,
        )
        raw = src.read(
            1,
            window=window,
            out_shape=(grid.height, grid.width),
            resampling=Resampling.mode,
            boundless=True,
            fill_value=ignore_index,
        )

    # Map the raster's own nodata sentinel (if declared) to ignore_index --
    # mirrors the nodata->void mapping sample_raster_to_points does for the
    # per-point `forest` task (preprocess_flair3d_v2.py). Without this, a
    # stray nodata value would be written verbatim into forest_2d.npy and
    # only surface later as an opaque CUDA nll_loss assert during training.
    if nodata is not None:
        raw = np.where(raw == nodata, ignore_index, raw)

    # rasterio reads north-up (row 0 = north/top); the Flair3D+ grid
    # convention used by network.npy / NetworkRasterToPointLabels is south-up
    # (row 0 = south, row index increases with northing) -- see
    # network_xy_raster_utils.mask_from_absolute_cells. Flip to match.
    forest = np.flipud(raw).astype(np.uint8, copy=False)
    forest = forest[np.newaxis, :, :]  # (1, H, W)

    valid_values = {0, 1, ignore_index}
    bad_values = sorted(int(v) for v in np.unique(forest) if int(v) not in valid_values)
    if bad_values:
        raise ValueError(
            f"{patch_dir}: forest_2d raster for {forest_tiff_path} contains "
            f"unexpected pixel value(s) {bad_values} outside the valid set "
            f"{sorted(valid_values)} (0=non-forest, 1=forest, "
            f"{ignore_index}=ignore_index/void). Check the source FOREST "
            "GeoTIFF's nodata handling before re-running preprocessing."
        )

    np.save(patch_dir / "forest_2d.npy", forest)

    meta["forest_2d"] = {
        "source": "FOREST_geotiff",
        "origin_x": float(grid.origin_x),
        "origin_y": float(grid.origin_y),
        "width": int(grid.width),
        "height": int(grid.height),
        "pixel_m": float(grid.pixel_m),
        "crs": "EPSG:2154",
        "channel_order": ["FOREST"],
        "abs_xy_bounds": [xmin, ymin, xmax, ymax],
        "positive_pixel_count": int((forest == 1).sum()),
        "void_pixel_count": int((forest == ignore_index).sum()),
    }
    _write_meta(patch_dir, meta)
    return {
        "patch": str(patch_dir),
        "shape": list(forest.shape),
        "positive_pixel_count": int((forest == 1).sum()),
        "void_pixel_count": int((forest == ignore_index).sum()),
    }


def run(
    data_root: Path,
    source_dataset_root: Path,
    split_manifest_csv: Path,
    *,
    splits: Optional[List[str]] = None,
    pixel_m: float = 0.5,
    ignore_index: int = 2,
    force_reload_bounds: bool = False,
    missing_tiles_file: Optional[Path] = None,
    num_workers: int = 1,
) -> None:
    if missing_tiles_file is None:
        missing_tiles_file = default_missing_coord_details_csv()
    known_missing = load_known_missing_tiles(
        missing_tiles_file if missing_tiles_file.is_file() else None
    )
    patches, n_skipped = load_manifest_patches(
        split_manifest_csv, splits=splits, known_missing=known_missing
    )
    workers = max(1, int(num_workers))
    print(
        f"Manifest: {len(patches) + n_skipped} LIDARHD rows "
        f"({n_skipped} known-missing skipped) -> {len(patches)} to process "
        f"(num_workers={workers})"
    )

    # Resolve patch_dir/forest_tiff_path and hard-fail on missing coord.npy
    # up front (cheap, sequential) -- only the actual raster read/write in
    # process_patch is worth parallelizing.
    tasks: List[Tuple[Path, str]] = []
    n_missing_tiff = 0
    for patch in patches:
        patch_dir = patch.patch_dir(data_root)
        if not (patch_dir / "coord.npy").is_file():
            raise FileNotFoundError(f"Manifest patch missing coord.npy: {patch_dir}")
        forest_tiff_path = build_modality_patch_path(
            dataset_root=str(source_dataset_root),
            modality="FOREST",
            dept_year=patch.dept_year,
            roi=patch.roi,
            lidar_patch_stem=patch.lidar_patch_stem(),
        )
        if not Path(forest_tiff_path).is_file():
            n_missing_tiff += 1
            print(f"WARNING: FOREST tiff not found, skipping: {forest_tiff_path}")
            continue
        tasks.append((patch_dir, forest_tiff_path))

    n_ok = 0
    if workers <= 1:
        for patch_dir, forest_tiff_path in tqdm(tasks, desc="patches", unit="patch"):
            process_patch(
                patch_dir,
                forest_tiff_path,
                pixel_m=pixel_m,
                ignore_index=ignore_index,
                force_reload_bounds=force_reload_bounds,
            )
            n_ok += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    process_patch,
                    patch_dir,
                    forest_tiff_path,
                    pixel_m=pixel_m,
                    ignore_index=ignore_index,
                    force_reload_bounds=force_reload_bounds,
                ): patch_dir
                for patch_dir, forest_tiff_path in tasks
            }
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="patches", unit="patch"
            ):
                patch_dir = futures[fut]
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001 — surface worker failures
                    raise RuntimeError(f"Patch worker failed for {patch_dir}: {exc}") from exc
                n_ok += 1

    print(f"Done. forest_2d.npy written for {n_ok} patches ({n_missing_tiff} missing tiffs).")
    if n_missing_tiff > 0:
        raise RuntimeError(
            f"{n_missing_tiff} FOREST tiff(s) were missing (see WARNING lines above); "
            "FOREST coverage is expected to be complete for every manifest patch."
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rasterize the FOREST GeoTIFF into forest_2d.npy on Flair3D+ tiles."
    )
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument(
        "--source_dataset_root",
        type=str,
        required=True,
        help="Root directory containing auxiliary modality GeoTIFFs (FOREST, etc.), "
        "same layout as preprocess_flair3d_v2's --dataset_root.",
    )
    p.add_argument("--split_manifest_csv", type=str, required=True)
    p.add_argument("--splits", type=str, nargs="*", default=None)
    p.add_argument("--pixel_m", type=float, default=0.5)
    p.add_argument("--ignore_index", type=int, default=2)
    p.add_argument("--force_reload_bounds", action="store_true")
    p.add_argument("--missing_tiles_file", type=str, default=None)
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for rasterizing patches (default: 1, sequential).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    missing = Path(args.missing_tiles_file).resolve() if args.missing_tiles_file else None
    run(
        Path(args.data_root).resolve(),
        Path(args.source_dataset_root).resolve(),
        Path(args.split_manifest_csv).resolve(),
        splits=args.splits,
        pixel_m=float(args.pixel_m),
        ignore_index=int(args.ignore_index),
        force_reload_bounds=bool(args.force_reload_bounds),
        missing_tiles_file=missing,
        num_workers=int(args.num_workers),
    )


if __name__ == "__main__":
    main()
