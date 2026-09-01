"""Standalone backfill: write network.npy (3, H, W) binary masks per Malibu3D+ tile.

Driven by the split manifest CSV (same contract as ``preprocess_malibu3d_v2``):
each ``LIDARHD=True`` row must already exist under ``data_root`` with
``coord.npy``. Missing patches are hard errors (manifest is the source of
truth; disk is only checked). Known-missing tiles listed in
``missing_coord_tiles.details.csv`` are skipped.

Loads graphs exported by Malibu3D-build ``export_network_graphs.py``, densifies
each ROI network once into unique 1 m Lambert cells (longitudinal step plus
optional lateral ``± line_width_m / 2`` offsets), then slices those cells into
per-patch grids. Empty masks write ``meta.network`` only (no ``network.npy``).

Availability flags ``ROADS`` / ``RAILROADS`` / ``TRANSMISSION_LINES`` must be
present in the split manifest (enriched by Malibu3D-build). A ``True`` flag with
a missing ``*_graph.gpkg`` is a hard error.

Example::

python pointcept/datasets/preprocessing/malibu3d_plus/rasterize_network.py \
    --data_root data/malibu3d_plus \
    --network_graphs_root data/malibu3d_build/data/network_graphs \
    --split_manifest_csv data/malibu3d_plus/raw/scene_split_manifest_D067.csv \
    --num_workers 24
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

# Prefer sibling imports so this script runs without the full Pointcept stack
# (geopandas env without torch / termcolor).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    from network_label_utils import (  # type: ignore
        NETWORK_TYPES,
        load_roi_exported_networks,
        parse_bool_flag,
    )
    from network_xy_raster_utils import (  # type: ignore
        abs_xy_bounds_from_coord,
        default_missing_coord_details_csv,
        densify_segments_to_absolute_cells,
        grid_from_xy_bounds,
        load_known_missing_tiles,
        mask_from_absolute_cells,
    )
except ImportError:  # pragma: no cover
    from pointcept.datasets.preprocessing.malibu3d_plus.network_label_utils import (
        NETWORK_TYPES,
        load_roi_exported_networks,
        parse_bool_flag,
    )
    from pointcept.datasets.preprocessing.malibu3d_plus.network_xy_raster_utils import (
        abs_xy_bounds_from_coord,
        default_missing_coord_details_csv,
        densify_segments_to_absolute_cells,
        grid_from_xy_bounds,
        load_known_missing_tiles,
        mask_from_absolute_cells,
    )


REQUIRED_MANIFEST_COLUMNS = frozenset(
    {
        "split",
        "dept_year",
        "roi",
        "patch_id",
        "LIDARHD",
        *NETWORK_TYPES,
    }
)


@dataclass(frozen=True)
class ManifestPatch:
    """One LIDARHD patch listed in the split manifest."""

    split: str
    dept_year: str
    roi: str
    patch_id: str
    flags: Tuple[Tuple[str, bool], ...]  # frozen network flags

    @property
    def flags_dict(self) -> Dict[str, bool]:
        return dict(self.flags)

    def patch_dir(self, data_root: Path) -> Path:
        return (
            data_root
            / self.split
            / f"{self.dept_year}_LIDARHD"
            / self.roi
            / self.patch_id
        )

    def roi_dir(self, data_root: Path) -> Path:
        return data_root / self.split / f"{self.dept_year}_LIDARHD" / self.roi


def load_manifest_patches(
    split_manifest_csv: Path,
    *,
    splits: Optional[Sequence[str]] = None,
    network_types: Sequence[str] = NETWORK_TYPES,
    known_missing: Optional[set[Tuple[str, str]]] = None,
) -> Tuple[List[ManifestPatch], int]:
    """Load LIDARHD=True rows from the manifest (optionally filtered by split).

    Skips ``(split, patch_id)`` in ``known_missing`` (same exclusions as training).
    Returns ``(patches, n_skipped_known_missing)``.
    """
    if not split_manifest_csv.is_file():
        raise FileNotFoundError(f"split_manifest_csv not found: {split_manifest_csv}")

    splits_set = {s.strip().lower() for s in splits} if splits else None
    types = tuple(network_types)
    skip = known_missing or set()
    patches: List[ManifestPatch] = []
    n_skipped = 0

    with split_manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {split_manifest_csv}")
        missing_cols = [
            c for c in REQUIRED_MANIFEST_COLUMNS if c not in reader.fieldnames
        ]
        if missing_cols:
            raise ValueError(
                f"split_manifest_csv missing columns {missing_cols}. "
                "Enrich network flags via Malibu3D-build export_network_graphs.py."
            )
        for row in reader:
            split = (row.get("split") or "").strip().lower()
            dept_year = (row.get("dept_year") or "").strip()
            roi = (row.get("roi") or "").strip()
            patch_id = (row.get("patch_id") or "").strip()
            if not split or not dept_year or not roi or not patch_id:
                continue
            if splits_set is not None and split not in splits_set:
                continue
            if not parse_bool_flag(row.get("LIDARHD")):
                continue
            if (split, patch_id) in skip:
                n_skipped += 1
                continue
            flags = tuple((t, parse_bool_flag(row.get(t))) for t in types)
            patches.append(
                ManifestPatch(
                    split=split,
                    dept_year=dept_year,
                    roi=roi,
                    patch_id=patch_id,
                    flags=flags,
                )
            )
    return patches, n_skipped


def group_by_roi(
    patches: List[ManifestPatch], data_root: Path
) -> List[Tuple[Path, Dict[str, bool], List[Path]]]:
    """Group manifest patches by ROI dir; resolve paths and hard-fail if missing.

    Returns a sorted list of ``(roi_dir, flags, patch_dirs)``. Existing patches in
    a ROI are kept even if siblings are missing — but any unexpected absence
    still raises after the full scan.
    """
    grouped: Dict[Path, List[ManifestPatch]] = defaultdict(list)
    for patch in patches:
        grouped[patch.roi_dir(data_root)].append(patch)

    out: List[Tuple[Path, Dict[str, bool], List[Path]]] = []
    missing: List[str] = []
    for roi_dir in sorted(grouped.keys(), key=str):
        roi_patches = sorted(grouped[roi_dir], key=lambda p: p.patch_id)
        flags = roi_patches[0].flags_dict
        for p in roi_patches[1:]:
            if p.flags_dict != flags:
                raise ValueError(
                    f"Inconsistent network flags within ROI {roi_dir}: "
                    f"{roi_patches[0].patch_id}={flags} vs "
                    f"{p.patch_id}={p.flags_dict}"
                )
        patch_dirs: List[Path] = []
        for p in roi_patches:
            patch_dir = p.patch_dir(data_root)
            coord = patch_dir / "coord.npy"
            if not coord.is_file():
                missing.append(str(coord))
            else:
                patch_dirs.append(patch_dir)
        if patch_dirs:
            out.append((roi_dir, flags, patch_dirs))

    if missing:
        preview = "\n  ".join(missing[:20])
        more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise FileNotFoundError(
            f"{len(missing)} manifest patch(es) missing coord.npy under {data_root} "
            f"(not listed in the missing-tiles exclusion file):\n"
            f"  {preview}{more}"
        )
    return out


def _load_cached_bounds(meta: dict) -> Optional[Tuple[float, float, float, float]]:
    net = meta.get("network")
    if not isinstance(net, dict):
        return None
    bounds = net.get("abs_xy_bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    try:
        xmin, ymin, xmax, ymax = (float(v) for v in bounds)
    except (TypeError, ValueError):
        return None
    if not np.isfinite([xmin, ymin, xmax, ymax]).all():
        return None
    if xmax < xmin or ymax < ymin:
        return None
    return xmin, ymin, xmax, ymax


def _read_meta(patch_dir: Path) -> dict:
    meta_path = patch_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(patch_dir: Path, meta: dict) -> None:
    meta_path = patch_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _densify_roi_cells(
    segments_by_type: Dict[str, np.ndarray],
    *,
    pixel_m: float,
    sample_step_m: float,
    line_width_m: float,
) -> Dict[str, np.ndarray]:
    """Densify each network type once into unique absolute 1 m cells."""
    cells_by_type: Dict[str, np.ndarray] = {}
    for network_type in NETWORK_TYPES:
        segments = segments_by_type.get(
            network_type, np.empty((0, 2, 3), dtype=np.float64)
        )
        cells_by_type[network_type] = densify_segments_to_absolute_cells(
            segments,
            pixel_m=pixel_m,
            sample_step_m=sample_step_m,
            line_width_m=line_width_m,
        )
    return cells_by_type


def process_patch(
    patch_dir: Path,
    cells_by_type: Dict[str, np.ndarray],
    *,
    flags: Dict[str, bool],
    pixel_m: float,
    sample_step_m: float,
    line_width_m: float,
    force_reload_bounds: bool = False,
) -> dict:
    """Write network.npy (if non-empty) and update meta.json. Returns stats dict."""
    meta = _read_meta(patch_dir)
    bounds = None if force_reload_bounds else _load_cached_bounds(meta)
    if bounds is None:
        bounds = abs_xy_bounds_from_coord(patch_dir)
    xmin, ymin, xmax, ymax = bounds
    grid = grid_from_xy_bounds(xmin, ymin, xmax, ymax, pixel_m=pixel_m)

    masks = []
    positive_counts = {}
    for network_type in NETWORK_TYPES:
        cells = cells_by_type.get(
            network_type, np.empty((0, 2), dtype=np.int64)
        )
        mask = mask_from_absolute_cells(cells, grid)
        masks.append(mask.astype(np.uint8, copy=False))
        positive_counts[network_type] = int(mask.sum())

    network = np.stack(masks, axis=0)  # (3, H, W)
    is_empty = all(c == 0 for c in positive_counts.values())
    npy_path = patch_dir / "network.npy"
    if is_empty:
        if npy_path.is_file():
            npy_path.unlink()
    else:
        np.save(npy_path, network)

    meta["network"] = {
        "source": "export_network_graphs",
        "origin_x": float(grid.origin_x),
        "origin_y": float(grid.origin_y),
        "width": int(grid.width),
        "height": int(grid.height),
        "pixel_m": float(grid.pixel_m),
        "crs": "EPSG:2154",
        "channel_order": list(NETWORK_TYPES),
        "sample_step_m": float(sample_step_m),
        "line_width_m": float(line_width_m),
        "manifest_flags": {k: bool(flags.get(k, False)) for k in NETWORK_TYPES},
        "abs_xy_bounds": [xmin, ymin, xmax, ymax],
        "positive_pixel_counts": positive_counts,
        "empty": bool(is_empty),
    }
    _write_meta(patch_dir, meta)
    return {
        "patch": str(patch_dir),
        "shape": list(network.shape),
        "positive_pixel_counts": positive_counts,
        "empty": bool(is_empty),
        "wrote_npy": not is_empty,
    }


def _process_roi_task(
    roi_dir: str,
    patch_dirs: List[str],
    flags: Dict[str, bool],
    network_graphs_root: str,
    pixel_m: float,
    sample_step_m: float,
    line_width_m: float,
    force_reload_bounds: bool,
) -> dict:
    """Worker entry: load ROI graphs once, densify once, process all patches."""
    roi_path = Path(roi_dir)
    segments_by_type = load_roi_exported_networks(
        Path(network_graphs_root),
        roi_path,
        flags=flags,
        network_types=NETWORK_TYPES,
    )
    cells_by_type = _densify_roi_cells(
        segments_by_type,
        pixel_m=pixel_m,
        sample_step_m=sample_step_m,
        line_width_m=line_width_m,
    )
    seg_counts = {
        t: int(segments_by_type[t].shape[0]) if segments_by_type[t].size else 0
        for t in NETWORK_TYPES
    }
    cell_counts = {t: int(cells_by_type[t].shape[0]) for t in NETWORK_TYPES}

    patch_stats: List[dict] = []
    n_wrote = 0
    n_empty = 0
    for patch_dir_s in patch_dirs:
        stats = process_patch(
            Path(patch_dir_s),
            cells_by_type,
            flags=flags,
            pixel_m=pixel_m,
            sample_step_m=sample_step_m,
            line_width_m=line_width_m,
            force_reload_bounds=force_reload_bounds,
        )
        patch_stats.append(stats)
        if stats["wrote_npy"]:
            n_wrote += 1
        else:
            n_empty += 1

    return {
        "roi": roi_path.name,
        "roi_dir": str(roi_path),
        "n_patches": len(patch_dirs),
        "n_wrote_npy": n_wrote,
        "n_empty_meta_only": n_empty,
        "segment_counts": seg_counts,
        "cell_counts": cell_counts,
        "flags": flags,
        "patch_stats": patch_stats,
    }


def _update_roi_pbar(pbar: tqdm, result: dict) -> None:
    pbar.set_postfix(
        roi=result["roi"],
        wrote=result["n_wrote_npy"],
        empty=result["n_empty_meta_only"],
        refresh=False,
    )
    pbar.update(1)


def run(
    data_root: Path,
    network_graphs_root: Path,
    split_manifest_csv: Path,
    *,
    splits: Optional[List[str]] = None,
    pixel_m: float = 1.0,
    sample_step_m: float = 0.25,
    line_width_m: float = 0.2,
    force_reload_bounds: bool = False,
    max_rois: Optional[int] = None,
    missing_tiles_file: Optional[Path] = None,
    num_workers: int = 1,
) -> None:
    if missing_tiles_file is None:
        missing_tiles_file = default_missing_coord_details_csv()
    known_missing = load_known_missing_tiles(
        missing_tiles_file if missing_tiles_file.is_file() else None
    )
    if known_missing:
        print(
            f"Known missing tiles: {len(known_missing)} from {missing_tiles_file}"
        )
    elif missing_tiles_file is not None:
        print(
            f"Warning: missing-tiles file not found ({missing_tiles_file}); "
            "unexpected absences will hard-fail."
        )

    patches, n_skipped = load_manifest_patches(
        split_manifest_csv,
        splits=splits,
        network_types=NETWORK_TYPES,
        known_missing=known_missing,
    )
    roi_items = group_by_roi(patches, data_root)
    if max_rois is not None:
        roi_items = roi_items[: max(0, int(max_rois))]

    n_patches = sum(len(pdirs) for _, _, pdirs in roi_items)
    workers = max(1, int(num_workers))
    print(
        f"Manifest: {len(patches) + n_skipped} LIDARHD rows "
        f"({n_skipped} known-missing skipped) → "
        f"{n_patches} on disk in {len(roi_items)} ROIs under {data_root}"
    )
    print(f"network_graphs_root: {network_graphs_root}")
    print(f"split_manifest_csv: {split_manifest_csv}")
    print(
        f"Raster params: pixel_m={pixel_m}, sample_step_m={sample_step_m}, "
        f"line_width_m={line_width_m}, num_workers={workers}"
    )

    n_rois = len(roi_items)
    n_wrote = 0
    n_empty = 0

    with tqdm(total=n_rois, desc="ROIs", unit="roi") as pbar:
        if workers <= 1:
            for roi_dir, flags, roi_patches in roi_items:
                result = _process_roi_task(
                    str(roi_dir),
                    [str(p) for p in roi_patches],
                    flags,
                    str(network_graphs_root),
                    pixel_m,
                    sample_step_m,
                    line_width_m,
                    force_reload_bounds,
                )
                n_wrote += int(result["n_wrote_npy"])
                n_empty += int(result["n_empty_meta_only"])
                _update_roi_pbar(pbar, result)
        else:
            futures = {}
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for roi_dir, flags, roi_patches in roi_items:
                    fut = pool.submit(
                        _process_roi_task,
                        str(roi_dir),
                        [str(p) for p in roi_patches],
                        flags,
                        str(network_graphs_root),
                        float(pixel_m),
                        float(sample_step_m),
                        float(line_width_m),
                        bool(force_reload_bounds),
                    )
                    futures[fut] = roi_dir.name
                for fut in as_completed(futures):
                    roi_name = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:  # noqa: BLE001 — surface worker failures
                        raise RuntimeError(
                            f"ROI worker failed for {roi_name}: {exc}"
                        ) from exc
                    n_wrote += int(result["n_wrote_npy"])
                    n_empty += int(result["n_empty_meta_only"])
                    _update_roi_pbar(pbar, result)

    print(
        f"Done. Patches={n_wrote + n_empty} "
        f"(network.npy={n_wrote}, empty meta-only={n_empty})."
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Rasterize Malibu3D-build exported network graphs into network.npy "
            "binary masks on Malibu3D+ tiles."
        )
    )
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Malibu3D+ data root (contains train/val/test/...)",
    )
    p.add_argument(
        "--network_graphs_root",
        type=str,
        required=True,
        help="Root of exported *_graph.gpkg files (Malibu3D-build network_graphs)",
    )
    p.add_argument(
        "--split_manifest_csv",
        type=str,
        required=True,
        help=(
            "scene_split_manifest.csv enriched with ROADS/RAILROADS/"
            "TRANSMISSION_LINES; drives which patches to process"
        ),
    )
    p.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=None,
        help="Optional split names to keep from the manifest (default: all)",
    )
    p.add_argument("--pixel_m", type=float, default=1.0)
    p.add_argument(
        "--sample_step_m",
        type=float,
        default=0.25,
        help="Longitudinal densification step along each edge (meters)",
    )
    p.add_argument(
        "--line_width_m",
        type=float,
        default=0.2,
        help=(
            "Total corridor width (meters): sample ± width/2 along the segment "
            "normal (no centerline point). <=0 falls back to centerline-only"
        ),
    )
    p.add_argument(
        "--force_reload_bounds",
        action="store_true",
        help="Ignore cached abs_xy_bounds in meta.json",
    )
    p.add_argument(
        "--max_rois",
        type=int,
        default=None,
        help="Optional limit on number of ROIs (debug)",
    )
    p.add_argument(
        "--missing_tiles_file",
        type=str,
        default=None,
        help=(
            "Exclusion list of known-missing (split, patch_id), same role as "
            "Malibu3DDataset missing tiles. Default: "
            "data/malibu3d_plus/missing_coord_tiles.details.csv"
        ),
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="ROI-level process workers (1 = sequential; try 12 on local machine, 24 on JZ)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    missing = (
        Path(args.missing_tiles_file).resolve()
        if args.missing_tiles_file
        else None
    )
    run(
        Path(args.data_root).resolve(),
        Path(args.network_graphs_root).resolve(),
        Path(args.split_manifest_csv).resolve(),
        splits=args.splits,
        pixel_m=float(args.pixel_m),
        sample_step_m=float(args.sample_step_m),
        line_width_m=float(args.line_width_m),
        force_reload_bounds=bool(args.force_reload_bounds),
        max_rois=args.max_rois,
        missing_tiles_file=missing,
        num_workers=int(args.num_workers),
    )


if __name__ == "__main__":
    main()
