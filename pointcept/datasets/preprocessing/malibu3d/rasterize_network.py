"""Write the ``network.npy`` (1, H, W) road masks of every MALiBU3D tile from the road graphs.

Driven by the tile CSV (``tiles.csv`` or a subset of its rows, read via
``pointcept/datasets/tile_catalog.py``): each tile must already exist under ``data_root`` with
``coord.npy`` (a missing tile is an error). Each mask is written on the tile's 1 m grid given by
the ``forest_origin_x/y`` and ``forest_width/height`` columns, the grid of ``forest_2d.npy``.

Ground-truth roads are graphs (``{dept}_{roi}_ROADS_graph.gpkg``, one per ROI, read from
each ROI directory, which is the Hugging Face layout, or from ``--network_graphs_root``).
Each ROI graph is densified once into unique 1 m Lambert-93 cells (longitudinal step plus
optional lateral ``± line_width_m / 2`` offsets), which are then sliced into each tile's
grid. The single channel is ``NETWORK_TYPES`` (ROADS). Every tile gets a ``network.npy``, all
zeros when no road crosses it, so a missing file always means the masks were not rasterized.

A tile whose catalog flag ``has_roads_graph`` (older ``ROADS``) is ``True`` but whose ROI
has no ROADS graph is an error.

Example::

    python pointcept/datasets/preprocessing/malibu3d/rasterize_network.py \
        --data_root data/malibu3d \
        --split_manifest_csv data/malibu3d/tiles.csv \
        --num_workers 16
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from pointcept.datasets.tile_catalog import ManifestPatch, load_manifest_patches
from pointcept.evaluation.network.label_utils import (
    NETWORK_TYPES,
    load_roi_exported_networks,
)
from pointcept.evaluation.network.xy_raster_utils import (
    GridSpec,
    densify_segments_to_absolute_cells,
    mask_from_absolute_cells,
)

# Pixel size of the tile grids of tiles.csv (forest_* columns).
PIXEL_M = 1.0


def group_by_roi(
    patches: List[ManifestPatch], data_root: Path
) -> List[Tuple[Path, Dict[str, bool], List[Tuple[Path, GridSpec]]]]:
    """Group tiles by ROI directory; raise if any tile lacks ``coord.npy`` or its grid.

    Returns a sorted list of ``(roi_dir, flags, [(patch_dir, grid), ...])``. Flags must
    agree within a ROI. All missing tiles are listed in a single error after the full scan.
    """
    grouped: Dict[Path, List[ManifestPatch]] = defaultdict(list)
    for patch in patches:
        grouped[patch.roi_dir(data_root)].append(patch)

    out: List[Tuple[Path, Dict[str, bool], List[Tuple[Path, GridSpec]]]] = []
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
        patch_dirs: List[Tuple[Path, GridSpec]] = []
        for p in roi_patches:
            if p.raster_grid is None:
                raise ValueError(
                    f"Tile {p.patch_id} has no raster grid: the tile CSV needs the "
                    "forest_origin_x, forest_origin_y, forest_width and forest_height "
                    "columns of tiles.csv."
                )
            patch_dir = p.patch_dir(data_root)
            coord = patch_dir / "coord.npy"
            if not coord.is_file():
                missing.append(str(coord))
            else:
                origin_x, origin_y, width, height = p.raster_grid
                grid = GridSpec(
                    origin_x=float(origin_x),
                    origin_y=float(origin_y),
                    width=int(width),
                    height=int(height),
                    pixel_m=PIXEL_M,
                )
                patch_dirs.append((patch_dir, grid))
        if patch_dirs:
            out.append((roi_dir, flags, patch_dirs))

    if missing:
        preview = "\n  ".join(missing[:20])
        more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise FileNotFoundError(
            f"{len(missing)} manifest patch(es) missing coord.npy under {data_root}:\n"
            f"  {preview}{more}"
        )
    return out


def _densify_roi_cells(
    segments_by_type: Dict[str, np.ndarray],
    *,
    pixel_m: float,
    sample_step_m: float,
    line_width_m: float,
) -> Dict[str, np.ndarray]:
    """Densify each network type of a ROI once into unique absolute ``pixel_m`` cells."""
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
    patch_dir: Path, grid: GridSpec, cells_by_type: Dict[str, np.ndarray]
) -> dict:
    """Write the tile's ``(1, H, W)`` network.npy (all zeros without any road); return stats."""
    masks = []
    positive_counts = {}
    for network_type in NETWORK_TYPES:
        cells = cells_by_type.get(network_type, np.empty((0, 2), dtype=np.int64))
        mask = mask_from_absolute_cells(cells, grid)
        masks.append(mask.astype(np.uint8, copy=False))
        positive_counts[network_type] = int(mask.sum())
    np.save(patch_dir / "network.npy", np.stack(masks, axis=0))
    return {
        "patch": str(patch_dir),
        "positive_pixel_counts": positive_counts,
        "empty": all(c == 0 for c in positive_counts.values()),
    }


def _process_roi_task(
    roi_dir: str,
    patches: List[Tuple[str, GridSpec]],
    flags: Dict[str, bool],
    network_graphs_root: Optional[str],
    sample_step_m: float,
    line_width_m: float,
) -> dict:
    """Worker entry: load and densify the ROI graphs once, then process all its tiles."""
    roi_path = Path(roi_dir)
    segments_by_type = load_roi_exported_networks(
        Path(network_graphs_root) if network_graphs_root else None,
        roi_path,
        flags=flags,
        network_types=NETWORK_TYPES,
    )
    cells_by_type = _densify_roi_cells(
        segments_by_type,
        pixel_m=PIXEL_M,
        sample_step_m=sample_step_m,
        line_width_m=line_width_m,
    )
    seg_counts = {
        t: int(segments_by_type[t].shape[0]) if segments_by_type[t].size else 0
        for t in NETWORK_TYPES
    }
    cell_counts = {t: int(cells_by_type[t].shape[0]) for t in NETWORK_TYPES}

    patch_stats: List[dict] = []
    n_empty = 0
    for patch_dir_s, grid in patches:
        stats = process_patch(Path(patch_dir_s), grid, cells_by_type)
        patch_stats.append(stats)
        n_empty += int(stats["empty"])

    return {
        "roi": roi_path.name,
        "roi_dir": str(roi_path),
        "n_patches": len(patches),
        "n_empty": n_empty,
        "segment_counts": seg_counts,
        "cell_counts": cell_counts,
        "flags": flags,
        "patch_stats": patch_stats,
    }


def _update_roi_pbar(pbar: tqdm, result: dict) -> None:
    pbar.set_postfix(
        roi=result["roi"],
        tiles=result["n_patches"],
        empty=result["n_empty"],
        refresh=False,
    )
    pbar.update(1)


def run(
    data_root: Path,
    network_graphs_root: Optional[Path],
    split_manifest_csv: Path,
    *,
    splits: Optional[List[str]] = None,
    sample_step_m: float = 0.25,
    line_width_m: float = 0.2,
    max_rois: Optional[int] = None,
    num_workers: int = 1,
) -> None:
    patches = load_manifest_patches(split_manifest_csv, splits=splits)
    roi_items = group_by_roi(patches, data_root)
    if max_rois is not None:
        roi_items = roi_items[: max(0, int(max_rois))]

    n_patches = sum(len(pdirs) for _, _, pdirs in roi_items)
    workers = max(1, int(num_workers))
    print(
        f"Manifest: {len(patches)} tiles → "
        f"{n_patches} on disk in {len(roi_items)} ROIs under {data_root}"
    )
    print(f"network_graphs_root: {network_graphs_root}")
    print(f"split_manifest_csv: {split_manifest_csv}")
    print(
        f"Raster params: pixel_m={PIXEL_M}, sample_step_m={sample_step_m}, "
        f"line_width_m={line_width_m}, num_workers={workers}"
    )

    n_rois = len(roi_items)
    n_tiles = 0
    n_empty = 0

    with tqdm(total=n_rois, desc="ROIs", unit="roi") as pbar:
        if workers <= 1:
            for roi_dir, flags, roi_patches in roi_items:
                result = _process_roi_task(
                    str(roi_dir),
                    [(str(p), g) for p, g in roi_patches],
                    flags,
                    str(network_graphs_root) if network_graphs_root else None,
                    sample_step_m,
                    line_width_m,
                )
                n_tiles += int(result["n_patches"])
                n_empty += int(result["n_empty"])
                _update_roi_pbar(pbar, result)
        else:
            futures = {}
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for roi_dir, flags, roi_patches in roi_items:
                    fut = pool.submit(
                        _process_roi_task,
                        str(roi_dir),
                        [(str(p), g) for p, g in roi_patches],
                        flags,
                        str(network_graphs_root) if network_graphs_root else None,
                        float(sample_step_m),
                        float(line_width_m),
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
                    n_tiles += int(result["n_patches"])
                    n_empty += int(result["n_empty"])
                    _update_roi_pbar(pbar, result)

    print(f"Done. network.npy written for {n_tiles} tiles ({n_empty} without any road).")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Rasterize the MALiBU3D road-network graphs into network.npy "
            "binary masks on MALiBU3D tiles."
        )
    )
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="MALiBU3D data root (contains train/val/test/...)",
    )
    p.add_argument(
        "--network_graphs_root",
        type=str,
        default=None,
        help="Directory of *_graph.gpkg files. Default: read each graph from its ROI directory "
        "(Hugging Face layout of the dataset).",
    )
    p.add_argument(
        "--split_manifest_csv",
        type=str,
        required=True,
        help=("tiles.csv (or a subset of its rows): drives which tiles to process"),
    )
    p.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=None,
        help="Optional split names to keep from the manifest (default: all)",
    )
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
        "--max_rois",
        type=int,
        default=None,
        help="Optional limit on number of ROIs (debug)",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="ROI-level process workers (1 = sequential)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    run(
        Path(args.data_root).resolve(),
        Path(args.network_graphs_root).resolve() if args.network_graphs_root else None,
        Path(args.split_manifest_csv).resolve(),
        splits=args.splits,
        sample_step_m=float(args.sample_step_m),
        line_width_m=float(args.line_width_m),
        max_rois=args.max_rois,
        num_workers=int(args.num_workers),
    )


if __name__ == "__main__":
    main()
