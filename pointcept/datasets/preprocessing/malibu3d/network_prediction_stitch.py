"""Stitch per-tile predicted network probability rasters into one ROI-level raster.

Ground-truth network graphs are per ROI, while models predict per tile (subtile of a
ROI). ``MultiTaskTester`` (``pointcept/engines/test.py``) writes one
``{patch_id}_logits_network.npy`` ``(C, H, W)`` probability raster per tile, on the grid
of that tile's ``meta.json["network"]`` (written by ``rasterize_network.py``). All tile
grids snap to the same absolute 1 m Lambert-93 lattice (``GridSpec`` origins are integer
multiples of ``pixel_m``), so stitching is integer-offset placement, no resampling.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    from network_xy_raster_utils import GridSpec  # type: ignore
except ImportError:  # pragma: no cover
    from pointcept.datasets.preprocessing.malibu3d.network_xy_raster_utils import (  # type: ignore
        GridSpec,
    )

# Re-exported so that tools/eval_network_apls.py imports tile grouping and
# known-missing handling from this single module.
try:
    from rasterize_network import (  # type: ignore  # noqa: F401
        ManifestPatch,
        default_missing_coord_details_csv,
        load_known_missing_tiles,
        group_by_roi,
        load_manifest_patches,
    )
except ImportError:  # pragma: no cover
    from pointcept.datasets.preprocessing.malibu3d.rasterize_network import (  # type: ignore  # noqa: F401
        ManifestPatch,
        default_missing_coord_details_csv,
        load_known_missing_tiles,
        group_by_roi,
        load_manifest_patches,
    )


def group_by_roi_complete_only(
    patches: Sequence["ManifestPatch"],
    data_root: Path,
    *,
    drop_incomplete: bool = False,
) -> Tuple[List[Tuple[Path, dict, List[Path], dict]], List[dict]]:
    """Group tiles by ROI for the stitched APLS evaluation.

    Returns ``(kept, excluded)``: ``kept`` holds ``(roi_dir, flags, patch_dirs,
    coverage)`` with only the tiles present on disk (``coord.npy``), ``coverage``
    counting the missing ones; ``excluded`` describes the dropped ROIs.

    With ``drop_incomplete=False`` (default), a ROI with missing tiles is kept and
    scored on the available ones (holes stay NaN in the stitch while the GT is the
    full-ROI graph, so the score can be pessimistic). ``drop_incomplete=True`` drops
    it. ROIs without any tile on disk are always excluded.
    """
    grouped: Dict[Path, List["ManifestPatch"]] = defaultdict(list)
    for patch in patches:
        grouped[patch.roi_dir(data_root)].append(patch)

    kept: List[Tuple[Path, dict, List[Path], dict]] = []
    excluded: List[dict] = []
    for roi_dir in sorted(grouped.keys(), key=str):
        roi_patches = sorted(grouped[roi_dir], key=lambda p: p.patch_id)
        flags = roi_patches[0].flags_dict
        for p in roi_patches[1:]:
            if p.flags_dict != flags:
                raise ValueError(
                    f"Inconsistent network flags within ROI {roi_dir}: "
                    f"{roi_patches[0].patch_id}={flags} vs {p.patch_id}={p.flags_dict}"
                )
        missing_patch_ids = [
            p.patch_id
            for p in roi_patches
            if not (p.patch_dir(data_root) / "coord.npy").is_file()
        ]
        present = [
            p for p in roi_patches if (p.patch_dir(data_root) / "coord.npy").is_file()
        ]
        coverage = {
            "n_subtiles_total": len(roi_patches),
            "n_subtiles_present": len(present),
            "n_subtiles_missing": len(missing_patch_ids),
            "missing_patch_ids": missing_patch_ids,
            "incomplete_local_mirror": bool(missing_patch_ids),
        }
        if not present:
            excluded.append(
                {
                    "roi": roi_dir.name,
                    "department": roi_dir.parent.name.removesuffix("_LIDARHD"),
                    "reason": "no_local_subtiles",
                    **coverage,
                }
            )
            continue
        if missing_patch_ids and drop_incomplete:
            excluded.append(
                {
                    "roi": roi_dir.name,
                    "department": roi_dir.parent.name.removesuffix("_LIDARHD"),
                    "reason": "incomplete_local_mirror",
                    **coverage,
                }
            )
            continue
        patch_dirs = [p.patch_dir(data_root) for p in present]
        kept.append((roi_dir, flags, patch_dirs, coverage))
    return kept, excluded


def _read_patch_grid(patch_dir: Path) -> GridSpec:
    meta_path = patch_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta.json under {patch_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    net = meta.get("network")
    if not isinstance(net, dict):
        raise ValueError(f"meta.json under {patch_dir} has no 'network' grid info")
    return GridSpec(
        origin_x=float(net["origin_x"]),
        origin_y=float(net["origin_y"]),
        width=int(net["width"]),
        height=int(net["height"]),
        pixel_m=float(net["pixel_m"]),
    )


def read_patch_meta(patch_dir: Path) -> dict | None:
    """Raw ``meta.json`` for one patch, or ``None`` if missing/unreadable."""
    meta_path = patch_dir / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def patch_network_is_confirmed_empty(
    patch_dir: Path, network_types: Sequence[str]
) -> bool | None:
    """Whether ``meta.json`` confirms zero GT pixels for every ``network_types`` channel.

    ``rasterize_network.py`` writes no ``network.npy`` for a tile without any positive
    pixel, which is the usual, harmless reason why a ``{patch_id}_logits_network.npy``
    prediction is absent. This lets callers tell that case apart from a real coverage
    gap.

    Returns ``None`` when it can't be determined (missing/malformed meta.json, or a
    requested channel has no recorded pixel count) -- treat that conservatively as
    "not confirmed empty", since it may still be a real gap.
    """
    meta = read_patch_meta(patch_dir)
    if meta is None:
        return None
    net = meta.get("network")
    if not isinstance(net, dict):
        return None
    counts = net.get("positive_pixel_counts")
    if not isinstance(counts, dict):
        return None
    for network_type in network_types:
        count = counts.get(network_type)
        if count is None:
            return None
        if count != 0:
            return False
    return True


def stitch_roi_predictions(
    patch_dirs: Sequence[Path],
    save_path: Path,
    *,
    pixel_m: float = 1.0,
    combine: str = "nanmean",
    allow_missing_predictions: bool = False,
) -> Tuple[np.ndarray, GridSpec]:
    """Place each tile's predicted ``(C,H,W)`` raster (read from ``save_path``) onto one ROI raster.

    Returns ``(roi_probs, roi_grid)``: ``roi_probs`` is ``(C, H_roi, W_roi)`` float32
    with NaN for cells covered by no tile grid (no LiDAR there, so neither GT nor
    prediction exists).
    ``C`` is inferred from the first available ``{patch_id}_logits_network.npy``.
    Overlapping subtile cells are combined per ``combine`` (``"nanmean"`` default;
    ``"max"``/``"first"`` for sensitivity checks).

    By default, **all** ``patch_dirs`` must have a logits file (raises
    ``FileNotFoundError`` listing missing ones). With
    ``allow_missing_predictions=True``, missing logits are skipped (those cells
    stay NaN); still raises if **no** logits file is present.
    """
    if combine not in ("nanmean", "max", "first"):
        raise ValueError(f"combine must be one of nanmean/max/first, got {combine!r}")

    missing_pred_files = [
        patch_dir.name
        for patch_dir in patch_dirs
        if not (save_path / f"{patch_dir.name}_logits_network.npy").is_file()
    ]
    if missing_pred_files and not allow_missing_predictions:
        preview = ", ".join(missing_pred_files[:10])
        more = (
            f", ... and {len(missing_pred_files) - 10} more"
            if len(missing_pred_files) > 10
            else ""
        )
        raise FileNotFoundError(
            f"{len(missing_pred_files)}/{len(patch_dirs)} subtile(s) missing "
            f"{{patch_id}}_logits_network.npy under {save_path}: {preview}{more}"
        )

    kept_patch_dirs: List[Path] = [
        patch_dir
        for patch_dir in patch_dirs
        if (save_path / f"{patch_dir.name}_logits_network.npy").is_file()
    ]
    if not kept_patch_dirs:
        raise FileNotFoundError(
            f"0/{len(patch_dirs)} subtile(s) have {{patch_id}}_logits_network.npy "
            f"under {save_path}"
        )

    grids: List[GridSpec] = []
    for patch_dir in kept_patch_dirs:
        grid = _read_patch_grid(patch_dir)
        if abs(grid.pixel_m - pixel_m) > 1e-9:
            raise ValueError(
                f"pixel_m mismatch for {patch_dir}: expected {pixel_m}, got {grid.pixel_m}"
            )
        grids.append(grid)

    ix0 = min(int(round(g.origin_x / pixel_m)) for g in grids)
    iy0 = min(int(round(g.origin_y / pixel_m)) for g in grids)
    ix1 = max(int(round(g.origin_x / pixel_m)) + g.width for g in grids)
    iy1 = max(int(round(g.origin_y / pixel_m)) + g.height for g in grids)
    roi_grid = GridSpec(
        origin_x=float(ix0 * pixel_m),
        origin_y=float(iy0 * pixel_m),
        width=int(ix1 - ix0),
        height=int(iy1 - iy0),
        pixel_m=float(pixel_m),
    )

    first_pred = np.load(save_path / f"{kept_patch_dirs[0].name}_logits_network.npy")
    if first_pred.ndim != 3:
        raise ValueError(
            f"{kept_patch_dirs[0].name}_logits_network.npy: expected (C,H,W), "
            f"got {first_pred.shape}"
        )
    n_channels = int(first_pred.shape[0])
    if combine == "nanmean":
        sum_ = np.zeros((n_channels, roi_grid.height, roi_grid.width), dtype=np.float64)
        cnt = np.zeros((n_channels, roi_grid.height, roi_grid.width), dtype=np.int32)
    else:
        acc = np.full(
            (n_channels, roi_grid.height, roi_grid.width), np.nan, dtype=np.float64
        )

    for patch_dir, grid in zip(kept_patch_dirs, grids):
        pred_path = save_path / f"{patch_dir.name}_logits_network.npy"
        arr = np.load(pred_path).astype(np.float64, copy=False)
        if arr.shape != (n_channels, grid.height, grid.width):
            raise ValueError(
                f"{pred_path}: shape {arr.shape} != expected "
                f"({n_channels}, {grid.height}, {grid.width})"
            )
        ox = int(round((grid.origin_x - roi_grid.origin_x) / pixel_m))
        oy = int(round((grid.origin_y - roi_grid.origin_y) / pixel_m))
        h, w = grid.height, grid.width

        if combine == "nanmean":
            finite = np.isfinite(arr)
            sum_[:, oy : oy + h, ox : ox + w] += np.nan_to_num(arr, nan=0.0)
            cnt[:, oy : oy + h, ox : ox + w] += finite.astype(np.int32)
        elif combine == "max":
            dst = acc[:, oy : oy + h, ox : ox + w]
            np.fmax(dst, arr, out=dst)
        else:  # "first"
            dst = acc[:, oy : oy + h, ox : ox + w]
            unset = np.isnan(dst)
            dst[unset] = arr[unset]

    if combine == "nanmean":
        roi_probs = np.where(cnt > 0, sum_ / np.maximum(cnt, 1), np.nan)
    else:
        roi_probs = acc

    return roi_probs.astype(np.float32, copy=False), roi_grid
