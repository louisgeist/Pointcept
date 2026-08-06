"""Stitch per-subtile predicted network probability rasters into one ROI-level raster.

The GT network graphs (exported by Flair3D-build) are per-ROI, but Pointcept trains and
predicts per-subtile (see ``rasterize_network.py`` module docstring for the subtile/ROI
distinction). ``MultiTaskTester`` (``pointcept/engines/test.py``) already writes one
``{patch_id}_logits_network.npy`` ``(C, H, W)`` soft-probability raster per subtile, using the
exact same grid as that subtile's ``meta.json["network"]`` (written at preprocessing time by
``rasterize_network.py:process_patch``). All subtile grids snap to the same global absolute
1 m Lambert-93 lattice (``GridSpec`` origins are always integer multiples of ``pixel_m``), so
stitching is pure integer-offset placement -- no reprojection needed.
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
    from pointcept.datasets.preprocessing.flair3d_plus.network_xy_raster_utils import (  # type: ignore
        GridSpec,
    )

# Re-exported for callers that only need manifest/ROI-grouping (kept here so
# tools/eval_network_apls.py has a single import surface for this pipeline).
try:
    from rasterize_network import (  # type: ignore  # noqa: F401
        ManifestPatch,
        group_by_roi,
        load_manifest_patches,
    )
except ImportError:  # pragma: no cover
    from pointcept.datasets.preprocessing.flair3d_plus.rasterize_network import (  # type: ignore  # noqa: F401
        ManifestPatch,
        group_by_roi,
        load_manifest_patches,
    )


def group_by_roi_complete_only(
    patches: Sequence["ManifestPatch"], data_root: Path
) -> Tuple[List[Tuple[Path, dict, List[Path]]], List[dict]]:
    """Group manifest patches by ROI, keeping only ROIs fully mirrored on disk locally.

    A ROI stitched from a *partial* set of subtiles has structural holes exactly where
    the missing subtiles would be -- while the GT graph (derived from vector BDTOPO
    data, independent of local LiDAR mirroring) stays complete for the whole ROI.
    Scoring that partial raster against the full GT graph would silently bias APLS
    downward for reasons that have nothing to do with model quality, purely due to
    which subtiles happen to be mirrored on this machine (see CLAUDE.md's
    Hecate-vs-Jean-Zay data availability note). So a ROI with even one subtile missing
    on disk is dropped ENTIRELY -- never partially included -- and returned separately
    as ``excluded`` so callers can report it plainly instead of it silently
    contaminating the metric. This does NOT reuse ``rasterize_network.py``'s
    ``group_by_roi`` (which hard-fails the whole run on any manifest/disk mismatch --
    correct for preprocessing, where disk state must match the manifest exactly, but
    impractical for an eval run against a partially-mirrored local manifest).
    """
    grouped: Dict[Path, List["ManifestPatch"]] = defaultdict(list)
    for patch in patches:
        grouped[patch.roi_dir(data_root)].append(patch)

    kept: List[Tuple[Path, dict, List[Path]]] = []
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
        if missing_patch_ids:
            excluded.append(
                {
                    "roi": roi_dir.name,
                    "reason": "incomplete_local_mirror",
                    "n_subtiles_total": len(roi_patches),
                    "n_subtiles_missing": len(missing_patch_ids),
                    "missing_patch_ids": missing_patch_ids,
                }
            )
            continue
        patch_dirs = [p.patch_dir(data_root) for p in roi_patches]
        kept.append((roi_dir, flags, patch_dirs))
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


def stitch_roi_predictions(
    patch_dirs: Sequence[Path],
    save_path: Path,
    *,
    pixel_m: float = 1.0,
    combine: str = "nanmean",
) -> Tuple[np.ndarray, GridSpec]:
    """Place each subtile's predicted (3,H,W) raster onto one shared ROI raster.

    Returns ``(roi_probs, roi_grid)``: ``roi_probs`` is ``(C, H_roi, W_roi)`` float32
    with NaN for cells no subtile *observed by LiDAR* (a real, model-independent gap --
    both GT and prediction are blind there, so treating it as unobserved is fair).
    ``C`` is inferred from the first ``{patch_id}_logits_network.npy`` (typically 2 for
    ROADS+RAILROADS training heads, or 3 for legacy three-channel dumps).
    Overlapping subtile cells are combined per ``combine`` (``"nanmean"`` default;
    ``"max"``/``"first"`` for sensitivity checks).

    Requires ALL of ``patch_dirs`` to have a ``{patch_id}_logits_network.npy`` under
    ``save_path`` -- raises ``FileNotFoundError`` listing every missing one otherwise,
    rather than silently stitching a partial ROI (which would bias APLS the same way an
    incomplete local subtile mirror would; see ``group_by_roi_complete_only``). Callers
    should treat this exception as "exclude the whole ROI from evaluation", not retry
    with a subset.
    """
    if combine not in ("nanmean", "max", "first"):
        raise ValueError(f"combine must be one of nanmean/max/first, got {combine!r}")

    missing_pred_files = [
        patch_dir.name
        for patch_dir in patch_dirs
        if not (save_path / f"{patch_dir.name}_logits_network.npy").is_file()
    ]
    if missing_pred_files:
        preview = ", ".join(missing_pred_files[:10])
        more = f", ... and {len(missing_pred_files) - 10} more" if len(missing_pred_files) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing_pred_files)}/{len(patch_dirs)} subtile(s) missing "
            f"{{patch_id}}_logits_network.npy under {save_path}: {preview}{more}"
        )

    grids: List[GridSpec] = []
    kept_patch_dirs: List[Path] = list(patch_dirs)
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
