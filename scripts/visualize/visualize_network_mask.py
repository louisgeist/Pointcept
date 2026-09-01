#!/usr/bin/env python3
"""Visualize Malibu3D network GT masks, optional pred probs/binary/graph, and mean RGB.

Two mutually exclusive modes:

* ``--tile``: one preprocessed subtile (local QA; predicted graph / APLS are
  subtile-local -- GT graph is clipped to the tile bounds).
* ``--roi``: full ROI (all child subtiles stitched onto one 1 m grid). Predictions
  come from ``--result-dir`` via ``network_prediction_stitch``; predicted graph and
  APLS match the official per-ROI metric in ``tools/eval_network_apls.py``.

Layout without predictions::

    [ ROADS ] [ RAILROADS ] [ TRANSMISSION_LINES ]   # GT
    [-------- mean-pooled RGB (same 1 m grid) --------]

Layout with predictions (add ``--network-graphs-root`` for the extra GT-graph row)::

    [ GT ROADS ] [ GT RAIL ] [ GT TL ]
    [ Prob ROADS ] [ Prob RAIL ] [ Prob TL ]
    [ Bin@thr ROADS ] [ Bin RAIL ] [ Bin TL ]
    [ Pred graph ROADS ] [ Pred graph RAIL ] [ Pred graph TL ]     # yellow/orange
    [ GT graph ROADS ]   [ GT graph RAIL ]   [ GT graph TL ]       # cyan/white
    [-------- mean-pooled RGB --------]

Example (GT only, subtile)::

    python scripts/visualize/visualize_network_mask.py \\
      --tile data/malibu3d_plus/train/D067-2021_LIDARHD/AF-S1-22/D067-2021_AF-S1-22_1-1 \\
      --out /tmp/network_mask_rgb.png

Example (GT + predictions + predicted graph, subtile)::

    python scripts/visualize/visualize_network_mask.py \\
      --tile data/malibu3d_plus/train/D067-2021_LIDARHD/AF-S1-22/D067-2021_AF-S1-22_1-1 \\
      --logits exp/default/result/D067-2021_AF-S1-22_1-1_logits_network.npy \\
      --threshold 0.2 \\
      --prob-autoscale \\
      --out /tmp/network_gt_pred.png

Example (full ROI, stitched logits + official APLS)::

    python scripts/visualize/visualize_network_mask.py \\
      --roi data/malibu3d_plus/train/D067-2021_LIDARHD/AF-S1-22 \\
      --result-dir exp/malibu3d/spunet_network_overfit_ROI/result \\
      --threshold 0.5 \\
      --prob-autoscale \\
      --network-graphs-root data/malibu3d_build/data/network_graphs \\
      --out /tmp/AF-S1-22_network_roi.png

The predicted graph is built from the binarized mask via the same mask -> graph
post-processing pipeline used to evaluate APLS (see ``tools/eval_network_apls.py``):
optional morphology -> pixel graph -> endpoint-fix -> RDP -> merge, defaults
mirroring the GT export preset ``network=v5``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_MALIBU3D_PLUS_DIR = os.path.join(
    REPO_ROOT, "pointcept", "datasets", "preprocessing", "malibu3d_plus"
)
if _MALIBU3D_PLUS_DIR not in sys.path:
    # Sibling import (bypasses pointcept/__init__.py, i.e. no torch import) --
    # network_xy_raster_utils / network_graph_pipeline / network_label_utils are all
    # standalone numpy/scipy(+geopandas for the GT loader) modules by design.
    sys.path.insert(0, _MALIBU3D_PLUS_DIR)

import network_graph_pipeline as ngp  # type: ignore
import network_prediction_stitch as nps  # type: ignore
import network_xy_raster_utils as xy_rast  # type: ignore

_FG_COLORS = {
    "ROADS": "#ff5555",
    "RAILROADS": "#55ff55",
    "TRANSMISSION_LINES": "#5599ff",
}
_LOGITS_SUFFIX = "_logits_network.npy"
_PRED_EDGE_RGB = np.array([255, 220, 40], dtype=np.uint8)  # yellow (matches raster-utils default)
_PRED_NODE_RGB = np.array([255, 140, 0], dtype=np.uint8)  # orange
_GT_EDGE_RGB = np.array([80, 220, 255], dtype=np.uint8)  # cyan
_GT_NODE_RGB = np.array([255, 255, 255], dtype=np.uint8)  # white


def _load_gt_graph(network_graphs_root: Path, roi_dir: Path, network_type: str):
    """Import network_label_utils lazily (needs geopandas; keep it optional)."""
    import network_label_utils as nlu  # type: ignore

    gt_path = nlu.expected_exported_graph_path(network_graphs_root, roi_dir, network_type)
    return nlu.load_roi_exported_network_graph(gt_path)


def _clip_segment_to_bbox(p0, p1, xmin, ymin, xmax, ymax):
    """Liang-Barsky clip of segment p0-p1 against an axis-aligned box.

    Returns ``(clipped_p0, clipped_p1)`` (points along the original line, at the box
    boundary where the segment exits) or ``None`` if the segment never touches the box.
    """
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, p0[0] - xmin),
        (dx, xmax - p0[0]),
        (-dy, p0[1] - ymin),
        (dy, ymax - p0[1]),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    if t0 > t1:
        return None
    return (p0[0] + t0 * dx, p0[1] + t0 * dy), (p0[0] + t1 * dx, p0[1] + t1 * dy)


def _clip_graph_to_grid(
    node_xy: np.ndarray,
    edges: np.ndarray,
    edge_length_m: np.ndarray,
    grid,
    *,
    truncate_partial_edges: bool = False,
):
    """Keep only nodes within this subtile's local grid.

    The GT graph is exported at full-ROI granularity (spans many subtiles), but the
    predicted graph here is subtile-local -- clip the GT graph down to this subtile's
    bounds first so a local diagnostic is comparable (not dominated by the rest of the
    ROI the predicted graph can never see). This is NOT the official per-ROI APLS
    definition (see ``tools/eval_network_apls.py``), only a local sanity-check.

    ``truncate_partial_edges=False`` (default, used for the local APLS score): drop any
    edge with a node outside the grid entirely -- keeps the diagnostic graph simple/exact.
    ``truncate_partial_edges=True`` (used for the GT-graph display panel): a real GT road
    edge often has its far endpoint several tens of meters away in a *neighboring*
    subtile (RDP-simplified nodes are sparse) -- dropping the whole edge then makes a
    road that visibly spans this subtile's full width look like it's "missing" from GT.
    Instead, truncate such edges at the subtile boundary (Liang-Barsky) and add a
    boundary-crossing point as a new node, so the displayed line reaches the tile edge.
    """
    ix, iy = xy_rast.xy_to_indices(node_xy, grid)
    inside = (ix >= 0) & (iy >= 0) & (ix < grid.width) & (iy < grid.height)
    keep_idx = np.flatnonzero(inside)
    remap = np.full(node_xy.shape[0], -1, dtype=np.int64)
    remap[keep_idx] = np.arange(keep_idx.shape[0], dtype=np.int64)

    out_xy: list = [tuple(p) for p in node_xy[keep_idx]]
    out_edges: list = []
    out_lengths: list = []

    def _add_virtual_node(pt) -> int:
        out_xy.append(pt)
        return len(out_xy) - 1

    xmin, ymin = grid.origin_x, grid.origin_y
    xmax = xmin + grid.width * grid.pixel_m
    ymax = ymin + grid.height * grid.pixel_m

    for k in range(edges.shape[0]):
        u, v = int(edges[k, 0]), int(edges[k, 1])
        u_in, v_in = bool(inside[u]), bool(inside[v])
        if u_in and v_in:
            out_edges.append((int(remap[u]), int(remap[v])))
            out_lengths.append(float(edge_length_m[k]))
            continue
        if not truncate_partial_edges:
            continue
        clipped = _clip_segment_to_bbox(node_xy[u], node_xy[v], xmin, ymin, xmax, ymax)
        if clipped is None:
            continue
        cp0, cp1 = clipped
        if u_in:
            a, b = int(remap[u]), _add_virtual_node(cp1)
        elif v_in:
            a, b = int(remap[v]), _add_virtual_node(cp0)
        else:
            a, b = _add_virtual_node(cp0), _add_virtual_node(cp1)
        length = float(np.hypot(out_xy[a][0] - out_xy[b][0], out_xy[a][1] - out_xy[b][1]))
        out_edges.append((min(a, b), max(a, b)))
        out_lengths.append(length)

    new_xy = np.asarray(out_xy, dtype=np.float64) if out_xy else np.empty((0, 2), dtype=np.float64)
    new_edges = np.asarray(out_edges, dtype=np.int64) if out_edges else np.empty((0, 2), dtype=np.int64)
    new_lengths = np.asarray(out_lengths, dtype=np.float64) if out_lengths else np.empty((0,), dtype=np.float64)
    return new_xy, new_edges, new_lengths


def _local_apls(
    pred_graph, gt_xy: np.ndarray, gt_edges: np.ndarray, gt_len: np.ndarray, *, roi_name: str, network_type: str
) -> float | None:
    """Subtile-local APLS diagnostic against an already-clipped GT graph, or None if empty."""
    import apls_metric as apls  # type: ignore

    if gt_xy.shape[0] == 0:
        return None
    gt_aps = apls.ApsGraph(node_xy=gt_xy, edges=gt_edges, edge_length_m=gt_len)
    pred_aps = apls.apls_graph_from_pixel_graph(pred_graph)
    result = apls.apls_pair_score(gt_aps, pred_aps, roi=roi_name, network_type=network_type)
    return None if result.denom == 0 else result.score


def _roi_apls(pred_graph, loaded_gt, *, roi_name: str, network_type: str) -> float | None:
    """Official per-ROI APLS (same adapters as ``tools/eval_network_apls.py``)."""
    import apls_metric as apls  # type: ignore

    if loaded_gt.node_xy.shape[0] == 0:
        return None
    gt_aps = apls.apls_graph_from_loaded_graph(loaded_gt)
    pred_aps = apls.apls_graph_from_pixel_graph(pred_graph)
    result = apls.apls_pair_score(gt_aps, pred_aps, roi=roi_name, network_type=network_type)
    return None if result.denom == 0 else result.score


def _load_network(tile: Path) -> tuple[np.ndarray, dict, list[str]]:
    meta = json.loads((tile / "meta.json").read_text())
    if "network" not in meta:
        raise KeyError(f"No meta.network in {tile / 'meta.json'}")
    net_meta = meta["network"]
    channel_order = list(net_meta["channel_order"])
    h, w = int(net_meta["height"]), int(net_meta["width"])
    net_path = tile / "network.npy"
    if net_path.is_file() and not net_meta.get("empty", False):
        network = np.load(net_path)
    else:
        network = np.zeros((len(channel_order), h, w), dtype=np.uint8)
    if network.shape != (len(channel_order), h, w):
        raise ValueError(
            f"network.npy shape {network.shape} != "
            f"({len(channel_order)}, {h}, {w}) from meta"
        )
    return network, net_meta, channel_order


def _load_logits(path: Path, n_channels: int, h: int, w: int) -> np.ndarray:
    logits = np.load(path)
    if logits.ndim != 3:
        raise ValueError(f"logits must be (C, H, W), got shape {logits.shape}")
    if logits.shape != (n_channels, h, w):
        raise ValueError(
            f"logits shape {logits.shape} != expected ({n_channels}, {h}, {w}) "
            f"from tile meta.network"
        )
    return logits.astype(np.float32, copy=False)


def _patch_id_from_logits(path: Path) -> str | None:
    name = path.name
    if name.endswith(_LOGITS_SUFFIX):
        return name[: -len(_LOGITS_SUFFIX)]
    return None


def _discover_roi_patch_dirs(roi_dir: Path) -> list[Path]:
    """Child dirs under ``roi_dir`` that look like preprocessed subtiles."""
    patch_dirs = sorted(
        p
        for p in roi_dir.iterdir()
        if p.is_dir() and (p / "meta.json").is_file() and (p / "coord.npy").is_file()
    )
    if not patch_dirs:
        raise FileNotFoundError(
            f"No preprocessed subtiles (meta.json + coord.npy) under {roi_dir}"
        )
    return patch_dirs


def _read_patch_grid(patch_dir: Path):
    """GridSpec from a subtile's meta.network (same as network_prediction_stitch)."""
    meta = json.loads((patch_dir / "meta.json").read_text())
    net = meta.get("network")
    if not isinstance(net, dict):
        raise ValueError(f"meta.json under {patch_dir} has no 'network' grid info")
    return xy_rast.GridSpec(
        origin_x=float(net["origin_x"]),
        origin_y=float(net["origin_y"]),
        width=int(net["width"]),
        height=int(net["height"]),
        pixel_m=float(net["pixel_m"]),
    ), list(net["channel_order"]), float(net["pixel_m"])


def _roi_grid_from_patch_grids(grids: list, pixel_m: float):
    ix0 = min(int(round(g.origin_x / pixel_m)) for g in grids)
    iy0 = min(int(round(g.origin_y / pixel_m)) for g in grids)
    ix1 = max(int(round(g.origin_x / pixel_m)) + g.width for g in grids)
    iy1 = max(int(round(g.origin_y / pixel_m)) + g.height for g in grids)
    return xy_rast.GridSpec(
        origin_x=float(ix0 * pixel_m),
        origin_y=float(iy0 * pixel_m),
        width=int(ix1 - ix0),
        height=int(iy1 - iy0),
        pixel_m=float(pixel_m),
    )


def _place_offset(grid, roi_grid) -> tuple[int, int]:
    pixel_m = roi_grid.pixel_m
    ox = int(round((grid.origin_x - roi_grid.origin_x) / pixel_m))
    oy = int(round((grid.origin_y - roi_grid.origin_y) / pixel_m))
    return ox, oy


def stitch_roi_gt_masks(
    patch_dirs: list[Path],
) -> tuple[np.ndarray, object, list[str]]:
    """OR-stitch per-subtile ``network.npy`` onto one ROI raster.

    Returns ``(network, roi_grid, channel_order)`` with ``network`` uint8 ``(C,H,W)``.
    """
    grids = []
    channel_order = None
    pixel_m = None
    for patch_dir in patch_dirs:
        grid, ch, pm = _read_patch_grid(patch_dir)
        if channel_order is None:
            channel_order = ch
            pixel_m = pm
        elif ch != channel_order:
            raise ValueError(
                f"channel_order mismatch in {patch_dir}: {ch} vs {channel_order}"
            )
        elif abs(pm - pixel_m) > 1e-9:
            raise ValueError(f"pixel_m mismatch in {patch_dir}: {pm} vs {pixel_m}")
        grids.append(grid)

    assert channel_order is not None and pixel_m is not None
    roi_grid = _roi_grid_from_patch_grids(grids, pixel_m)
    n_ch = len(channel_order)
    acc = np.zeros((n_ch, roi_grid.height, roi_grid.width), dtype=np.uint8)

    for patch_dir, grid in zip(patch_dirs, grids):
        network, _, _ = _load_network(patch_dir)
        ox, oy = _place_offset(grid, roi_grid)
        h, w = grid.height, grid.width
        dst = acc[:, oy : oy + h, ox : ox + w]
        np.maximum(dst, network.astype(np.uint8), out=dst)

    return acc, roi_grid, channel_order


def stitch_roi_mean_rgb(
    patch_dirs: list[Path],
    roi_grid,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate per-subtile mean-RGB onto the ROI grid (weighted by occupancy).

    Returns ``(mean_rgb, count)`` with shapes ``(H,W,3)`` float64 and ``(H,W)`` int32.
    Avoids loading all LiDAR points of the ROI at once.
    """
    sum_rgb = np.zeros((roi_grid.height, roi_grid.width, 3), dtype=np.float64)
    count = np.zeros((roi_grid.height, roi_grid.width), dtype=np.int32)
    pixel_m = roi_grid.pixel_m

    for patch_dir in patch_dirs:
        grid, _, pm = _read_patch_grid(patch_dir)
        if abs(pm - pixel_m) > 1e-9:
            raise ValueError(f"pixel_m mismatch in {patch_dir}: {pm} vs {pixel_m}")
        coord = np.load(patch_dir / "coord.npy")
        color = np.load(patch_dir / "color.npy")
        transl = np.load(patch_dir / "coord_translation.npy")
        abs_xy = coord[:, :2].astype(np.float64) + transl[:2].astype(np.float64)
        tile_rgb, tile_count = xy_rast.mean_rgb_raster(abs_xy, color, grid)
        ox, oy = _place_offset(grid, roi_grid)
        h, w = grid.height, grid.width
        dst_sum = sum_rgb[oy : oy + h, ox : ox + w]
        dst_cnt = count[oy : oy + h, ox : ox + w]
        has = tile_count > 0
        dst_sum[has] += tile_rgb[has] * tile_count[has, None]
        dst_cnt += tile_count

    mean_rgb = np.zeros_like(sum_rgb)
    occupied = count > 0
    mean_rgb[occupied] = sum_rgb[occupied] / count[occupied, None]
    return mean_rgb, count


def _imshow_binary(ax, mask: np.ndarray, fg_hex: str) -> None:
    cmap = ListedColormap(["#111111", fg_hex])
    ax.imshow(
        np.flipud(mask.astype(np.uint8)),
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )


def _imshow_prob(
    ax,
    prob: np.ndarray,
    *,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
):
    """Show soft probs; NaN (unobserved) rendered black.

    Pass ``vmin=None, vmax=None`` to let matplotlib autoscale on finite values
    for that panel (prefer a shared range via explicit vmin/vmax when comparing
    channels).
    """
    show = np.flipud(prob.astype(np.float32, copy=True))
    masked = np.ma.masked_invalid(show)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#111111")
    return ax.imshow(
        masked, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
    )


def _prob_display_range(
    logits: np.ndarray, *, autoscale: bool
) -> tuple[float | None, float | None]:
    """Return (vmin, vmax) for pred-prob panels.

    Default: fixed [0, 1]. With ``autoscale=True``: shared min/max over all
    finite values across channels (so the colorbar stays comparable).
    """
    if not autoscale:
        return 0.0, 1.0
    finite = logits[np.isfinite(logits)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmin == vmax:
        # Avoid a collapsed colormap when all finite values are equal.
        eps = 1e-6 if vmin == 0.0 else abs(vmin) * 1e-6
        return vmin - eps, vmax + eps
    return vmin, vmax


def _binarize(logits: np.ndarray, threshold: float) -> np.ndarray:
    """Foreground where prob >= threshold; NaN / unobserved -> background."""
    valid = np.isfinite(logits)
    return (valid & (logits >= threshold)).astype(np.uint8)


def _build_predicted_graph(
    binary_mask: np.ndarray,
    grid,
    *,
    connectivity: int,
    rdp_epsilon_m: float,
    endpoint_fix_stage: str,
    merge_weight_threshold: float,
):
    """Predicted graph for one channel, via the same pipeline used for APLS eval."""
    processed = ngp.build_processed_network_graph_from_mask(
        binary_mask.astype(bool),
        grid,
        connectivity=connectivity,
        rdp_epsilon_m=rdp_epsilon_m,
        endpoint_fix_enabled=True,
        endpoint_fix_stage=endpoint_fix_stage,
        merge_enabled=True,
        merge_weight_threshold=merge_weight_threshold,
    )
    return processed.graph_final


def _as_pixel_graph_for_display(node_xy: np.ndarray, edges: np.ndarray, grid):
    """Wrap raw (node_xy, edges) as a PixelGraph for rasterize_graph_edges (node_rc unused there)."""
    return xy_rast.PixelGraph(
        node_rc=np.zeros((node_xy.shape[0], 2), dtype=np.int64),
        node_xy=node_xy,
        edges=edges,
        grid=grid,
    )


def _compose_rgb_with_graphs(mean_rgb: np.ndarray, count: np.ndarray, grid, layers):
    """Mean-RGB background with one or more graph layers composited on top.

    ``layers`` is a list of ``(pixel_graph, edge_rgb, node_rgb)``, drawn in order so
    later layers win on overlapping pixels.
    """
    rgba = np.zeros((grid.height, grid.width, 4), dtype=np.uint8)
    has_pts = count > 0
    if np.any(has_pts):
        rgba[has_pts, :3] = np.clip(np.rint(mean_rgb[has_pts]), 0, 255).astype(np.uint8)
        rgba[has_pts, 3] = 255
    for pixel_graph, edge_rgb, node_rgb in layers:
        overlay = xy_rast.rasterize_graph_edges(
            pixel_graph, grid=grid, color_rgb=edge_rgb, node_color_rgb=node_rgb
        )
        hit = overlay[:, :, 3] == 255
        if np.any(hit):
            rgba[hit] = overlay[hit]
    return rgba


def render_arrays(
    out: Path,
    *,
    title_name: str,
    network: np.ndarray,
    channel_order: list[str],
    grid,
    mean_rgb: np.ndarray,
    count: np.ndarray,
    logits: np.ndarray | None = None,
    logits_label: str | None = None,
    threshold: float = 0.2,
    prob_autoscale: bool = False,
    network_graphs_root: Path | None = None,
    graph_roi_dir: Path | None = None,
    apls_mode: str = "subtile",
    connectivity: int = 4,
    rdp_epsilon_m: float = 2.0,
    endpoint_fix_stage: str = "pre_rdp",
    merge_weight_threshold: float = 2.5,
    dpi: int = 150,
) -> Path:
    """Shared figure layout for subtile and ROI modes.

    ``apls_mode``:
      - ``"subtile"``: clip GT to ``grid``, label APLS as local sanity-check
      - ``"roi"``: full GT graph, label APLS as official per-ROI score
    """
    if apls_mode not in ("subtile", "roi"):
        raise ValueError(f"apls_mode must be 'subtile' or 'roi', got {apls_mode!r}")
    if network_graphs_root is not None and graph_roi_dir is None:
        raise ValueError("graph_roi_dir is required when network_graphs_root is set")

    h, w = grid.height, grid.width
    pixel_m = grid.pixel_m
    binary = _binarize(logits, threshold) if logits is not None else None
    rgb_show = np.flipud(np.clip(np.round(mean_rgb), 0, 255).astype(np.uint8))
    apls_scores: dict[str, float] = {}

    show_gt_graph_row = logits is not None and network_graphs_root is not None
    n_mask_rows = (4 + (1 if show_gt_graph_row else 0)) if logits is not None else 1
    n_rows = n_mask_rows + 1  # + RGB
    fig_h = 4.0 * n_rows
    n_cols = 4 if logits is not None else 3
    width_ratios = [1.0, 1.0, 1.0, 0.06] if logits is not None else [1.0, 1.0, 1.0]
    fig = plt.figure(figsize=(12.5 if logits is not None else 12, fig_h), facecolor="white")
    gs = fig.add_gridspec(
        n_rows, n_cols, height_ratios=[1.0] * n_mask_rows + [1.2], width_ratios=width_ratios
    )

    for i, name in enumerate(channel_order):
        ax = fig.add_subplot(gs[0, i])
        mask = network[i].astype(bool)
        _imshow_binary(ax, mask, _FG_COLORS.get(name, "#ffffff"))
        ax.set_title(f"GT {name}\npositives={int(mask.sum())}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    last_im = None
    if logits is not None:
        vmin, vmax = _prob_display_range(logits, autoscale=prob_autoscale)
        scale_note = (
            f"autoscale [{vmin:.4g}, {vmax:.4g}]"
            if prob_autoscale
            else "[0, 1]"
        )
        for i, name in enumerate(channel_order):
            ax = fig.add_subplot(gs[1, i])
            last_im = _imshow_prob(ax, logits[i], vmin=vmin, vmax=vmax)
            finite = np.isfinite(logits[i])
            ax.set_title(
                f"Pred prob {name}\n"
                f"finite={int(finite.sum())}, "
                f"max={float(np.nanmax(logits[i])) if np.any(finite) else float('nan'):.3f}",
                fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        for i, name in enumerate(channel_order):
            ax = fig.add_subplot(gs[2, i])
            assert binary is not None
            bin_mask = binary[i].astype(bool)
            _imshow_binary(ax, bin_mask, _FG_COLORS.get(name, "#ffffff"))
            ax.set_title(
                f"Pred bin @{threshold:g} {name}\npositives={int(bin_mask.sum())}",
                fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        if last_im is not None:
            cax = fig.add_subplot(gs[1, 3])
            fig.colorbar(last_im, cax=cax, label=f"P(fg) {scale_note}")

        for i, name in enumerate(channel_order):
            ax = fig.add_subplot(gs[3, i])
            assert binary is not None
            pred_graph = _build_predicted_graph(
                binary[i],
                grid,
                connectivity=connectivity,
                rdp_epsilon_m=rdp_epsilon_m,
                endpoint_fix_stage=endpoint_fix_stage,
                merge_weight_threshold=merge_weight_threshold,
            )
            n_gt_nodes = None
            apls_score = None
            if network_graphs_root is not None:
                assert graph_roi_dir is not None
                loaded_gt = _load_gt_graph(network_graphs_root, graph_roi_dir, name)

                if apls_mode == "roi":
                    n_gt_nodes = int(loaded_gt.node_xy.shape[0])
                    apls_score = _roi_apls(
                        pred_graph,
                        loaded_gt,
                        roi_name=graph_roi_dir.name,
                        network_type=name,
                    )
                    gt_xy_disp = loaded_gt.node_xy
                    gt_edges_disp = loaded_gt.edges
                else:
                    gt_xy, gt_edges, gt_len = _clip_graph_to_grid(
                        loaded_gt.node_xy, loaded_gt.edges, loaded_gt.edge_length_m, grid
                    )
                    n_gt_nodes = int(gt_xy.shape[0])
                    apls_score = _local_apls(
                        pred_graph,
                        gt_xy,
                        gt_edges,
                        gt_len,
                        roi_name=graph_roi_dir.name,
                        network_type=name,
                    )
                    gt_xy_disp, gt_edges_disp, _ = _clip_graph_to_grid(
                        loaded_gt.node_xy,
                        loaded_gt.edges,
                        loaded_gt.edge_length_m,
                        grid,
                        truncate_partial_edges=True,
                    )

                if apls_score is not None:
                    apls_scores[name] = apls_score

                ax_gt = fig.add_subplot(gs[4, i])
                if gt_xy_disp.shape[0] > 0:
                    gt_pixel_graph = _as_pixel_graph_for_display(
                        gt_xy_disp, gt_edges_disp, grid
                    )
                    gt_rgba = _compose_rgb_with_graphs(
                        mean_rgb,
                        count,
                        grid,
                        [(gt_pixel_graph, _GT_EDGE_RGB, _GT_NODE_RGB)],
                    )
                    ax_gt.imshow(np.flipud(gt_rgba), interpolation="nearest")
                else:
                    ax_gt.imshow(rgb_show, interpolation="nearest")
                gt_node_note = (
                    f"nodes={n_gt_nodes}"
                    if apls_mode == "roi"
                    else f"nodes (local, strict)={n_gt_nodes}"
                )
                ax_gt.set_title(
                    f"GT graph {name} (cyan/white)\n{gt_node_note}",
                    fontsize=10,
                )
                ax_gt.set_xticks([])
                ax_gt.set_yticks([])
                ax_gt.set_aspect("equal")

            pred_pixel_graph = _as_pixel_graph_for_display(
                pred_graph.node_xy, pred_graph.edges, grid
            )
            pred_rgba = _compose_rgb_with_graphs(
                mean_rgb,
                count,
                grid,
                [(pred_pixel_graph, _PRED_EDGE_RGB, _PRED_NODE_RGB)],
            )
            ax.imshow(np.flipud(pred_rgba), interpolation="nearest")
            n_pred_nodes = int(pred_graph.node_xy.shape[0])
            subtitle = f"pred nodes={n_pred_nodes}"
            if apls_score is not None:
                if apls_mode == "roi":
                    subtitle += f"\nAPLS vs GT={apls_score:.3f}"
                else:
                    subtitle += f"\nAPLS vs GT (subtile-local)={apls_score:.3f}"
            ax.set_title(f"Pred graph {name} (yellow/orange)\n{subtitle}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

    ax_rgb = fig.add_subplot(gs[n_mask_rows, :3])
    ax_rgb.imshow(rgb_show, interpolation="nearest")
    ax_rgb.set_title(
        f"Mean-pooled RGB (pixel_m={pixel_m}, grid={w}x{h}, "
        f"occupied={int((count > 0).sum())}/{h * w})",
        fontsize=12,
    )
    ax_rgb.set_xticks([])
    ax_rgb.set_yticks([])
    ax_rgb.set_aspect("equal")

    title = f"Network mask: {title_name}"
    if logits is not None:
        if logits_label:
            title += f"  |  logits: {logits_label}"
        title += "  |  graph: pred=yellow/orange"
        if network_graphs_root is not None:
            title += ", GT=cyan/white"
        if apls_scores:
            macro = float(np.mean(list(apls_scores.values())))
            per_channel = ", ".join(f"{k}={v:.3f}" for k, v in apls_scores.items())
            if apls_mode == "roi":
                title += f"  |  APLS: macro={macro:.3f} [{per_channel}]"
            else:
                title += (
                    f"  |  APLS (subtile-local, NOT official): "
                    f"macro={macro:.3f} [{per_channel}]"
                )
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1.0, 0.97])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def render(
    tile: Path,
    out: Path,
    *,
    logits_path: Path | None = None,
    threshold: float = 0.2,
    prob_autoscale: bool = False,
    network_graphs_root: Path | None = None,
    connectivity: int = 4,
    rdp_epsilon_m: float = 2.0,
    endpoint_fix_stage: str = "pre_rdp",
    merge_weight_threshold: float = 2.5,
    dpi: int = 150,
) -> Path:
    """Subtile mode: load one tile and render."""
    network, net_meta, channel_order = _load_network(tile)
    h, w = int(net_meta["height"]), int(net_meta["width"])
    pixel_m = float(net_meta["pixel_m"])
    grid = xy_rast.GridSpec(
        origin_x=float(net_meta["origin_x"]),
        origin_y=float(net_meta["origin_y"]),
        width=w,
        height=h,
        pixel_m=pixel_m,
    )

    logits = None
    logits_label = None
    if logits_path is not None:
        logits = _load_logits(logits_path, len(channel_order), h, w)
        logits_label = logits_path.name

    coord = np.load(tile / "coord.npy")
    color = np.load(tile / "color.npy")
    transl = np.load(tile / "coord_translation.npy")
    abs_xy = coord[:, :2].astype(np.float64) + transl[:2].astype(np.float64)
    mean_rgb, count = xy_rast.mean_rgb_raster(abs_xy, color, grid)

    return render_arrays(
        out,
        title_name=tile.name,
        network=network,
        channel_order=channel_order,
        grid=grid,
        mean_rgb=mean_rgb,
        count=count,
        logits=logits,
        logits_label=logits_label,
        threshold=threshold,
        prob_autoscale=prob_autoscale,
        network_graphs_root=network_graphs_root,
        graph_roi_dir=tile.parent,
        apls_mode="subtile",
        connectivity=connectivity,
        rdp_epsilon_m=rdp_epsilon_m,
        endpoint_fix_stage=endpoint_fix_stage,
        merge_weight_threshold=merge_weight_threshold,
        dpi=dpi,
    )


def render_roi(
    roi_dir: Path,
    out: Path,
    *,
    result_dir: Path | None = None,
    threshold: float = 0.2,
    prob_autoscale: bool = False,
    network_graphs_root: Path | None = None,
    connectivity: int = 4,
    rdp_epsilon_m: float = 2.0,
    endpoint_fix_stage: str = "pre_rdp",
    merge_weight_threshold: float = 2.5,
    dpi: int = 150,
) -> Path:
    """ROI mode: stitch all subtiles, optionally stitch logits, render."""
    patch_dirs = _discover_roi_patch_dirs(roi_dir)
    network, roi_grid, channel_order = stitch_roi_gt_masks(patch_dirs)
    mean_rgb, count = stitch_roi_mean_rgb(patch_dirs, roi_grid)

    logits = None
    logits_label = None
    if result_dir is not None:
        logits, stitched_grid = nps.stitch_roi_predictions(patch_dirs, result_dir)
        if (
            stitched_grid.width != roi_grid.width
            or stitched_grid.height != roi_grid.height
            or abs(stitched_grid.origin_x - roi_grid.origin_x) > 1e-6
            or abs(stitched_grid.origin_y - roi_grid.origin_y) > 1e-6
        ):
            raise ValueError(
                f"Stitched logits grid {stitched_grid} != GT/RGB roi_grid {roi_grid}"
            )
        if logits.shape[0] != len(channel_order):
            raise ValueError(
                f"Stitched logits channels {logits.shape[0]} != {len(channel_order)}"
            )
        logits_label = f"{result_dir.name}/ (*{len(patch_dirs)} subtiles)"

    return render_arrays(
        out,
        title_name=roi_dir.name,
        network=network,
        channel_order=channel_order,
        grid=roi_grid,
        mean_rgb=mean_rgb,
        count=count,
        logits=logits,
        logits_label=logits_label,
        threshold=threshold,
        prob_autoscale=prob_autoscale,
        network_graphs_root=network_graphs_root,
        graph_roi_dir=roi_dir,
        apls_mode="roi",
        connectivity=connectivity,
        rdp_epsilon_m=rdp_epsilon_m,
        endpoint_fix_stage=endpoint_fix_stage,
        merge_weight_threshold=merge_weight_threshold,
        dpi=dpi,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot network GT masks (and optional pred probs / binarized / graphs) "
            "above mean-pooled LiDAR RGB. Use --tile for one subtile or --roi for a "
            "full stitched ROI."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--tile",
        type=Path,
        default=None,
        help="Path to a preprocessed Malibu3D+ subtile directory "
        "(must contain meta.json, coord.npy, color.npy, coord_translation.npy).",
    )
    mode.add_argument(
        "--roi",
        type=Path,
        default=None,
        help="Path to a Malibu3D+ ROI directory containing preprocessed subtiles. "
        "GT masks and mean-RGB are stitched; use --result-dir for stitched preds.",
    )
    parser.add_argument(
        "--logits",
        type=Path,
        default=None,
        help="(tile mode) Optional path to `{tile}_logits_network.npy` (C, H, W) soft "
        "probs; NaN = unobserved. Adds prob + binarized + graph rows.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="(roi mode) Directory of `{patch_id}_logits_network.npy` files "
        "(e.g. exp/.../result). Stitched via network_prediction_stitch.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Foreground threshold for binarized pred row (default: 0.2).",
    )
    parser.add_argument(
        "--prob-autoscale",
        action="store_true",
        help="Color-scale pred probs over shared finite min/max across channels "
        "instead of fixed [0, 1] (useful when probs are peaked near 0).",
    )
    parser.add_argument(
        "--network-graphs-root",
        type=Path,
        default=None,
        help="Optional Malibu3D-build exported-graphs root "
        "(e.g. data/malibu3d_build/data/network_graphs). If set, shows the "
        "GT graph row and APLS scores; requires predictions (--logits or "
        "--result-dir). Only usable in the geopandas-enabled `pointcept` env.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=4,
        choices=[4, 8],
        help="Pixel-graph adjacency for the predicted-graph pipeline (default: 4).",
    )
    parser.add_argument(
        "--rdp-epsilon-m",
        type=float,
        default=2.0,
        help="RDP simplification tolerance in meters for the predicted graph (default: 2.0).",
    )
    parser.add_argument(
        "--endpoint-fix-stage",
        type=str,
        default="pre_rdp",
        choices=["pre_rdp", "post_rdp"],
        help="When to run degree-1 endpoint repair in the predicted-graph pipeline.",
    )
    parser.add_argument(
        "--merge-weight-threshold",
        type=float,
        default=2.5,
        help="Neighbor-node merge weight threshold for the predicted graph (default: 2.5).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: /tmp/<name>_network_mask_rgb.png).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if args.tile is not None and args.result_dir is not None:
        raise SystemExit("--result-dir is for --roi mode; use --logits with --tile")
    if args.roi is not None and args.logits is not None:
        raise SystemExit("--logits is for --tile mode; use --result-dir with --roi")

    has_preds = False
    if args.tile is not None:
        has_preds = args.logits is not None
    else:
        has_preds = args.result_dir is not None

    if not (0.0 <= args.threshold <= 1.0):
        raise SystemExit(f"--threshold must be in [0, 1], got {args.threshold}")

    network_graphs_root = None
    if args.network_graphs_root is not None:
        if not has_preds:
            raise SystemExit(
                "--network-graphs-root requires predictions "
                "(--logits with --tile, or --result-dir with --roi)."
            )
        network_graphs_root = args.network_graphs_root.resolve()
        if not network_graphs_root.is_dir():
            raise SystemExit(f"--network-graphs-root not found: {network_graphs_root}")

    shared_kw = dict(
        threshold=args.threshold,
        prob_autoscale=args.prob_autoscale,
        network_graphs_root=network_graphs_root,
        connectivity=args.connectivity,
        rdp_epsilon_m=args.rdp_epsilon_m,
        endpoint_fix_stage=args.endpoint_fix_stage,
        merge_weight_threshold=args.merge_weight_threshold,
        dpi=args.dpi,
    )

    if args.tile is not None:
        tile = args.tile.resolve()
        if not tile.is_dir():
            raise SystemExit(f"Tile directory not found: {tile}")
        for name in ("meta.json", "coord.npy", "color.npy", "coord_translation.npy"):
            if not (tile / name).is_file():
                raise SystemExit(f"Missing {name} in {tile}")

        logits_path = None
        if args.logits is not None:
            logits_path = args.logits.resolve()
            if not logits_path.is_file():
                raise SystemExit(f"Logits file not found: {logits_path}")
            patch_id = _patch_id_from_logits(logits_path)
            if patch_id is not None and patch_id != tile.name:
                print(
                    f"warning: logits patch_id={patch_id!r} != tile name={tile.name!r}",
                    file=sys.stderr,
                )

        out = args.out
        if out is None:
            suffix = "_gt_pred" if logits_path is not None else "_network_mask_rgb"
            out = Path("/tmp") / f"{tile.name}{suffix}.png"

        written = render(tile, out, logits_path=logits_path, **shared_kw)
    else:
        roi_dir = args.roi.resolve()
        if not roi_dir.is_dir():
            raise SystemExit(f"ROI directory not found: {roi_dir}")

        result_dir = None
        if args.result_dir is not None:
            result_dir = args.result_dir.resolve()
            if not result_dir.is_dir():
                raise SystemExit(f"--result-dir not found: {result_dir}")

        out = args.out
        if out is None:
            suffix = "_gt_pred_roi" if result_dir is not None else "_network_mask_rgb_roi"
            out = Path("/tmp") / f"{roi_dir.name}{suffix}.png"

        written = render_roi(roi_dir, out, result_dir=result_dir, **shared_kw)

    print(f"wrote {written}")


if __name__ == "__main__":
    main()
