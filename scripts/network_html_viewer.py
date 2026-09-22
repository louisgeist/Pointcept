#!/usr/bin/env python3
"""Interactive HTML viewer for Flair3D network APLS debugging (full ROI only).

Per channel, renders one large main viewport (the aerial image, always visible) with
checkbox-toggleable overlay layers (GT graph, pred graph, GT/pred masks, pred prob,
node path-error, APLS worst-paths) drawn with large, clearly visible nodes so overlays
stay legible even when zoomed out. A secondary strip of small static thumbnails (one per
layer, flattened onto the aerial image) sits below the main viewport for quick reference.
Below that: a length-vs-similarity scatter plot over every node pair APLS actually scores
(not just the worst-K), to see where along the length axis APLS loses points, and a
per-channel timing breakdown (predicted-graph build / APLS diagnostics / worst-path
export) to see which part of the computation is costly.

Reuses the exact same data path as ``scripts/visualize_network_mask.py`` --roi mode
(stitching, predicted-graph pipeline, APLS diagnostics) so panels/scores match 1:1; this
script only changes how the result is rendered.

Example::

python scripts/network_html_viewer.py \
  --roi data/flair3d_plus/test/D075-2021_LIDARHD/UU-S1-4 \
  --result-dir /data/geist/superpixel_transformer_dev/local/temp/network_UU-S1-4 \
  --threshold 0.2 \
  --network-graphs-root /data/geist/Flair3D-build/data/network_graphs \
  --out-dir outputs/html_viewer

Logits trained without TRANSMISSION_LINES (2 channels) are auto-mapped onto
ROADS/RAILROADS when GT still has 3 channels; override with
``--network-types ROADS RAILROADS`` if needed.

Then open ``/tmp/AF-S1-22_viewer/index.html`` in a browser, or send that one file
around -- everything (manifest + every panel PNG) is embedded inline, plain
``file://`` works, no fetch()/server needed.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import visualize_network_mask as vnm  # type: ignore  # also inserts flair3d_plus onto sys.path

import apls_metric as apls  # type: ignore
import network_graph_pipeline as ngp  # type: ignore
import network_xy_raster_utils as xy_rast  # type: ignore

_MASK_ALPHA = 190  # translucent fill alpha (0-255) for mask/prob overlay layers

# Layers rendered in this z-order (bottom -> top) in the main viewport.
_LAYER_ORDER = [
    "pred_prob",
    "gt_mask",
    "pred_binary",
    "gt_graph",
    "pred_graph",
    "node_error",
]
_LAYER_LABELS = {
    "pred_prob": "pred prob (heatmap)",
    "gt_mask": "GT mask",
    "pred_binary": "pred binary",
    "gt_graph": "GT graph",
    "pred_graph": "pred graph",
    "node_error": "node path-error",
}
# Layers checked on by default when the page loads.
_DEFAULT_ON = {"gt_graph", "pred_graph"}

# Which checkbox group each layer is listed under in the layer-controls panel; layers with
# no entry here fall back to _LAYER_GROUP_OVERLAY. Populated for the base layers above at
# import time; per-run pipeline-stage layers (see build_viewer) get added to this at call
# time, one entry per distinct ngp.ProcessedNetworkGraph.mask_stages/graph_stages name.
_LAYER_GROUP_OVERLAY = "overlay layers"
_LAYER_GROUP_MASK_STAGES = "pipeline stages: mask"
_LAYER_GROUP_GRAPH_STAGES = "pipeline stages: graph"
_LAYER_GROUP_DIFF_STAGES = "pipeline stages: removed by this step"
_LAYER_GROUP_DIFF_ADDED_STAGES = "pipeline stages: added by this step"
_LAYER_GROUPS: dict[str, str] = {name: _LAYER_GROUP_OVERLAY for name in _LAYER_ORDER}

# Fixed (not per-channel _FG_COLORS) highlight color for "removed vs previous mask stage"
# diff layers -- deliberately the same across every channel/stage so the eye learns one
# color for "this is what got stripped here", rather than blending into the normal mask fill.
_DIFF_REMOVED_HEX = "#ff2d55"

# Human-readable suffix per raw ngp stage name (shared by mask_stages/graph_stages entries
# that reuse the same conceptual step, e.g. RDP only ever appears in graph_stages).
_STAGE_LABELS = {
    "binarized": "1. binarized",
    "open": "2. open",
    "close": "3. close",
    "small_objects_removed": "4. small objects removed",
    "skeleton": "5. skeleton",
    "pixel_graph_raw": "6. raw pixel graph",
    "endpoint_fix": "7. endpoint-fix (diagonal)",
    "rdp": "8. RDP simplified",
    "merge": "9. node cluster merge",
    "node_linking": "10. node linking (radius-fix)",
    "small_components_removed": "11. small components removed",
    "final": "12. final",
}

# graph_stages entries listed here are skipped entirely (no rasterization, no PNG, no
# checkbox) -- just noted as plain non-interactive text in the layer-controls panel so the
# step isn't forgotten, without paying its render/HTML-size cost or cluttering the toggle
# list with a layer nobody actually wants to inspect.
_SKIP_GRAPH_STAGE_VIZ = {"pixel_graph_raw", "endpoint_fix"}

# Mask stages that can only ever go one direction relative to the previous stage, by
# construction -- their diff in the *other* direction is always an empty layer, so don't
# bother generating/showing it: open(A) subset-of A (can only remove/keep, never add);
# remove_small_objects and skeletonize are both pure thinning (never add); close(A)
# superset-of-or-equal-to A (can only add/keep, never remove).
_SKIP_MASK_DIFF_ADDED = {"open", "small_objects_removed", "skeleton"}
_SKIP_MASK_DIFF_REMOVED = {"close"}

# Fixed colors for specific mask stages (overriding the per-channel _FG_COLORS fill) --
# every mask stage otherwise renders in the same channel color, making it hard to tell
# which stage you're looking at when flipping checkboxes; a couple of stages get a
# distinct, always-the-same color instead so they're recognizable at a glance.
_STAGE_MASK_COLOR_OVERRIDE = {
    "binarized": "#ff3b30",  # red
    "open": "#0a84ff",  # blue
    "close": "#0a84ff",  # blue
}

# Fixed highlight color for "edges added vs previous graph stage" diff layers (e.g. node
# linking only ever adds edges) -- green, paired with the red _DIFF_REMOVED_HEX used for
# mask-pixel removal, so "added" vs "removed" reads consistently across every diff layer.
_DIFF_ADDED_HEX = "#00e676"
_DIFF_ADDED_RGB = np.array([0, 230, 118], dtype=np.uint8)

# Same densify/snap-to-edge config as the official metric (tools/eval_network_apls.py's
# own defaults) -- the header score badge and the pair diagnostics (worst paths, scatter,
# node-error) are computed under this config, not the older unrestricted
# nearest-node / non-densified matching.
_APLS_DENSIFY_M = 50.0
_APLS_SNAP_TO_EDGE_M = 4.0


def _parse_optional_float(s: str) -> float | None:
    """CLI helper: ``none``/``null``/``false`` -> None (disabled), else float meters."""
    if str(s).strip().lower() in ("none", "null", "false", ""):
        return None
    return float(s)


def _parse_optional_int(s: str) -> int | None:
    """CLI helper: ``none``/``null``/``false`` -> None (disabled), else int."""
    if str(s).strip().lower() in ("none", "null", "false", ""):
        return None
    return int(s)


def _parse_bool(s: str) -> bool:
    """CLI helper: truthy/falsy strings -> bool."""
    v = str(s).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {s!r}")


# --------------------------------------------------------------------------------------
# Array -> PNG helpers
# --------------------------------------------------------------------------------------


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (int(c) for c in np.asarray(rgb).reshape(3))
    return f"#{r:02x}{g:02x}{b:02x}"


def _png_data_uri(rgb: np.ndarray) -> str:
    """Encode an RGB(A) array as a PNG and return it as a ``data:`` URI (base64) --
    lets every panel be embedded directly in ``index.html`` instead of a sibling file,
    so the viewer is a single portable file (still works over ``file://``)."""
    buf = io.BytesIO()
    plt.imsave(buf, np.flipud(rgb), format="png")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# --------------------------------------------------------------------------------------
# Transparent-background overlay builders (alpha=0 outside the drawn feature, so they
# stack on top of the aerial base layer in the main viewport). Alpha is set explicitly at
# painted pixels (not inferred from color != black) -- inferring alpha from color would be
# fragile against whichever colormap a given overlay uses (e.g. one that legitimately maps
# a value to near-black).
# --------------------------------------------------------------------------------------


def _binary_overlay_rgba(mask: np.ndarray, fg_hex: str, *, alpha: int = _MASK_ALPHA) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    hit = mask.astype(bool)
    rgba[hit, :3] = np.asarray(_hex_to_rgb(fg_hex), dtype=np.uint8)
    rgba[hit, 3] = alpha
    return rgba


def _prob_overlay_rgba(prob: np.ndarray, vmin: float, vmax: float, *, alpha: int = _MASK_ALPHA) -> np.ndarray:
    cmap = plt.cm.viridis
    finite = np.isfinite(prob)
    rgba = np.zeros((*prob.shape, 4), dtype=np.uint8)
    if np.any(finite):
        norm = np.clip((prob - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
        colored = cmap(norm)
        rgba[finite, :3] = np.clip(np.rint(colored[finite][:, :3] * 255), 0, 255).astype(np.uint8)
        rgba[finite, 3] = alpha
    return rgba


def _paint_nodes_rgba(rgba: np.ndarray, node_xy: np.ndarray, grid, colors: np.ndarray, *, radius_px: int) -> None:
    ix, iy = xy_rast.xy_to_indices(node_xy, grid)
    h, w = rgba.shape[:2]
    r = int(max(0, radius_px))
    for k in range(node_xy.shape[0]):
        x, y = int(ix[k]), int(iy[k])
        if not (0 <= x < w and 0 <= y < h):
            continue
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        rgba[y0:y1, x0:x1, :3] = np.asarray(colors[k], dtype=np.uint8).reshape(3)
        rgba[y0:y1, x0:x1, 3] = 255


def _node_error_overlay_rgba(grid, gt_aps, diag, *, radius_px: int) -> np.ndarray:
    rgba = np.zeros((grid.height, grid.width, 4), dtype=np.uint8)
    node_rgb = vnm._error_to_rgb(diag.node_mean_error)
    unused = ~np.isfinite(diag.node_mean_error)
    node_rgb = node_rgb.copy()
    node_rgb[unused] = vnm._GT_NODE_RGB
    _paint_nodes_rgba(rgba, gt_aps.node_xy, grid, node_rgb, radius_px=radius_px)
    return rgba


def _graphs_equal(a, b) -> bool:
    """True iff two PixelGraph snapshots are pixel-for-pixel identical (same node set, same
    edges) -- used to detect a pipeline stage that changed nothing vs. the previous one
    (e.g. "final" == "node linking" whenever node linking is the last stage that ran)."""
    return (
        a.node_rc.shape == b.node_rc.shape
        and np.array_equal(a.node_rc, b.node_rc)
        and a.edges.shape == b.edges.shape
        and np.array_equal(a.edges, b.edges)
    )


def _added_edges_pixel_graph(prev_graph, cur_graph, grid) -> "xy_rast.PixelGraph":
    """Compact PixelGraph containing only the edges present in ``cur_graph`` but not in
    ``prev_graph`` (plus their endpoint nodes) -- valid only when both graphs share the same
    node indexing (true for e.g. node-linking/endpoint-fix, which only append edges; NOT
    true across RDP/merge, which renumber nodes -- caller must check via _graphs_equal-style
    node_rc comparison before calling this for a meaningful result)."""
    prev_edges = {(int(u), int(v)) for u, v in np.asarray(prev_graph.edges).tolist()}
    added = [
        (int(u), int(v))
        for u, v in np.asarray(cur_graph.edges).tolist()
        if (int(u), int(v)) not in prev_edges
    ]
    if not added:
        return xy_rast.PixelGraph(
            node_rc=np.zeros((0, 2), dtype=np.int64),
            node_xy=np.zeros((0, 2), dtype=np.float64),
            edges=np.zeros((0, 2), dtype=np.int64),
            grid=grid,
        )
    node_ids = sorted({i for pair in added for i in pair})
    remap = {old: new for new, old in enumerate(node_ids)}
    edges_compact = np.array([[remap[u], remap[v]] for u, v in added], dtype=np.int64)
    node_xy_compact = cur_graph.node_xy[node_ids]
    return xy_rast.PixelGraph(
        node_rc=np.zeros((len(node_ids), 2), dtype=np.int64),
        node_xy=node_xy_compact,
        edges=edges_compact,
        grid=grid,
    )


def _downsample_for_thumb(rgb: np.ndarray, *, max_dim: int = 240) -> np.ndarray:
    """Shrink an opaque RGB panel before PNG-encoding it as a thumbnail -- the strip
    displays these at ~150 CSS px, so embedding them at full native (often 1024+ px)
    resolution just bloats the self-contained HTML for no visual benefit."""
    h, w = rgb.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return rgb
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return np.asarray(Image.fromarray(rgb, mode="RGB").resize((new_w, new_h), Image.BILINEAR))


def _flatten_over_base(base_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """Alpha-composite a transparent overlay onto an opaque base -- used only to build the
    static secondary thumbnails (the main viewport composites layers live via CSS stacking)."""
    a = overlay_rgba[..., 3:4].astype(np.float32) / 255.0
    out = base_rgb.astype(np.float32) * (1.0 - a) + overlay_rgba[..., :3].astype(np.float32) * a
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# World XY <-> image-pixel coordinates (must mirror the flipud() used when saving PNGs)
# --------------------------------------------------------------------------------------


def _world_to_img_xy(xy: np.ndarray, grid) -> list[list[float]]:
    ix, iy = xy_rast.xy_to_indices(np.asarray(xy, dtype=np.float64).reshape(-1, 2), grid)
    px = ix.astype(np.float64)
    py = (grid.height - 1 - iy).astype(np.float64)
    return np.stack([px, py], axis=1).tolist()


# --------------------------------------------------------------------------------------
# Main export
# --------------------------------------------------------------------------------------


def build_viewer(
    roi_dir: Path,
    result_dir: Path,
    network_graphs_root: Path,
    out_dir: Path,
    *,
    threshold: float,
    prob_autoscale: bool,
    connectivity: int,
    morph_connectivity: int,
    open_iterations: int,
    close_iterations: int,
    remove_small_objects_enabled: bool,
    remove_small_objects_min_size_px: int,
    rdp_epsilon_m: float,
    endpoint_fix_enabled: bool,
    endpoint_fix_stage: str,
    merge_hop_threshold: float,
    radius_fix_radius_m: float | None,
    min_component_nodes: int,
    apls_min_path_length_m: float | None,
    apls_max_nodes_exact: int | None,
    worst_paths_export: int,
    node_radius_px: int,
    network_types: list[str] | None = None,
) -> Path:
    t_load0 = time.perf_counter()
    patch_dirs = vnm._discover_roi_patch_dirs(roi_dir)
    network, roi_grid, channel_order = vnm.stitch_roi_gt_masks(patch_dirs)
    mean_rgb, count = vnm.stitch_roi_mean_rgb(patch_dirs, roi_grid)
    logits, stitched_grid = vnm.nps.stitch_roi_predictions(patch_dirs, result_dir)
    if (
        stitched_grid.width != roi_grid.width
        or stitched_grid.height != roi_grid.height
        or abs(stitched_grid.origin_x - roi_grid.origin_x) > 1e-6
        or abs(stitched_grid.origin_y - roi_grid.origin_y) > 1e-6
    ):
        raise ValueError(f"Stitched logits grid {stitched_grid} != GT/RGB roi_grid {roi_grid}")

    channel_mapping = vnm.resolve_pred_channel_mapping(
        channel_order, logits.shape[0], network_types=network_types
    )

    vmin, vmax = vnm._prob_display_range(logits, autoscale=prob_autoscale)
    binary = vnm._binarize(logits, threshold)
    base_rgb = vnm._rgb_canvas(mean_rgb, count)
    mean_rgb_panel = _png_data_uri(base_rgb)
    load_stitch_s = time.perf_counter() - t_load0

    channels_manifest: dict[str, dict] = {}
    # Extended with one entry per pipeline-stage layer the first time it's seen below (same
    # run config for every channel, so the set of stage names is identical across channels).
    layer_order = list(_LAYER_ORDER)
    layer_labels = dict(_LAYER_LABELS)
    layer_groups = dict(_LAYER_GROUPS)

    for gt_i, logit_i, name in channel_mapping:
        mask = network[gt_i].astype(bool)
        prob = logits[logit_i]
        bin_mask = binary[logit_i].astype(bool)

        loaded_gt = vnm._load_gt_graph(network_graphs_root, roi_dir, name)
        n_gt_edges_avail = int(loaded_gt.edges.shape[0])
        if mask.sum() == 0 and bin_mask.sum() == 0 and n_gt_edges_avail == 0:
            continue  # nothing to show for this channel on this ROI

        t_chan0 = time.perf_counter()

        t0 = time.perf_counter()
        # Call the pipeline directly (not vnm._build_predicted_graph, which only returns
        # .graph_final) so every intermediate stage (mask_stages/graph_stages) is available
        # to render as its own layer below.
        radius_fix_extra: dict = {}
        if radius_fix_radius_m is not None:
            radius_fix_extra["radius_fix_enabled"] = True
            radius_fix_extra["radius_fix_radius_m"] = float(radius_fix_radius_m)
        processed = ngp.build_processed_network_graph_from_mask(
            bin_mask,
            roi_grid,
            connectivity=connectivity,
            morph_connectivity=morph_connectivity,
            open_iterations=open_iterations,
            close_iterations=close_iterations,
            remove_small_objects_enabled=remove_small_objects_enabled,
            remove_small_objects_min_size_px=remove_small_objects_min_size_px,
            rdp_epsilon_m=rdp_epsilon_m,
            endpoint_fix_enabled=endpoint_fix_enabled,
            endpoint_fix_stage=endpoint_fix_stage,
            merge_enabled=True,
            merge_hop_threshold=merge_hop_threshold,
            min_component_nodes=min_component_nodes,
            **radius_fix_extra,
        )
        pred_graph = processed.graph_final
        pred_aps = apls.apls_graph_from_pixel_graph(pred_graph)
        gt_aps = apls.apls_graph_from_loaded_graph(loaded_gt)
        build_pred_graph_s = time.perf_counter() - t0

        t_panels0 = time.perf_counter()
        overlays: dict[str, np.ndarray] = {
            "gt_mask": _binary_overlay_rgba(mask, vnm._FG_COLORS.get(name, "#ffffff")),
            "pred_binary": _binary_overlay_rgba(bin_mask, vnm._FG_COLORS.get(name, "#ffffff")),
            "pred_prob": _prob_overlay_rgba(prob, vmin, vmax),
        }
        # Per-channel: which graph stages were skipped/redundant for THIS channel's data --
        # e.g. an empty channel makes every stage trivially "identical", which shouldn't
        # pollute another channel's notes, so this must not be shared across channels.
        graph_stage_notes: list[str] = []
        pred_pixel_graph = vnm._as_pixel_graph_for_display(pred_graph.node_xy, pred_graph.edges, roi_grid)
        overlays["pred_graph"] = xy_rast.rasterize_graph_edges(
            pred_pixel_graph,
            grid=roi_grid,
            color_rgb=vnm._PRED_EDGE_RGB,
            node_color_rgb=vnm._PRED_NODE_RGB,
            node_radius_px=node_radius_px,
        )
        gt_pixel_graph = vnm._as_pixel_graph_for_display(gt_aps.node_xy, gt_aps.edges, roi_grid)
        overlays["gt_graph"] = xy_rast.rasterize_graph_edges(
            gt_pixel_graph,
            grid=roi_grid,
            color_rgb=vnm._GT_EDGE_RGB,
            node_color_rgb=vnm._GT_NODE_RGB,
            node_radius_px=node_radius_px,
        )

        # One overlay layer per pipeline stage actually executed (see network_graph_pipeline
        # docstring/CLAUDE.md for the full binarize -> ... -> node-linking chain) -- lets you
        # toggle through exactly what each stage did to this channel's mask/graph.
        prev_stage_name, prev_stage_mask = None, None
        for stage_name, stage_mask in processed.mask_stages:
            lname = f"stage_mask_{stage_name}"
            stage_color = _STAGE_MASK_COLOR_OVERRIDE.get(stage_name, vnm._FG_COLORS.get(name, "#ffffff"))
            overlays[lname] = _binary_overlay_rgba(stage_mask, stage_color)
            if lname not in layer_labels:
                layer_order.append(lname)
                layer_labels[lname] = f"{_STAGE_LABELS.get(stage_name, stage_name)} (mask)"
                layer_groups[lname] = _LAYER_GROUP_MASK_STAGES

            # Diff vs the previous mask stage. Only generated in the direction that stage can
            # actually move a pixel in -- e.g. "open" can only remove/keep (never add), so its
            # "added" diff would always be an empty layer by construction; skip it rather than
            # show a checkbox that can never do anything (see _SKIP_MASK_DIFF_*).
            if prev_stage_mask is not None:
                cur_lbl = _STAGE_LABELS.get(stage_name, stage_name)

                if stage_name not in _SKIP_MASK_DIFF_REMOVED:
                    removed = prev_stage_mask & ~stage_mask
                    dlname = f"stage_diff_removed_{stage_name}"
                    overlays[dlname] = _binary_overlay_rgba(removed, _DIFF_REMOVED_HEX, alpha=230)
                    if dlname not in layer_labels:
                        layer_order.append(dlname)
                        layer_labels[dlname] = f"removed by {cur_lbl}"
                        layer_groups[dlname] = _LAYER_GROUP_DIFF_STAGES

                if stage_name not in _SKIP_MASK_DIFF_ADDED:
                    added = ~prev_stage_mask & stage_mask
                    alname = f"stage_diff_added_{stage_name}"
                    overlays[alname] = _binary_overlay_rgba(added, _DIFF_ADDED_HEX, alpha=230)
                    if alname not in layer_labels:
                        layer_order.append(alname)
                        layer_labels[alname] = f"added by {cur_lbl}"
                        layer_groups[alname] = _LAYER_GROUP_DIFF_ADDED_STAGES
            prev_stage_name, prev_stage_mask = stage_name, stage_mask
        prev_graph_stage_name, prev_graph_stage = None, None
        for stage_name, stage_graph in processed.graph_stages:
            # Auto-skip a stage that changed nothing vs. the previous one (e.g. "final" is
            # literally the same PixelGraph as "node linking" whenever node linking is the
            # last stage that ran) -- still noted, just not rendered as a redundant layer.
            redundant_of = None
            if prev_graph_stage is not None and _graphs_equal(prev_graph_stage, stage_graph):
                redundant_of = _STAGE_LABELS.get(prev_graph_stage_name, prev_graph_stage_name)

            if stage_name in _SKIP_GRAPH_STAGE_VIZ or redundant_of is not None:
                label = _STAGE_LABELS.get(stage_name, stage_name)
                if redundant_of is not None:
                    label = f"{label} (identical to {redundant_of})"
                if label not in graph_stage_notes:
                    graph_stage_notes.append(label)
                prev_graph_stage_name, prev_graph_stage = stage_name, stage_graph
                continue

            lname = f"stage_graph_{stage_name}"
            stage_pixel_graph = vnm._as_pixel_graph_for_display(
                stage_graph.node_xy, stage_graph.edges, roi_grid
            )
            overlays[lname] = xy_rast.rasterize_graph_edges(
                stage_pixel_graph,
                grid=roi_grid,
                color_rgb=vnm._PRED_EDGE_RGB,
                node_color_rgb=vnm._PRED_NODE_RGB,
                node_radius_px=node_radius_px,
            )
            if lname not in layer_labels:
                layer_order.append(lname)
                layer_labels[lname] = f"{_STAGE_LABELS.get(stage_name, stage_name)} (graph)"
                layer_groups[lname] = _LAYER_GROUP_GRAPH_STAGES

            # Added-edges diff vs. the previous graph stage -- only meaningful when both
            # stages share the exact same node set (e.g. node-linking only appends edges);
            # RDP/merge renumber nodes, so a plain edge-set diff wouldn't mean anything there.
            if (
                prev_graph_stage is not None
                and prev_graph_stage.node_rc.shape[0] == stage_graph.node_rc.shape[0]
                and np.array_equal(prev_graph_stage.node_rc, stage_graph.node_rc)
            ):
                added_graph = _added_edges_pixel_graph(prev_graph_stage, stage_graph, roi_grid)
                dlname = f"stage_diff_added_{stage_name}"
                overlays[dlname] = xy_rast.rasterize_graph_edges(
                    added_graph,
                    grid=roi_grid,
                    color_rgb=_DIFF_ADDED_RGB,
                    node_color_rgb=_DIFF_ADDED_RGB,
                    node_radius_px=node_radius_px,
                )
                if dlname not in layer_labels:
                    layer_order.append(dlname)
                    cur_lbl = _STAGE_LABELS.get(stage_name, stage_name)
                    layer_labels[dlname] = f"added by {cur_lbl}"
                    layer_groups[dlname] = _LAYER_GROUP_DIFF_ADDED_STAGES
            prev_graph_stage_name, prev_graph_stage = stage_name, stage_graph

        entry: dict = {
            "apls_score": None,
            "apls_score_gt_to_pred": None,
            "apls_score_pred_to_gt": None,
            "n_nodes_gt": int(gt_aps.node_xy.shape[0]),
            "n_nodes_pred": int(pred_aps.node_xy.shape[0]),
            "mask_hex": vnm._FG_COLORS.get(name, "#ffffff"),
            "layers": {},
            "thumbnails": {},
            "graph_stage_notes": graph_stage_notes,
            "worst_paths": [],
            "pair_scatter": {"L_gt": [], "similarity": []},
            "timings": {},
        }
        for lname, ov in overlays.items():
            entry["layers"][lname] = _png_data_uri(ov)
            entry["thumbnails"][lname] = _png_data_uri(_downsample_for_thumb(_flatten_over_base(base_rgb, ov)))
        panels_s = time.perf_counter() - t_panels0

        # Official tile score: same densify/snap-to-edge/bidirectional-harmonic-mean config
        # as tools/eval_network_apls.py, so this badge matches the real reported metric.
        t_official = time.perf_counter()
        sym = apls.apls_symmetric_score(
            gt_aps, pred_aps, roi=roi_dir.name, network_type=name,
            min_path_length_m=apls_min_path_length_m,
            max_nodes_exact=apls_max_nodes_exact,
        )
        if np.isfinite(sym.score):
            entry["apls_score"] = float(sym.score)
        if np.isfinite(sym.score_gt_to_pred):
            entry["apls_score_gt_to_pred"] = float(sym.score_gt_to_pred)
        if np.isfinite(sym.score_pred_to_gt):
            entry["apls_score_pred_to_gt"] = float(sym.score_pred_to_gt)
        official_score_s = time.perf_counter() - t_official

        # Per-pair diagnostics (worst paths / scatter / node-error) only break down
        # the GT->pred direction, under the same densify/snap-to-edge config as above -- so
        # diag.result.score == sym.score_gt_to_pred, but the bidirectional pred->GT half
        # folded into the badge above isn't visualized pair-by-pair here.
        t1 = time.perf_counter()
        diag = apls.apls_pair_diagnostics(
            gt_aps, pred_aps, roi=roi_dir.name, network_type=name,
            densify=_APLS_DENSIFY_M, snap_to_edge=_APLS_SNAP_TO_EDGE_M,
            min_path_length_m=apls_min_path_length_m,
            max_nodes_exact=apls_max_nodes_exact,
        )
        apls_diagnostics_s = time.perf_counter() - t1
        worst_paths_s = 0.0

        if diag is not None:
            gt_used = diag.gt_used

            t_panels1 = time.perf_counter()
            node_error_overlay = _node_error_overlay_rgba(roi_grid, gt_used, diag, radius_px=node_radius_px)
            entry["layers"]["node_error"] = _png_data_uri(node_error_overlay)
            entry["thumbnails"]["node_error"] = _png_data_uri(
                _downsample_for_thumb(_flatten_over_base(base_rgb, node_error_overlay))
            )
            panels_s += time.perf_counter() - t_panels1

            entry["pair_scatter"] = {
                "L_gt": diag.pair_L_gt.tolist(),
                "similarity": (1.0 - diag.pair_error).tolist(),
            }

            t2 = time.perf_counter()
            if diag.pair_error.shape[0] > 0:
                order = np.argsort(-diag.pair_error)
                k_export = min(int(worst_paths_export), int(order.shape[0]))
                worst_paths = []
                for rank, pi in enumerate(order[:k_export]):
                    u, v = int(diag.pair_u[pi]), int(diag.pair_v[pi])
                    node_path = apls.reconstruct_gt_shortest_path(diag, u, v)
                    if not node_path or len(node_path) < 2:
                        continue
                    pts = _world_to_img_xy(gt_used.node_xy[node_path], roi_grid)
                    worst_paths.append(
                        {
                            "rank": rank,
                            "u": u,
                            "v": v,
                            "pair_error": float(diag.pair_error[pi]),
                            "L_gt_m": float(diag.pair_L_gt[pi]),
                            "L_pred_m": float(diag.pair_L_pred[pi]),
                            "points": pts,
                        }
                    )
                entry["worst_paths"] = worst_paths
            worst_paths_s = time.perf_counter() - t2

        entry["timings"] = {
            "build_pred_graph_s": build_pred_graph_s,
            "panels_s": panels_s,
            "official_score_s": official_score_s,
            "apls_diagnostics_s": apls_diagnostics_s,
            "worst_paths_s": worst_paths_s,
            "total_s": time.perf_counter() - t_chan0,
        }

        channels_manifest[name] = entry

    total_s = time.perf_counter() - t_load0

    manifest = {
        "roi": roi_dir.name,
        "grid": {
            "width": int(roi_grid.width),
            "height": int(roi_grid.height),
            "pixel_m": float(roi_grid.pixel_m),
        },
        "threshold": threshold,
        "mean_rgb_panel": mean_rgb_panel,
        "channels": channels_manifest,
        "layer_order": layer_order,
        "layer_labels": layer_labels,
        "layer_groups": layer_groups,
        "default_on": sorted(_DEFAULT_ON),
        "node_radius_px": node_radius_px,
        "radius_fix_radius_m": radius_fix_radius_m,
        "apls_min_path_length_m": apls_min_path_length_m,
        "run_config": {
            "roi": roi_dir.name,
            "result_dir": str(result_dir),
            "network_graphs_root": str(network_graphs_root),
            "threshold": threshold,
            "prob_autoscale": prob_autoscale,
            "connectivity": connectivity,
            "morph_connectivity": morph_connectivity,
            "open_iterations": open_iterations,
            "close_iterations": close_iterations,
            "remove_small_objects_enabled": remove_small_objects_enabled,
            "remove_small_objects_min_size_px": remove_small_objects_min_size_px,
            "rdp_epsilon_m": rdp_epsilon_m,
            "endpoint_fix_enabled": endpoint_fix_enabled,
            "endpoint_fix_stage": endpoint_fix_stage,
            "merge_hop_threshold": merge_hop_threshold,
            "radius_fix_radius_m": radius_fix_radius_m,
            "min_component_nodes": min_component_nodes,
            "node_radius_px": node_radius_px,
            "worst_paths_export": worst_paths_export,
            "apls_densify_m": _APLS_DENSIFY_M,
            "apls_snap_to_edge_m": _APLS_SNAP_TO_EDGE_M,
            "apls_min_path_length_m": apls_min_path_length_m,
            "apls_max_nodes_exact": apls_max_nodes_exact,
        },
        "timings": {
            "load_stitch_s": load_stitch_s,
            "total_s": total_s,
            "per_channel": {name: e["timings"] for name, e in channels_manifest.items()},
        },
        "legend": {
            "prob_range": [float(vmin), float(vmax)],
            "pred_edge": _rgb_to_hex(vnm._PRED_EDGE_RGB),
            "pred_node": _rgb_to_hex(vnm._PRED_NODE_RGB),
            "gt_edge": _rgb_to_hex(vnm._GT_EDGE_RGB),
            "gt_node": _rgb_to_hex(vnm._GT_NODE_RGB),
            "diff_removed": _DIFF_REMOVED_HEX,
            "diff_added": _DIFF_ADDED_HEX,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "index.html"
    html_path.write_text(_render_html(manifest))
    return html_path


# --------------------------------------------------------------------------------------
# HTML/CSS/JS (self-contained, no CDN, manifest embedded inline -> works over file://)
# --------------------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Network APLS viewer: __ROI__</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #14161a; color: #e8e8ec; }
  header { position: sticky; top: 0; z-index: 20; background: #1c1f26; padding: 10px 16px; border-bottom: 1px solid #333; display: flex; flex-wrap: wrap; align-items: center; gap: 16px; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  .badge { padding: 2px 8px; border-radius: 10px; background: #2a2e38; font-size: 12px; }
  .badge b { color: #fff; }
  .badge.timing { color: #9aa0ac; }
  section.channel { padding: 14px 16px 24px; border-bottom: 1px solid #22252c; }
  section.channel h2 { font-size: 14px; margin: 0 0 4px; color: #ffd479; }
  .timing-line { font-size: 11px; color: #6a6f7a; margin: 0 0 10px; }
  .main-row { display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-start; }
  .layer-controls { display: flex; flex-direction: column; gap: 4px; font-size: 12px; background: #1c1f26; border: 1px solid #2e323c; border-radius: 6px; padding: 8px 10px; min-width: 190px; }
  .layer-controls .group-title { color: #6a6f7a; font-size: 10px; text-transform: uppercase; letter-spacing: .04em; margin: 6px 0 2px; }
  .layer-controls .group-title:first-child { margin-top: 0; }
  .layer-controls label { display: flex; align-items: center; gap: 6px; cursor: pointer; padding: 1px 0; }
  .layer-controls .group-note { color: #565b66; font-size: 11px; font-style: italic; padding: 1px 0 1px 22px; }
  .main-viewport { flex: 1 1 520px; min-width: 320px; }
  .viewport { position: relative; width: 100%; aspect-ratio: var(--ar); overflow: hidden; cursor: grab; background: #000; border-radius: 6px; border: 1px solid #2e323c; }
  .viewport.small { border-radius: 4px; }
  .viewport:active { cursor: grabbing; }
  .stage { position: absolute; top: 0; left: 0; transform-origin: 0 0; }
  .stage img { position: absolute; top: 0; left: 0; image-rendering: pixelated; user-select: none; -webkit-user-drag: none; }
  .stage img.layer-img { display: none; }
  .stage img.layer-img.on { display: block; }
  .stage svg { position: absolute; top: 0; left: 0; pointer-events: none; display: none; }
  .stage svg.on { display: block; }
  .stage svg .hit { pointer-events: stroke; }
  polyline.worst-path { fill: none; }
  #resetView { margin-top: 6px; }
  button { background: #2a2e38; border: 1px solid #444; color: #e8e8ec; border-radius: 5px; padding: 4px 10px; font-size: 12px; cursor: pointer; }
  button:hover { background: #383e4a; }
  .thumb-strip { display: flex; gap: 8px; overflow-x: auto; margin-top: 12px; padding-bottom: 4px; }
  .thumb { flex: none; width: 150px; background: #1c1f26; border: 1px solid #2e323c; border-radius: 5px; overflow: hidden; cursor: pointer; }
  .thumb:hover { border-color: #ffd479; }
  .thumb img { width: 100%; display: block; image-rendering: pixelated; }
  .thumb .tcap { font-size: 10px; padding: 3px 5px; color: #9aa0ac; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .diag-row { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 14px; align-items: flex-start; }
  .scatter-box { background: #1c1f26; border: 1px solid #2e323c; border-radius: 6px; padding: 8px 10px; }
  .scatter-box .cap { font-size: 11px; color: #9aa0ac; margin-bottom: 4px; }
  canvas.scatter { display: block; background: #101216; border-radius: 4px; }
  #legend { background: #191c22; padding: 8px 16px; border-bottom: 1px solid #2e323c; display: flex; flex-wrap: wrap; gap: 8px 22px; align-items: center; font-size: 11px; color: #b8bcc4; }
  #legend .group { display: flex; align-items: center; gap: 6px; }
  #legend .group-label { color: #6a6f7a; margin-right: 2px; }
  #legend .item { display: flex; align-items: center; gap: 4px; white-space: nowrap; }
  #legend .swatch { display: inline-block; flex: none; }
  #legend .swatch.box { width: 12px; height: 12px; border-radius: 3px; }
  #legend .swatch.line { width: 18px; height: 3px; border-radius: 2px; }
  #legend .swatch.dot { width: 9px; height: 9px; border-radius: 50%; }
  #legend .swatch.grad { width: 60px; height: 9px; border-radius: 2px; }
  #runConfig { background: #14161a; padding: 6px 16px 8px; border-bottom: 1px solid #2e323c; display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: baseline; font-size: 11px; color: #9aa0ac; }
  #runConfig .cfg-title { color: #6a6f7a; text-transform: uppercase; letter-spacing: .04em; font-size: 10px; margin-right: 4px; }
  #runConfig .cfg-item { white-space: nowrap; }
  #runConfig .cfg-item b { color: #dfe2e8; font-weight: 600; }
  #runConfig .cfg-item.disabled { opacity: 0.55; }
  #tooltip { position: fixed; z-index: 50; background: #14161acc; border: 1px solid #555; border-radius: 5px; padding: 6px 9px; font-size: 11px; pointer-events: none; display: none; max-width: 280px; line-height: 1.5; backdrop-filter: blur(2px); }
  #tooltip b { color: #ffd479; }
  footer { padding: 10px 16px 30px; font-size: 11px; color: #6a6f7a; }
</style>
</head>
<body>
<header>
  <h1>__ROI__ &middot; network APLS viewer</h1>
  __BADGES__
  <span class="badge timing">total build: <b>__TOTAL_S__s</b></span>
</header>
<section id="legend"></section>
<section id="runConfig"></section>
<div id="tooltip"></div>
<div id="root"></div>
<footer>Drag to pan, scroll to zoom on any viewport. Layer checkboxes toggle overlays on the aerial base image; nodes are drawn large (radius __NODE_RADIUS_PX__ px) so they stay visible while zoomed out. APLS is computed with the same config as the official metric (<code>tools/eval_network_apls.py</code>'s own defaults: densify=50m, snap_to_edge=4m, bidirectional harmonic mean) &mdash; the header badge and h2 "APLS=" are that official tile score; "gt→pred"/"pred→gt" are its two unidirectional halves. __MIN_PATH_LENGTH_NOTE__Per-pair diagnostics below (node path-error, worst paths, scatter) only break down the gt→pred half, over the densified GT graph (so there are more GT node dots than the raw GT graph has vertices &mdash; densify inserts a node every ≤50m along long edges). Worst paths are ranked by APLS pair_error (1&nbsp;=&nbsp;full miss); rank 0 is worst, and the slider in each channel's layer-controls panel picks which single rank is drawn. Scatter plot: one point per unordered GT-connected node pair actually scored (gt→pred direction) that passes the short-path filter. __RADIUS_FIX_NOTE__</footer>
<script>
const MANIFEST = __MANIFEST_JSON__;
const GRID = MANIFEST.grid;

function el(tag, attrs, children) {
  const e = document.createElement(tag);
  for (const k in (attrs || {})) {
    if (k === "class") e.className = attrs[k];
    else if (k === "text") e.textContent = attrs[k];
    else e.setAttribute(k, attrs[k]);
  }
  (children || []).forEach((c) => e.appendChild(c));
  return e;
}

function svgEl(tag, attrs) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in (attrs || {})) e.setAttribute(k, attrs[k]);
  return e;
}

function errColor(e) {
  const t = Math.max(0, Math.min(1, e));
  const r = Math.round(120 + 135 * t);
  const g = Math.round(40 * (1 - t));
  const b = Math.round(40 * (1 - t));
  return `rgb(${r},${g},${b})`;
}

function buildRunConfig() {
  const sec = document.getElementById("runConfig");
  const c = MANIFEST.run_config;

  function item(label, value, opts) {
    opts = opts || {};
    const disabled = value === null || value === undefined;
    const valTxt = disabled ? (opts.offLabel || "off") : (opts.fmt ? opts.fmt(value) : String(value));
    return el("span", { class: "cfg-item" + (disabled ? " disabled" : "") }, [
      el("span", { text: label + ": " }),
      el("b", { text: valTxt }),
    ]);
  }

  sec.appendChild(el("span", { class: "cfg-title", text: "run config:" }));
  sec.appendChild(item("ROI", c.roi));
  sec.appendChild(item("threshold", c.threshold));
  sec.appendChild(item("connectivity", c.connectivity));
  sec.appendChild(item("morph_connectivity", c.morph_connectivity));
  sec.appendChild(item("open_iterations", c.open_iterations));
  sec.appendChild(item("close_iterations", c.close_iterations));
  sec.appendChild(item("remove_small_objects_enabled", c.remove_small_objects_enabled));
  sec.appendChild(item("remove_small_objects_min_size_px", c.remove_small_objects_min_size_px));
  sec.appendChild(item("rdp_epsilon_m", c.rdp_epsilon_m));
  sec.appendChild(item("endpoint_fix_enabled", c.endpoint_fix_enabled));
  sec.appendChild(item("endpoint_fix_stage", c.endpoint_fix_stage));
  sec.appendChild(item("merge_hop_threshold", c.merge_hop_threshold));
  sec.appendChild(item("radius_fix_radius_m", c.radius_fix_radius_m, { offLabel: "disabled" }));
  sec.appendChild(item("min_component_nodes", c.min_component_nodes));
  sec.appendChild(item("node_radius_px", c.node_radius_px));
  sec.appendChild(item("worst_paths_export", c.worst_paths_export));
  sec.appendChild(item("apls_densify_m", c.apls_densify_m, { offLabel: "disabled" }));
  sec.appendChild(item("apls_snap_to_edge_m", c.apls_snap_to_edge_m, { offLabel: "disabled" }));
  sec.appendChild(item("apls_min_path_length_m", c.apls_min_path_length_m, { offLabel: "disabled" }));
  sec.appendChild(item("apls_max_nodes_exact", c.apls_max_nodes_exact));
  sec.appendChild(item("prob_autoscale", c.prob_autoscale));
  sec.appendChild(item("result_dir", c.result_dir));
  sec.appendChild(item("network_graphs_root", c.network_graphs_root));
}

function buildLegend() {
  const sec = document.getElementById("legend");
  const L = MANIFEST.legend;

  function item(swatchClass, style, label) {
    return el("span", { class: "item" }, [
      el("span", { class: `swatch ${swatchClass}`, style }),
      el("span", { text: label }),
    ]);
  }
  function group(label, items) {
    return el("span", { class: "group" }, [el("span", { class: "group-label", text: label }), ...items]);
  }

  const maskItems = Object.entries(MANIFEST.channels).map(([name, chan]) =>
    item("box", `background:${chan.mask_hex}`, name)
  );
  sec.appendChild(group("mask / binary:", maskItems));

  sec.appendChild(
    group("pred prob:", [
      item("grad", "background:linear-gradient(90deg,#440154,#414487,#2a788e,#22a884,#7ad151,#fde725)",
        `${L.prob_range[0].toFixed(2)} → ${L.prob_range[1].toFixed(2)} (viridis, translucent)`),
    ])
  );

  sec.appendChild(
    group("graph:", [
      item("line", `background:${L.pred_edge}`, "pred edge"),
      item("dot", `background:${L.pred_node}`, "pred node"),
      item("line", `background:${L.gt_edge}`, "GT edge"),
      item("dot", `background:${L.gt_node}`, "GT node"),
    ])
  );

  sec.appendChild(
    group("node path-error:", [
      item("grad", "background:linear-gradient(90deg,#00A3FF,#ff3b30)",
        "0 (perfect) → 1 (full miss)"),
    ])
  );

  sec.appendChild(
    group("pipeline diff:", [
      item("box", `background:${L.diff_removed}`, "pixels removed by this stage vs. the previous one"),
      item("line", `background:${L.diff_added}`, "edges added by this stage vs. the previous one"),
    ])
  );

  sec.appendChild(
    group("worst paths:", [
      item("grad", `background:linear-gradient(90deg,${errColor(0)},${errColor(1)})`,
        "low → high pair_error (rank 0 = worst)"),
    ])
  );
}

// ---- Main synced-zoom viewport (one per channel) --------------------------------------

function makeMainViewport(channelName, chan) {
  const viewport = el("div", { class: "viewport" });
  viewport.style.setProperty("--ar", `${GRID.width} / ${GRID.height}`);
  const stage = el("div", { class: "stage" });
  stage.style.width = GRID.width + "px";
  stage.style.height = GRID.height + "px";

  const base = el("img", { src: MANIFEST.mean_rgb_panel, width: GRID.width, height: GRID.height });
  stage.appendChild(base);

  const layerImgs = {};
  for (const lname of MANIFEST.layer_order) {
    const src = chan.layers[lname];
    if (!src) continue;
    const img = el("img", {
      class: "layer-img" + (MANIFEST.default_on.includes(lname) ? " on" : ""),
      src, width: GRID.width, height: GRID.height,
    });
    stage.appendChild(img);
    layerImgs[lname] = img;
  }

  const worstSvg = svgEl("svg", { viewBox: `0 0 ${GRID.width} ${GRID.height}`, width: GRID.width, height: GRID.height });
  makeWorstPathOverlay(channelName, chan)(worstSvg);
  stage.appendChild(worstSvg);

  viewport.appendChild(stage);
  attachPanBehavior(viewport);

  return { viewport, stage, layerImgs, worstSvg };
}

// Per-viewport independent pan/zoom (each channel's main view + each thumbnail is small
// and self-contained; syncing across *all* of them isn't needed once layers replace the
// old flat panel grid, so each viewport just tracks and applies its own transform).
function attachPanBehavior(viewport) {
  const stage = viewport.querySelector(".stage");
  const view = { scale: 1, tx: 0, ty: 0 };
  let dragging = false, lastX = 0, lastY = 0, didFit = false;

  function apply() {
    stage.style.transform = `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`;
    const strokePx = 2.0 / view.scale;
    viewport.querySelectorAll("polyline.worst-path").forEach((p) => p.setAttribute("stroke-width", strokePx));
  }
  function fit() {
    const rect = viewport.getBoundingClientRect();
    if (rect.width === 0) return;
    const s = Math.min(rect.width / GRID.width, rect.height / GRID.height);
    view.scale = s;
    view.tx = (rect.width - GRID.width * s) / 2;
    view.ty = (rect.height - GRID.height * s) / 2;
    apply();
    didFit = true;
  }
  viewport.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = viewport.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const factor = Math.exp(-e.deltaY * 0.0015);
    const newScale = Math.min(Math.max(view.scale * factor, 0.02), 400);
    const localX = (cx - view.tx) / view.scale;
    const localY = (cy - view.ty) / view.scale;
    view.tx = cx - localX * newScale;
    view.ty = cy - localY * newScale;
    view.scale = newScale;
    apply();
  }, { passive: false });
  viewport.addEventListener("mousedown", (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY; e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    view.tx += e.clientX - lastX;
    view.ty += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    apply();
  });
  window.addEventListener("mouseup", () => (dragging = false));
  window.addEventListener("resize", () => { if (!didFit) fit(); });
  viewport._resetView = fit;
  setTimeout(fit, 30);
}

const tooltip = document.getElementById("tooltip");
function showTip(html, x, y) {
  tooltip.innerHTML = html;
  tooltip.style.left = (x + 14) + "px";
  tooltip.style.top = (y + 14) + "px";
  tooltip.style.display = "block";
}
function hideTip() { tooltip.style.display = "none"; }

function makeWorstPathOverlay(channelName, chan) {
  return (svg) => {
    chan.worst_paths.forEach((wp) => {
      const pts = wp.points.map((p) => p.join(",")).join(" ");
      const pl = svgEl("polyline", {
        class: "worst-path hit",
        points: pts,
        stroke: errColor(wp.pair_error),
        "stroke-width": 2.0,
        "data-rank": wp.rank,
      });
      // Only rank 0 (the single worst path) is shown by default -- the worst-path slider
      // in buildLayerControls picks exactly one rank to display at a time.
      pl.style.display = wp.rank === 0 ? "" : "none";
      pl.addEventListener("mousemove", (e) => {
        showTip(
          `<b>${channelName}</b> worst-path rank ${wp.rank}<br>` +
          `pair_error = <b>${wp.pair_error.toFixed(3)}</b><br>` +
          `L_gt = ${wp.L_gt_m.toFixed(1)} m &nbsp; L_pred = ${isFinite(wp.L_pred_m) ? wp.L_pred_m.toFixed(1) + " m" : "&infin; (disconnected)"}<br>` +
          `GT nodes u=${wp.u} v=${wp.v}`,
          e.clientX, e.clientY
        );
      });
      pl.addEventListener("mouseleave", hideTip);
      svg.appendChild(pl);
    });
  };
}

// ---- Layer checkboxes / worst-path toggles ---------------------------------------------

function buildLayerControls(channelName, chan, mv) {
  const box = el("div", { class: "layer-controls" });
  // Bucket by group first (rather than only starting a new header when the group differs
  // from the immediately preceding layer) so groups render as clean contiguous sections
  // even when layer_order interleaves them (e.g. mask stage / removed-diff / mask stage).
  const groups = new Map();
  for (const lname of MANIFEST.layer_order) {
    const img = mv.layerImgs[lname];
    if (!img) continue;
    const group = (MANIFEST.layer_groups && MANIFEST.layer_groups[lname]) || "overlay layers";
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push({ lname, img });
  }
  // Per-channel (not global): skipped/redundant graph-stage notes -- always belong to the
  // "pipeline stages: graph" group, since that's the only group with skippable stages.
  const GRAPH_STAGES_GROUP = "pipeline stages: graph";
  const graphStageNotes = chan.graph_stage_notes || [];
  const appendNotes = (group) => {
    if (group !== GRAPH_STAGES_GROUP) return;
    for (const note of graphStageNotes) {
      box.appendChild(el("div", { class: "group-note", text: `${note} (not visualized)` }));
    }
  };
  for (const [group, items] of groups) {
    box.appendChild(el("div", { class: "group-title", text: group }));
    for (const { lname, img } of items) {
      const checked = img.classList.contains("on");
      const cb = el("input", { type: "checkbox" });
      cb.checked = checked;
      cb.addEventListener("change", () => img.classList.toggle("on", cb.checked));
      box.appendChild(el("label", {}, [cb, el("span", { text: MANIFEST.layer_labels[lname] || lname })]));
      img._checkbox = cb;
    }
    appendNotes(group);
  }
  // Group exists purely as notes for this channel (no visualized layer in it at all --
  // e.g. an empty channel where every graph stage was redundant).
  if (graphStageNotes.length > 0 && !groups.has(GRAPH_STAGES_GROUP)) {
    box.appendChild(el("div", { class: "group-title", text: GRAPH_STAGES_GROUP }));
    appendNotes(GRAPH_STAGES_GROUP);
  }
  box.appendChild(el("div", { class: "group-title", text: "vector overlays" }));
  const cbWorst = el("input", { type: "checkbox" });
  cbWorst.checked = false;
  cbWorst.addEventListener("change", () => mv.worstSvg.classList.toggle("on", cbWorst.checked));
  box.appendChild(el("label", {}, [cbWorst, el("span", { text: "APLS worst paths" })]));

  if (chan.worst_paths.length > 0) {
    // One rank shown at a time (not cumulative top-K) -- easier to read a single worst
    // path than N overlapping polylines; slide through ranks to inspect each in turn.
    const maxRank = chan.worst_paths.length - 1;
    const wrap = el("label", { style: "flex-direction:column; align-items:stretch; gap:2px;" }, [
      el("span", { text: `worst path shown: rank 0 / ${maxRank}` }),
      el("input", { type: "range", min: "0", max: String(maxRank), value: "0" }),
    ]);
    const slider = wrap.querySelector("input");
    const lbl = wrap.querySelector("span");
    slider.addEventListener("input", () => {
      const k = parseInt(slider.value, 10);
      lbl.textContent = `worst path shown: rank ${k} / ${maxRank}`;
      mv.worstSvg.querySelectorAll("polyline.worst-path").forEach((pl) => {
        pl.style.display = parseInt(pl.dataset.rank, 10) === k ? "" : "none";
      });
    });
    box.appendChild(wrap);
  }

  const resetBtn = el("button", { id: "resetView", text: "reset view" });
  resetBtn.addEventListener("click", () => mv.viewport._resetView());
  box.appendChild(resetBtn);

  return box;
}

function buildThumbStrip(channelName, chan, mv) {
  const strip = el("div", { class: "thumb-strip" });
  const addThumb = (label, src, onClick) => {
    const t = el("div", { class: "thumb" }, [
      el("img", { src }),
      el("div", { class: "tcap", text: label }),
    ]);
    if (onClick) t.addEventListener("click", onClick);
    strip.appendChild(t);
  };
  addThumb("aerial (mean RGB)", MANIFEST.mean_rgb_panel);
  for (const lname of MANIFEST.layer_order) {
    const src = chan.thumbnails[lname];
    if (!src) continue;
    const img = mv.layerImgs[lname];
    addThumb(MANIFEST.layer_labels[lname] || lname, src, () => {
      img.classList.add("on");
      if (img._checkbox) img._checkbox.checked = true;
    });
  }
  return strip;
}

// ---- Length-vs-similarity scatter (canvas; every scored pair, not just worst-K) -------

function buildScatter(channelName, chan) {
  const data = chan.pair_scatter;
  const n = data.L_gt.length;
  const wrap = el("div", { class: "scatter-box" });
  wrap.appendChild(el("div", { class: "cap", text: `APLS pair length vs. similarity (N=${n} scored pairs)` }));
  const W = 460, H = 260, pad = { l: 40, r: 10, t: 8, b: 26 };
  const canvas = el("canvas", { class: "scatter", width: String(W), height: String(H) });
  wrap.appendChild(canvas);
  const ctx = canvas.getContext("2d");

  if (n === 0) {
    ctx.fillStyle = "#6a6f7a";
    ctx.font = "12px sans-serif";
    ctx.fillText("no scored pairs", pad.l, H / 2);
    return wrap;
  }

  const maxL = Math.max(...data.L_gt, 1e-6);
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const xOf = (l) => pad.l + (l / maxL) * plotW;
  const yOf = (s) => pad.t + (1 - s) * plotH;

  // axes + gridlines
  ctx.strokeStyle = "#2e323c";
  ctx.fillStyle = "#6a6f7a";
  ctx.font = "10px sans-serif";
  ctx.lineWidth = 1;
  for (let gy = 0; gy <= 1.0001; gy += 0.25) {
    const y = yOf(gy);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillText(gy.toFixed(2), 4, y + 3);
  }
  for (let i = 0; i <= 4; i++) {
    const l = (maxL * i) / 4;
    const x = xOf(l);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
    ctx.fillText(l.toFixed(0) + "m", x - 10, H - pad.b + 14);
  }
  ctx.fillText("similarity", 4, pad.t - 0);
  ctx.fillText("GT shortest-path length", W - pad.r - 90, H - 4);

  const pts = [];
  for (let i = 0; i < n; i++) {
    const l = data.L_gt[i], s = data.similarity[i];
    const x = xOf(l), y = yOf(s);
    pts.push([x, y, l, s]);
    ctx.fillStyle = errColor(1 - s);
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    ctx.arc(x, y, 2.2, 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const my = (e.clientY - rect.top) * (H / rect.height);
    let best = null, bestD = 64; // px^2 radius
    for (const p of pts) {
      const d = (p[0] - mx) * (p[0] - mx) + (p[1] - my) * (p[1] - my);
      if (d < bestD) { bestD = d; best = p; }
    }
    if (best) {
      showTip(
        `<b>${channelName}</b> pair<br>L_gt = ${best[2].toFixed(1)} m<br>similarity = ${best[3].toFixed(3)}`,
        e.clientX, e.clientY
      );
    } else {
      hideTip();
    }
  });
  canvas.addEventListener("mouseleave", hideTip);

  return wrap;
}

function buildTimingLine(chan) {
  const t = chan.timings;
  if (!t || Object.keys(t).length === 0) return null;
  const parts = [
    `build pred graph ${t.build_pred_graph_s.toFixed(3)}s`,
    `panel render/encode ${t.panels_s.toFixed(3)}s`,
    `official score (densify+snap) ${t.official_score_s.toFixed(3)}s`,
    `APLS diagnostics ${t.apls_diagnostics_s.toFixed(3)}s`,
    `worst-path export ${t.worst_paths_s.toFixed(3)}s`,
    `channel total ${t.total_s.toFixed(3)}s`,
  ];
  return el("p", { class: "timing-line", text: "⏱ " + parts.join("  ·  ") });
}

// ---- Page assembly ----------------------------------------------------------------------

buildRunConfig();
buildLegend();

const root = document.getElementById("root");
for (const [name, chan] of Object.entries(MANIFEST.channels)) {
  const mv = makeMainViewport(name, chan);
  const controls = buildLayerControls(name, chan, mv);
  const mainRow = el("div", { class: "main-row" }, [
    controls,
    el("div", { class: "main-viewport" }, [mv.viewport]),
  ]);

  const scoreTxt = chan.apls_score !== null ? chan.apls_score.toFixed(3) : "n/a";
  const gpTxt = chan.apls_score_gt_to_pred !== null ? chan.apls_score_gt_to_pred.toFixed(3) : "n/a";
  const pgTxt = chan.apls_score_pred_to_gt !== null ? chan.apls_score_pred_to_gt.toFixed(3) : "n/a";
  const h2 = el("h2", {
    text: `${name}   APLS=${scoreTxt} (gt→pred=${gpTxt}, pred→gt=${pgTxt})   nodes gt=${chan.n_nodes_gt} pred=${chan.n_nodes_pred}`,
  });

  const children = [h2];
  const timingLine = buildTimingLine(chan);
  if (timingLine) children.push(timingLine);
  children.push(mainRow, buildThumbStrip(name, chan, mv));
  children.push(el("div", { class: "diag-row" }, [buildScatter(name, chan)]));

  root.appendChild(el("section", { class: "channel" }, children));
}
</script>
</body>
</html>
"""


def _render_html(manifest: dict) -> str:
    badges = []
    for name, chan in manifest["channels"].items():
        score = chan["apls_score"]
        txt = f"{score:.3f}" if score is not None else "n/a"
        badges.append(f'<span class="badge">{name}: <b>{txt}</b></span>')
    html = _HTML_TEMPLATE
    html = html.replace("__ROI__", manifest["roi"])
    html = html.replace("__BADGES__", "\n  ".join(badges))
    html = html.replace("__TOTAL_S__", f'{manifest["timings"]["total_s"]:.2f}')
    html = html.replace("__NODE_RADIUS_PX__", str(manifest["node_radius_px"]))
    apls_min_path_length_m = manifest["apls_min_path_length_m"]
    min_path_note = (
        f"Pairs whose GT shortest path is under {apls_min_path_length_m:g}m are excluded from scoring/diagnostics "
        "(short-path filter). "
        if apls_min_path_length_m is not None
        else ""
    )
    html = html.replace("__MIN_PATH_LENGTH_NOTE__", min_path_note)
    radius_fix_m = manifest["radius_fix_radius_m"]
    radius_fix_note = (
        f"Predicted graph: endpoints/isolated nodes within {radius_fix_m:g}m of each other were connected "
        "(radius-fix, applied after merge)."
        if radius_fix_m is not None
        else ""
    )
    html = html.replace("__RADIUS_FIX_NOTE__", radius_fix_note)
    html = html.replace("__MANIFEST_JSON__", json.dumps(manifest))
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roi", type=Path, required=True, help="Flair3D+ ROI directory (preprocessed subtiles).")
    parser.add_argument("--result-dir", type=Path, required=True, help="Dir of {patch_id}_logits_network.npy files.")
    parser.add_argument(
        "--network-graphs-root", type=Path, required=True, help="Flair3D-build exported-graphs root."
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Foreground threshold (default: 0.5).")
    parser.add_argument("--prob-autoscale", action="store_true")
    parser.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    parser.add_argument("--morph-connectivity", type=int, default=4, choices=[4, 8])
    parser.add_argument(
        "--open-iterations", type=int, default=1,
        help="Erode-then-dilate pass count before skeletonizing; 0 disables opening "
        "(default: 1). Opening always runs before closing -- see "
        "network_graph_pipeline.build_processed_network_graph_from_mask.",
    )
    parser.add_argument(
        "--close-iterations", type=int, default=5,
        help="Dilate-then-erode pass count before skeletonizing, applied after opening; "
        "0 disables closing (default: 5).",
    )
    parser.add_argument(
        "--remove-small-objects-enabled",
        type=_parse_bool,
        default=True,
        help="Drop small connected components before skeletonize (default: true). "
        "Pass false/0/off to skip.",
    )
    parser.add_argument(
        "--remove-small-objects-min-size-px", type=int, default=8,
        help="Connected-component pixel-count threshold below which noise specks are dropped before skeletonizing (default: 8).",
    )
    parser.add_argument("--rdp-epsilon-m", type=float, default=2.0)
    parser.add_argument(
        "--endpoint-fix-enabled",
        type=_parse_bool,
        default=True,
        help="Run degree-1 diagonal endpoint repair (default: true). Pass false/0/off to skip.",
    )
    parser.add_argument(
        "--endpoint-fix-stage",
        type=str,
        default="pre_rdp",
        choices=["pre_rdp", "post_rdp"],
        help="When to run endpoint-fix relative to RDP (ignored if --endpoint-fix-enabled=false).",
    )
    parser.add_argument("--merge-hop-threshold", type=float, default=2.5)
    parser.add_argument(
        "--radius-fix-radius-m",
        type=_parse_optional_float,
        default=5.0,
        help=(
            "Radius (meters) to connect every predicted-graph endpoint/isolated node to every "
            "other one within that radius, applied after merge (extension of endpoint-fix). "
            "Default: 5.0. Pass none/null to disable."
        ),
    )
    parser.add_argument(
        "--min-component-nodes",
        type=int,
        default=5,
        help=(
            "Drop predicted-graph connected components with fewer than this many nodes, "
            "applied last (after merge/radius-fix). Default: 5. Pass 0 or 1 to disable."
        ),
    )
    parser.add_argument(
        "--apls-min-path-length-m",
        type=_parse_optional_float,
        default=5.0,
        help=(
            "SpaceNet-style short-path filter: GT/pred pairs whose shortest path is under this "
            "many meters are excluded from APLS scoring and diagnostics (default: 5.0). "
            "Pass none/null to disable."
        ),
    )
    parser.add_argument(
        "--apls-max-nodes-exact",
        type=_parse_optional_int,
        default=None,
        help=(
            "Hard cap on exact O(V^2) APLS (checked after densification): raises rather than "
            "silently subsampling if the source graph has more nodes than this (default: "
            "none = no cap). Pass a positive int to enable."
        ),
    )
    parser.add_argument(
        "--worst-paths-export",
        type=int,
        default=150,
        help="Max worst-error GT pairs (per channel) to export as hoverable path overlays (default: 150).",
    )
    parser.add_argument(
        "--node-radius-px",
        type=int,
        default=1,
        help="Node marker radius in raster pixels for graph/node-error overlays -- drawn as a "
        "(2*radius+1)-side square, so the default of 1 renders 3x3px nodes (bigger = more "
        "visible when zoomed out).",
    )
    parser.add_argument(
        "--network-types",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Channel names / order matching logits_network.npy channels. "
            "Default: identity when C matches GT channel_order, else "
            "ROADS RAILROADS when logits have 2 channels (training drop of "
            "TRANSMISSION_LINES). Pass e.g. ROADS RAILROADS explicitly if needed."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: /tmp/<roi>_viewer).")
    args = parser.parse_args()

    roi_dir = args.roi.resolve()
    if not roi_dir.is_dir():
        raise SystemExit(f"ROI directory not found: {roi_dir}")
    result_dir = args.result_dir.resolve()
    if not result_dir.is_dir():
        raise SystemExit(f"--result-dir not found: {result_dir}")
    network_graphs_root = args.network_graphs_root.resolve()
    if not network_graphs_root.is_dir():
        raise SystemExit(f"--network-graphs-root not found: {network_graphs_root}")
    if not (0.0 <= args.threshold <= 1.0):
        raise SystemExit(f"--threshold must be in [0, 1], got {args.threshold}")

    out_dir = args.out_dir.resolve() if args.out_dir is not None else Path("/tmp") / f"{roi_dir.name}_viewer"

    html_path = build_viewer(
        roi_dir,
        result_dir,
        network_graphs_root,
        out_dir,
        threshold=args.threshold,
        prob_autoscale=args.prob_autoscale,
        connectivity=args.connectivity,
        morph_connectivity=args.morph_connectivity,
        open_iterations=args.open_iterations,
        close_iterations=args.close_iterations,
        remove_small_objects_enabled=args.remove_small_objects_enabled,
        remove_small_objects_min_size_px=args.remove_small_objects_min_size_px,
        rdp_epsilon_m=args.rdp_epsilon_m,
        endpoint_fix_enabled=args.endpoint_fix_enabled,
        endpoint_fix_stage=args.endpoint_fix_stage,
        merge_hop_threshold=args.merge_hop_threshold,
        radius_fix_radius_m=args.radius_fix_radius_m,
        min_component_nodes=args.min_component_nodes,
        apls_min_path_length_m=args.apls_min_path_length_m,
        apls_max_nodes_exact=args.apls_max_nodes_exact,
        worst_paths_export=args.worst_paths_export,
        node_radius_px=args.node_radius_px,
        network_types=args.network_types,
    )
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
