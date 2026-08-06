"""Mask/segment -> pixel-graph post-processing pipeline (numpy/scipy only).

Mirrors ``Flair3D-build``'s ``scripts/network_graph_pipeline.py`` orchestration
(``build_processed_network_graph``: mask -> optional morphology -> pixel graph ->
endpoint-fix -> RDP -> optional merge) but adds ``build_processed_network_graph_from_mask``,
which starts directly from an already-thresholded boolean mask (e.g. a predicted
probability raster) instead of vector line segments -- skipping the
``centerline_pixel_mask`` step. No geopandas/GDAL dependency here (unlike Flair3D-build's
version, which also handles GeoPackage export); this module only builds/simplifies graphs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Prefer sibling imports so this module runs without the full Pointcept stack,
# matching rasterize_network.py's import contract.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    import network_xy_raster_utils as xy_rast  # type: ignore
except ImportError:  # pragma: no cover
    from pointcept.datasets.preprocessing.flair3d_plus import (  # type: ignore
        network_xy_raster_utils as xy_rast,
    )


@dataclass(frozen=True)
class ProcessedNetworkGraph:
    """Pixel graph after the configured RDP / endpoint-fix / merge pipeline."""

    grid: xy_rast.GridSpec
    line_mask: np.ndarray
    graph_raw: xy_rast.PixelGraph
    graph_rdp_only: xy_rast.PixelGraph
    graph_pre_rdp_fix: xy_rast.PixelGraph | None
    graph_after_endpoint_fix: xy_rast.PixelGraph
    graph_after_radius_fix: xy_rast.PixelGraph | None
    graph_final: xy_rast.PixelGraph
    endpoint_fix_info: dict[str, Any] | None
    merged_info: dict[str, int] | None
    radius_fix_info: dict[str, Any] | None
    n_centerline_pixels_raw: int
    n_centerline_pixels: int


def _build_processed_network_graph_from_line_mask(
    line_mask: np.ndarray,
    grid: xy_rast.GridSpec,
    *,
    connectivity: int,
    morph_enabled: bool,
    morph_operation: str,
    morph_iterations: int,
    morph_connectivity: int,
    rdp_epsilon_m: float,
    endpoint_fix_enabled: bool,
    endpoint_fix_stage: str,
    endpoint_fix_added_edge_weight: float,
    endpoint_fix_include_isolated_nodes: bool,
    merge_enabled: bool,
    merge_weight_threshold: float,
    radius_fix_enabled: bool,
    radius_fix_radius_m: float,
    radius_fix_added_edge_weight: float,
    radius_fix_include_isolated_nodes: bool,
) -> ProcessedNetworkGraph:
    """Shared body: optional morphology -> pixel graph -> endpoint-fix -> RDP -> merge."""
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    morph_operation = str(morph_operation).strip().lower()
    if morph_operation not in ("none", "dilate", "erode"):
        raise ValueError(
            f"morphology.operation must be one of none/dilate/erode, got {morph_operation!r}"
        )
    if morph_connectivity not in (4, 8):
        raise ValueError(
            f"morphology.connectivity must be 4 or 8, got {morph_connectivity}"
        )
    endpoint_fix_stage = str(endpoint_fix_stage).strip().lower()
    if endpoint_fix_stage not in ("pre_rdp", "post_rdp"):
        raise ValueError(
            "endpoint_fix_stage must be one of 'pre_rdp'|'post_rdp', got "
            f"{endpoint_fix_stage!r}"
        )

    n_line_raw = int(np.count_nonzero(line_mask))
    if morph_enabled and morph_operation != "none" and morph_iterations > 0:
        if morph_operation == "dilate":
            line_mask = xy_rast.morph_dilate_mask(
                line_mask,
                iterations=morph_iterations,
                connectivity=morph_connectivity,
            )
        else:
            line_mask = xy_rast.morph_erode_mask(
                line_mask,
                iterations=morph_iterations,
                connectivity=morph_connectivity,
            )
    n_line = int(np.count_nonzero(line_mask))

    graph = xy_rast.build_pixel_graph(line_mask, grid, connectivity=connectivity)
    simplified_rdp_only = xy_rast.simplify_pixel_graph_rdp(
        graph, epsilon_m=rdp_epsilon_m
    )

    endpoint_fix_info: dict[str, Any] | None = None
    graph_pre_rdp_fix: xy_rast.PixelGraph | None = None
    graph_after_endpoint_fix = simplified_rdp_only
    if endpoint_fix_enabled and endpoint_fix_stage == "pre_rdp":
        graph_pre_rdp_fix, endpoint_fix_info = (
            xy_rast.repair_degree1_endpoints_diagonal_opposed(
                graph,
                added_edge_weight=endpoint_fix_added_edge_weight,
                include_isolated_nodes=endpoint_fix_include_isolated_nodes,
            )
        )
        graph_after_endpoint_fix = xy_rast.simplify_pixel_graph_rdp(
            graph_pre_rdp_fix, epsilon_m=rdp_epsilon_m
        )
    elif endpoint_fix_enabled and endpoint_fix_stage == "post_rdp":
        graph_after_endpoint_fix, endpoint_fix_info = (
            xy_rast.repair_degree1_endpoints_diagonal_opposed(
                simplified_rdp_only,
                added_edge_weight=endpoint_fix_added_edge_weight,
                include_isolated_nodes=endpoint_fix_include_isolated_nodes,
            )
        )

    graph_final = graph_after_endpoint_fix
    merged_info: dict[str, int] | None = None
    if merge_enabled:
        simplified_weights = (
            np.asarray(graph_after_endpoint_fix.edge_weights, dtype=np.float64)
            if graph_after_endpoint_fix.edge_weights is not None
            else np.ones(
                (graph_after_endpoint_fix.edges.shape[0],), dtype=np.float64
            )
        )
        edge_keep_mask = simplified_weights > merge_weight_threshold
        n_edges_filtered = int(np.count_nonzero(edge_keep_mask))
        n_comp_before = len(
            xy_rast.connected_components_from_edges(
                int(graph_after_endpoint_fix.node_rc.shape[0]),
                graph_after_endpoint_fix.edges[edge_keep_mask],
            )
        )
        merged = xy_rast.merge_neighbor_nodes(
            graph_after_endpoint_fix,
            weight_threshold=merge_weight_threshold,
        )
        n_comp_after = len(xy_rast.connected_components_nodes(merged))
        graph_final = merged
        merged_info = {
            "n_components_before": int(n_comp_before),
            "n_components_after": int(n_comp_after),
            "n_nodes_before": int(graph_after_endpoint_fix.node_rc.shape[0]),
            "n_edges_before": int(graph_after_endpoint_fix.edges.shape[0]),
            "n_nodes_after": int(merged.node_rc.shape[0]),
            "n_edges_after": int(merged.edges.shape[0]),
            "n_edges_weight_filtered": n_edges_filtered,
        }

    # Radius-based extension of endpoint-fix: runs last (after merge), in real XY
    # meters rather than raw pixel-diagonal adjacency, to bridge larger prediction gaps
    # between dangling ends that survived RDP simplification + merge.
    radius_fix_info: dict[str, Any] | None = None
    graph_after_radius_fix: xy_rast.PixelGraph | None = None
    if radius_fix_enabled:
        graph_after_radius_fix, radius_fix_info = (
            xy_rast.repair_degree1_endpoints_within_radius(
                graph_final,
                radius_m=radius_fix_radius_m,
                added_edge_weight=radius_fix_added_edge_weight,
                include_isolated_nodes=radius_fix_include_isolated_nodes,
            )
        )
        graph_final = graph_after_radius_fix

    return ProcessedNetworkGraph(
        grid=grid,
        line_mask=line_mask,
        graph_raw=graph,
        graph_rdp_only=simplified_rdp_only,
        graph_pre_rdp_fix=graph_pre_rdp_fix,
        graph_after_endpoint_fix=graph_after_endpoint_fix,
        graph_after_radius_fix=graph_after_radius_fix,
        graph_final=graph_final,
        endpoint_fix_info=endpoint_fix_info,
        merged_info=merged_info,
        radius_fix_info=radius_fix_info,
        n_centerline_pixels_raw=n_line_raw,
        n_centerline_pixels=n_line,
    )


def build_processed_network_graph_from_mask(
    mask: np.ndarray,
    grid: xy_rast.GridSpec,
    *,
    connectivity: int = 4,
    morph_enabled: bool = False,
    morph_operation: str = "none",
    morph_iterations: int = 1,
    morph_connectivity: int = 4,
    rdp_epsilon_m: float = 2.0,
    endpoint_fix_enabled: bool = True,
    endpoint_fix_stage: str = "pre_rdp",
    endpoint_fix_added_edge_weight: float = 1.0,
    endpoint_fix_include_isolated_nodes: bool = True,
    merge_enabled: bool = True,
    merge_weight_threshold: float = 2.5,
    radius_fix_enabled: bool = False,
    radius_fix_radius_m: float = 5.0,
    radius_fix_added_edge_weight: float = 1.0,
    radius_fix_include_isolated_nodes: bool = True,
) -> ProcessedNetworkGraph:
    """Build the final pixel graph from an already-thresholded boolean mask.

    Same optional morphology -> pixel graph -> endpoint-fix -> RDP -> merge chain as
    Flair3D-build's ``build_processed_network_graph``, but starting directly from
    ``mask`` instead of deriving it from vector segments via ``centerline_pixel_mask``
    -- used to turn a predicted probability raster (already thresholded by the caller)
    into a graph comparable to the GT graph. Defaults mirror the GT export preset
    ``network=v5`` (``connectivity=4``, ``rdp_epsilon_m=2.0``, endpoint-fix enabled
    pre-RDP incl. isolated nodes, merge enabled post-RDP at ``weight_threshold=2.5``,
    morphology disabled) for an apples-to-apples topological comparison.

    ``radius_fix_*`` (disabled by default -- opt in explicitly, it changes the graph and
    therefore any APLS numbers derived from it): radius-based extension of endpoint-fix,
    applied last (after merge) -- connects every endpoint/isolated node to every other
    endpoint/isolated node within ``radius_fix_radius_m`` straight-line meters, not just
    the diagonal-pixel-adjacent case the earlier endpoint-fix stage handles. See
    ``network_xy_raster_utils.repair_degree1_endpoints_within_radius``.
    """
    line_mask = np.asarray(mask, dtype=bool)
    if line_mask.shape != (grid.height, grid.width):
        raise ValueError(
            f"mask shape {line_mask.shape} != grid (H,W)=({grid.height}, {grid.width})"
        )
    return _build_processed_network_graph_from_line_mask(
        line_mask,
        grid,
        connectivity=connectivity,
        morph_enabled=morph_enabled,
        morph_operation=morph_operation,
        morph_iterations=morph_iterations,
        morph_connectivity=morph_connectivity,
        rdp_epsilon_m=rdp_epsilon_m,
        endpoint_fix_enabled=endpoint_fix_enabled,
        endpoint_fix_stage=endpoint_fix_stage,
        endpoint_fix_added_edge_weight=endpoint_fix_added_edge_weight,
        endpoint_fix_include_isolated_nodes=endpoint_fix_include_isolated_nodes,
        merge_enabled=merge_enabled,
        merge_weight_threshold=merge_weight_threshold,
        radius_fix_enabled=radius_fix_enabled,
        radius_fix_radius_m=radius_fix_radius_m,
        radius_fix_added_edge_weight=radius_fix_added_edge_weight,
        radius_fix_include_isolated_nodes=radius_fix_include_isolated_nodes,
    )


def edge_length_m(graph: xy_rast.PixelGraph) -> np.ndarray:
    """Straight-line XY length (m) per edge -- same formula GT edges' `distance` uses."""
    edges = np.asarray(graph.edges, dtype=np.int64)
    if edges.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    xy = np.asarray(graph.node_xy, dtype=np.float64)
    delta = xy[edges[:, 1]] - xy[edges[:, 0]]
    return np.linalg.norm(delta, axis=1)
