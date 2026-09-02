"""Mask/segment -> pixel-graph post-processing pipeline (numpy/scipy only).

Mirrors ``Malibu3D-build``'s ``scripts/network_graph_pipeline.py`` orchestration
(``build_processed_network_graph``: mask -> optional morphology -> pixel graph ->
endpoint-fix -> RDP -> optional merge) but adds ``build_processed_network_graph_from_mask``,
which starts directly from an already-thresholded boolean mask (e.g. a predicted
probability raster) instead of vector line segments -- skipping the
``centerline_pixel_mask`` step. No geopandas/GDAL dependency here (unlike Malibu3D-build's
version, which also handles GeoPackage export); this module only builds/simplifies graphs.

The from-mask path also depends on ``scikit-image`` (unlike ``network_xy_raster_utils``,
which stays a dependency-light, copy-into-other-repos module): a predicted probability
raster thresholded into a boolean mask is a blob several pixels wide, not the 1px-wide
centerline ``build_pixel_graph`` assumes, so it's skeletonized (Zhang-Suen thinning) down
to a topological centerline before graph-building -- otherwise every mask pixel becomes a
node in a dense 2D grid graph, and ``merge_neighbor_nodes`` collapses whole blobs into a
single hub node (star-shaped artifacts in the resulting graph).
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from skimage.morphology import remove_small_objects, skeletonize


class StageTimer:
    """Optional wall-clock stage timer for ``build_processed_network_graph_*``.

    When ``enabled`` is False, ``stage(...)`` is a near-no-op. When enabled, each
    entered stage accumulates seconds into ``timings`` (skipped stages are absent,
    not recorded as 0.0). Re-entering the same name adds to the existing total
    (e.g. two RDP passes under ``pre_rdp`` endpoint-fix).
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + (
                time.perf_counter() - t0
            )

# Prefer sibling imports so this module runs without the full Pointcept stack,
# matching rasterize_network.py's import contract.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    import network_xy_raster_utils as xy_rast  # type: ignore
except ImportError:  # pragma: no cover
    from pointcept.datasets.preprocessing.malibu3d import (  # type: ignore
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
    small_components_info: dict[str, int] | None
    n_centerline_pixels_raw: int
    n_centerline_pixels: int
    n_mask_pixels_pre_skeleton: int
    # Ordered (stage_name, array/graph) snapshots, one per pipeline stage actually executed
    # (skipped/disabled stages are omitted, not included as no-op duplicates) -- lets a
    # viewer render "what did stage N do" without recomputing anything itself.
    mask_stages: list[tuple[str, np.ndarray]]
    graph_stages: list[tuple[str, xy_rast.PixelGraph]]
    # Wall-clock seconds per executed stage when ``collect_timings=True``; else None.
    stage_timings: dict[str, float] | None = None


def _build_processed_network_graph_from_line_mask(
    line_mask: np.ndarray,
    grid: xy_rast.GridSpec,
    *,
    connectivity: int,
    morph_connectivity: int,
    open_iterations: int,
    close_iterations: int,
    remove_small_objects_enabled: bool,
    remove_small_objects_min_size_px: int,
    skeletonize_enabled: bool,
    rdp_epsilon_m: float,
    endpoint_fix_enabled: bool,
    endpoint_fix_stage: str,
    endpoint_fix_added_edge_weight: float,
    endpoint_fix_include_isolated_nodes: bool,
    merge_enabled: bool,
    merge_hop_threshold: float,
    radius_fix_enabled: bool,
    radius_fix_radius_m: float,
    radius_fix_added_edge_weight: float,
    radius_fix_include_isolated_nodes: bool,
    min_component_nodes: int,
    collect_timings: bool = False,
) -> ProcessedNetworkGraph:
    """Shared body: optional morphology -> pixel graph -> endpoint-fix -> RDP -> merge."""
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
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

    timer = StageTimer(enabled=collect_timings)
    mask_stages: list[tuple[str, np.ndarray]] = [("binarized", np.array(line_mask, dtype=bool))]

    n_line_raw = int(np.count_nonzero(line_mask))

    def _dilate(m: np.ndarray, iterations: int) -> np.ndarray:
        return xy_rast.morph_dilate_mask(
            m, iterations=iterations, connectivity=morph_connectivity
        )

    def _erode(m: np.ndarray, iterations: int) -> np.ndarray:
        return xy_rast.morph_erode_mask(
            m, iterations=iterations, connectivity=morph_connectivity
        )

    # Morphology is always opening (if open_iterations > 0) followed by closing (if
    # close_iterations > 0) -- there's no separate "operation" selector: 0 iterations
    # disables that step, both can run together (open first), and standalone dilate/erode
    # aren't exposed since they're rarely what you actually want here (they only ever
    # grow or only ever shrink, with no noise/gap-specific behavior).
    if open_iterations > 0:
        # Opening: erode `open_iterations` times then dilate the same number of times --
        # shrinks away small protrusions/noise blobs (thinner than the structuring element
        # after that many passes) while restoring the surviving mask's extent, without
        # silently dropping isolated blobs the way remove_small_objects does by pixel count
        # alone. No border-padding needed here: erosion is always a subset of its input
        # regardless of boundary convention, so open(A) subset-of A holds unconditionally --
        # the analogous artifact for opening would be *under*-recovering real content near a
        # tile edge, which isn't fixable without assuming what's beyond the tile, so it's
        # left as the honest, conservative default.
        with timer.stage("open"):
            line_mask = _dilate(_erode(line_mask, open_iterations), open_iterations)
        mask_stages.append(("open", line_mask.copy()))
    if close_iterations > 0:
        # Closing: dilate `close_iterations` times then erode the same number of times --
        # bridges small gaps/breaks in the mask (common in noisy predictions) without
        # growing the overall extent.
        #
        # Plain dilate-then-erode is NOT a correct closing at the array's edges: the erode
        # pass treats "outside the array" as background, so real mask content within
        # `close_iterations` pixels of a tile edge gets spuriously eaten away even though
        # closing must never remove pixels (A subset-of close(A)). Standard fix (mirrors
        # scipy.ndimage.binary_closing's border_value=1 recommendation): pad the *dilated*
        # result with True before eroding -- simulates "foreground continues past this
        # tile" so the erosion can't shrink real edge content -- then crop back to the
        # original shape. (Padding before the dilate step instead would be wrong: the
        # artificial True border would itself dilate inward and contaminate real
        # background near the edge.)
        with timer.stage("close"):
            dilated = _dilate(line_mask, close_iterations)
            padded = np.pad(dilated, close_iterations, mode="constant", constant_values=True)
            eroded = _erode(padded, close_iterations)
            line_mask = eroded[close_iterations:-close_iterations, close_iterations:-close_iterations]
        mask_stages.append(("close", line_mask.copy()))

    if remove_small_objects_enabled and remove_small_objects_min_size_px > 0:
        with timer.stage("remove_small_objects"):
            line_mask = remove_small_objects(
                line_mask, min_size=remove_small_objects_min_size_px
            )
        mask_stages.append(("small_objects_removed", line_mask.copy()))

    n_mask_pixels_pre_skeleton = int(np.count_nonzero(line_mask))
    if skeletonize_enabled:
        # A thresholded predicted-probability mask is a blob several pixels wide, not the
        # 1px-wide centerline build_pixel_graph assumes -- thin it down first, or every mask
        # pixel becomes a graph node and merge_neighbor_nodes collapses whole blobs into a
        # single hub node (star-shaped artifacts).
        with timer.stage("skeletonize"):
            line_mask = skeletonize(line_mask)
        mask_stages.append(("skeleton", line_mask.copy()))
    n_line = int(np.count_nonzero(line_mask))

    graph_stages: list[tuple[str, xy_rast.PixelGraph]] = []

    with timer.stage("pixel_graph"):
        graph = xy_rast.build_pixel_graph(line_mask, grid, connectivity=connectivity)
    graph_stages.append(("pixel_graph_raw", graph))
    with timer.stage("rdp"):
        simplified_rdp_only = xy_rast.simplify_pixel_graph_rdp(
            graph, epsilon_m=rdp_epsilon_m
        )

    endpoint_fix_info: dict[str, Any] | None = None
    graph_pre_rdp_fix: xy_rast.PixelGraph | None = None
    graph_after_endpoint_fix = simplified_rdp_only
    if endpoint_fix_enabled and endpoint_fix_stage == "pre_rdp":
        with timer.stage("endpoint_fix"):
            graph_pre_rdp_fix, endpoint_fix_info = (
                xy_rast.repair_degree1_endpoints_diagonal_opposed(
                    graph,
                    added_edge_weight=endpoint_fix_added_edge_weight,
                    include_isolated_nodes=endpoint_fix_include_isolated_nodes,
                )
            )
        graph_stages.append(("endpoint_fix", graph_pre_rdp_fix))
        with timer.stage("rdp"):
            graph_after_endpoint_fix = xy_rast.simplify_pixel_graph_rdp(
                graph_pre_rdp_fix, epsilon_m=rdp_epsilon_m
            )
        graph_stages.append(("rdp", graph_after_endpoint_fix))
    elif endpoint_fix_enabled and endpoint_fix_stage == "post_rdp":
        graph_stages.append(("rdp", simplified_rdp_only))
        with timer.stage("endpoint_fix"):
            graph_after_endpoint_fix, endpoint_fix_info = (
                xy_rast.repair_degree1_endpoints_diagonal_opposed(
                    simplified_rdp_only,
                    added_edge_weight=endpoint_fix_added_edge_weight,
                    include_isolated_nodes=endpoint_fix_include_isolated_nodes,
                )
            )
        graph_stages.append(("endpoint_fix", graph_after_endpoint_fix))
    else:
        graph_stages.append(("rdp", simplified_rdp_only))

    graph_final = graph_after_endpoint_fix
    merged_info: dict[str, int] | None = None
    if merge_enabled:
        with timer.stage("merge"):
            simplified_weights = (
                np.asarray(graph_after_endpoint_fix.edge_weights, dtype=np.float64)
                if graph_after_endpoint_fix.edge_weights is not None
                else np.ones(
                    (graph_after_endpoint_fix.edges.shape[0],), dtype=np.float64
                )
            )
            edge_keep_mask = simplified_weights > merge_hop_threshold
            n_edges_filtered = int(np.count_nonzero(edge_keep_mask))
            n_comp_before = len(
                xy_rast.connected_components_from_edges(
                    int(graph_after_endpoint_fix.node_rc.shape[0]),
                    graph_after_endpoint_fix.edges[edge_keep_mask],
                )
            )
            merged = xy_rast.merge_neighbor_nodes(
                graph_after_endpoint_fix,
                hop_threshold=merge_hop_threshold,
            )
            n_comp_after = len(xy_rast.connected_components_nodes(merged))
        graph_final = merged
        graph_stages.append(("merge", merged))
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
        with timer.stage("radius_fix"):
            graph_after_radius_fix, radius_fix_info = (
                xy_rast.repair_degree1_endpoints_within_radius(
                    graph_final,
                    radius_m=radius_fix_radius_m,
                    added_edge_weight=radius_fix_added_edge_weight,
                    include_isolated_nodes=radius_fix_include_isolated_nodes,
                )
            )
        graph_final = graph_after_radius_fix
        graph_stages.append(("node_linking", graph_after_radius_fix))

    # Final cleanup step: drop whole connected components smaller than
    # `min_component_nodes` -- runs last so it catches small fragments regardless of
    # which earlier stages produced them (isolated skeleton specks, leftover merge
    # remnants, unlinked radius-fix endpoints, ...).
    small_components_info: dict[str, int] | None = None
    if min_component_nodes > 1:
        with timer.stage("min_component_nodes"):
            n_nodes_before_prune = int(graph_final.node_rc.shape[0])
            n_comp_before_prune = len(xy_rast.connected_components_nodes(graph_final))
            graph_final = xy_rast.drop_small_components(
                graph_final, min_nodes=min_component_nodes
            )
        graph_stages.append(("small_components_removed", graph_final))
        small_components_info = {
            "min_component_nodes": int(min_component_nodes),
            "n_components_before": n_comp_before_prune,
            "n_components_after": len(xy_rast.connected_components_nodes(graph_final)),
            "n_nodes_before": n_nodes_before_prune,
            "n_nodes_after": int(graph_final.node_rc.shape[0]),
        }

    graph_stages.append(("final", graph_final))

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
        small_components_info=small_components_info,
        n_centerline_pixels_raw=n_line_raw,
        n_centerline_pixels=n_line,
        n_mask_pixels_pre_skeleton=n_mask_pixels_pre_skeleton,
        mask_stages=mask_stages,
        graph_stages=graph_stages,
        stage_timings=dict(timer.timings) if collect_timings else None,
    )


def build_processed_network_graph_from_mask(
    mask: np.ndarray,
    grid: xy_rast.GridSpec,
    *,
    connectivity: int = 4,
    morph_connectivity: int = 4,
    open_iterations: int = 1,
    close_iterations: int = 5,
    remove_small_objects_enabled: bool = True,
    remove_small_objects_min_size_px: int = 8,
    skeletonize_enabled: bool = True,
    rdp_epsilon_m: float = 2.0,
    endpoint_fix_enabled: bool = True,
    endpoint_fix_stage: str = "pre_rdp",
    endpoint_fix_added_edge_weight: float = 1.0,
    endpoint_fix_include_isolated_nodes: bool = True,
    merge_enabled: bool = True,
    merge_hop_threshold: float = 2.5,
    radius_fix_enabled: bool = False,
    radius_fix_radius_m: float = 5.0,
    radius_fix_added_edge_weight: float = 1.0,
    radius_fix_include_isolated_nodes: bool = True,
    min_component_nodes: int = 5,
    collect_timings: bool = False,
) -> ProcessedNetworkGraph:
    """Build the final pixel graph from an already-thresholded boolean mask.

    Same optional morphology -> pixel graph -> endpoint-fix -> RDP -> merge chain as
    Malibu3D-build's ``build_processed_network_graph``, but starting directly from
    ``mask`` instead of deriving it from vector segments via ``centerline_pixel_mask``
    -- used to turn a predicted probability raster (already thresholded by the caller)
    into a graph comparable to the GT graph. Defaults mirror the GT export preset
    ``network=v5`` (``connectivity=4``, ``rdp_epsilon_m=2.0``, endpoint-fix enabled
    pre-RDP incl. isolated nodes, merge enabled post-RDP at ``merge_hop_threshold=2.5``)
    for an apples-to-apples topological comparison, except morphology, which -- unlike
    the GT export path -- is on by default here since predicted masks (not vector-derived
    GT ones) are the only caller of this function and benefit from it; see
    ``open_iterations``/``close_iterations`` below.

    Morphology is always **opening then closing** -- there's no separate operation
    selector; ``open_iterations``/``close_iterations`` (independent of each other) each
    control one step, and ``0`` disables that step entirely (so ``open_iterations=0,
    close_iterations=0`` disables morphology altogether). Standalone dilate/erode aren't
    exposed here since they only ever grow or only ever shrink the mask, with no
    noise/gap-specific behavior -- opening and closing are what you actually want for
    cleaning up a noisy predicted mask.

    ``open_iterations`` (erode that many times, then dilate the same number of times)
    shrinks away small protrusions and noise blobs thinner than the structuring element
    after that many passes, without growing the mask past its original extent.
    ``close_iterations`` (dilate that many times, then erode the same number of times)
    bridges small gaps/breaks in the mask (common in noisy predictions) without shrinking
    it -- note closing never removes pixels, so a mask-diff between its input and output is
    always empty by construction. Opening and closing serve different purposes and
    typically need different pass counts -- defaults (``open_iterations=1``,
    ``close_iterations=5``) are a starting point, not derived from any particular dataset
    calibration.

    ``remove_small_objects_*`` / ``skeletonize_enabled`` (both on by default -- this is the
    from-mask path's whole reason for existing): a thresholded predicted-probability mask is
    a blob several pixels wide, not the 1px-wide centerline ``build_pixel_graph`` assumes.
    ``remove_small_objects`` first drops isolated noise specks under
    ``remove_small_objects_min_size_px`` connected pixels (cheap to do on the mask, before
    they'd otherwise become spurious isolated graph nodes/components), then ``skeletonize``
    (Zhang-Suen thinning) reduces the remaining blob(s) to a topological centerline. Skipping
    this on a wide mask means every mask pixel becomes a graph node in a dense 2D grid graph,
    and ``merge_neighbor_nodes`` then collapses whole blobs into a single hub node (visible as
    star-shaped artifacts -- many nodes directly "connected" to one central node). A mask
    that's already thin (e.g. a rasterized GT mask) passes through skeletonize unchanged, so
    this is safe to leave on unconditionally.

    ``radius_fix_*`` (disabled by default -- opt in explicitly, it changes the graph and
    therefore any APLS numbers derived from it): radius-based extension of endpoint-fix,
    applied last (after merge) -- connects every endpoint/isolated node to every other
    endpoint/isolated node within ``radius_fix_radius_m`` straight-line meters, not just
    the diagonal-pixel-adjacent case the earlier endpoint-fix stage handles. See
    ``network_xy_raster_utils.repair_degree1_endpoints_within_radius``.

    ``min_component_nodes`` (default ``5``): final cleanup step, applied after everything
    else (incl. ``radius_fix``) -- drops whole connected components with fewer than this
    many nodes, since a fragment that small is more likely a spurious prediction artifact
    (isolated skeleton speck, leftover merge remnant, unlinked endpoint) than real,
    separately-standing network. Set to ``0`` or ``1`` to disable. See
    ``network_xy_raster_utils.drop_small_components``.
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
        morph_connectivity=morph_connectivity,
        open_iterations=open_iterations,
        close_iterations=close_iterations,
        remove_small_objects_enabled=remove_small_objects_enabled,
        remove_small_objects_min_size_px=remove_small_objects_min_size_px,
        skeletonize_enabled=skeletonize_enabled,
        rdp_epsilon_m=rdp_epsilon_m,
        endpoint_fix_enabled=endpoint_fix_enabled,
        endpoint_fix_stage=endpoint_fix_stage,
        endpoint_fix_added_edge_weight=endpoint_fix_added_edge_weight,
        endpoint_fix_include_isolated_nodes=endpoint_fix_include_isolated_nodes,
        merge_enabled=merge_enabled,
        merge_hop_threshold=merge_hop_threshold,
        radius_fix_enabled=radius_fix_enabled,
        radius_fix_radius_m=radius_fix_radius_m,
        radius_fix_added_edge_weight=radius_fix_added_edge_weight,
        radius_fix_include_isolated_nodes=radius_fix_include_isolated_nodes,
        min_component_nodes=min_component_nodes,
        collect_timings=collect_timings,
    )


def edge_length_m(graph: xy_rast.PixelGraph) -> np.ndarray:
    """Straight-line XY length (m) per edge -- same formula GT edges' `distance` uses."""
    edges = np.asarray(graph.edges, dtype=np.int64)
    if edges.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    xy = np.asarray(graph.node_xy, dtype=np.float64)
    delta = xy[edges[:, 1]] - xy[edges[:, 0]]
    return np.linalg.norm(delta, axis=1)
