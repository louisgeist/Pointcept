"""APLS (Average Path Length Similarity) between a ground-truth and a predicted graph.

SpaceNet / CosmiQ-aligned pipeline (defaults)::

    densify both graphs (``densify=50`` m max edge length; ``None`` disables)
    -> match control points (``snap_to_edge=4`` m snap radius; ``None`` = unrestricted NN)
    -> APLS(G, G') and APLS(G', G)
    -> tile score = harmonic mean (0 if either side is <= 0 / non-finite)

Unidirectional primitive (still used by viewers / diagnostics)::

    APLS(G,G') = average over unordered connected pairs {u,v} in G of
        clip(1 - |L_uv - L'_u'v'| / L_uv, 0, 1)

where ``u'``/``v'`` are matched nodes in ``G'`` (nearest node, or snap-to-edge), and a
missing path / unmatched control node scores 0 for that pair.

Practical sum (undirected graphs, ``D`` symmetric):
- **Self-pairs excluded** (``u == v``).
- **Each unordered pair once** (``u < v`` only).
- **Source-disconnected pairs excluded**: if ``u`` and ``v`` lie in different connected
  components of the *source* graph (``L_uv = inf``), those pairs are dropped from both
  numerator and denominator.

Dataset-level aggregation is a pair-count-weighted average across ROIs (using the
GT->pred denom), then an unweighted macro-average across network channels.

numpy/scipy only -- no geopandas/shapely dependency.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    import network_xy_raster_utils as xy_rast  # type: ignore
    from network_label_utils import LoadedNetworkGraph  # type: ignore
except ImportError:  # pragma: no cover
    from pointcept.datasets.preprocessing.flair3d_plus import (  # type: ignore
        network_xy_raster_utils as xy_rast,
    )
    from pointcept.datasets.preprocessing.flair3d_plus.network_label_utils import (  # type: ignore
        LoadedNetworkGraph,
    )

NETWORK_TYPES = ("ROADS", "RAILROADS", "TRANSMISSION_LINES")

# Treat projections within this fraction of edge length as landing on an endpoint.
_ENDPOINT_T_EPS = 1e-9
# Absolute XY tolerance when preferring an existing node over an edge projection.
_NODE_REUSE_EPS_M = 1e-6


@dataclass(frozen=True)
class ApsGraph:
    """Minimal graph representation APLS needs: XY positions + weighted edges."""

    node_xy: np.ndarray        # (N, 2) float64
    edges: np.ndarray          # (E, 2) int64, u < v
    edge_length_m: np.ndarray  # (E,) float64 >= 0 -- Dijkstra weight


def apls_graph_from_pixel_graph(graph: "xy_rast.PixelGraph") -> ApsGraph:
    """Predicted-graph adapter: edge length from XY, NOT ``graph.edge_weights`` (hop count)."""
    from network_graph_pipeline import edge_length_m as _edge_length_m  # local import: avoid cycle

    return ApsGraph(
        node_xy=np.asarray(graph.node_xy, dtype=np.float64),
        edges=np.asarray(graph.edges, dtype=np.int64),
        edge_length_m=_edge_length_m(graph),
    )


def apls_graph_from_loaded_graph(loaded: LoadedNetworkGraph) -> ApsGraph:
    """GT-graph adapter: edge length = the gpkg `distance` field (already Euclidean)."""
    return ApsGraph(
        node_xy=np.asarray(loaded.node_xy, dtype=np.float64),
        edges=np.asarray(loaded.edges, dtype=np.int64),
        edge_length_m=np.asarray(loaded.edge_length_m, dtype=np.float64),
    )


def densify_apls_graph(graph: ApsGraph, max_edge_len_m: float = 50.0) -> ApsGraph:
    """Insert mid-edge nodes so no edge is longer than ``max_edge_len_m`` (SpaceNet smoothing).

    For an edge of length ``L > max_edge_len_m``::

        n_seg = ceil(L / max_edge_len_m)
        insert n_seg - 1 nodes at fractions i/n_seg along the straight XY segment,
        replacing the edge by n_seg sub-edges of length L/n_seg each.

    Examples (max_edge_len_m=50): 50 m -> no insert; 120 m -> nodes at 40 and 80 m;
    200 m -> nodes at 50, 100, 150 m. Straight edges are always densified (no CosmiQ
    curvature skip).
    """
    if max_edge_len_m <= 0:
        raise ValueError(f"max_edge_len_m must be > 0, got {max_edge_len_m}")

    node_xy = [np.asarray(p, dtype=np.float64) for p in graph.node_xy]
    edges = np.asarray(graph.edges, dtype=np.int64).reshape(-1, 2)
    lengths = np.asarray(graph.edge_length_m, dtype=np.float64).reshape(-1)
    if edges.shape[0] == 0:
        return ApsGraph(
            node_xy=np.asarray(node_xy, dtype=np.float64).reshape(-1, 2)
            if node_xy
            else np.empty((0, 2), dtype=np.float64),
            edges=np.empty((0, 2), dtype=np.int64),
            edge_length_m=np.empty((0,), dtype=np.float64),
        )

    new_edges: list[tuple[int, int]] = []
    new_lengths: list[float] = []

    for (u, v), length in zip(edges, lengths):
        u_i, v_i = int(u), int(v)
        L = float(length)
        if not np.isfinite(L) or L < 0:
            raise ValueError(f"Invalid edge length {L} between nodes {u_i} and {v_i}")
        if L <= max_edge_len_m:
            a, b = (u_i, v_i) if u_i < v_i else (v_i, u_i)
            new_edges.append((a, b))
            new_lengths.append(L)
            continue

        n_seg = int(math.ceil(L / max_edge_len_m))
        seg_len = L / n_seg
        pu = np.asarray(node_xy[u_i], dtype=np.float64)
        pv = np.asarray(node_xy[v_i], dtype=np.float64)
        prev = u_i
        for i in range(1, n_seg):
            t = i / n_seg
            p = (1.0 - t) * pu + t * pv
            new_idx = len(node_xy)
            node_xy.append(p)
            a, b = (prev, new_idx) if prev < new_idx else (new_idx, prev)
            new_edges.append((a, b))
            new_lengths.append(seg_len)
            prev = new_idx
        a, b = (prev, v_i) if prev < v_i else (v_i, prev)
        new_edges.append((a, b))
        new_lengths.append(seg_len)

    return ApsGraph(
        node_xy=np.asarray(node_xy, dtype=np.float64).reshape(-1, 2),
        edges=np.asarray(new_edges, dtype=np.int64).reshape(-1, 2),
        edge_length_m=np.asarray(new_lengths, dtype=np.float64),
    )


def _closest_point_on_skeleton(
    node_xy: np.ndarray,
    edges: np.ndarray,
    edge_length_m: np.ndarray,
    query: np.ndarray,
) -> Tuple[str, float, dict]:
    """Closest point on the undirected straight-edge skeleton to ``query``.

    Returns ``(kind, dist, info)`` where ``kind`` is ``'node'`` or ``'edge'``.
    Prefers an existing node when distances are within ``_NODE_REUSE_EPS_M``.

    The edge search is fully vectorized over all edges at once (a plain Python loop here
    is the dominant cost of ``match_control_points`` -- this function is called once per
    query point, and used to re-loop over every edge in Python each time).
    """
    q = np.asarray(query, dtype=np.float64).reshape(2)
    n = int(node_xy.shape[0])
    best_dist = float("inf")
    best_kind = "none"
    best_info: dict = {}

    if n > 0:
        d_nodes = np.linalg.norm(node_xy - q[None, :], axis=1)
        i_node = int(np.argmin(d_nodes))
        best_dist = float(d_nodes[i_node])
        best_kind = "node"
        best_info = {"node": i_node}

    n_edges = int(edges.shape[0])
    if n_edges > 0:
        u = edges[:, 0]
        v = edges[:, 1]
        a = node_xy[u]  # (E, 2)
        b = node_xy[v]  # (E, 2)
        ab = b - a  # (E, 2)
        lab2 = np.einsum("ij,ij->i", ab, ab)  # (E,)
        degenerate = lab2 <= (_NODE_REUSE_EPS_M * _NODE_REUSE_EPS_M)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.einsum("ij,ij->i", q[None, :] - a, ab) / lab2
        interior = (~degenerate) & (t > _ENDPOINT_T_EPS) & (t < 1.0 - _ENDPOINT_T_EPS)
        if np.any(interior):
            p = a + t[:, None] * ab
            dist = np.linalg.norm(q[None, :] - p, axis=1)
            dist = np.where(interior, dist, np.inf)
            e_idx = int(np.argmin(dist))
            d = float(dist[e_idx])
            # Prefer existing node on ties / near-ties.
            if d + _NODE_REUSE_EPS_M < best_dist:
                L = float(edge_length_m[e_idx])
                lab = float(np.sqrt(lab2[e_idx]))
                best_dist = d
                best_kind = "edge"
                best_info = {
                    "edge_idx": e_idx,
                    "u": int(u[e_idx]),
                    "v": int(v[e_idx]),
                    "t": float(t[e_idx]),
                    "point": p[e_idx],
                    "length": L if np.isfinite(L) and L > 0 else lab,
                }

    return best_kind, best_dist, best_info


def _split_edge_at_point(
    node_xy: np.ndarray,
    edges: np.ndarray,
    lengths: np.ndarray,
    edge_idx: int,
    point: np.ndarray,
    t: float,
    edge_length: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Split ``edges[edge_idx]`` at ``point`` (fraction ``t`` along u->v).

    Functional (numpy-array) style -- returns the updated ``(node_xy, edges, lengths,
    new_node_idx)`` rather than mutating Python lists in place, since growing a plain
    numpy array by concatenation is far cheaper here than round-tripping through
    per-element Python lists on every call (this runs once per matched query point).
    """
    u, v = int(edges[edge_idx, 0]), int(edges[edge_idx, 1])
    new_idx = node_xy.shape[0]
    node_xy = np.concatenate([node_xy, np.asarray(point, dtype=np.float64).reshape(1, 2)], axis=0)
    len_u = max(float(edge_length) * float(t), 0.0)
    len_v = max(float(edge_length) * (1.0 - float(t)), 0.0)
    a1, b1 = (u, new_idx) if u < new_idx else (new_idx, u)
    a2, b2 = (new_idx, v) if new_idx < v else (v, new_idx)
    keep = np.ones(edges.shape[0], dtype=bool)
    keep[edge_idx] = False
    edges = np.concatenate([edges[keep], np.array([[a1, b1], [a2, b2]], dtype=np.int64)], axis=0)
    lengths = np.concatenate([lengths[keep], np.array([len_u, len_v], dtype=np.float64)])
    return node_xy, edges, lengths, new_idx


def match_control_points(
    target: ApsGraph,
    query_xy: np.ndarray,
    max_snap_m: float = 4.0,
) -> Tuple[ApsGraph, np.ndarray]:
    """Snap each query point onto ``target``'s skeleton (SpaceNet snap-to-edge).

    For each query:
    - if closest skeleton point is farther than ``max_snap_m`` -> unmatched (-1);
    - if closest is an existing node -> reuse it (no injection);
    - if closest is mid-edge -> split that edge and insert a node.

    Returns ``(possibly_augmented_graph, match_idx)`` with ``match_idx`` shape ``(Q,)``.
    numpy only.
    """
    query_xy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    n_q = int(query_xy.shape[0])
    if n_q == 0:
        return target, np.empty((0,), dtype=np.int64)

    node_xy = np.asarray(target.node_xy, dtype=np.float64).reshape(-1, 2).copy()
    edges = np.asarray(target.edges, dtype=np.int64).reshape(-1, 2).copy()
    lengths = np.asarray(target.edge_length_m, dtype=np.float64).reshape(-1).copy()
    match_idx = np.full(n_q, -1, dtype=np.int64)

    if node_xy.shape[0] == 0:
        return (
            ApsGraph(
                node_xy=np.empty((0, 2), dtype=np.float64),
                edges=np.empty((0, 2), dtype=np.int64),
                edge_length_m=np.empty((0,), dtype=np.float64),
            ),
            match_idx,
        )

    for qi in range(n_q):
        kind, dist, info = _closest_point_on_skeleton(node_xy, edges, lengths, query_xy[qi])
        if kind == "none" or dist > max_snap_m:
            continue
        if kind == "node":
            match_idx[qi] = int(info["node"])
            continue
        # Mid-edge injection.
        node_xy, edges, lengths, new_idx = _split_edge_at_point(
            node_xy,
            edges,
            lengths,
            int(info["edge_idx"]),
            info["point"],
            float(info["t"]),
            float(info["length"]),
        )
        match_idx[qi] = new_idx

    return ApsGraph(node_xy=node_xy, edges=edges, edge_length_m=lengths), match_idx


def match_nn_nodes(
    target: ApsGraph,
    query_xy: np.ndarray,
    max_snap_m: Optional[float] = None,
) -> Tuple[ApsGraph, np.ndarray]:
    """Nearest-node matching; optional hard distance radius (``None`` = unrestricted)."""
    query_xy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    n_q = int(query_xy.shape[0])
    match_idx = np.full(n_q, -1, dtype=np.int64)
    if n_q == 0 or target.node_xy.shape[0] == 0:
        return target, match_idx
    tree = cKDTree(np.asarray(target.node_xy, dtype=np.float64))
    dist, idx = tree.query(query_xy, k=1)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if max_snap_m is None:
        ok = np.isfinite(dist)
    else:
        ok = np.isfinite(dist) & (dist <= float(max_snap_m))
    match_idx[ok] = idx[ok]
    return target, match_idx


def _adjacency_csr(graph: ApsGraph) -> csr_matrix:
    n = int(graph.node_xy.shape[0])
    edges = graph.edges
    if n == 0 or edges.shape[0] == 0:
        return csr_matrix((n, n), dtype=np.float64)
    weights = np.maximum(graph.edge_length_m, 1e-9)
    u = edges[:, 0]
    v = edges[:, 1]
    data = np.concatenate([weights, weights])
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def _shortest_path_all_pairs(graph: ApsGraph) -> np.ndarray:
    """(N,N) float64 all-pairs shortest path; np.inf for disconnected pairs, 0 diagonal."""
    n = int(graph.node_xy.shape[0])
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)
    if graph.edges.shape[0] == 0:
        d = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(d, 0.0)
        return d
    return shortest_path(_adjacency_csr(graph), method="D", directed=False)


def _shortest_path_all_pairs_with_predecessors(
    graph: ApsGraph,
) -> tuple[np.ndarray, np.ndarray]:
    """Like ``_shortest_path_all_pairs`` plus a predecessors matrix for path reconstruction."""
    n = int(graph.node_xy.shape[0])
    if n == 0:
        return np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.int32)
    if graph.edges.shape[0] == 0:
        d = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(d, 0.0)
        pred = np.full((n, n), -9999, dtype=np.int32)
        return d, pred
    return shortest_path(
        _adjacency_csr(graph), method="D", directed=False, return_predecessors=True
    )


def _reconstruct_path(predecessors: np.ndarray, src: int, dst: int) -> list[int] | None:
    """Node index path ``src -> ... -> dst`` from a scipy predecessors matrix, or None."""
    if src == dst:
        return [int(src)]
    if predecessors.shape[0] == 0:
        return None
    path = [int(dst)]
    while path[-1] != src:
        prev = int(predecessors[src, path[-1]])
        if prev < 0:
            return None
        path.append(prev)
        if len(path) > predecessors.shape[0] + 1:
            return None  # cycle guard
    path.reverse()
    return path


def _nearest_node_indices(query_xy: np.ndarray, ref_xy: np.ndarray) -> np.ndarray:
    """For each row of ``query_xy``, index into ``ref_xy`` of its nearest neighbor."""
    tree = cKDTree(ref_xy)
    _, idx = tree.query(query_xy, k=1)
    return np.asarray(idx, dtype=np.int64).reshape(-1)


def _harmonic_mean(a: float, b: float) -> float:
    """CosmiQ-style: 0 if either side is non-positive or non-finite."""
    if (
        not np.isfinite(a)
        or not np.isfinite(b)
        or a <= 0.0
        or b <= 0.0
    ):
        return 0.0
    return float(2.0 * a * b / (a + b))


@dataclass(frozen=True)
class ApsPairResult:
    roi: str
    network_type: str
    score: float  # numerator / denom, or NaN if denom == 0
    numerator: float
    denom: int  # # unordered source pairs with a finite shortest path (same component)
    n_nodes_gt: int
    n_nodes_pred: int
    n_edges_gt: int
    n_edges_pred: int


@dataclass(frozen=True)
class ApsSymmetricResult:
    """Tile-level APLS with optional bidirectional harmonic mean."""

    roi: str
    network_type: str
    score: float  # hmean of both directions when symmetric, else gt->pred
    score_gt_to_pred: float
    score_pred_to_gt: float
    numerator: float  # gt->pred numerator (dataset weighting)
    denom: int  # gt->pred denom
    numerator_pred_to_gt: float
    denom_pred_to_gt: int
    n_nodes_gt: int
    n_nodes_pred: int
    n_edges_gt: int
    n_edges_pred: int


@dataclass(frozen=True)
class ApsDiagnostics:
    """Per-pair / per-node breakdown for visualizing what hurts APLS."""

    result: ApsPairResult
    gt_used: ApsGraph  # GT graph actually indexed by pair_u/pair_v/match_idx/node_*/gt_predecessors
    pred_used: ApsGraph  # pred graph actually indexed by match_idx (grown if snap-to-edge injected nodes)
    match_idx: np.ndarray  # (N_gt,) -> pred node index; -1 if no pred nodes / unmatched
    match_collapse_count: np.ndarray  # (N_gt,) # GT nodes sharing the same pred match
    pair_u: np.ndarray  # (P,) GT node indices
    pair_v: np.ndarray
    pair_error: np.ndarray  # (P,) in [0, 1]; 1 = full miss / max relative error
    pair_L_gt: np.ndarray  # (P,) GT shortest-path length for each scored pair
    pair_L_pred: np.ndarray  # (P,) matched predicted-graph shortest-path length; inf if unmatched
    node_mean_error: np.ndarray  # (N_gt,) nan if node in no scored pair
    node_n_pairs: np.ndarray  # (N_gt,) int
    gt_predecessors: np.ndarray  # for reconstructing GT shortest paths (indexes gt_used)


def _score_directed(
    source: ApsGraph,
    target: ApsGraph,
    match_idx: np.ndarray,
    *,
    max_nodes_exact: Optional[int],
    roi: str,
    network_type: str,
    role: str,
    min_path_length_m: Optional[float] = None,
) -> Tuple[float, float, int]:
    """APLS(source, target) given a precomputed ``match_idx`` into ``target``.

    ``min_path_length_m``: SpaceNet-style short-path filter -- source pairs whose GT
    shortest-path length is below this (meters) are excluded from both numerator and
    denominator, same as the existing source-disconnected exclusion. ``None`` (default)
    keeps every GT-connected pair, matching prior behavior.

    ``max_nodes_exact``: hard cap on ``|V_source|`` (after densification); ``None``
    disables the cap.

    Returns ``(score, numerator, denom)``. ``score`` is NaN when denom == 0.
    """
    n_src = int(source.node_xy.shape[0])
    if n_src < 2:
        return float("nan"), 0.0, 0
    if max_nodes_exact is not None and n_src > max_nodes_exact:
        raise ValueError(
            f"ROI {roi!r} network_type {network_type!r} ({role}): source graph has "
            f"{n_src} nodes, exceeding max_nodes_exact={max_nodes_exact}. Exact O(V^2) "
            "APLS would be too expensive here; no silent subsampling is done by default "
            "since it would affect reported numbers -- raise max_nodes_exact (or set it "
            "to None to disable the cap) if this ROI/channel is expected to be this "
            "large. Note: densify increases |V|; the limit applies after densification."
        )

    D_src = _shortest_path_all_pairs(source)
    triu_i, triu_j = np.triu_indices(n_src, k=1)
    L_src_all = D_src[triu_i, triu_j]
    connected = np.isfinite(L_src_all)
    if min_path_length_m is not None:
        connected = connected & (L_src_all >= float(min_path_length_m))
    denom = int(connected.sum())
    if denom == 0:
        return float("nan"), 0.0, 0

    pair_u = triu_i[connected]
    pair_v = triu_j[connected]
    L_src_pairs = L_src_all[connected]

    match_idx = np.asarray(match_idx, dtype=np.int64).reshape(-1)
    if match_idx.shape[0] != n_src:
        raise ValueError(
            f"match_idx length {match_idx.shape[0]} != source nodes {n_src}"
        )

    mu = match_idx[pair_u]
    mv = match_idx[pair_v]
    matched = (mu >= 0) & (mv >= 0)

    not_error = np.zeros(denom, dtype=np.float64)
    if matched.any() and target.node_xy.shape[0] > 0:
        D_tgt = _shortest_path_all_pairs(target)
        L_tgt = np.full(denom, np.inf, dtype=np.float64)
        L_tgt[matched] = D_tgt[mu[matched], mv[matched]]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.abs(L_src_pairs - L_tgt) / L_src_pairs
        ne = np.clip(1.0 - rel_err, 0.0, 1.0)
        ne = np.where(np.isfinite(ne), ne, 0.0)
        # Unmatched control nodes -> full miss (already 0); keep matched scores.
        not_error = np.where(matched, ne, 0.0)

    numerator = float(not_error.sum())
    return numerator / denom, numerator, denom


def apls_pair_diagnostics(
    gt: ApsGraph,
    pred: ApsGraph,
    *,
    roi: str,
    network_type: str,
    max_nodes_exact: Optional[int] = 4000,
    densify: Optional[float] = None,
    snap_to_edge: Optional[float] = None,
    min_path_length_m: Optional[float] = None,
) -> ApsDiagnostics | None:
    """Like ``apls_pair_score`` but also returns per-pair errors and GT predecessors.

    Returns ``None`` when there is nothing to score (``denom == 0``).
    ``pair_error = 1 - pair_score`` so high values hurt APLS.

    ``densify``/``snap_to_edge`` (meters): when set, mirrors the SpaceNet-aligned
    preprocessing ``apls_symmetric_score`` uses (there, default ``50.0``/``4.0``) before
    diagnosing the GT->pred direction, so ``result.score`` matches
    ``apls_symmetric_score(...).score_gt_to_pred`` under the same parameters. ``None``
    (default, for existing callers) keeps the legacy unrestricted nearest-node,
    non-densified matching. Either way this only diagnoses the GT->pred direction --
    the symmetric (harmonic-mean) tile score also folds in pred->GT, which this
    function does not break down; use ``apls_symmetric_score`` for the tile-level number.

    ``min_path_length_m``: SpaceNet-style short-path filter -- GT pairs whose shortest
    path is shorter than this (meters) are excluded from ``pair_u``/``pair_v``/etc, same
    as ``apls_symmetric_score``/``apls_pair_score``. ``None`` (default) keeps every
    GT-connected pair.

    Since ``densify``/snap-to-edge injection can grow the node set, the returned
    ``gt_used``/``pred_used`` are the (possibly grown) graphs actually indexed by
    ``pair_u``/``pair_v``/``match_idx``/``node_mean_error``/``gt_predecessors`` -- use
    those, not the original ``gt``/``pred``, to look up node positions for the result.
    """
    if densify is not None:
        max_edge_len_m = float(densify)
        gt = densify_apls_graph(gt, max_edge_len_m=max_edge_len_m)
        pred = densify_apls_graph(pred, max_edge_len_m=max_edge_len_m)

    n_gt = int(gt.node_xy.shape[0])
    empty_match = np.full(n_gt, -1, dtype=np.int64)
    empty_collapse = np.ones(n_gt, dtype=np.int64)

    if pred.node_xy.shape[0] == 0:
        match_idx = empty_match.copy()
    elif snap_to_edge is not None:
        pred, match_idx = match_control_points(pred, gt.node_xy, max_snap_m=float(snap_to_edge))
    else:
        _, match_idx = match_nn_nodes(pred, gt.node_xy, max_snap_m=None)
    n_pred = int(pred.node_xy.shape[0])

    result = apls_pair_score(
        gt,
        pred,
        roi=roi,
        network_type=network_type,
        max_nodes_exact=max_nodes_exact,
        match_idx=match_idx,
        min_path_length_m=min_path_length_m,
    )

    if result.denom == 0:
        return None

    D_gt, gt_pred = _shortest_path_all_pairs_with_predecessors(gt)
    triu_i, triu_j = np.triu_indices(n_gt, k=1)
    L_gt_all = D_gt[triu_i, triu_j]
    connected = np.isfinite(L_gt_all)
    if min_path_length_m is not None:
        connected = connected & (L_gt_all >= float(min_path_length_m))
    pair_u = triu_i[connected].astype(np.int64)
    pair_v = triu_j[connected].astype(np.int64)
    L_gt_pairs = L_gt_all[connected]

    # How many GT nodes snap to the same pred node (collapse > 1 hurts paths). Unmatched
    # (-1, only possible with a snap radius) nodes are excluded from collapse counting --
    # they don't share a pred node with anyone, they're simply not matched to one.
    valid = match_idx >= 0
    collapse = empty_collapse.copy()
    if valid.any():
        _, inv, counts = np.unique(match_idx[valid], return_inverse=True, return_counts=True)
        collapse[valid] = counts[inv].astype(np.int64)

    if n_pred == 0 or not valid.any():
        pair_error = np.ones(pair_u.shape[0], dtype=np.float64)
        pair_L_pred = np.full(pair_u.shape[0], np.inf, dtype=np.float64)
    else:
        D_pred = _shortest_path_all_pairs(pred)
        mu = match_idx[pair_u]
        mv = match_idx[pair_v]
        matched_pair = (mu >= 0) & (mv >= 0)
        L_pred_pairs = np.full(pair_u.shape[0], np.inf, dtype=np.float64)
        L_pred_pairs[matched_pair] = D_pred[mu[matched_pair], mv[matched_pair]]
        pair_L_pred = L_pred_pairs
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.abs(L_gt_pairs - L_pred_pairs) / L_gt_pairs
        not_error = np.clip(1.0 - rel_err, 0.0, 1.0)
        not_error = np.where(np.isfinite(not_error) & matched_pair, not_error, 0.0)
        pair_error = 1.0 - not_error

    node_sum = np.zeros(n_gt, dtype=np.float64)
    node_n = np.zeros(n_gt, dtype=np.int64)
    np.add.at(node_sum, pair_u, pair_error)
    np.add.at(node_sum, pair_v, pair_error)
    np.add.at(node_n, pair_u, 1)
    np.add.at(node_n, pair_v, 1)
    node_mean = np.full(n_gt, np.nan, dtype=np.float64)
    has = node_n > 0
    node_mean[has] = node_sum[has] / node_n[has]

    return ApsDiagnostics(
        result=result,
        gt_used=gt,
        pred_used=pred,
        match_idx=match_idx,
        match_collapse_count=collapse,
        pair_u=pair_u,
        pair_v=pair_v,
        pair_error=pair_error,
        pair_L_gt=L_gt_pairs,
        pair_L_pred=pair_L_pred,
        node_mean_error=node_mean,
        node_n_pairs=node_n,
        gt_predecessors=gt_pred,
    )


def reconstruct_gt_shortest_path(
    diagnostics: ApsDiagnostics, u: int, v: int
) -> list[int] | None:
    """GT node-index path for a scored pair, using diagnostics predecessors."""
    return _reconstruct_path(diagnostics.gt_predecessors, int(u), int(v))


def apls_pair_score(
    gt: ApsGraph,
    pred: ApsGraph,
    *,
    roi: str,
    network_type: str,
    max_nodes_exact: Optional[int] = 4000,
    match_idx: Optional[np.ndarray] = None,
    min_path_length_m: Optional[float] = None,
) -> ApsPairResult:
    """Exact O(|V_gt|^2) unidirectional APLS(G, G').

    Sums over unordered GT node pairs ``{u,v}, u != v`` that have a **finite** shortest
    path in ``gt`` (same connected component). GT-disconnected pairs are skipped entirely
    (see module docstring). A missing path in ``pred`` for such a pair scores 0.

    If ``match_idx`` is None, unrestricted nearest-node matching is used (legacy).
    Otherwise ``match_idx[i]`` is the pred node for GT node ``i`` (``-1`` = unmatched).

    ``max_nodes_exact``: hard cap on ``|V_gt|``; ``None`` disables the cap.

    ``min_path_length_m``: SpaceNet-style short-path filter -- GT pairs whose shortest
    path is shorter than this (meters) are also excluded. ``None`` (default) scores
    every GT-connected pair, matching prior behavior.
    """
    n_gt = int(gt.node_xy.shape[0])
    n_pred = int(pred.node_xy.shape[0])
    n_edges_gt = int(gt.edges.shape[0])
    n_edges_pred = int(pred.edges.shape[0])

    def _empty_result(*, denom: int, score: float) -> ApsPairResult:
        return ApsPairResult(
            roi=roi,
            network_type=network_type,
            score=score,
            numerator=0.0,
            denom=denom,
            n_nodes_gt=n_gt,
            n_nodes_pred=n_pred,
            n_edges_gt=n_edges_gt,
            n_edges_pred=n_edges_pred,
        )

    if match_idx is None:
        if n_pred == 0:
            # Need denom first.
            if n_gt < 2:
                return _empty_result(denom=0, score=float("nan"))
            if max_nodes_exact is not None and n_gt > max_nodes_exact:
                raise ValueError(
                    f"ROI {roi!r} network_type {network_type!r}: GT graph has {n_gt} "
                    f"nodes, exceeding max_nodes_exact={max_nodes_exact}."
                )
            D_gt = _shortest_path_all_pairs(gt)
            triu_i, triu_j = np.triu_indices(n_gt, k=1)
            L_gt_all = D_gt[triu_i, triu_j]
            connected = np.isfinite(L_gt_all)
            if min_path_length_m is not None:
                connected = connected & (L_gt_all >= float(min_path_length_m))
            denom = int(connected.sum())
            if denom == 0:
                return _empty_result(denom=0, score=float("nan"))
            return ApsPairResult(
                roi=roi,
                network_type=network_type,
                score=0.0,
                numerator=0.0,
                denom=denom,
                n_nodes_gt=n_gt,
                n_nodes_pred=n_pred,
                n_edges_gt=n_edges_gt,
                n_edges_pred=n_edges_pred,
            )
        match_idx = _nearest_node_indices(gt.node_xy, pred.node_xy)

    score, numerator, denom = _score_directed(
        gt,
        pred,
        match_idx,
        max_nodes_exact=max_nodes_exact,
        roi=roi,
        network_type=network_type,
        role="gt_to_pred",
        min_path_length_m=min_path_length_m,
    )
    return ApsPairResult(
        roi=roi,
        network_type=network_type,
        score=score,
        numerator=numerator,
        denom=denom,
        n_nodes_gt=n_gt,
        n_nodes_pred=n_pred,
        n_edges_gt=n_edges_gt,
        n_edges_pred=n_edges_pred,
    )


def apls_symmetric_score(
    gt: ApsGraph,
    pred: ApsGraph,
    *,
    roi: str,
    network_type: str,
    densify: Optional[float] = 50.0,
    snap_to_edge: Optional[float] = 4.0,
    symmetric: bool = True,
    max_nodes_exact: Optional[int] = 4000,
    min_path_length_m: Optional[float] = None,
) -> ApsSymmetricResult:
    """SpaceNet-aligned APLS: optional densify, snap-to-edge, bidirectional harmonic mean.

    ``densify``: max edge length in meters (default ``50``), or ``None`` to skip.
    ``snap_to_edge``: snap radius in meters (default ``4``), or ``None`` for unrestricted
    nearest-node matching (no edge projection). Legacy bools are coerced
    (``True``→default meters, ``False``→``None``).
    ``max_nodes_exact`` is checked **after** densification on each source graph used for
    a directed score; ``None`` disables the cap (exact APLS runs regardless of ``|V|``).
    ``min_path_length_m``: SpaceNet-style short-path filter, applied to each direction's
    own source graph (e.g. ``5.0`` for roads excludes GT/pred pairs whose shortest path
    is under 5 m). ``None`` (default) scores every connected pair, matching prior
    behavior.
    """
    n_edges_gt_raw = int(gt.edges.shape[0])
    n_edges_pred_raw = int(pred.edges.shape[0])

    # SpaceNet empty handling: both empty -> 1; exactly one empty -> 0.
    # Edgeless graphs (even with isolated nodes) count as empty for routing.
    gt_no_route = n_edges_gt_raw == 0
    pred_no_route = n_edges_pred_raw == 0
    if gt_no_route and pred_no_route:
        return ApsSymmetricResult(
            roi=roi,
            network_type=network_type,
            score=1.0,
            score_gt_to_pred=1.0,
            score_pred_to_gt=1.0,
            numerator=0.0,
            denom=0,
            numerator_pred_to_gt=0.0,
            denom_pred_to_gt=0,
            n_nodes_gt=int(gt.node_xy.shape[0]),
            n_nodes_pred=int(pred.node_xy.shape[0]),
            n_edges_gt=n_edges_gt_raw,
            n_edges_pred=n_edges_pred_raw,
        )
    if gt_no_route or pred_no_route:
        return ApsSymmetricResult(
            roi=roi,
            network_type=network_type,
            score=0.0,
            score_gt_to_pred=0.0,
            score_pred_to_gt=0.0,
            numerator=0.0,
            denom=0,
            numerator_pred_to_gt=0.0,
            denom_pred_to_gt=0,
            n_nodes_gt=int(gt.node_xy.shape[0]),
            n_nodes_pred=int(pred.node_xy.shape[0]),
            n_edges_gt=n_edges_gt_raw,
            n_edges_pred=n_edges_pred_raw,
        )

    if densify is not None:
        if isinstance(densify, bool):
            max_edge_len_m = 50.0 if densify else None
        else:
            max_edge_len_m = float(densify)
        if max_edge_len_m is not None and max_edge_len_m <= 0:
            raise ValueError(f"densify must be > 0 or None, got {densify!r}")
        if max_edge_len_m is None:
            gt_d, pred_d = gt, pred
        else:
            gt_d = densify_apls_graph(gt, max_edge_len_m=max_edge_len_m)
            pred_d = densify_apls_graph(pred, max_edge_len_m=max_edge_len_m)
    else:
        gt_d = gt
        pred_d = pred

    if isinstance(snap_to_edge, bool):
        snap_radius = 4.0 if snap_to_edge else None
    elif snap_to_edge is None:
        snap_radius = None
    else:
        snap_radius = float(snap_to_edge)
        if snap_radius <= 0:
            raise ValueError(f"snap_to_edge must be > 0 or None, got {snap_to_edge!r}")

    if snap_radius is not None:
        pred_aug, match_gt = match_control_points(
            pred_d, gt_d.node_xy, max_snap_m=snap_radius
        )
    else:
        pred_aug, match_gt = match_nn_nodes(pred_d, gt_d.node_xy, max_snap_m=None)
    score_gp, num_gp, den_gp = _score_directed(
        gt_d,
        pred_aug,
        match_gt,
        max_nodes_exact=max_nodes_exact,
        roi=roi,
        network_type=network_type,
        role="gt_to_pred",
        min_path_length_m=min_path_length_m,
    )

    if symmetric:
        if snap_radius is not None:
            gt_aug, match_pred = match_control_points(
                gt_d, pred_d.node_xy, max_snap_m=snap_radius
            )
        else:
            gt_aug, match_pred = match_nn_nodes(
                gt_d, pred_d.node_xy, max_snap_m=None
            )
        score_pg, num_pg, den_pg = _score_directed(
            pred_d,
            gt_aug,
            match_pred,
            max_nodes_exact=max_nodes_exact,
            roi=roi,
            network_type=network_type,
            role="pred_to_gt",
            min_path_length_m=min_path_length_m,
        )
        # If a direction has nothing to score (denom 0) but the other is finite, CosmiQ
        # treats non-positive/NaN as forcing total 0 when composing hmean -- keep that.
        score = _harmonic_mean(score_gp, score_pg)
    else:
        score_pg = float("nan")
        num_pg = 0.0
        den_pg = 0
        score = score_gp if np.isfinite(score_gp) else 0.0

    return ApsSymmetricResult(
        roi=roi,
        network_type=network_type,
        score=float(score) if np.isfinite(score) else float("nan"),
        score_gt_to_pred=float(score_gp) if np.isfinite(score_gp) else float("nan"),
        score_pred_to_gt=float(score_pg) if np.isfinite(score_pg) else float("nan"),
        numerator=float(num_gp),
        denom=int(den_gp),
        numerator_pred_to_gt=float(num_pg),
        denom_pred_to_gt=int(den_pg),
        n_nodes_gt=int(gt_d.node_xy.shape[0]),
        n_nodes_pred=int(pred_d.node_xy.shape[0]),
        n_edges_gt=int(gt_d.edges.shape[0]),
        n_edges_pred=int(pred_d.edges.shape[0]),
    )


def aggregate_dataset_apls(
    results: Sequence[Union[ApsPairResult, ApsSymmetricResult]],
) -> Dict[str, object]:
    """Per-channel weighted dataset APLS + unweighted macro-average across channels.

    Per channel: sum(numerator) / sum(denom) across all ROIs (excluding denom==0 entries)
    -- each ROI's weight is its count of GT-connected unordered pairs (gt->pred denom).
    Headline metric: unweighted mean of the per-channel scores (macro-average).

    For ``ApsSymmetricResult``, ``score`` is the harmonic-mean tile score; weighting still
    uses gt->pred ``numerator``/``denom``. Also reports unidirectional channel averages
    when those fields are present.
    """
    per_channel: Dict[str, float] = {}
    per_channel_gt_to_pred: Dict[str, float] = {}
    per_channel_pred_to_gt: Dict[str, float] = {}

    for network_type in NETWORK_TYPES:
        rs = [r for r in results if r.network_type == network_type and r.denom > 0]
        if not rs:
            per_channel[network_type] = float("nan")
            per_channel_gt_to_pred[network_type] = float("nan")
            per_channel_pred_to_gt[network_type] = float("nan")
            continue

        # Weight tile-level ``score`` (hmean or uni) by gt->pred pair count.
        w_sum = 0.0
        w_den = 0
        for r in rs:
            w_sum += float(r.score) * int(r.denom) if np.isfinite(r.score) else 0.0
            w_den += int(r.denom)
        per_channel[network_type] = (w_sum / w_den) if w_den > 0 else float("nan")

        # Unidirectional diagnostics (pair-count weighted).
        num = sum(r.numerator for r in rs)
        den = sum(r.denom for r in rs)
        per_channel_gt_to_pred[network_type] = (num / den) if den > 0 else float("nan")

        rs_pg = [
            r
            for r in results
            if r.network_type == network_type
            and isinstance(r, ApsSymmetricResult)
            and r.denom_pred_to_gt > 0
        ]
        if rs_pg:
            num_pg = sum(r.numerator_pred_to_gt for r in rs_pg)
            den_pg = sum(r.denom_pred_to_gt for r in rs_pg)
            per_channel_pred_to_gt[network_type] = (
                (num_pg / den_pg) if den_pg > 0 else float("nan")
            )
        else:
            per_channel_pred_to_gt[network_type] = float("nan")

    finite_scores = [v for v in per_channel.values() if np.isfinite(v)]
    macro_apls = float(np.mean(finite_scores)) if finite_scores else float("nan")

    return {
        "per_channel": per_channel,
        "per_channel_gt_to_pred": per_channel_gt_to_pred,
        "per_channel_pred_to_gt": per_channel_pred_to_gt,
        "macro_apls": macro_apls,
    }
