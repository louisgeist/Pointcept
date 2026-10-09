"""APLS (Average Path Length Similarity) between a ground-truth and a predicted graph.

**The metric is fixed.** Its parameters are the module constants below
(``APLS_DENSIFY_M``, ``APLS_SNAP_TO_EDGE_M``, ``APLS_MIN_PATH_LENGTH_M``, bidirectional
harmonic mean) and are deliberately *not* exposed by the public API, so that every reported
number is comparable. Changing them means a new ``APLS_PROTOCOL_VERSION``. The public
entry points are :func:`apls_symmetric_score` (tile/ROI score), :func:`apls_pair_diagnostics`
(GT->pred breakdown for visualization) and :func:`aggregate_dataset_apls`.

SpaceNet / CosmiQ-aligned pipeline::

    densify both graphs (edges longer than ``APLS_DENSIFY_M`` = 50 m are split in equal parts)
    -> match control points (snap to the other graph's skeleton within ``APLS_SNAP_TO_EDGE_M`` = 4 m)
    -> APLS(G, G') and APLS(G', G), pairs with GT path < ``APLS_MIN_PATH_LENGTH_M`` = 5 m dropped
    -> tile score = harmonic mean (0 if either side is <= 0 / non-finite)

Unidirectional primitive (used internally and by diagnostics)::

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
- **Empty graphs**: both empty -> score 1 with zero weight; GT with routes but an empty
  prediction -> score 0 weighted by the GT pair count (the ROI is penalized, not dropped).

Dataset-level aggregation is a pair-count-weighted average across ROIs (using the
GT->pred denom), then an unweighted macro-average across network channels.

numpy/scipy only -- no geopandas/shapely dependency.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    import network_xy_raster_utils as xy_rast  # type: ignore
    from network_label_utils import NETWORK_TYPES, LoadedNetworkGraph  # type: ignore
except ImportError:  # pragma: no cover
    from pointcept.datasets.preprocessing.flair3d_plus import (  # type: ignore
        network_xy_raster_utils as xy_rast,
    )
    from pointcept.datasets.preprocessing.flair3d_plus.network_label_utils import (  # type: ignore
        NETWORK_TYPES,
        LoadedNetworkGraph,
    )

# --- Fixed APLS protocol -------------------------------------------------------------
# Do not make these configurable: they define the metric. Bump APLS_PROTOCOL_VERSION if
# any of them (or the scoring conventions in the module docstring) ever changes.
APLS_PROTOCOL_VERSION: Final = "1.0"
APLS_DENSIFY_M: Final = 50.0  # max edge length after densification (m)
APLS_SNAP_TO_EDGE_M: Final = 4.0  # control-point snap radius (m)
APLS_MIN_PATH_LENGTH_M: Final = 5.0  # pairs with a shorter GT/source path are ignored (m)
APLS_PROTOCOL: Final = {
    "version": APLS_PROTOCOL_VERSION,
    "densify_m": APLS_DENSIFY_M,
    "snap_to_edge_m": APLS_SNAP_TO_EDGE_M,
    "min_path_length_m": APLS_MIN_PATH_LENGTH_M,
    "bidirectional": True,
    "aggregation": "harmonic_mean",
}

# Treat projections within this fraction of edge length as landing on an endpoint.
_ENDPOINT_T_EPS = 1e-9
# Absolute XY tolerance when preferring an existing node over an edge projection.
_NODE_REUSE_EPS_M = 1e-6


@dataclass(frozen=True)
class AplsGraph:
    """Minimal graph representation APLS needs: XY positions + weighted edges."""

    node_xy: np.ndarray        # (N, 2) float64
    edges: np.ndarray          # (E, 2) int64, u < v
    edge_length_m: np.ndarray  # (E,) float64 >= 0 -- Dijkstra weight


def apls_graph_from_pixel_graph(graph: "xy_rast.PixelGraph") -> AplsGraph:
    """Predicted-graph adapter: edge length from XY, NOT ``graph.edge_weights`` (hop count)."""
    from network_graph_pipeline import edge_length_m as _edge_length_m  # local import: avoid cycle

    return AplsGraph(
        node_xy=np.asarray(graph.node_xy, dtype=np.float64),
        edges=np.asarray(graph.edges, dtype=np.int64),
        edge_length_m=_edge_length_m(graph),
    )


def apls_graph_from_loaded_graph(loaded: LoadedNetworkGraph) -> AplsGraph:
    """GT-graph adapter: edge length = the gpkg `distance` field (already Euclidean)."""
    return AplsGraph(
        node_xy=np.asarray(loaded.node_xy, dtype=np.float64),
        edges=np.asarray(loaded.edges, dtype=np.int64),
        edge_length_m=np.asarray(loaded.edge_length_m, dtype=np.float64),
    )


def densify_apls_graph(graph: AplsGraph, max_edge_len_m: float = 50.0) -> AplsGraph:
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
        return AplsGraph(
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

    return AplsGraph(
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
    target: AplsGraph,
    query_xy: np.ndarray,
    max_snap_m: float = 4.0,
) -> Tuple[AplsGraph, np.ndarray]:
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
            AplsGraph(
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

    return AplsGraph(node_xy=node_xy, edges=edges, edge_length_m=lengths), match_idx



def _adjacency_csr(graph: AplsGraph) -> csr_matrix:
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
class AplsResult:
    """Official APLS of one (ROI, channel): both directions plus their harmonic mean.

    "gt_to_pred" is APLS(G, G'): shortest paths between GT nodes compared with the same paths in
    the prediction. "pred_to_gt" is APLS(G', G): the converse. ``score`` is their harmonic mean.
    """

    roi: str
    network_type: str
    score: float  # harmonic mean of both directions
    score_gt_to_pred: float
    score_pred_to_gt: float
    numerator: float  # gt->pred numerator (dataset weighting)
    denom: int  # gt->pred denom: # unordered GT node pairs in one component, >= min path length
    numerator_pred_to_gt: float
    denom_pred_to_gt: int
    n_nodes_gt: int
    n_nodes_pred: int
    n_edges_gt: int
    n_edges_pred: int


@dataclass(frozen=True)
class AplsDiagnostics:
    """GT->pred breakdown (per pair / per node) for visualizing what hurts APLS."""

    score: float  # == AplsResult.score_gt_to_pred
    numerator: float
    denom: int
    gt_used: AplsGraph  # GT graph actually indexed by pair_u/pair_v/match_idx/node_*/gt_predecessors
    pred_used: AplsGraph  # pred graph actually indexed by match_idx (grown if snap-to-edge injected nodes)
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


def _shortest_paths(graph: AplsGraph, *, return_predecessors: bool = False):
    """(N,N) all-pairs Dijkstra distances; ``np.inf`` if disconnected, 0 on the diagonal.

    With ``return_predecessors`` also returns scipy's predecessors matrix (path reconstruction).
    """
    n = int(graph.node_xy.shape[0])
    if n == 0:
        d = np.empty((0, 0), dtype=np.float64)
        return (d, np.empty((0, 0), dtype=np.int32)) if return_predecessors else d
    if graph.edges.shape[0] == 0:
        d = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(d, 0.0)
        return (d, np.full((n, n), -9999, dtype=np.int32)) if return_predecessors else d
    return shortest_path(
        _adjacency_csr(graph),
        method="D",
        directed=False,
        return_predecessors=return_predecessors,
    )


@dataclass(frozen=True)
class _DirectedPairs:
    """Per-pair quantities of one APLS direction ``source -> target`` (the single kernel)."""

    pair_u: np.ndarray  # (P,) source node indices, u < v
    pair_v: np.ndarray
    L_src: np.ndarray  # (P,) shortest-path length in the source graph
    L_tgt: np.ndarray  # (P,) same pair's path length in the target graph; inf if unmatched/unreachable
    not_error: np.ndarray  # (P,) clip(1 - |L_src - L_tgt| / L_src, 0, 1); 0 if unmatched
    predecessors: Optional[np.ndarray]  # source-graph predecessors (only if requested)

    @property
    def denom(self) -> int:
        return int(self.pair_u.shape[0])

    @property
    def numerator(self) -> float:
        return float(self.not_error.sum())


def _directed_pairs(
    source: AplsGraph,
    target: AplsGraph,
    match_idx: np.ndarray,
    *,
    return_predecessors: bool = False,
) -> _DirectedPairs:
    """Score every eligible source pair against the target, given ``match_idx`` into ``target``.

    Eligible pairs: unordered source node pairs ``u < v`` in the same connected component whose
    source path is at least ``APLS_MIN_PATH_LENGTH_M`` (others are excluded from numerator and
    denominator alike). A pair with an unmatched endpoint scores 0.
    """
    n_src = int(source.node_xy.shape[0])
    if return_predecessors:
        D_src, predecessors = _shortest_paths(source, return_predecessors=True)
    else:
        D_src, predecessors = _shortest_paths(source), None

    triu_i, triu_j = np.triu_indices(n_src, k=1)
    L_src_all = D_src[triu_i, triu_j]
    eligible = np.isfinite(L_src_all) & (L_src_all >= APLS_MIN_PATH_LENGTH_M)
    pair_u = triu_i[eligible].astype(np.int64)
    pair_v = triu_j[eligible].astype(np.int64)
    L_src = L_src_all[eligible]
    n_pairs = int(pair_u.shape[0])

    match_idx = np.asarray(match_idx, dtype=np.int64).reshape(-1)
    if match_idx.shape[0] != n_src:
        raise ValueError(f"match_idx length {match_idx.shape[0]} != source nodes {n_src}")
    mu = match_idx[pair_u]
    mv = match_idx[pair_v]
    matched = (mu >= 0) & (mv >= 0)

    L_tgt = np.full(n_pairs, np.inf, dtype=np.float64)
    not_error = np.zeros(n_pairs, dtype=np.float64)
    if matched.any() and target.node_xy.shape[0] > 0:
        D_tgt = _shortest_paths(target)
        L_tgt[matched] = D_tgt[mu[matched], mv[matched]]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.abs(L_src - L_tgt) / L_src
        ne = np.clip(1.0 - rel_err, 0.0, 1.0)
        ne = np.where(np.isfinite(ne), ne, 0.0)
        # Unmatched control nodes -> full miss (0); keep matched scores.
        not_error = np.where(matched, ne, 0.0)

    return _DirectedPairs(pair_u, pair_v, L_src, L_tgt, not_error, predecessors)


def _score_directed(
    source: AplsGraph, target: AplsGraph, match_idx: np.ndarray
) -> Tuple[float, float, int]:
    """``(score, numerator, denom)`` of APLS(source, target); ``score`` is NaN when denom == 0."""
    pairs = _directed_pairs(source, target, match_idx)
    if pairs.denom == 0:
        return float("nan"), 0.0, 0
    return pairs.numerator / pairs.denom, pairs.numerator, pairs.denom


def apls_pair_diagnostics(
    gt: AplsGraph,
    pred: AplsGraph,
    *,
    roi: str,
    network_type: str,
) -> AplsDiagnostics | None:
    """Per-pair / per-node breakdown of the GT->pred direction under the fixed APLS protocol.

    ``score`` equals ``apls_symmetric_score(...).score_gt_to_pred``. Only the GT->pred
    direction is broken down -- the official tile score (harmonic mean) also folds in
    pred->GT; use :func:`apls_symmetric_score` for that number.
    Returns ``None`` when there is nothing to score (``denom == 0``).
    ``pair_error = 1 - pair_score`` so high values hurt APLS.

    Densification and snap-to-edge injection can grow the node set, so the returned
    ``gt_used``/``pred_used`` are the graphs actually indexed by
    ``pair_u``/``pair_v``/``match_idx``/``node_mean_error``/``gt_predecessors`` -- use
    those, not the original ``gt``/``pred``, to look up node positions for the result.
    """
    gt = densify_apls_graph(gt, max_edge_len_m=APLS_DENSIFY_M)
    pred = densify_apls_graph(pred, max_edge_len_m=APLS_DENSIFY_M)
    pred, match_idx = match_control_points(pred, gt.node_xy, max_snap_m=APLS_SNAP_TO_EDGE_M)
    n_gt = int(gt.node_xy.shape[0])

    pairs = _directed_pairs(gt, pred, match_idx, return_predecessors=True)
    if pairs.denom == 0:
        return None
    pair_error = 1.0 - pairs.not_error

    # How many GT nodes snap to the same pred node (collapse > 1 hurts paths). Unmatched
    # (-1) nodes are excluded from collapse counting -- they don't share a pred node with
    # anyone, they're simply not matched to one.
    valid = match_idx >= 0
    collapse = np.ones(n_gt, dtype=np.int64)
    if valid.any():
        _, inv, counts = np.unique(match_idx[valid], return_inverse=True, return_counts=True)
        collapse[valid] = counts[inv].astype(np.int64)

    node_sum = np.zeros(n_gt, dtype=np.float64)
    node_n = np.zeros(n_gt, dtype=np.int64)
    np.add.at(node_sum, pairs.pair_u, pair_error)
    np.add.at(node_sum, pairs.pair_v, pair_error)
    np.add.at(node_n, pairs.pair_u, 1)
    np.add.at(node_n, pairs.pair_v, 1)
    node_mean = np.full(n_gt, np.nan, dtype=np.float64)
    has = node_n > 0
    node_mean[has] = node_sum[has] / node_n[has]

    return AplsDiagnostics(
        score=pairs.numerator / pairs.denom,
        numerator=pairs.numerator,
        denom=pairs.denom,
        gt_used=gt,
        pred_used=pred,
        match_idx=match_idx,
        match_collapse_count=collapse,
        pair_u=pairs.pair_u,
        pair_v=pairs.pair_v,
        pair_error=pair_error,
        pair_L_gt=pairs.L_src,
        pair_L_pred=pairs.L_tgt,
        node_mean_error=node_mean,
        node_n_pairs=node_n,
        gt_predecessors=pairs.predecessors,
    )


def reconstruct_gt_shortest_path(
    diagnostics: AplsDiagnostics, u: int, v: int
) -> list[int] | None:
    """GT node-index path for a scored pair, using diagnostics predecessors."""
    return _reconstruct_path(diagnostics.gt_predecessors, int(u), int(v))


def apls_symmetric_score(
    gt: AplsGraph,
    pred: AplsGraph,
    *,
    roi: str,
    network_type: str,
) -> AplsResult:
    """Official APLS between a GT and a predicted graph (fixed protocol, see ``APLS_PROTOCOL``).

    Densify both graphs (``APLS_DENSIFY_M``), snap control points onto the other graph
    (``APLS_SNAP_TO_EDGE_M``), score both directions ignoring pairs whose source path is
    shorter than ``APLS_MIN_PATH_LENGTH_M``, and combine them with a harmonic mean.
    Intentionally takes no tuning parameters.
    """
    # Edgeless graphs (even with isolated nodes) count as empty for routing.
    # Both empty -> nothing to score: perfect (1.0) but zero weight (denom 0).
    # Exactly one empty falls through to the general path on purpose: the GT pairs are still
    # counted in ``denom`` and all score 0, so a ROI whose routes were entirely missed (empty
    # prediction) lowers the dataset average instead of silently dropping out of it.
    n_edges_gt_raw = int(gt.edges.shape[0])
    n_edges_pred_raw = int(pred.edges.shape[0])
    if n_edges_gt_raw == 0 and n_edges_pred_raw == 0:
        return AplsResult(
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
            n_edges_gt=0,
            n_edges_pred=0,
        )

    gt_d = densify_apls_graph(gt, max_edge_len_m=APLS_DENSIFY_M)
    pred_d = densify_apls_graph(pred, max_edge_len_m=APLS_DENSIFY_M)

    pred_aug, match_gt = match_control_points(pred_d, gt_d.node_xy, max_snap_m=APLS_SNAP_TO_EDGE_M)
    score_gp, num_gp, den_gp = _score_directed(gt_d, pred_aug, match_gt)

    gt_aug, match_pred = match_control_points(gt_d, pred_d.node_xy, max_snap_m=APLS_SNAP_TO_EDGE_M)
    score_pg, num_pg, den_pg = _score_directed(pred_d, gt_aug, match_pred)

    # If a direction has nothing to score (denom 0) but the other is finite, CosmiQ
    # treats non-positive/NaN as forcing total 0 when composing hmean -- keep that.
    score = _harmonic_mean(score_gp, score_pg)

    return AplsResult(
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


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def aggregate_dataset_apls(
    results: Sequence[AplsResult],
) -> Dict[str, object]:
    """Per-channel pair-count-weighted dataset APLS + unweighted macro-average across channels.

    Per channel, each ROI is weighted by its count of GT-connected unordered pairs
    (gt->pred ``denom``; ROIs with ``denom == 0`` are excluded). The channel score is the
    weighted mean of the tile-level (harmonic-mean) ``score``; the unidirectional channel
    scores are ``sum(numerator) / sum(denom)`` for each direction.
    Headline metric: unweighted mean of the finite per-channel scores (macro-average).
    """
    per_channel: Dict[str, float] = {}
    per_channel_gt_to_pred: Dict[str, float] = {}
    per_channel_pred_to_gt: Dict[str, float] = {}

    for network_type in NETWORK_TYPES:
        chan = [r for r in results if r.network_type == network_type]
        rs = [r for r in chan if r.denom > 0]
        rs_pg = [r for r in chan if r.denom_pred_to_gt > 0]

        per_channel[network_type] = _ratio(
            sum(float(r.score) * r.denom if np.isfinite(r.score) else 0.0 for r in rs),
            sum(r.denom for r in rs),
        )
        per_channel_gt_to_pred[network_type] = _ratio(
            sum(r.numerator for r in rs), sum(r.denom for r in rs)
        )
        per_channel_pred_to_gt[network_type] = _ratio(
            sum(r.numerator_pred_to_gt for r in rs_pg), sum(r.denom_pred_to_gt for r in rs_pg)
        )

    finite_scores = [v for v in per_channel.values() if np.isfinite(v)]
    macro_apls = float(np.mean(finite_scores)) if finite_scores else float("nan")

    return {
        "per_channel": per_channel,
        "per_channel_gt_to_pred": per_channel_gt_to_pred,
        "per_channel_pred_to_gt": per_channel_pred_to_gt,
        "macro_apls": macro_apls,
    }
