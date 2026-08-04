"""APLS (Average Path Length Similarity) between a ground-truth and a predicted graph.

Based on the two (algebraically equivalent) formulas from the paper draft::

    APLS(G,G') = 1 - (1/|V|^2) * sum_{(u,v) in V^2} min(1, |L_uv - L'_u'v'| / L_uv)
    APLS(script-G, script-G-hat)
        = [sum_G |V_G|^2 * APLS(G, G-hat)] / [sum_G |V_G|^2]

where ``u'``/``v'`` are the nodes in ``G'`` spatially nearest to ``u``/``v`` in ``G``, and a
missing path in the **prediction** (``L'_u'v'`` infinite while ``L_uv`` is finite) gets
relative error 1.

Practical sum (undirected graphs, ``D`` symmetric):
- **Self-pairs excluded** (``u == v``): trivially ``L_uv = 0``, uninformative.
- **Each unordered pair once** (``u < v`` only).
- **GT-disconnected pairs excluded**: if ``u`` and ``v`` lie in different connected
  components of ``G`` (``L_uv = inf``), there is no ground-truth path length to compare
  against -- those pairs are dropped from both numerator and denominator. Otherwise a
  fragmented GT graph would drive APLS(G,G) well below 1 and crush every prediction
  score for reasons unrelated to model quality.

So the sum runs over unordered pairs ``{u,v}`` with a **finite** GT shortest path,
normalized by that pair count (not by ``|V|*(|V|-1)/2``). Dataset-level aggregation is a
global numerator/denominator sum across ROIs (pair-count-weighted average), then an
unweighted macro-average across network channels.

numpy/scipy only -- no geopandas dependency, so this module is usable from lightweight unit
tests as well as the full driver script.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

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


@dataclass(frozen=True)
class ApsPairResult:
    roi: str
    network_type: str
    score: float  # numerator / denom, or NaN if denom == 0
    numerator: float
    denom: int  # # unordered GT pairs with a finite shortest path (same component)
    n_nodes_gt: int
    n_nodes_pred: int
    n_edges_gt: int
    n_edges_pred: int


@dataclass(frozen=True)
class ApsDiagnostics:
    """Per-pair / per-node breakdown for visualizing what hurts APLS."""

    result: ApsPairResult
    match_idx: np.ndarray  # (N_gt,) -> pred node index; -1 if no pred nodes
    match_collapse_count: np.ndarray  # (N_gt,) # GT nodes sharing the same pred match
    pair_u: np.ndarray  # (P,) GT node indices
    pair_v: np.ndarray
    pair_error: np.ndarray  # (P,) in [0, 1]; 1 = full miss / max relative error
    node_mean_error: np.ndarray  # (N_gt,) nan if node in no scored pair
    node_n_pairs: np.ndarray  # (N_gt,) int
    gt_predecessors: np.ndarray  # for reconstructing GT shortest paths


def apls_pair_diagnostics(
    gt: ApsGraph,
    pred: ApsGraph,
    *,
    roi: str,
    network_type: str,
    max_nodes_exact: int = 4000,
) -> ApsDiagnostics | None:
    """Like ``apls_pair_score`` but also returns per-pair errors and GT predecessors.

    Returns ``None`` when there is nothing to score (``denom == 0``).
    ``pair_error = 1 - pair_score`` so high values hurt APLS.
    """
    result = apls_pair_score(
        gt,
        pred,
        roi=roi,
        network_type=network_type,
        max_nodes_exact=max_nodes_exact,
    )
    n_gt = int(gt.node_xy.shape[0])
    n_pred = int(pred.node_xy.shape[0])
    empty_match = np.full(n_gt, -1, dtype=np.int64)
    empty_collapse = np.ones(n_gt, dtype=np.int64)

    if result.denom == 0:
        return None

    D_gt, gt_pred = _shortest_path_all_pairs_with_predecessors(gt)
    triu_i, triu_j = np.triu_indices(n_gt, k=1)
    L_gt_all = D_gt[triu_i, triu_j]
    connected = np.isfinite(L_gt_all)
    pair_u = triu_i[connected].astype(np.int64)
    pair_v = triu_j[connected].astype(np.int64)
    L_gt_pairs = L_gt_all[connected]

    if n_pred == 0:
        pair_error = np.ones(pair_u.shape[0], dtype=np.float64)
        match_idx = empty_match
        collapse = empty_collapse
    else:
        match_idx = _nearest_node_indices(gt.node_xy, pred.node_xy)
        # How many GT nodes snap to the same pred node (collapse > 1 hurts paths).
        _, inv, counts = np.unique(match_idx, return_inverse=True, return_counts=True)
        collapse = counts[inv].astype(np.int64)

        D_pred = _shortest_path_all_pairs(pred)
        D_pred_matched = D_pred[np.ix_(match_idx, match_idx)]
        L_pred_pairs = D_pred_matched[pair_u, pair_v]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.abs(L_gt_pairs - L_pred_pairs) / L_gt_pairs
        not_error = np.clip(1.0 - rel_err, 0.0, 1.0)
        not_error = np.where(np.isfinite(not_error), not_error, 0.0)
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
        match_idx=match_idx if n_pred > 0 else empty_match,
        match_collapse_count=collapse if n_pred > 0 else empty_collapse,
        pair_u=pair_u,
        pair_v=pair_v,
        pair_error=pair_error,
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
    max_nodes_exact: int = 4000,
) -> ApsPairResult:
    """Exact O(|V_gt|^2) APLS between a ground-truth graph and a predicted graph.

    Sums over unordered GT node pairs ``{u,v}, u != v`` that have a **finite** shortest
    path in ``gt`` (same connected component). GT-disconnected pairs are skipped entirely
    (see module docstring). A missing path in ``pred`` for such a pair scores 0.
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

    if n_gt < 2:
        # 0 or 1 GT nodes -> no non-self pair exists to score; exclude from aggregation.
        return _empty_result(denom=0, score=float("nan"))
    if n_gt > max_nodes_exact:
        raise ValueError(
            f"ROI {roi!r} network_type {network_type!r}: GT graph has {n_gt} nodes, "
            f"exceeding max_nodes_exact={max_nodes_exact}. Exact O(V^2) APLS would be too "
            "expensive here; no silent subsampling is done by default since it would "
            "affect reported numbers -- raise max_nodes_exact explicitly if this ROI/"
            "channel is expected to be this large."
        )

    D_gt = _shortest_path_all_pairs(gt)
    triu_i, triu_j = np.triu_indices(n_gt, k=1)  # unordered pairs u < v
    L_gt_all = D_gt[triu_i, triu_j]
    connected = np.isfinite(L_gt_all)
    denom = int(connected.sum())
    if denom == 0:
        # All GT nodes are isolates (or otherwise pairwise disconnected) -- nothing to score.
        return _empty_result(denom=0, score=float("nan"))

    if n_pred == 0:
        # Every GT-connected pair is missing in G' -> relative error 1 -> contributes 0.
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

    D_pred = _shortest_path_all_pairs(pred)
    match_idx = _nearest_node_indices(gt.node_xy, pred.node_xy)
    D_pred_matched = D_pred[np.ix_(match_idx, match_idx)]

    L_gt_pairs = L_gt_all[connected]
    L_pred_pairs = D_pred_matched[triu_i, triu_j][connected]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.abs(L_gt_pairs - L_pred_pairs) / L_gt_pairs
    not_error = np.clip(1.0 - rel_err, 0.0, 1.0)
    # Pred-disconnected (L_pred inf) -> nan rel_err -> score 0 for that pair.
    not_error = np.where(np.isfinite(not_error), not_error, 0.0)
    numerator = float(not_error.sum())

    return ApsPairResult(
        roi=roi,
        network_type=network_type,
        score=numerator / denom,
        numerator=numerator,
        denom=denom,
        n_nodes_gt=n_gt,
        n_nodes_pred=n_pred,
        n_edges_gt=n_edges_gt,
        n_edges_pred=n_edges_pred,
    )


def aggregate_dataset_apls(results: Sequence[ApsPairResult]) -> Dict[str, object]:
    """Per-channel weighted dataset APLS + unweighted macro-average across channels.

    Per channel: sum(numerator) / sum(denom) across all ROIs (excluding denom==0 entries)
    -- each ROI's weight is its count of GT-connected unordered pairs. Headline metric:
    unweighted mean of the per-channel scores (macro-average), so a dense network type
    like ROADS does not dominate a pooled score.
    """
    per_channel: Dict[str, float] = {}
    for network_type in NETWORK_TYPES:
        rs = [r for r in results if r.network_type == network_type and r.denom > 0]
        num = sum(r.numerator for r in rs)
        den = sum(r.denom for r in rs)
        per_channel[network_type] = (num / den) if den > 0 else float("nan")

    finite_scores = [v for v in per_channel.values() if np.isfinite(v)]
    macro_apls = float(np.mean(finite_scores)) if finite_scores else float("nan")

    return {
        "per_channel": per_channel,
        "macro_apls": macro_apls,
    }
