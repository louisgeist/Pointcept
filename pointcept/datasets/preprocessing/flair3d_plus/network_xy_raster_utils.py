"""Project linear networks onto a 1 m XY raster aligned to Lambert 93 meters.

Assumes point and segment coordinates are already in EPSG:2154 (metric).
Pixel (i, j) covers world cell::

    [origin_x + i * pixel_m, origin_x + (i + 1) * pixel_m)
    × [origin_y + j * pixel_m, origin_y + (j + 1) * pixel_m)

With ``pixel_m=1``, that is the absolute meter square
``[floor(E), floor(E)+1) × [floor(N), floor(N)+1)``.

Standalone module (numpy + Pillow) intended to be copied into other repos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_PIXEL_M = 1.0
# Fine densification along centerlines before binning into 1 m Lambert cells.
# 0.01 m is denser than needed for 1 m pixels but cheap vs graph/RDP; ensures
# thin diagonals still touch every crossed cell.
DEFAULT_CENTERLINE_SAMPLE_STEP_M = 0.01
DEFAULT_RDP_EPSILON_M = 2.0
# Distinct marker color for retained RDP nodes (edges keep network color).
DEFAULT_NODE_MARKER_RGB = np.array([255, 140, 0], dtype=np.uint8)  # orange
# Single-pixel markers (radius 0) so edges stay readable between nodes.
DEFAULT_NODE_MARKER_RADIUS_PX = 0
# Edge stroke color for RDP graph overlays (distinct from LiDAR RGB).
DEFAULT_EDGE_RGB = np.array([255, 220, 40], dtype=np.uint8)  # yellow
# Densification for drawing graph edges only (independent of centerline mask step).
DEFAULT_EDGE_SAMPLE_STEP_M = 0.5

# 4-connectivity (von Neumann) and 8-connectivity (Moore), excluding (0, 0).
_NEIGH4_OFFSETS = np.array(
    [(-1, 0), (0, -1), (0, 1), (1, 0)],
    dtype=np.int64,
)
_NEIGH8_OFFSETS = np.array(
    [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class GridSpec:
    """Axis-aligned XY grid snapped to absolute metric coordinates.

    ``origin_x`` / ``origin_y`` are the south-west corner of local pixel (0, 0),
    always a multiple of ``pixel_m`` (Lambert meter alignment when CRS is 2154).
    Array layout is ``(height, width)`` with row ``j`` = northing index.
    """

    origin_x: float
    origin_y: float
    width: int
    height: int
    pixel_m: float = DEFAULT_PIXEL_M

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"GridSpec width/height must be positive, got {self.width}x{self.height}"
            )
        if self.pixel_m <= 0:
            raise ValueError(f"pixel_m must be > 0, got {self.pixel_m}")


def grid_from_xy_bounds(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    pixel_m: float = DEFAULT_PIXEL_M,
) -> GridSpec:
    """Build a grid covering ``[xmin, xmax] × [ymin, ymax]`` with snapped origins.

    Origins use ``floor(min / pixel_m) * pixel_m``. Extent goes up to
    ``ceil(max / pixel_m)`` so the last meter cell that touches ``max`` is included.
    """
    step = float(pixel_m)
    if step <= 0:
        raise ValueError(f"pixel_m must be > 0, got {pixel_m}")
    if not np.isfinite([xmin, ymin, xmax, ymax]).all():
        raise ValueError("Bounds must be finite")
    if xmax < xmin or ymax < ymin:
        raise ValueError(
            f"Invalid bounds: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}"
        )

    ix0 = int(np.floor(xmin / step))
    iy0 = int(np.floor(ymin / step))
    ix1 = int(np.ceil(xmax / step))
    iy1 = int(np.ceil(ymax / step))
    # Point exactly on an integer max boundary: still include that edge cell.
    if ix1 <= ix0:
        ix1 = ix0 + 1
    if iy1 <= iy0:
        iy1 = iy0 + 1

    return GridSpec(
        origin_x=float(ix0 * step),
        origin_y=float(iy0 * step),
        width=int(ix1 - ix0),
        height=int(iy1 - iy0),
        pixel_m=step,
    )


def xy_to_indices(
    xy: np.ndarray,
    grid: GridSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Map world XY to local integer pixel indices (may fall outside the grid)."""
    pts = np.asarray(xy, dtype=np.float64)
    if pts.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"xy must have shape (N, >=2), got {pts.shape}")
    inv = 1.0 / float(grid.pixel_m)
    ix = np.floor((pts[:, 0] - grid.origin_x) * inv).astype(np.int64)
    iy = np.floor((pts[:, 1] - grid.origin_y) * inv).astype(np.int64)
    return ix, iy


def _in_grid_mask(
    ix: np.ndarray,
    iy: np.ndarray,
    grid: GridSpec,
) -> np.ndarray:
    return (ix >= 0) & (iy >= 0) & (ix < grid.width) & (iy < grid.height)


def sample_segments_xy(
    segments: np.ndarray,
    sample_step_m: float = DEFAULT_CENTERLINE_SAMPLE_STEP_M,
) -> np.ndarray:
    """Densely sample XY points along line segments (centerline only).

    ``segments`` shape ``(N, 2, >=2)``. Returns ``(K, 2)``.
    """
    if segments.size == 0 or segments.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    segs = np.asarray(segments, dtype=np.float64)
    p0 = segs[:, 0, :2]
    p1 = segs[:, 1, :2]
    step = float(sample_step_m)
    if step <= 0:
        step = DEFAULT_CENTERLINE_SAMPLE_STEP_M
    pieces: list[np.ndarray] = []
    for i in range(segs.shape[0]):
        a = p0[i]
        b = p1[i]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        dist = float(np.linalg.norm(b - a))
        if dist < 1e-9:
            pieces.append(a[None, :])
            continue
        n_steps = max(1, int(np.ceil(dist / step)))
        t = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64)
        pieces.append(a[None, :] + t[:, None] * (b - a)[None, :])
    if not pieces:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(pieces, axis=0)


def sample_segments_xy_with_width(
    segments: np.ndarray,
    sample_step_m: float = DEFAULT_CENTERLINE_SAMPLE_STEP_M,
    line_width_m: float = 0.0,
) -> np.ndarray:
    """Sample XY along segments, optionally as a thin lateral corridor.

    When ``line_width_m > 0``, each longitudinal sample ``p`` emits only the two
    offset points ``p ± (width/2) * n`` along the unit segment normal (no
    centerline point). When ``line_width_m <= 0``, falls back to centerline-only
    sampling (same as ``sample_segments_xy``).

    ``segments`` shape ``(N, 2, >=2)``. Returns ``(K, 2)``.
    """
    width = float(line_width_m)
    if width <= 0:
        return sample_segments_xy(segments, sample_step_m=sample_step_m)

    if segments.size == 0 or segments.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    segs = np.asarray(segments, dtype=np.float64)
    p0 = segs[:, 0, :2]
    p1 = segs[:, 1, :2]
    step = float(sample_step_m)
    if step <= 0:
        step = DEFAULT_CENTERLINE_SAMPLE_STEP_M
    half = 0.5 * width
    pieces: list[np.ndarray] = []
    for i in range(segs.shape[0]):
        a = p0[i]
        b = p1[i]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        delta = b - a
        dist = float(np.linalg.norm(delta))
        if dist < 1e-9:
            # Degenerate segment: no well-defined normal → keep the point.
            pieces.append(a[None, :])
            continue
        t_hat = delta / dist
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=np.float64)
        n_steps = max(1, int(np.ceil(dist / step)))
        ts = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64)
        center = a[None, :] + ts[:, None] * delta[None, :]
        offset = half * n_hat
        pieces.append(center + offset[None, :])
        pieces.append(center - offset[None, :])
    if not pieces:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(pieces, axis=0)


def mean_rgb_raster(
    points_xy: np.ndarray,
    rgb: np.ndarray,
    grid: GridSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Average RGB per grid cell.

    Returns
    -------
    mean_rgb : (H, W, 3) float32 — 0 where ``count == 0``
    count : (H, W) int32
    """
    h, w = grid.height, grid.width
    mean = np.zeros((h, w, 3), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.int32)
    pts = np.asarray(points_xy, dtype=np.float64)
    colors = np.asarray(rgb)
    if pts.size == 0:
        return mean, count
    if pts.shape[0] != colors.shape[0]:
        raise ValueError(
            f"points_xy length ({pts.shape[0]}) != rgb length ({colors.shape[0]})"
        )
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"rgb must have shape (N, 3), got {colors.shape}")

    finite = np.isfinite(pts[:, :2]).all(axis=1)
    if not np.any(finite):
        return mean, count
    ix, iy = xy_to_indices(pts[finite, :2], grid)
    inside = _in_grid_mask(ix, iy, grid)
    if not np.any(inside):
        return mean, count
    ix = ix[inside]
    iy = iy[inside]
    cols = colors[finite][inside].astype(np.float64, copy=False)
    flat = iy * w + ix
    n_cells = h * w
    sum_r = np.bincount(flat, weights=cols[:, 0], minlength=n_cells)
    sum_g = np.bincount(flat, weights=cols[:, 1], minlength=n_cells)
    sum_b = np.bincount(flat, weights=cols[:, 2], minlength=n_cells)
    count_flat = np.bincount(flat, minlength=n_cells).astype(np.int32)
    count = count_flat.reshape(h, w)
    nz = count_flat > 0
    mean_flat = np.zeros((n_cells, 3), dtype=np.float32)
    if np.any(nz):
        mean_flat[nz, 0] = (sum_r[nz] / count_flat[nz]).astype(np.float32)
        mean_flat[nz, 1] = (sum_g[nz] / count_flat[nz]).astype(np.float32)
        mean_flat[nz, 2] = (sum_b[nz] / count_flat[nz]).astype(np.float32)
    return mean_flat.reshape(h, w, 3), count


def absolute_meter_cells_from_xy(
    xy: np.ndarray,
    pixel_m: float = DEFAULT_PIXEL_M,
) -> np.ndarray:
    """Unique absolute Lambert cell indices ``(ix, iy)`` for world XY samples.

    Cell ``(ix, iy)`` is the meter square
    ``[ix * pixel_m, (ix + 1) * pixel_m) × [iy * pixel_m, (iy + 1) * pixel_m)``.
    Returns ``(N, 2)`` int64, sorted unique, or empty ``(0, 2)``.
    """
    pts = np.asarray(xy, dtype=np.float64)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"xy must have shape (N, >=2), got {pts.shape}")
    step = float(pixel_m)
    if step <= 0:
        raise ValueError(f"pixel_m must be > 0, got {pixel_m}")
    finite = np.isfinite(pts[:, :2]).all(axis=1)
    if not np.any(finite):
        return np.empty((0, 2), dtype=np.int64)
    inv = 1.0 / step
    ix = np.floor(pts[finite, 0] * inv).astype(np.int64)
    iy = np.floor(pts[finite, 1] * inv).astype(np.int64)
    stacked = np.stack([ix, iy], axis=1)
    return np.unique(stacked, axis=0)


def densify_segments_to_absolute_cells(
    segments: np.ndarray,
    *,
    pixel_m: float = DEFAULT_PIXEL_M,
    sample_step_m: float = DEFAULT_CENTERLINE_SAMPLE_STEP_M,
    line_width_m: float = 0.0,
) -> np.ndarray:
    """Densify segments once and return unique absolute ``(ix, iy)`` meter cells.

    ``line_width_m > 0`` samples lateral offsets ``± width/2`` (no centerline);
    see ``sample_segments_xy_with_width``.
    """
    samples = sample_segments_xy_with_width(
        segments,
        sample_step_m=sample_step_m,
        line_width_m=line_width_m,
    )
    return absolute_meter_cells_from_xy(samples, pixel_m=pixel_m)


def mask_from_absolute_cells(
    cells: np.ndarray,
    grid: GridSpec,
) -> np.ndarray:
    """Paint a boolean local mask from absolute meter cells ``(N, 2)`` of ``(ix, iy)``."""
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    if cells.size == 0:
        return mask
    arr = np.asarray(cells, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"cells must have shape (N, 2), got {arr.shape}")
    step = float(grid.pixel_m)
    ix0 = int(np.floor(grid.origin_x / step + 1e-9))
    iy0 = int(np.floor(grid.origin_y / step + 1e-9))
    local_ix = arr[:, 0] - ix0
    local_iy = arr[:, 1] - iy0
    inside = _in_grid_mask(local_ix, local_iy, grid)
    if not np.any(inside):
        return mask
    mask[local_iy[inside], local_ix[inside]] = True
    return mask


def centerline_pixel_mask(
    segments: np.ndarray,
    grid: GridSpec,
    sample_step_m: float = DEFAULT_CENTERLINE_SAMPLE_STEP_M,
) -> np.ndarray:
    """Boolean mask of pixels touched by densified centerline samples."""
    cells = densify_segments_to_absolute_cells(
        segments, pixel_m=grid.pixel_m, sample_step_m=sample_step_m
    )
    return mask_from_absolute_cells(cells, grid)


@dataclass(frozen=True)
class PixelGraph:
    """Undirected 8-neighborhood graph over centerline pixels.

    ``node_rc`` rows are ``(row=iy, col=ix)`` in local grid indices.
    ``edges`` stores each undirected edge once with ``u < v``.
    """

    node_rc: np.ndarray
    node_xy: np.ndarray
    edges: np.ndarray
    grid: GridSpec


def pixel_centers_xy(node_rc: np.ndarray, grid: GridSpec) -> np.ndarray:
    """World XY of pixel centers: ``origin + (index + 0.5) * pixel_m``."""
    rc = np.asarray(node_rc, dtype=np.float64)
    if rc.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if rc.ndim != 2 or rc.shape[1] != 2:
        raise ValueError(f"node_rc must have shape (N, 2), got {rc.shape}")
    step = float(grid.pixel_m)
    # node_rc columns: (row=iy, col=ix) → world (x, y)
    x = grid.origin_x + (rc[:, 1] + 0.5) * step
    y = grid.origin_y + (rc[:, 0] + 0.5) * step
    return np.stack([x, y], axis=1)


def build_pixel_graph(
    mask: np.ndarray,
    grid: GridSpec,
    connectivity: int = 4,
) -> PixelGraph:
    """Build an undirected neighborhood graph from a boolean centerline mask.

    ``connectivity`` is ``4`` (orthogonal) or ``8`` (incl. diagonals).
    Nodes are ``True`` pixels; an edge links two adjacent mask pixels.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    offsets = _NEIGH4_OFFSETS if connectivity == 4 else _NEIGH8_OFFSETS

    m = np.asarray(mask, dtype=bool)
    if m.shape != (grid.height, grid.width):
        raise ValueError(
            f"mask shape {m.shape} != grid (H,W)=({grid.height}, {grid.width})"
        )
    node_rc = np.argwhere(m).astype(np.int64, copy=False)
    n = int(node_rc.shape[0])
    if n == 0:
        return PixelGraph(
            node_rc=node_rc,
            node_xy=np.empty((0, 2), dtype=np.float64),
            edges=np.empty((0, 2), dtype=np.int64),
            grid=grid,
        )

    id_grid = np.full((grid.height, grid.width), -1, dtype=np.int64)
    id_grid[node_rc[:, 0], node_rc[:, 1]] = np.arange(n, dtype=np.int64)

    rows = node_rc[:, 0]
    cols = node_rc[:, 1]
    edge_u: list[np.ndarray] = []
    edge_v: list[np.ndarray] = []
    src = np.arange(n, dtype=np.int64)
    for dr, dc in offsets:
        nr = rows + int(dr)
        nc = cols + int(dc)
        inside = (nr >= 0) & (nc >= 0) & (nr < grid.height) & (nc < grid.width)
        if not np.any(inside):
            continue
        nbr = np.full(n, -1, dtype=np.int64)
        nbr[inside] = id_grid[nr[inside], nc[inside]]
        valid = nbr >= 0
        # Undirected: keep each edge once with u < v.
        keep = valid & (src < nbr)
        if not np.any(keep):
            continue
        edge_u.append(src[keep])
        edge_v.append(nbr[keep])

    if edge_u:
        edges = np.stack(
            [np.concatenate(edge_u), np.concatenate(edge_v)], axis=1
        ).astype(np.int64, copy=False)
    else:
        edges = np.empty((0, 2), dtype=np.int64)

    return PixelGraph(
        node_rc=node_rc,
        node_xy=pixel_centers_xy(node_rc, grid),
        edges=edges,
        grid=grid,
    )


def build_pixel_graph_4(mask: np.ndarray, grid: GridSpec) -> PixelGraph:
    """4-connected pixel graph (orthogonal neighbors only)."""
    return build_pixel_graph(mask, grid, connectivity=4)


def build_pixel_graph_8(mask: np.ndarray, grid: GridSpec) -> PixelGraph:
    """8-connected pixel graph (Moore neighborhood)."""
    return build_pixel_graph(mask, grid, connectivity=8)


def adjacency_list(edges: np.ndarray, n_nodes: int) -> list[list[int]]:
    """Build an undirected adjacency list from ``edges`` shape ``(E, 2)``."""
    adj: list[list[int]] = [[] for _ in range(int(n_nodes))]
    if n_nodes <= 0:
        return adj
    e = np.asarray(edges, dtype=np.int64)
    if e.size == 0:
        return adj
    for u, v in e:
        ui = int(u)
        vi = int(v)
        if ui < 0 or vi < 0 or ui >= n_nodes or vi >= n_nodes or ui == vi:
            continue
        adj[ui].append(vi)
        adj[vi].append(ui)
    return adj


def node_degrees(edges: np.ndarray, n_nodes: int) -> np.ndarray:
    """Per-node undirected degree."""
    deg = np.zeros(int(n_nodes), dtype=np.int64)
    if n_nodes <= 0:
        return deg
    e = np.asarray(edges, dtype=np.int64)
    if e.size == 0:
        return deg
    for u, v in e:
        ui = int(u)
        vi = int(v)
        if ui < 0 or vi < 0 or ui >= n_nodes or vi >= n_nodes or ui == vi:
            continue
        deg[ui] += 1
        deg[vi] += 1
    return deg


def _walk_polyline(
    start: int,
    nxt: int,
    adj: list[list[int]],
    degree: np.ndarray,
    used_undirected: set[tuple[int, int]],
) -> list[int]:
    """Walk from ``start`` through ``nxt`` until a critical node (or cycle close)."""
    path = [start, nxt]
    u, v = start, nxt
    used_undirected.add((min(u, v), max(u, v)))
    while True:
        if degree[v] != 2:
            break
        neighbors = adj[v]
        candidates = [w for w in neighbors if w != u]
        if not candidates:
            break
        w = candidates[0]
        key = (min(v, w), max(v, w))
        if key in used_undirected:
            break
        used_undirected.add(key)
        path.append(w)
        u, v = v, w
        if v == start:
            # Closed cycle walked from an arbitrary break point.
            break
    return path


def extract_polylines(graph: PixelGraph) -> list[np.ndarray]:
    """Split the graph into polylines between critical nodes (degree != 2).

    Isolated nodes (degree 0) are returned as length-1 paths. Pure cycles of
    only degree-2 nodes are broken at an arbitrary node and returned as a
    closed path (first == last).
    """
    n = int(graph.node_rc.shape[0])
    if n == 0:
        return []
    adj = adjacency_list(graph.edges, n)
    deg = node_degrees(graph.edges, n)
    used: set[tuple[int, int]] = set()
    polylines: list[np.ndarray] = []

    # Paths starting at critical nodes (degree != 2).
    for s in range(n):
        if deg[s] == 2:
            continue
        if deg[s] == 0:
            polylines.append(np.array([s], dtype=np.int64))
            continue
        for nbr in adj[s]:
            key = (min(s, nbr), max(s, nbr))
            if key in used:
                continue
            path = _walk_polyline(s, nbr, adj, deg, used)
            polylines.append(np.asarray(path, dtype=np.int64))

    # Remaining pure cycles (all degree-2).
    for s in range(n):
        if deg[s] != 2:
            continue
        for nbr in adj[s]:
            key = (min(s, nbr), max(s, nbr))
            if key in used:
                continue
            path = _walk_polyline(s, nbr, adj, deg, used)
            # Ensure closed representation when we returned to start.
            if path[0] != path[-1] and path[-1] in adj[path[0]]:
                close = (min(path[-1], path[0]), max(path[-1], path[0]))
                if close not in used:
                    used.add(close)
                    path = path + [path[0]]
            polylines.append(np.asarray(path, dtype=np.int64))

    return polylines


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from ``point`` to segment ``a``–``b``."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-18:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def ramer_douglas_peucker(
    points_xy: np.ndarray,
    epsilon_m: float,
) -> np.ndarray:
    """Ramer–Douglas–Peucker keep-mask for an open or closed polyline.

    Returns a boolean array of shape ``(M,)``. Endpoints are always kept.
    For a closed ring (``points[0] == points[-1]`` within 1e-9), the first and
    last entries are both kept and correspond to the same vertex.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    m = int(pts.shape[0])
    keep = np.zeros(m, dtype=bool)
    if m == 0:
        return keep
    if m <= 2:
        keep[:] = True
        return keep
    eps = float(epsilon_m)
    if eps < 0:
        raise ValueError(f"epsilon_m must be >= 0, got {epsilon_m}")

    # Iterative stack of (i0, i1) index ranges inclusive.
    stack: list[tuple[int, int]] = [(0, m - 1)]
    keep[0] = True
    keep[m - 1] = True
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        a = pts[i0]
        b = pts[i1]
        segment = pts[i0 + 1 : i1]
        if segment.shape[0] == 0:
            continue
        # Distances for interior points.
        dists = np.empty(segment.shape[0], dtype=np.float64)
        for k in range(segment.shape[0]):
            dists[k] = _point_segment_distance(segment[k], a, b)
        k_max = int(np.argmax(dists))
        d_max = float(dists[k_max])
        if d_max > eps:
            i_max = i0 + 1 + k_max
            keep[i_max] = True
            stack.append((i0, i_max))
            stack.append((i_max, i1))
    return keep


def simplify_pixel_graph_rdp(
    graph: PixelGraph,
    epsilon_m: float = DEFAULT_RDP_EPSILON_M,
) -> PixelGraph:
    """Simplify a pixel graph by RDP on each extracted polyline.

    Critical nodes (degree != 2) are always retained. Returns a new
    ``PixelGraph`` with reindexed nodes and undirected deduplicated edges.
    """
    n = int(graph.node_rc.shape[0])
    if n == 0:
        return graph

    polylines = extract_polylines(graph)
    keep_nodes = np.zeros(n, dtype=bool)
    new_edge_pairs: set[tuple[int, int]] = set()

    for path in polylines:
        if path.shape[0] == 0:
            continue
        if path.shape[0] == 1:
            keep_nodes[int(path[0])] = True
            continue
        # Drop duplicate closing vertex for RDP coords if present.
        closed = (
            path.shape[0] >= 3
            and int(path[0]) == int(path[-1])
        )
        idx = path[:-1] if closed else path
        pts = graph.node_xy[idx]
        mask = ramer_douglas_peucker(pts, epsilon_m=epsilon_m)
        kept_idx = idx[mask]
        for node_id in kept_idx:
            keep_nodes[int(node_id)] = True
        # Chain edges along simplified vertices.
        for a, b in zip(kept_idx[:-1], kept_idx[1:]):
            ua, ub = int(a), int(b)
            if ua == ub:
                continue
            new_edge_pairs.add((min(ua, ub), max(ua, ub)))
        if closed and kept_idx.shape[0] >= 2:
            ua, ub = int(kept_idx[-1]), int(kept_idx[0])
            if ua != ub:
                new_edge_pairs.add((min(ua, ub), max(ua, ub)))

    # Always keep isolated / critical leftovers already marked; ensure at least
    # endpoints from empty-path cases.
    old_ids = np.flatnonzero(keep_nodes).astype(np.int64)
    if old_ids.shape[0] == 0:
        return PixelGraph(
            node_rc=np.empty((0, 2), dtype=np.int64),
            node_xy=np.empty((0, 2), dtype=np.float64),
            edges=np.empty((0, 2), dtype=np.int64),
            grid=graph.grid,
        )

    remap = np.full(n, -1, dtype=np.int64)
    remap[old_ids] = np.arange(old_ids.shape[0], dtype=np.int64)
    node_rc = graph.node_rc[old_ids]
    node_xy = graph.node_xy[old_ids]
    edges_list = [
        (int(remap[u]), int(remap[v]))
        for u, v in sorted(new_edge_pairs)
        if remap[u] >= 0 and remap[v] >= 0 and remap[u] != remap[v]
    ]
    if edges_list:
        edges = np.asarray(edges_list, dtype=np.int64)
        # Normalize u < v and unique.
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
    else:
        edges = np.empty((0, 2), dtype=np.int64)

    return PixelGraph(
        node_rc=node_rc,
        node_xy=node_xy,
        edges=edges,
        grid=graph.grid,
    )


def _paint_disks(
    rgba: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    color: np.ndarray,
    radius_px: int,
) -> None:
    """Paint filled disks of ``radius_px`` (Chebyshev) at valid (ix, iy) in-place."""
    h, w = rgba.shape[:2]
    r = max(0, int(radius_px))
    color = np.asarray(color, dtype=np.uint8).reshape(3)
    for x, y in zip(ix.tolist(), iy.tolist()):
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        rgba[y0:y1, x0:x1, :3] = color
        rgba[y0:y1, x0:x1, 3] = 255


def rasterize_graph_edges(
    graph: PixelGraph,
    grid: GridSpec | None = None,
    color_rgb: np.ndarray | None = None,
    node_color_rgb: np.ndarray | None = None,
    node_radius_px: int = DEFAULT_NODE_MARKER_RADIUS_PX,
    sample_step_m: float = DEFAULT_EDGE_SAMPLE_STEP_M,
) -> np.ndarray:
    """Rasterize graph edges + small node markers (no centerline fill).

    Draws only straight edge strokes between graph nodes (``color_rgb``, yellow
    by default) and orange markers at retained nodes. Does **not** paint the
    original dense centerline mask.
    """
    g = grid if grid is not None else graph.grid
    rgba = np.zeros((g.height, g.width, 4), dtype=np.uint8)
    n_nodes = int(graph.node_xy.shape[0])
    if n_nodes == 0:
        return rgba
    edge_color = (
        np.asarray(color_rgb, dtype=np.uint8).reshape(3)
        if color_rgb is not None
        else DEFAULT_EDGE_RGB.copy()
    )
    node_color = (
        np.asarray(node_color_rgb, dtype=np.uint8).reshape(3)
        if node_color_rgb is not None
        else DEFAULT_NODE_MARKER_RGB.copy()
    )

    if graph.edges.size > 0:
        segs = []
        for u, v in np.asarray(graph.edges, dtype=np.int64):
            segs.append([graph.node_xy[int(u)], graph.node_xy[int(v)]])
        segments = np.asarray(segs, dtype=np.float64)  # (E, 2, 2)
        segments3 = np.concatenate(
            [segments, np.zeros((segments.shape[0], 2, 1), dtype=np.float64)],
            axis=2,
        )
        samples = sample_segments_xy(segments3, sample_step_m=sample_step_m)
        if samples.shape[0] > 0:
            ix, iy = xy_to_indices(samples, g)
            inside = _in_grid_mask(ix, iy, g)
            if np.any(inside):
                rgba[iy[inside], ix[inside], :3] = edge_color
                rgba[iy[inside], ix[inside], 3] = 255

    # Node markers on top of edges (1 px orange by default).
    nix, niy = xy_to_indices(graph.node_xy, g)
    n_inside = _in_grid_mask(nix, niy, g)
    if np.any(n_inside):
        _paint_disks(
            rgba,
            nix[n_inside],
            niy[n_inside],
            node_color,
            radius_px=node_radius_px,
        )
    return rgba


def compose_rgb_with_graph(
    points_xy: np.ndarray,
    rgb: np.ndarray,
    graph: PixelGraph,
    color_rgb: np.ndarray | None = None,
    grid: GridSpec | None = None,
    sample_step_m: float = DEFAULT_EDGE_SAMPLE_STEP_M,
    node_color_rgb: np.ndarray | None = None,
    node_radius_px: int = DEFAULT_NODE_MARKER_RADIUS_PX,
) -> np.ndarray:
    """Mean-RGB background + yellow graph edges + small orange node markers.

    Does not paint the dense centerline network mask — only RDP/graph edges.
    """
    g = grid if grid is not None else graph.grid
    mean_rgb, count = mean_rgb_raster(points_xy, rgb, g)
    rgba = np.zeros((g.height, g.width, 4), dtype=np.uint8)
    has_pts = count > 0
    if np.any(has_pts):
        rgba[has_pts, :3] = np.clip(np.rint(mean_rgb[has_pts]), 0, 255).astype(
            np.uint8
        )
        rgba[has_pts, 3] = 255
    edge_color = (
        np.asarray(color_rgb, dtype=np.uint8).reshape(3)
        if color_rgb is not None
        else DEFAULT_EDGE_RGB.copy()
    )
    overlay = rasterize_graph_edges(
        graph,
        grid=g,
        color_rgb=edge_color,
        node_color_rgb=node_color_rgb,
        node_radius_px=node_radius_px,
        sample_step_m=sample_step_m,
    )
    hit = overlay[:, :, 3] == 255
    if np.any(hit):
        rgba[hit] = overlay[hit]
    return rgba


def project_network_to_rgba(
    points_xy: np.ndarray,
    rgb: np.ndarray,
    segments: np.ndarray,
    network_color_rgb: np.ndarray,
    grid: GridSpec,
    sample_step_m: float = DEFAULT_CENTERLINE_SAMPLE_STEP_M,
) -> np.ndarray:
    """Build RGBA raster: centerline color > mean point RGB > transparent.

    Returns ``(H, W, 4)`` uint8.
    """
    mean_rgb, count = mean_rgb_raster(points_xy, rgb, grid)
    line_mask = centerline_pixel_mask(
        segments, grid, sample_step_m=sample_step_m
    )
    rgba = np.zeros((grid.height, grid.width, 4), dtype=np.uint8)
    has_pts = count > 0
    if np.any(has_pts):
        rgba[has_pts, :3] = np.clip(np.rint(mean_rgb[has_pts]), 0, 255).astype(
            np.uint8
        )
        rgba[has_pts, 3] = 255
    if np.any(line_mask):
        color = np.asarray(network_color_rgb, dtype=np.uint8).reshape(3)
        rgba[line_mask, :3] = color
        rgba[line_mask, 3] = 255
    return rgba


def bounds_from_points_and_segments(
    points_xy: np.ndarray,
    segments: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Axis-aligned bounds of finite point XY and segment endpoints, or None."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.size and pts.ndim == 2 and pts.shape[1] >= 2:
        finite = np.isfinite(pts[:, :2]).all(axis=1)
        if np.any(finite):
            xs.append(pts[finite, 0])
            ys.append(pts[finite, 1])
    segs = np.asarray(segments, dtype=np.float64)
    if segs.size and segs.ndim == 3 and segs.shape[1] == 2 and segs.shape[2] >= 2:
        ends = segs[:, :, :2].reshape(-1, 2)
        finite = np.isfinite(ends).all(axis=1)
        if np.any(finite):
            xs.append(ends[finite, 0])
            ys.append(ends[finite, 1])
    if not xs:
        return None
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    return (
        float(np.min(x_all)),
        float(np.min(y_all)),
        float(np.max(x_all)),
        float(np.max(y_all)),
    )


def save_rgba_png(
    path: Path | str,
    rgba: np.ndarray,
    flip_y: bool = True,
) -> Path:
    """Write an RGBA uint8 array as PNG. ``flip_y=True`` puts north at the top."""
    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"rgba must have shape (H, W, 4), got {arr.shape}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img_arr = np.flipud(arr) if flip_y else arr
    Image.fromarray(img_arr, mode="RGBA").save(out)
    return out
