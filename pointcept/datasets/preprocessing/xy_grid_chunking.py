"""
Regular XY grid tiling for Pointcept preprocessing.

Splits a scene dict (coord + arbitrary aligned npy arrays) into spatial tiles
on the horizontal plane, using the point cloud axis-aligned bounding box.

- ``split_scene_xy_regular``: fixed ``N × N`` cells (same count along X and Y).
- ``split_scene_xy_by_chunk_size``: physical step in XY ``(nx × ny`` may differ).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


def split_scene_xy_regular(
    scene: Dict[str, np.ndarray],
    chunking: int,
) -> List[Tuple[str, Dict[str, np.ndarray]]]:
    """
    Split one scene into a regular XY grid of ``chunking x chunking`` tiles.

    All arrays in ``scene`` must have the same leading size as ``coord``; each
    is masked identically. Requires key ``coord`` with shape (N, 3+).

    Last row/column use inclusive upper bounds so boundary points are not dropped.

    When ``chunking <= 1``, returns ``[("0-0", scene)]`` (same dict reference).

    Tiles that contain no points are omitted from the returned list (no empty subtiles).
    """
    coord = scene["coord"]
    if coord.shape[0] == 0:
        return []

    if chunking <= 1:
        return [("0-0", scene)]

    x_min, y_min = coord[:, 0].min(), coord[:, 1].min()
    x_max, y_max = coord[:, 0].max(), coord[:, 1].max()

    x_edges = np.linspace(x_min, x_max, chunking + 1, dtype=np.float32)
    y_edges = np.linspace(y_min, y_max, chunking + 1, dtype=np.float32)

    parts: List[Tuple[str, Dict[str, np.ndarray]]] = []
    for row in range(chunking):
        for col in range(chunking):
            x0, x1 = x_edges[row], x_edges[row + 1]
            y0, y1 = y_edges[col], y_edges[col + 1]

            if row == chunking - 1:
                x_mask = (coord[:, 0] >= x0) & (coord[:, 0] <= x1)
            else:
                x_mask = (coord[:, 0] >= x0) & (coord[:, 0] < x1)

            if col == chunking - 1:
                y_mask = (coord[:, 1] >= y0) & (coord[:, 1] <= y1)
            else:
                y_mask = (coord[:, 1] >= y0) & (coord[:, 1] < y1)

            mask = x_mask & y_mask
            sub_scene = {key: scene[key][mask] for key in scene}
            if sub_scene["coord"].shape[0] == 0:
                continue
            parts.append((f"{row}-{col}", sub_scene))
    return parts


def split_scene_xy_by_chunk_size(
    scene: Dict[str, np.ndarray],
    chunk_size: float,
) -> List[Tuple[str, Dict[str, np.ndarray]]]:
    """
    Split one scene into an XY grid with approximate tile size ``chunk_size`` along each axis.

    Uses the scene AABB in XY; ``nx = ceil(dx / chunk_size)``, ``ny = ceil(dy / chunk_size)``,
    so rectangular scenes get ``nx × ny`` tiles (not necessarily square).

    ``chunk_size`` must use the same length unit as ``coord[:, 0]`` and ``coord[:, 1]`` (e.g. meters).

    Positive infinity is treated like an arbitrarily large tile: a single tile ``0-0`` covering the bbox.

    All arrays in ``scene`` must align with ``coord`` on the leading dimension. Requires ``coord``.

    Interior tiles use ``[lower, upper)`` on X and Y; the last slice along each axis uses a closed
    upper bound so boundary points are not dropped.

    Tiles that contain no points are omitted from the returned list (no empty subtiles).
    """
    if chunk_size <= 0 or math.isnan(chunk_size):
        raise ValueError("chunk_size must be positive and finite, or positive infinity.")

    coord = scene["coord"]
    if coord.shape[0] == 0:
        return []

    x_min, y_min = coord[:, 0].min(), coord[:, 1].min()
    x_max, y_max = coord[:, 0].max(), coord[:, 1].max()
    dx = float(x_max - x_min)
    dy = float(y_max - y_min)

    if math.isinf(chunk_size):
        return [("0-0", scene)]

    nx = max(1, math.ceil(dx / chunk_size))
    ny = max(1, math.ceil(dy / chunk_size))

    if nx == 1 and ny == 1:
        return [("0-0", scene)]

    x_edges = np.linspace(x_min, x_max, nx + 1, dtype=np.float32)
    y_edges = np.linspace(y_min, y_max, ny + 1, dtype=np.float32)

    parts: List[Tuple[str, Dict[str, np.ndarray]]] = []
    for row in range(nx):
        for col in range(ny):
            x0, x1 = x_edges[row], x_edges[row + 1]
            y0, y1 = y_edges[col], y_edges[col + 1]

            if row == nx - 1:
                x_mask = (coord[:, 0] >= x0) & (coord[:, 0] <= x1)
            else:
                x_mask = (coord[:, 0] >= x0) & (coord[:, 0] < x1)

            if col == ny - 1:
                y_mask = (coord[:, 1] >= y0) & (coord[:, 1] <= y1)
            else:
                y_mask = (coord[:, 1] >= y0) & (coord[:, 1] < y1)

            mask = x_mask & y_mask
            sub_scene = {key: scene[key][mask] for key in scene}
            if sub_scene["coord"].shape[0] == 0:
                continue
            parts.append((f"{row}-{col}", sub_scene))
    return parts
