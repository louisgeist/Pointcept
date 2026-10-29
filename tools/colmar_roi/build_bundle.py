"""Assemble the Colmar city-centre ROI (D068-2021 / UU-S1-12) into one point-cloud bundle.

Reads the 50 preprocessed sub-tiles of the ROI (Pointcept scene folders), recenters them in
float64 on a common ROI-local frame, voxel-subsamples (default: the 0.1 m delivery grid) and writes
a single ``bundle.npz`` consumed by ``viser_viewer.py`` and ``export_html.py``.

Per-point layers stored: RGB, semantic (v20), forest, the four natural-habitat ecological axes
(habitat type, moisture regime, soil chemistry, bioclimatic zone), elevation (height above DTM, m)
and LiDAR intensity ("strength"). The road network is stored as a graph (nodes with their degree,
edges cut in pieces of <= 6 m), both draped on the road surface (impervious / bridge points, DTM from ``z - elevation`` elsewhere).

Colours are not stored here: see palette_paper.json / common.palette.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

# v2 (cm-accurate coordinates), copied from Jean Zay $SCRATCH/data_pointcept/flair3d_plus_v2/test/D068-2021_LIDARHD/UU-S1-12;
# the v1 hecate copy (Y snapped to 0.5 m) is data/flair3d_plus/test/D068-2021_LIDARHD/UU-S1-12
DEFAULT_ROI_DIR = "outputs/colmar_roi/roi_v2/UU-S1-12"
DEFAULT_GRAPH = "/data/geist/Flair3D-build/data/network_graphs/D068_UU-S1-12_ROADS_graph.gpkg"
DEFAULT_REMAP = "/data/geist/Pointcept/pointcept/datasets/preprocessing/malibu3d/malibu3d_label_remap.py"

# bundle key -> remap definition name (colours live in palette_paper.json, see common.palette)
AXES = {
    "nh_habitat": "by_habitat_type_ecological",
    "nh_moisture": "by_moisture_regime",
    "nh_soil": "by_soil_chemistry",
    "nh_climate": "by_climatic_domain",
}
EDGE_SEG_M = 6.0


def _load_remap(path: str):
    spec = importlib.util.spec_from_file_location("malibu3d_label_remap_viz", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    return mod


def _scene_dirs(roi_dir: Path) -> list[Path]:
    dirs = sorted(p for p in roi_dir.iterdir() if (p / "coord.npy").exists())
    if not dirs:
        raise SystemExit(f"No scene with coord.npy under {roi_dir}")
    return dirs


def _load_scene(scene: Path) -> dict:
    # coord.npy is float32 relative to coord_translation (float64, absolute Lambert-93)
    trans = np.load(scene / "coord_translation.npy").astype(np.float64)
    out = {"abs_xyz": np.load(scene / "coord.npy").astype(np.float64) + trans}
    for key, fname in (("rgb", "color"), ("segment", "segment"), ("forest", "forest"),
                       ("nathab", "natural_habitat"), ("elevation", "elevation"), ("strength", "strength")):
        out[key] = np.load(scene / f"{fname}.npy")
    meta = json.loads((scene / "meta.json").read_text())
    nh_def = meta["label_definitions"]["natural_habitat"]
    # v1 stores one CarHab id per point ("default"); v2 stores the four axes directly (N x 4, AXES order)
    assert nh_def in ("default", "ecological_axes"), meta["label_definitions"]
    if nh_def == "ecological_axes":
        assert meta["natural_habitat_layout"]["definitions"] == list(AXES.values()), meta["natural_habitat_layout"]
    out["nathab_axes"] = np.full(len(out["abs_xyz"]), nh_def == "ecological_axes")
    return out


def _dtm_grid(xyz: np.ndarray, elevation: np.ndarray, cell: float = 1.0) -> np.ndarray:
    """Ground height per ``cell`` from z - elevation (elevation is height above DTM); gaps filled by nearest."""
    ok = ~np.isnan(elevation)
    ij = np.floor(xyz[ok, :2] / cell).astype(int)
    shape = ij.max(0) + 1
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    ground = (xyz[ok, 2] - elevation[ok]).astype(np.float64)
    s = np.bincount(flat, ground, minlength=shape[0] * shape[1])
    c = np.bincount(flat, minlength=shape[0] * shape[1])
    grid = np.where(c > 0, s / np.maximum(c, 1), np.nan).reshape(shape)
    idx = ndimage.distance_transform_edt(np.isnan(grid), return_distances=False, return_indices=True)
    return grid[tuple(idx)]


def _road_grid(xyz: np.ndarray, segment: np.ndarray, dtm: np.ndarray, cell: float = 1.0) -> np.ndarray:
    """Height of the road surface per cell: mean z of impervious-surface and bridge points (so edges follow bridge
    decks), DTM where there is none."""
    sel = np.isin(segment, (2, 12))
    ij = np.clip(np.floor(xyz[sel, :2] / cell).astype(int), 0, np.array(dtm.shape) - 1)
    flat = ij[:, 0] * dtm.shape[1] + ij[:, 1]
    size = dtm.shape[0] * dtm.shape[1]
    s = np.bincount(flat, xyz[sel, 2].astype(np.float64), minlength=size)
    c = np.bincount(flat, minlength=size)
    return np.where(c.reshape(dtm.shape) > 0, (s / np.maximum(c, 1)).reshape(dtm.shape), dtm)


def _ground_z(grid: np.ndarray, xy: np.ndarray, cell: float = 1.0) -> np.ndarray:
    ij = np.clip(np.floor(xy / cell).astype(int), 0, np.array(grid.shape) - 1)
    return grid[ij[:, 0], ij[:, 1]]


def _build_graph(path: str, origin: np.ndarray, grid: np.ndarray) -> dict:
    """Nodes and edges (straight node-to-node segments) in ROI-local coordinates, draped on the road surface.

    Each edge is cut into pieces of at most ``EDGE_SEG_M`` so the drawn tube follows the terrain; z is smoothed
    along the edge (3-vertex moving average, ends kept). ``graph_seg`` rows are (x0, y0, z0, x1, y1, z1).
    """
    import geopandas as gpd

    nodes = gpd.read_file(path, layer="nodes")
    edges = gpd.read_file(path, layer="edges")
    node_xy = np.stack([nodes.x.values - origin[0], nodes.y.values - origin[1]], axis=1)
    segs = []
    for geom in edges.geometry:
        assert len(geom.coords) == 2, "edges are expected to be straight node-to-node segments"
        (x0, y0), (x1, y1) = (np.array(c) - origin[:2] for c in geom.coords)
        k = max(1, int(np.ceil(geom.length / EDGE_SEG_M)))
        t = np.linspace(0, 1, k + 1)[:, None]
        xy = (1 - t) * np.array([x0, y0]) + t * np.array([x1, y1])
        z = _ground_z(grid, xy)
        if k > 1:
            z[1:-1] = (z[:-2] + z[1:-1] + z[2:]) / 3
        pts = np.column_stack([xy, z])
        segs.append(np.hstack([pts[:-1], pts[1:]]))
    seg = np.concatenate(segs).astype(np.float32)
    print(f"road graph: {len(nodes)} nodes, {len(edges)} edges, {len(seg)} tube pieces")
    return {"graph_nodes": np.column_stack([node_xy, _ground_z(grid, node_xy)]).astype(np.float32), "graph_seg": seg,
            "graph_n_edges": np.int32(len(edges))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--roi-dir", default=DEFAULT_ROI_DIR)
    ap.add_argument("--graph", default=DEFAULT_GRAPH)
    ap.add_argument("--remap-module", default=DEFAULT_REMAP)
    ap.add_argument("--out", default="outputs/colmar_roi/bundle.npz")
    ap.add_argument("--voxel", type=float, default=0.1, help="voxel size (m); 0 keeps every point")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scenes = [_load_scene(d) for d in _scene_dirs(Path(args.roi_dir))]
    print(f"{len(scenes)} sub-tiles, {sum(len(s['abs_xyz']) for s in scenes):,} points")
    cat = {k: np.concatenate([s[k] for s in scenes]) for k in scenes[0]}
    abs_xyz = cat.pop("abs_xyz")

    if args.voxel > 0:
        perm = np.random.default_rng(args.seed).permutation(len(abs_xyz))  # random representative per voxel
        vox = np.floor((abs_xyz[perm] - abs_xyz.min(0)) / args.voxel).astype(np.int64)
        dims = vox.max(0) + 1
        key = (vox[:, 0] * dims[1] + vox[:, 1]) * dims[2] + vox[:, 2]
        _, first = np.unique(key, return_index=True)
        keep = np.sort(perm[first])
        abs_xyz = abs_xyz[keep]
        cat = {k: v[keep] for k, v in cat.items()}
        print(f"voxel {args.voxel} m -> {len(keep):,} points")

    origin = np.floor(abs_xyz.min(0))
    xyz = (abs_xyz - origin).astype(np.float32)  # float64 subtraction first, then cast
    elev = cat["elevation"].astype(np.float32)

    # natural-habitat ecological axes: stored as such (v2), or from the stored ("default") ids via the remap (v1)
    axes_layout = cat.pop("nathab_axes")
    assert axes_layout.all() or not axes_layout.any(), "mixed natural-habitat layouts across sub-tiles"
    arrays = {}
    if axes_layout.all():
        arrays = {key: cat["nathab"][:, j].astype(np.uint8) for j, key in enumerate(AXES)}
    else:
        remap = _load_remap(args.remap_module)
        storage = remap.get_definition("natural_habitat", "default")
        for key, definition in AXES.items():
            lut = remap.build_stored_to_train_lut(storage, remap.get_definition("natural_habitat", definition))
            arrays[key] = lut[np.clip(cat["nathab"].astype(int), 0, len(lut) - 1)].astype(np.uint8)

    graph = _build_graph(args.graph, origin, _road_grid(xyz, cat["segment"], _dtm_grid(xyz, elev)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, xyz=xyz, rgb=cat["rgb"].astype(np.uint8), segment=cat["segment"].astype(np.uint8),
        forest=cat["forest"].astype(np.uint8), elevation=elev, strength=cat["strength"].astype(np.float32),
        origin_l93=origin, **arrays, **graph,
        meta=json.dumps({"roi": "D068-2021_UU-S1-12", "crs": "EPSG:2154", "voxel": args.voxel}),
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    for name, arr in [("segment", cat["segment"]), ("forest", cat["forest"])] + list(arrays.items()):
        u, c = np.unique(arr, return_counts=True)
        print(f"  {name}: " + ", ".join(f"{int(a)}={b / len(arr):.1%}" for a, b in zip(u, c)))
    print(f"  elevation: nan={np.isnan(elev).mean():.2%} p2/p98={np.nanpercentile(elev, [2, 98]).round(2).tolist()}")
    print(f"  extent (m): {xyz.max(0).round(1).tolist()}")


if __name__ == "__main__":
    main()
