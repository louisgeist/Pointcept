"""Load and write Flair3D-build network graphs for Pointcept preprocessing.

Expects / produces GeoPackages with layers ``nodes`` / ``edges`` / ``metadata``
(EPSG:2154)::

    {network_graphs_root}/{dept}_{zone}_{NETWORK}_graph.gpkg
    {out_dir}/{dept}_{zone}_{NETWORK}_pred_graph.gpkg
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    from network_xy_raster_utils import PixelGraph  # type: ignore

CRS_EPSG = "EPSG:2154"
_OGR_EPSG = 2154

NETWORK_TYPES = ("ROADS", "RAILROADS", "TRANSMISSION_LINES")
NETWORK_FLAG_COLUMNS = NETWORK_TYPES


def dept_code_from_lidarhd_dirname(dirname: str) -> Optional[str]:
    """``D075-2021_LIDARHD`` → ``D075``; ``D059062-2021_LIDARHD`` → ``D059062``."""
    suffix = "_LIDARHD"
    if not dirname.endswith(suffix):
        return None
    stem = dirname[: -len(suffix)]
    if not stem:
        return None
    return stem.split("-", 1)[0]


def dept_code_from_dept_year(dept_year: str) -> str:
    """``D075-2021`` → ``D075``."""
    text = str(dept_year).strip()
    if not text:
        return ""
    return text.split("-", 1)[0]


def graph_output_stem(roi_dir: Path | str) -> Optional[str]:
    """``D075_UU-S1-4`` style stem from a ROI directory, or None if unknown."""
    roi_path = Path(roi_dir)
    dept = dept_code_from_lidarhd_dirname(roi_path.parent.name)
    if dept is None:
        return None
    return f"{dept}_{roi_path.name}"


def resolve_exported_graph_gpkg(
    network_graphs_root: Path | str,
    roi_dir: Path | str,
    network_type: str,
) -> Optional[Path]:
    """Resolve ``{root}/{dept}_{zone}_{NETWORK}_graph.gpkg`` if the file exists."""
    if network_type not in NETWORK_TYPES:
        raise ValueError(
            f"Unknown network type {network_type!r}; expected one of {NETWORK_TYPES}"
        )
    stem = graph_output_stem(roi_dir)
    if stem is None:
        return None
    path = Path(network_graphs_root) / f"{stem}_{network_type}_graph.gpkg"
    return path if path.is_file() else None


def expected_exported_graph_path(
    network_graphs_root: Path | str,
    roi_dir: Path | str,
    network_type: str,
) -> Path:
    """Return the expected graph path (may not exist yet)."""
    if network_type not in NETWORK_TYPES:
        raise ValueError(
            f"Unknown network type {network_type!r}; expected one of {NETWORK_TYPES}"
        )
    stem = graph_output_stem(roi_dir)
    if stem is None:
        raise ValueError(
            f"Cannot derive graph stem from ROI directory: {roi_dir}"
        )
    return Path(network_graphs_root) / f"{stem}_{network_type}_graph.gpkg"


def _iter_linestrings(geom: BaseGeometry) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if isinstance(g, LineString) and not g.is_empty]
    if hasattr(geom, "geoms"):
        out: list[LineString] = []
        for part in geom.geoms:
            out.extend(_iter_linestrings(part))
        return out
    return []


def _linestring_to_segments(geom: LineString) -> np.ndarray:
    """Convert a LineString to ``(N, 2, 3)`` XYZ segments (Z=0 if absent)."""
    coords = np.asarray(geom.coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] < 2:
        return np.empty((0, 2, 3), dtype=np.float64)
    if coords.shape[1] >= 3:
        xyz = coords[:, :3]
    else:
        z = np.zeros((coords.shape[0], 1), dtype=np.float64)
        xyz = np.concatenate([coords[:, :2], z], axis=1)
    return np.stack([xyz[:-1], xyz[1:]], axis=1)


def load_graph_edge_segments(gpkg_path: Path | str) -> np.ndarray:
    """Load the ``edges`` layer of an exported graph GPKG as ``(N, 2, 3)`` segments."""
    path = Path(gpkg_path)
    empty = np.empty((0, 2, 3), dtype=np.float64)
    try:
        gdf = gpd.read_file(path, layer="edges")
    except Exception:  # noqa: BLE001 — missing layer / corrupt file
        return empty
    if gdf.empty or "geometry" not in gdf.columns:
        return empty

    pieces: list[np.ndarray] = []
    for geom in gdf.geometry:
        for line in _iter_linestrings(geom):
            segs = _linestring_to_segments(line)
            if segs.shape[0] > 0:
                pieces.append(segs)
    if not pieces:
        return empty
    return np.concatenate(pieces, axis=0)


def parse_bool_flag(value: object) -> bool:
    """Parse CSV / JSON style bools."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Cannot parse boolean flag from {value!r}")


def load_roi_network_flags_from_manifest(
    split_manifest_csv: Path | str,
    roi_dir: Path | str,
    network_types: Sequence[str] = NETWORK_TYPES,
) -> Dict[str, bool]:
    """Read per-network availability flags for one ROI from the split manifest.

    Raises if required columns are missing or no row matches the ROI.
    """
    import csv

    csv_path = Path(split_manifest_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"split_manifest_csv not found: {csv_path}")

    roi_path = Path(roi_dir)
    dept = dept_code_from_lidarhd_dirname(roi_path.parent.name)
    zone = roi_path.name
    if dept is None:
        raise ValueError(f"Cannot derive department code from ROI: {roi_dir}")

    types = tuple(network_types)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = [c for c in types if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"split_manifest_csv missing network columns {missing}. "
                "Enrich it via Flair3D-build export_network_graphs.py "
                "(split_manifest_csv=...)."
            )
        for row in reader:
            row_dept = dept_code_from_dept_year(row.get("dept_year", ""))
            row_roi = str(row.get("roi", "")).strip()
            if row_dept == dept and row_roi == zone:
                return {t: parse_bool_flag(row.get(t)) for t in types}

    raise ValueError(
        f"No manifest row for ROI dept={dept!r} zone={zone!r} in {csv_path}"
    )


@dataclass(frozen=True)
class LoadedNetworkGraph:
    """True graph topology of an exported GPKG (not flattened to raw segments).

    Unlike ``load_graph_edge_segments`` (which discards node identity and just
    returns raw LineString endpoints), this preserves the actual node-shared
    topology written by Flair3D-build's ``write_pixel_graph_gpkg`` -- this IS the
    ground-truth graph for APLS, no re-derivation/re-rasterization needed.
    """

    node_xy: np.ndarray        # (N, 2) float64, row order of the `nodes` layer
    node_id: np.ndarray        # (N,) int64 -- original gpkg node_id (debugging/export)
    edges: np.ndarray          # (E, 2) int64 -- array-index pairs into node_xy, u < v
    edge_length_m: np.ndarray  # (E,) float64 -- from the `distance` field (NOT `weight`,
                                # which is a hop-count used only for the merge threshold)


def load_roi_exported_network_graph(gpkg_path: Path | str) -> LoadedNetworkGraph:
    """Read the `nodes` + `edges` layers of an exported GPKG and reconstruct topology.

    Empty node/edge layers (or a missing file) return a 0-node/0-edge graph rather
    than raising -- callers decide whether an empty GT graph is meaningful for their
    ROI/network_type combination.
    """
    path = Path(gpkg_path)
    empty = LoadedNetworkGraph(
        node_xy=np.empty((0, 2), dtype=np.float64),
        node_id=np.empty((0,), dtype=np.int64),
        edges=np.empty((0, 2), dtype=np.int64),
        edge_length_m=np.empty((0,), dtype=np.float64),
    )
    if not path.is_file():
        return empty

    try:
        nodes = gpd.read_file(path, layer="nodes")
    except Exception:  # noqa: BLE001 -- missing layer / corrupt file
        return empty
    if nodes.empty or "node_id" not in nodes.columns:
        return empty

    node_id = nodes["node_id"].to_numpy(dtype=np.int64, copy=False)
    node_xy = np.stack(
        [
            nodes["x"].to_numpy(dtype=np.float64, copy=False),
            nodes["y"].to_numpy(dtype=np.float64, copy=False),
        ],
        axis=1,
    )
    id_to_idx = {int(nid): i for i, nid in enumerate(node_id.tolist())}

    try:
        edges_df = gpd.read_file(path, layer="edges")
    except Exception:  # noqa: BLE001
        edges_df = None
    if edges_df is None or edges_df.empty or "u" not in edges_df.columns:
        return LoadedNetworkGraph(
            node_xy=node_xy,
            node_id=node_id,
            edges=np.empty((0, 2), dtype=np.int64),
            edge_length_m=np.empty((0,), dtype=np.float64),
        )

    u_raw = edges_df["u"].to_numpy(dtype=np.int64, copy=False)
    v_raw = edges_df["v"].to_numpy(dtype=np.int64, copy=False)
    distance = edges_df["distance"].to_numpy(dtype=np.float64, copy=False)

    pairs: dict[tuple[int, int], float] = {}
    for u_id, v_id, dist in zip(u_raw.tolist(), v_raw.tolist(), distance.tolist()):
        ui = id_to_idx.get(int(u_id))
        vi = id_to_idx.get(int(v_id))
        if ui is None or vi is None or ui == vi:
            continue
        key = (ui, vi) if ui < vi else (vi, ui)
        # Defensive de-dup: keep the first occurrence's distance.
        pairs.setdefault(key, float(dist))

    if pairs:
        sorted_keys = sorted(pairs.keys())
        edges = np.asarray(sorted_keys, dtype=np.int64)
        edge_length_m = np.asarray([pairs[k] for k in sorted_keys], dtype=np.float64)
    else:
        edges = np.empty((0, 2), dtype=np.int64)
        edge_length_m = np.empty((0,), dtype=np.float64)

    return LoadedNetworkGraph(
        node_xy=node_xy,
        node_id=node_id,
        edges=edges,
        edge_length_m=edge_length_m,
    )


def load_roi_exported_networks(
    network_graphs_root: Path | str,
    roi_dir: Path | str,
    *,
    flags: Dict[str, bool],
    network_types: Sequence[str] = NETWORK_TYPES,
) -> Dict[str, np.ndarray]:
    """Load edge segments for a ROI, hard-failing when a flagged graph is missing.

    Parameters
    ----------
    flags :
        Per-network availability from the split manifest. ``True`` requires the
        corresponding ``*_graph.gpkg`` to exist.
    """
    out: Dict[str, np.ndarray] = {}
    root = Path(network_graphs_root)
    for network_type in network_types:
        expect = bool(flags.get(network_type, False))
        path = resolve_exported_graph_gpkg(root, roi_dir, network_type)
        if expect and path is None:
            expected = expected_exported_graph_path(root, roi_dir, network_type)
            raise FileNotFoundError(
                f"Manifest flag {network_type}=True for ROI {Path(roi_dir).name} "
                f"but graph file is missing: {expected}"
            )
        if path is None:
            out[network_type] = np.empty((0, 2, 3), dtype=np.float64)
        else:
            out[network_type] = load_graph_edge_segments(path)
    return out


def pred_graph_output_stem_paths(
    out_dir: Path | str,
    roi_dir: Path | str,
    network_type: str,
) -> tuple[Path, Path, str]:
    """Return ``(gpkg_path, apls_json_path, stem)`` for a predicted-graph dump."""
    if network_type not in NETWORK_TYPES:
        raise ValueError(
            f"Unknown network type {network_type!r}; expected one of {NETWORK_TYPES}"
        )
    stem = graph_output_stem(roi_dir)
    if stem is None:
        raise ValueError(f"Cannot derive graph stem from ROI directory: {roi_dir}")
    out = Path(out_dir)
    prefix = f"{stem}_{network_type}"
    return out / f"{prefix}_pred_graph.gpkg", out / f"{prefix}_apls.json", stem


def edge_euclidean_distances(graph: "PixelGraph") -> np.ndarray:
    """Euclidean XY distance (m) for each undirected edge ``(u, v)``."""
    edges = np.asarray(graph.edges, dtype=np.int64)
    n_edges = int(edges.shape[0]) if edges.size else 0
    if n_edges == 0:
        return np.empty((0,), dtype=np.float64)
    xy = np.asarray(graph.node_xy, dtype=np.float64)
    delta = xy[edges[:, 1]] - xy[edges[:, 0]]
    return np.linalg.norm(delta, axis=1)


def pixel_graph_to_geodataframes(
    graph: "PixelGraph",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Convert a ``PixelGraph`` to ``nodes`` / ``edges`` GeoDataFrames (EPSG:2154)."""
    n_nodes = int(graph.node_rc.shape[0])
    node_xy = np.asarray(graph.node_xy, dtype=np.float64)
    node_rc = np.asarray(graph.node_rc, dtype=np.int64)
    if n_nodes == 0:
        nodes = gpd.GeoDataFrame(
            {
                "node_id": np.empty((0,), dtype=np.int64),
                "x": np.empty((0,), dtype=np.float64),
                "y": np.empty((0,), dtype=np.float64),
                "row": np.empty((0,), dtype=np.int64),
                "col": np.empty((0,), dtype=np.int64),
            },
            geometry=[],
            crs=CRS_EPSG,
        )
    else:
        nodes = gpd.GeoDataFrame(
            {
                "node_id": np.arange(n_nodes, dtype=np.int64),
                "x": node_xy[:, 0],
                "y": node_xy[:, 1],
                "row": node_rc[:, 0],
                "col": node_rc[:, 1],
            },
            geometry=[Point(float(x), float(y)) for x, y in node_xy],
            crs=CRS_EPSG,
        )

    edges_arr = np.asarray(graph.edges, dtype=np.int64)
    n_edges = int(edges_arr.shape[0]) if edges_arr.size else 0
    if n_edges == 0:
        edges = gpd.GeoDataFrame(
            {
                "edge_id": np.empty((0,), dtype=np.int64),
                "u": np.empty((0,), dtype=np.int64),
                "v": np.empty((0,), dtype=np.int64),
                "weight": np.empty((0,), dtype=np.float64),
                "distance": np.empty((0,), dtype=np.float64),
            },
            geometry=[],
            crs=CRS_EPSG,
        )
        return nodes, edges

    weights = (
        np.asarray(graph.edge_weights, dtype=np.float64)
        if graph.edge_weights is not None
        else np.ones((n_edges,), dtype=np.float64)
    )
    if weights.shape[0] != n_edges:
        raise ValueError(
            f"edge_weights length mismatch: {weights.shape[0]} vs {n_edges}"
        )
    distances = edge_euclidean_distances(graph)
    edge_geoms = [
        LineString(
            [
                (float(node_xy[int(u), 0]), float(node_xy[int(u), 1])),
                (float(node_xy[int(v), 0]), float(node_xy[int(v), 1])),
            ]
        )
        for u, v in edges_arr
    ]
    edges = gpd.GeoDataFrame(
        {
            "edge_id": np.arange(n_edges, dtype=np.int64),
            "u": edges_arr[:, 0],
            "v": edges_arr[:, 1],
            "weight": weights,
            "distance": distances,
        },
        geometry=edge_geoms,
        crs=CRS_EPSG,
    )
    return nodes, edges


def _spatial_ref_epsg2154():
    from osgeo import osr

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(_OGR_EPSG)
    try:
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except AttributeError:
        pass
    return srs


def _ogr_create_layer(ds, name: str, geom_type: int, fields: list[tuple[str, int]], *, srs):
    from osgeo import ogr

    if geom_type == ogr.wkbNone:
        layer = ds.CreateLayer(name, srs=srs, geom_type=geom_type)
    else:
        layer = ds.CreateLayer(
            name, srs=srs, geom_type=geom_type, options=["GEOMETRY_NAME=geom"]
        )
    if layer is None:
        raise RuntimeError(f"Failed to create GPKG layer {name!r}")
    for field_name, field_type in fields:
        if layer.CreateField(ogr.FieldDefn(field_name, field_type)) != 0:
            raise RuntimeError(
                f"Failed to create field {field_name!r} on layer {name!r}"
            )
    return layer


def _write_nodes_layer(ds, nodes: gpd.GeoDataFrame, srs) -> None:
    from osgeo import ogr

    layer = _ogr_create_layer(
        ds,
        "nodes",
        ogr.wkbPoint,
        [
            ("node_id", ogr.OFTInteger64),
            ("x", ogr.OFTReal),
            ("y", ogr.OFTReal),
            ("row", ogr.OFTInteger64),
            ("col", ogr.OFTInteger64),
        ],
        srs=srs,
    )
    if len(nodes) == 0:
        return
    defn = layer.GetLayerDefn()
    node_id = nodes["node_id"].to_numpy(dtype=np.int64, copy=False)
    xs = nodes["x"].to_numpy(dtype=np.float64, copy=False)
    ys = nodes["y"].to_numpy(dtype=np.float64, copy=False)
    rows = nodes["row"].to_numpy(dtype=np.int64, copy=False)
    cols = nodes["col"].to_numpy(dtype=np.int64, copy=False)
    for i, geom in enumerate(nodes.geometry):
        feat = ogr.Feature(defn)
        feat.SetField("node_id", int(node_id[i]))
        feat.SetField("x", float(xs[i]))
        feat.SetField("y", float(ys[i]))
        feat.SetField("row", int(rows[i]))
        feat.SetField("col", int(cols[i]))
        feat.SetGeometry(ogr.CreateGeometryFromWkb(geom.wkb))
        if layer.CreateFeature(feat) != 0:
            raise RuntimeError("Failed to write node feature")
        feat = None


def _write_edges_layer(ds, edges: gpd.GeoDataFrame, srs) -> None:
    from osgeo import ogr

    layer = _ogr_create_layer(
        ds,
        "edges",
        ogr.wkbLineString,
        [
            ("edge_id", ogr.OFTInteger64),
            ("u", ogr.OFTInteger64),
            ("v", ogr.OFTInteger64),
            ("weight", ogr.OFTReal),
            ("distance", ogr.OFTReal),
        ],
        srs=srs,
    )
    if len(edges) == 0:
        return
    defn = layer.GetLayerDefn()
    edge_id = edges["edge_id"].to_numpy(dtype=np.int64, copy=False)
    u = edges["u"].to_numpy(dtype=np.int64, copy=False)
    v = edges["v"].to_numpy(dtype=np.int64, copy=False)
    weight = edges["weight"].to_numpy(dtype=np.float64, copy=False)
    distance = edges["distance"].to_numpy(dtype=np.float64, copy=False)
    for i, geom in enumerate(edges.geometry):
        feat = ogr.Feature(defn)
        feat.SetField("edge_id", int(edge_id[i]))
        feat.SetField("u", int(u[i]))
        feat.SetField("v", int(v[i]))
        feat.SetField("weight", float(weight[i]))
        feat.SetField("distance", float(distance[i]))
        feat.SetGeometry(ogr.CreateGeometryFromWkb(geom.wkb))
        if layer.CreateFeature(feat) != 0:
            raise RuntimeError("Failed to write edge feature")
        feat = None


def _metadata_value_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    if isinstance(value, (bool, int, float, str)):
        return str(value)
    return str(value)


def flatten_metadata_for_gpkg(meta: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested metadata dicts into dotted keys for the GPKG table."""
    out: dict[str, Any] = {}
    for key, value in meta.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_metadata_for_gpkg(value, full))
        else:
            out[full] = value
    return out


def _write_metadata_layer(ds, metadata: dict[str, Any]) -> None:
    from osgeo import ogr

    layer = _ogr_create_layer(
        ds,
        "metadata",
        ogr.wkbNone,
        [("key", ogr.OFTString), ("value", ogr.OFTString)],
        srs=None,
    )
    defn = layer.GetLayerDefn()
    for key, value in flatten_metadata_for_gpkg(metadata).items():
        feat = ogr.Feature(defn)
        feat.SetField("key", str(key))
        feat.SetField("value", _metadata_value_to_str(value))
        if layer.CreateFeature(feat) != 0:
            raise RuntimeError("Failed to write metadata feature")
        feat = None


def write_pixel_graph_gpkg(
    path: Path | str,
    graph: "PixelGraph",
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """Write ``nodes``, ``edges``, and ``metadata`` layers to a GeoPackage.

    Uses GDAL/OGR in a single CreateDataSource session (avoids Fiona ``mode='a'``
    NULL pointer failures on GDAL 3.6 / Fiona 1.9). Empty graphs write empty
    layers rather than raising.
    """
    from osgeo import ogr

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    nodes, edges = pixel_graph_to_geodataframes(graph)
    srs = _spatial_ref_epsg2154()
    driver = ogr.GetDriverByName("GPKG")
    if driver is None:
        raise RuntimeError("GDAL GPKG driver is unavailable")

    ds = driver.CreateDataSource(str(out))
    if ds is None:
        raise RuntimeError(f"Failed to create GeoPackage: {out}")

    try:
        _write_nodes_layer(ds, nodes, srs)
        _write_edges_layer(ds, edges, srs)
        _write_metadata_layer(ds, metadata or {})
    finally:
        ds = None

    return out
