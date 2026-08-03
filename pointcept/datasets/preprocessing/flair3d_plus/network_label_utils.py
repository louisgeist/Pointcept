"""Load exported Flair3D-build network graphs for Pointcept preprocessing.

Expects GeoPackages produced by ``scripts/export_network_graphs.py``::

    {network_graphs_root}/{dept}_{zone}_{NETWORK}_graph.gpkg

with layers ``nodes`` / ``edges`` / ``metadata`` (EPSG:2154).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

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
