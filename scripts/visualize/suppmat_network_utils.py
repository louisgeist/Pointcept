"""Network-corridor helpers for the standalone Malibu3D sample-tile viewer.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

NETWORK_TYPES = ("ROADS", "RAILROADS", "TRANSMISSION_LINES")

# Distinct RGB colors per infrastructure type (viser line_segments).
NETWORK_COLORS: dict[str, np.ndarray] = {
    "ROADS": np.array([255, 220, 40], dtype=np.uint8),  # yellow
    "RAILROADS": np.array([200, 170, 230], dtype=np.uint8),  # pale violet
    "TRANSMISSION_LINES": np.array([0, 120, 255], dtype=np.uint8),  # vivid blue
}
# BDTOPO attribute codes used for filters / PREC_ALTI display height.
PREC_ALTI_UNKNOWN = 9999.0

# Per-segment / per-face attribute codes (filters + PREC_ALTI height pin).
ATTR_NORMAL = 0
ATTR_PREC_UNKNOWN = 1  # PREC_ALTI == 9999
ATTR_BELOW_GROUND = 2  # POS_SOL < 0
ATTR_BOTH = 3  # PREC_ALTI == 9999 and POS_SOL < 0

ATTR_FLAG_SCENE_SUFFIX: dict[int, str] = {
    ATTR_NORMAL: "normal",
    ATTR_PREC_UNKNOWN: "prec_alti",
    ATTR_BELOW_GROUND: "below_ground",
    ATTR_BOTH: "prec_and_below",
}

# Soft / non-carriageway ROADS natures (BDTOPO ``NATURE``).
PATH_NATURES: frozenset[str] = frozenset({"Chemin", "Sentier"})
# Unpaved / gravel carriageway (BDTOPO ``NATURE``).
UNPAVED_ROAD_NATURE = "Route empierrée"
# Ferry / maritime link (BDTOPO ``NATURE``) — not a physical road.
FERRY_NATURE = "Bac ou liaison maritime"
# BDTOPO ``ACCES_VL`` value: no light-vehicle access (stairs, paths, etc.).
ACCES_VL_IMPOSSIBLE = "Physiquement impossible"
# BDTOPO ``ETAT`` value meaning the segment is actually built and open.
ETAT_EN_SERVICE = "En service"
# BDTOPO ``FICTIF`` value: purely topological segment, no physical trace on the ground.
FICTIF_OUI = "Oui"

# Display height for PREC_ALTI=9999 segments: mean Z of points at/above this
# percentile of the ROI point cloud (avoids ground / canopy mix).
PREC_ALTI_HIGH_Z_PERCENTILE = 90.0

# LiDAR corridor display: paint points near network XY (any Z).
DEFAULT_CORRIDOR_RADIUS_M = 2.5
DEFAULT_CORRIDOR_SAMPLE_STEP_M = 1.0
DEFAULT_CORRIDOR_ALPHA = 0.7


def high_points_mean_z(
    points_xyz: np.ndarray,
    percentile: float = PREC_ALTI_HIGH_Z_PERCENTILE,
) -> float:
    """Mean Z of points at or above ``percentile`` of the cloud Z."""
    if points_xyz.size == 0:
        return 0.0
    z = np.asarray(points_xyz[:, 2], dtype=np.float64)
    finite = np.isfinite(z)
    if not np.any(finite):
        return 0.0
    z = z[finite]
    thr = float(np.percentile(z, percentile))
    high = z[z >= thr]
    if high.size == 0:
        return float(np.mean(z))
    return float(np.mean(high))


def attr_flag_has_prec_alti_unknown(flag: int) -> bool:
    """True if layer carries PREC_ALTI=9999 (red or pink)."""
    return flag in (ATTR_PREC_UNKNOWN, ATTR_BOTH)


def is_path_nature(nature: object) -> bool:
    """True if BDTOPO ``NATURE`` is Chemin or Sentier."""
    if nature is None:
        return False
    text = str(nature).strip()
    return text in PATH_NATURES


def is_unpaved_road_nature(nature: object) -> bool:
    """True if BDTOPO ``NATURE`` is Route empierrée."""
    if nature is None:
        return False
    return str(nature).strip() == UNPAVED_ROAD_NATURE


def is_ferry_nature(nature: object) -> bool:
    """True if BDTOPO ``NATURE`` is Bac ou liaison maritime."""
    if nature is None:
        return False
    return str(nature).strip() == FERRY_NATURE


def is_acces_vl_impossible(acces_vl: object) -> bool:
    """True if BDTOPO ``ACCES_VL`` is physically impossible."""
    if acces_vl is None:
        return False
    return str(acces_vl).strip() == ACCES_VL_IMPOSSIBLE


def is_fictif_oui(fictif: object) -> bool:
    """True if BDTOPO ``FICTIF`` is Oui."""
    if fictif is None:
        return False
    return str(fictif).strip() == FICTIF_OUI


def is_etat_not_en_service(etat: object) -> bool:
    """True if BDTOPO ``ETAT`` is set and not En service."""
    if etat is None:
        return False
    text = str(etat).strip()
    return bool(text) and text != ETAT_EN_SERVICE


def attr_flag_filtered(
    flag: int,
    hide_prec_alti_9999: bool,
    hide_pos_sol_lt0: bool,
) -> bool:
    """True if an ``ATTR_*`` layer should be hidden by GUI filters."""
    if flag == ATTR_PREC_UNKNOWN:
        return bool(hide_prec_alti_9999)
    if flag == ATTR_BELOW_GROUND:
        return bool(hide_pos_sol_lt0)
    if flag == ATTR_BOTH:
        return bool(hide_prec_alti_9999) or bool(hide_pos_sol_lt0)
    return False


def layer_filtered(
    flag: int,
    is_path: bool,
    is_acces_impossible: bool,
    hide_prec_alti_9999: bool,
    hide_pos_sol_lt0: bool,
    hide_paths: bool,
    hide_acces_impossible: bool,
    is_unpaved: bool = False,
    hide_unpaved_roads: bool = False,
    is_ferry: bool = False,
    hide_ferry: bool = False,
    is_fictif: bool = False,
    hide_fictif: bool = False,
    is_not_en_service: bool = False,
    hide_not_en_service: bool = False,
) -> bool:
    """Apply attribute + NATURE/FICTIF/ETAT + ACCES_VL filters for a layer."""
    if hide_paths and is_path:
        return True
    if hide_unpaved_roads and is_unpaved:
        return True
    if hide_ferry and is_ferry:
        return True
    if hide_fictif and is_fictif:
        return True
    if hide_not_en_service and is_not_en_service:
        return True
    if hide_acces_impossible and is_acces_impossible:
        return True
    return attr_flag_filtered(flag, hide_prec_alti_9999, hide_pos_sol_lt0)

NETWORK_SCENE_NAMES: dict[str, str] = {
    "ROADS": "/networks/roads",
    "RAILROADS": "/networks/railroads",
    "TRANSMISSION_LINES": "/networks/transmission",
}

NETWORK_GUI_LABELS: dict[str, str] = {
    "ROADS": "Roads",
    "RAILROADS": "Railroads",
    "TRANSMISSION_LINES": "Transmission lines",
}

DEFAULT_MAX_SEGMENT_LENGTH_M = 25.0
# BDTOPO often stores unknown altitude as -1000 (or similar sentinels).
INVALID_Z_THRESHOLD_M = -50.0
# Railway LARGEUR is a gauge class, not metres — map to a visual track width.
RAIL_GAUGE_WIDTH_M: dict[str, float] = {
    "Normale": 1.5,
    "Etroite": 1.0,
    "Large": 2.0,
}

# Network centerline coloring: one color per infrastructure type, or one per polyline.
NETWORK_COLOR_MODES = ("type", "polyline")
# Golden-ratio hue step for distinct per-polyline colors.
_POLYLINE_HUE_STEP = 0.618033988749895


def dept_stem_from_lidarhd_dirname(dirname: str) -> str | None:
    """Department stem from a LIDARHD folder name (e.g. D075-2021_LIDARHD → D075-2021)."""
    suffix = "_LIDARHD"
    if not dirname.endswith(suffix):
        return None
    stem = dirname[: -len(suffix)]
    return stem if stem else None


def resolve_network_gpkg(
    networks_root: Path | str,
    roi_dir: Path | str,
    network_type: str,
) -> Path | None:
    """Resolve ``{root}/{TYPE}/{dept}_{TYPE}/{zone}.gpkg`` from a ROI PLY directory."""
    if network_type not in NETWORK_TYPES:
        raise ValueError(
            f"Unknown network type {network_type!r}; expected one of {NETWORK_TYPES}"
        )
    roi_path = Path(roi_dir)
    dept_stem = dept_stem_from_lidarhd_dirname(roi_path.parent.name)
    if dept_stem is None:
        return None
    zone_name = roi_path.name
    gpkg = (
        Path(networks_root)
        / network_type
        / f"{dept_stem}_{network_type}"
        / f"{zone_name}.gpkg"
    )
    return gpkg if gpkg.is_file() else None


def _coords_xyz(geom: LineString) -> np.ndarray:
    """Return (M, 3) XYZ for a LineString (Z=0 if 2D)."""
    coords = np.asarray(geom.coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    if coords.shape[1] >= 3:
        return coords[:, :3].copy()
    z = np.zeros((coords.shape[0], 1), dtype=np.float64)
    return np.concatenate([coords[:, :2], z], axis=1)


def sanitize_polyline_z(
    coords: np.ndarray,
    fallback_z: float,
    invalid_z_threshold_m: float = INVALID_Z_THRESHOLD_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace sentinel / missing Z along a polyline; return coords and imputed mask."""
    if coords.shape[0] == 0:
        return coords, np.empty((0,), dtype=bool)
    out = coords.copy()
    z = out[:, 2]
    valid = np.isfinite(z) & (z >= invalid_z_threshold_m)
    imputed = ~valid
    if np.all(valid):
        return out, imputed
    if not np.any(valid):
        out[:, 2] = float(fallback_z)
        return out, imputed
    idx = np.arange(z.shape[0], dtype=np.float64)
    z_fixed = z.copy()
    z_fixed[~valid] = np.interp(idx[~valid], idx[valid], z[valid])
    out[:, 2] = z_fixed
    return out, imputed


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


def _densify_polyline(
    coords: np.ndarray,
    max_segment_length_m: float,
    imputed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert vertices so consecutive points are at most ``max_segment_length_m`` apart."""
    if coords.shape[0] == 0:
        return coords, np.empty((0,), dtype=bool)
    if imputed is None:
        imputed = np.zeros(coords.shape[0], dtype=bool)
    if coords.shape[0] < 2 or max_segment_length_m <= 0:
        return coords, imputed.astype(bool, copy=False)

    coord_pieces: list[np.ndarray] = [coords[0:1]]
    imputed_pieces: list[np.ndarray] = [imputed[0:1]]
    for i in range(coords.shape[0] - 1):
        p0 = coords[i]
        p1 = coords[i + 1]
        edge_imputed = bool(imputed[i] or imputed[i + 1])
        dist = float(np.linalg.norm(p1 - p0))
        if dist <= max_segment_length_m or dist < 1e-9:
            coord_pieces.append(p1[None, :])
            imputed_pieces.append(np.array([bool(imputed[i + 1])], dtype=bool))
            continue
        n_extra = int(np.ceil(dist / max_segment_length_m)) - 1
        for k in range(1, n_extra + 1):
            t = k / (n_extra + 1)
            coord_pieces.append((p0 + t * (p1 - p0))[None, :])
            imputed_pieces.append(np.array([edge_imputed], dtype=bool))
        coord_pieces.append(p1[None, :])
        imputed_pieces.append(np.array([bool(imputed[i + 1])], dtype=bool))
    return (
        np.concatenate(coord_pieces, axis=0),
        np.concatenate(imputed_pieces, axis=0),
    )


def polyline_to_segments(
    coords: np.ndarray,
    max_segment_length_m: float = DEFAULT_MAX_SEGMENT_LENGTH_M,
    imputed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a polyline (M, 3) into consecutive segments (N, 2, 3)."""
    if coords.shape[0] < 2:
        return (
            np.empty((0, 2, 3), dtype=np.float64),
            np.empty((0,), dtype=bool),
        )
    densified, densified_imputed = _densify_polyline(
        coords, max_segment_length_m, imputed=imputed
    )
    if densified.shape[0] < 2:
        return (
            np.empty((0, 2, 3), dtype=np.float64),
            np.empty((0,), dtype=bool),
        )
    segments = np.stack([densified[:-1], densified[1:]], axis=1)
    seg_imputed = densified_imputed[:-1] | densified_imputed[1:]
    return segments, seg_imputed


def parse_pos_sol(value: object) -> float | None:
    """Parse BDTOPO ``POS_SOL`` to float, or None if unknown."""
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        v = float(value)
        return v if np.isfinite(v) else None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", ""}:
        return None
    try:
        v = float(text.replace(",", "."))
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def feature_attr_flag(prec_alti: object, pos_sol: object) -> int:
    """Map BDTOPO ``PREC_ALTI`` / ``POS_SOL`` to an ``ATTR_*`` highlight code."""
    is_prec_unknown = False
    if prec_alti is not None:
        try:
            prec = float(prec_alti)
            is_prec_unknown = np.isfinite(prec) and prec == PREC_ALTI_UNKNOWN
        except (TypeError, ValueError):
            is_prec_unknown = False
    pos = parse_pos_sol(pos_sol)
    is_below = pos is not None and pos < 0.0
    if is_prec_unknown and is_below:
        return ATTR_BOTH
    if is_prec_unknown:
        return ATTR_PREC_UNKNOWN
    if is_below:
        return ATTR_BELOW_GROUND
    return ATTR_NORMAL


def parse_largeur_m(
    largeur: object,
    network_type: str,
    nb_voies: object = None,
) -> float | None:
    """Parse BDTOPO ``LARGEUR`` into a metric ribbon width, or None."""
    if largeur is None:
        return None
    if isinstance(largeur, (float, int, np.floating, np.integer)):
        if not np.isfinite(float(largeur)) or float(largeur) <= 0:
            return None
        return float(largeur)
    text = str(largeur).strip()
    if not text or text.lower() in {"nan", "none", ""}:
        return None
    # Numeric string (roads sometimes stored as object)
    try:
        value = float(text.replace(",", "."))
        return value if np.isfinite(value) and value > 0 else None
    except ValueError:
        pass
    if network_type == "RAILROADS":
        base = RAIL_GAUGE_WIDTH_M.get(text)
        if base is None:
            return None
        n_tracks = 1
        if nb_voies is not None and str(nb_voies).lower() not in {"nan", "none", ""}:
            try:
                n_tracks = max(int(float(nb_voies)), 1)
            except (TypeError, ValueError):
                n_tracks = 1
        # Approximate multi-track corridor: gauge + ~3.5 m between track axes.
        return base + (n_tracks - 1) * 3.5
    return None


def polyline_to_ribbon_mesh(
    coords: np.ndarray,
    width_m: float,
    imputed: np.ndarray | None = None,
    max_segment_length_m: float = DEFAULT_MAX_SEGMENT_LENGTH_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a horizontal ribbon mesh of metric width along a polyline."""
    empty_v = np.empty((0, 3), dtype=np.float64)
    empty_f = np.empty((0, 3), dtype=np.int32)
    empty_i = np.empty((0,), dtype=bool)
    if coords.shape[0] < 2 or width_m <= 0:
        return empty_v, empty_f, empty_i

    densified, densified_imputed = _densify_polyline(
        coords, max_segment_length_m, imputed=imputed
    )
    if densified.shape[0] < 2:
        return empty_v, empty_f, empty_i

    half = 0.5 * float(width_m)
    vertices: list[np.ndarray] = []
    faces: list[list[int]] = []
    face_imputed: list[bool] = []
    base = 0

    for i in range(densified.shape[0] - 1):
        p0 = densified[i]
        p1 = densified[i + 1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        horiz = float(np.hypot(dx, dy))
        if horiz < 1e-9:
            continue
        nx = -dy / horiz * half
        ny = dx / horiz * half
        # Quad: left0, right0, right1, left1
        v0 = np.array([p0[0] + nx, p0[1] + ny, p0[2]], dtype=np.float64)
        v1 = np.array([p0[0] - nx, p0[1] - ny, p0[2]], dtype=np.float64)
        v2 = np.array([p1[0] - nx, p1[1] - ny, p1[2]], dtype=np.float64)
        v3 = np.array([p1[0] + nx, p1[1] + ny, p1[2]], dtype=np.float64)
        vertices.extend([v0, v1, v2, v3])
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])
        edge_imp = bool(densified_imputed[i] or densified_imputed[i + 1])
        face_imputed.extend([edge_imp, edge_imp])
        base += 4

    if not vertices:
        return empty_v, empty_f, empty_i
    return (
        np.stack(vertices, axis=0),
        np.asarray(faces, dtype=np.int32),
        np.asarray(face_imputed, dtype=bool),
    )


def load_network_line_segments(
    gpkg_path: Path | str,
    max_segment_length_m: float = DEFAULT_MAX_SEGMENT_LENGTH_M,
    fallback_z: float = 0.0,
    invalid_z_threshold_m: float = INVALID_Z_THRESHOLD_M,
    build_width_ribbons: bool = True,
) -> tuple[
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load a GeoPackage into densified line segments and optional width ribbons."""
    path = Path(gpkg_path)
    empty_seg = np.empty((0, 2, 3), dtype=np.float64)
    empty_imp = np.empty((0,), dtype=bool)
    empty_flags = np.empty((0,), dtype=np.uint8)
    empty_bool = np.empty((0,), dtype=bool)
    empty_ids = np.empty((0,), dtype=np.int32)
    empty_v = np.empty((0, 3), dtype=np.float64)
    empty_f = np.empty((0, 3), dtype=np.int32)
    empty_pack = (
        empty_seg,
        0.0,
        empty_imp,
        empty_v,
        empty_f,
        empty_imp,
        empty_flags,
        empty_flags,
        empty_bool,
        empty_bool,
        empty_bool,
        empty_bool,
        empty_ids,
        empty_ids,
        empty_bool,
        empty_bool,
        empty_bool,
        empty_bool,
        empty_bool,
        empty_bool,
        empty_bool,
        empty_bool,
    )
    gdf = gpd.read_file(path)
    if gdf.empty or "geometry" not in gdf.columns:
        return empty_pack

    # Infer network type from parent folder name (ROADS / RAILROADS / …).
    network_type = path.parent.parent.name
    if network_type not in NETWORK_TYPES:
        network_type = path.parent.name.rsplit("_", 1)[-1]
    length_m = float(gdf.geometry.length.sum())
    has_largeur = "LARGEUR" in gdf.columns
    has_nb_voies = "NB_VOIES" in gdf.columns
    has_prec_alti = "PREC_ALTI" in gdf.columns
    has_pos_sol = "POS_SOL" in gdf.columns
    has_nature = "NATURE" in gdf.columns
    has_acces_vl = "ACCES_VL" in gdf.columns
    has_fictif = "FICTIF" in gdf.columns
    has_etat = "ETAT" in gdf.columns

    segment_list: list[np.ndarray] = []
    imputed_list: list[np.ndarray] = []
    seg_flag_list: list[np.ndarray] = []
    seg_path_list: list[np.ndarray] = []
    seg_acces_list: list[np.ndarray] = []
    seg_unpaved_list: list[np.ndarray] = []
    seg_ferry_list: list[np.ndarray] = []
    seg_fictif_list: list[np.ndarray] = []
    seg_not_en_service_list: list[np.ndarray] = []
    seg_poly_list: list[np.ndarray] = []
    ribbon_verts: list[np.ndarray] = []
    ribbon_faces: list[np.ndarray] = []
    ribbon_imp: list[np.ndarray] = []
    ribbon_flag_list: list[np.ndarray] = []
    ribbon_path_list: list[np.ndarray] = []
    ribbon_acces_list: list[np.ndarray] = []
    ribbon_unpaved_list: list[np.ndarray] = []
    ribbon_ferry_list: list[np.ndarray] = []
    ribbon_fictif_list: list[np.ndarray] = []
    ribbon_not_en_service_list: list[np.ndarray] = []
    ribbon_poly_list: list[np.ndarray] = []
    vert_offset = 0
    next_polyline_id = 0

    for idx, geom in enumerate(gdf.geometry):
        prec_alti = gdf["PREC_ALTI"].iloc[idx] if has_prec_alti else None
        pos_sol = gdf["POS_SOL"].iloc[idx] if has_pos_sol else None
        attr_flag = feature_attr_flag(prec_alti, pos_sol)
        nature = gdf["NATURE"].iloc[idx] if has_nature else None
        path_feat = is_path_nature(nature)
        unpaved_feat = is_unpaved_road_nature(nature)
        ferry_feat = is_ferry_nature(nature)
        acces_vl = gdf["ACCES_VL"].iloc[idx] if has_acces_vl else None
        acces_imp = is_acces_vl_impossible(acces_vl)
        fictif_feat = is_fictif_oui(gdf["FICTIF"].iloc[idx] if has_fictif else None)
        not_en_service_feat = is_etat_not_en_service(
            gdf["ETAT"].iloc[idx] if has_etat else None
        )

        width_m = None
        if build_width_ribbons and has_largeur:
            nb = gdf["NB_VOIES"].iloc[idx] if has_nb_voies else None
            width_m = parse_largeur_m(gdf["LARGEUR"].iloc[idx], network_type, nb)

        for line in _iter_linestrings(geom):
            polyline_id = next_polyline_id
            next_polyline_id += 1
            coords, imputed = sanitize_polyline_z(
                _coords_xyz(line),
                fallback_z=fallback_z,
                invalid_z_threshold_m=invalid_z_threshold_m,
            )
            segs, seg_imputed = polyline_to_segments(
                coords, max_segment_length_m, imputed=imputed
            )
            if segs.shape[0] > 0:
                segment_list.append(segs)
                imputed_list.append(seg_imputed)
                seg_flag_list.append(
                    np.full(segs.shape[0], attr_flag, dtype=np.uint8)
                )
                seg_path_list.append(
                    np.full(segs.shape[0], path_feat, dtype=bool)
                )
                seg_acces_list.append(
                    np.full(segs.shape[0], acces_imp, dtype=bool)
                )
                seg_unpaved_list.append(
                    np.full(segs.shape[0], unpaved_feat, dtype=bool)
                )
                seg_ferry_list.append(
                    np.full(segs.shape[0], ferry_feat, dtype=bool)
                )
                seg_fictif_list.append(
                    np.full(segs.shape[0], fictif_feat, dtype=bool)
                )
                seg_not_en_service_list.append(
                    np.full(segs.shape[0], not_en_service_feat, dtype=bool)
                )
                seg_poly_list.append(
                    np.full(segs.shape[0], polyline_id, dtype=np.int32)
                )

            if width_m is not None and width_m > 0:
                verts, faces, face_imp = polyline_to_ribbon_mesh(
                    coords,
                    width_m,
                    imputed=imputed,
                    max_segment_length_m=max_segment_length_m,
                )
                if verts.shape[0] > 0:
                    ribbon_verts.append(verts)
                    ribbon_faces.append(faces + vert_offset)
                    ribbon_imp.append(face_imp)
                    ribbon_flag_list.append(
                        np.full(faces.shape[0], attr_flag, dtype=np.uint8)
                    )
                    ribbon_path_list.append(
                        np.full(faces.shape[0], path_feat, dtype=bool)
                    )
                    ribbon_acces_list.append(
                        np.full(faces.shape[0], acces_imp, dtype=bool)
                    )
                    ribbon_unpaved_list.append(
                        np.full(faces.shape[0], unpaved_feat, dtype=bool)
                    )
                    ribbon_ferry_list.append(
                        np.full(faces.shape[0], ferry_feat, dtype=bool)
                    )
                    ribbon_fictif_list.append(
                        np.full(faces.shape[0], fictif_feat, dtype=bool)
                    )
                    ribbon_not_en_service_list.append(
                        np.full(faces.shape[0], not_en_service_feat, dtype=bool)
                    )
                    ribbon_poly_list.append(
                        np.full(faces.shape[0], polyline_id, dtype=np.int32)
                    )
                    vert_offset += verts.shape[0]

    if not segment_list:
        return (
            empty_seg,
            length_m,
            empty_imp,
            empty_v,
            empty_f,
            empty_imp,
            empty_flags,
            empty_flags,
            empty_bool,
            empty_bool,
            empty_bool,
            empty_bool,
            empty_ids,
            empty_ids,
            empty_bool,
            empty_bool,
            empty_bool,
            empty_bool,
            empty_bool,
            empty_bool,
            empty_bool,
            empty_bool,
        )

    segments = np.concatenate(segment_list, axis=0)
    imputed_z = np.concatenate(imputed_list, axis=0)
    seg_attr_flags = np.concatenate(seg_flag_list, axis=0)
    seg_is_path = np.concatenate(seg_path_list, axis=0)
    seg_acces_impossible = np.concatenate(seg_acces_list, axis=0)
    seg_is_unpaved = np.concatenate(seg_unpaved_list, axis=0)
    seg_is_ferry = np.concatenate(seg_ferry_list, axis=0)
    seg_is_fictif = np.concatenate(seg_fictif_list, axis=0)
    seg_is_not_en_service = np.concatenate(seg_not_en_service_list, axis=0)
    seg_polyline_ids = np.concatenate(seg_poly_list, axis=0)
    if ribbon_verts:
        r_verts = np.concatenate(ribbon_verts, axis=0)
        r_faces = np.concatenate(ribbon_faces, axis=0)
        r_imp = np.concatenate(ribbon_imp, axis=0)
        r_flags = np.concatenate(ribbon_flag_list, axis=0)
        r_path = np.concatenate(ribbon_path_list, axis=0)
        r_acces = np.concatenate(ribbon_acces_list, axis=0)
        r_unpaved = np.concatenate(ribbon_unpaved_list, axis=0)
        r_ferry = np.concatenate(ribbon_ferry_list, axis=0)
        r_fictif = np.concatenate(ribbon_fictif_list, axis=0)
        r_not_en_service = np.concatenate(ribbon_not_en_service_list, axis=0)
        r_poly = np.concatenate(ribbon_poly_list, axis=0)
    else:
        (
            r_verts,
            r_faces,
            r_imp,
            r_flags,
            r_path,
            r_acces,
            r_unpaved,
            r_ferry,
            r_fictif,
            r_not_en_service,
            r_poly,
        ) = (
            empty_v,
            empty_f,
            empty_imp,
            empty_flags,
            empty_bool,
            empty_bool,
            empty_bool,
            empty_bool,
            empty_bool,
            empty_bool,
            empty_ids,
        )
    return (
        segments,
        length_m,
        imputed_z,
        r_verts,
        r_faces,
        r_imp,
        seg_attr_flags,
        r_flags,
        seg_is_path,
        r_path,
        seg_acces_impossible,
        r_acces,
        seg_polyline_ids,
        r_poly,
        seg_is_unpaved,
        r_unpaved,
        seg_is_ferry,
        r_ferry,
        seg_is_fictif,
        r_fictif,
        seg_is_not_en_service,
        r_not_en_service,
    )


def format_network_length_m(length_m: float) -> str:
    """Human-readable length for GUI labels (metres or kilometres)."""
    if not np.isfinite(length_m) or length_m <= 0:
        return ""
    if length_m >= 1000.0:
        return f"{length_m / 1000.0:.1f} km"
    return f"{length_m:.0f} m"


def sample_segments_xy(
    segments: np.ndarray,
    sample_step_m: float = DEFAULT_CORRIDOR_SAMPLE_STEP_M,
) -> np.ndarray:
    """Densely sample XY points along line segments for corridor queries."""
    samples_xy, _seg_ids = sample_segments_xy_with_ids(
        segments, sample_step_m=sample_step_m
    )
    return samples_xy


def sample_segments_xy_with_ids(
    segments: np.ndarray,
    sample_step_m: float = DEFAULT_CORRIDOR_SAMPLE_STEP_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample XY along segments; also return the source segment index per sample."""
    if segments.size == 0 or segments.shape[0] == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.int32),
        )
    segs = np.asarray(segments, dtype=np.float64)
    p0 = segs[:, 0, :2]
    p1 = segs[:, 1, :2]
    step = float(sample_step_m)
    if step <= 0:
        step = DEFAULT_CORRIDOR_SAMPLE_STEP_M
    xy_pieces: list[np.ndarray] = []
    id_pieces: list[np.ndarray] = []
    for i in range(segs.shape[0]):
        a = p0[i]
        b = p1[i]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        dist = float(np.linalg.norm(b - a))
        if dist < 1e-9:
            xy_pieces.append(a[None, :])
            id_pieces.append(np.array([i], dtype=np.int32))
            continue
        n_steps = max(1, int(np.ceil(dist / step)))
        t = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64)
        xy_pieces.append(a[None, :] + t[:, None] * (b - a)[None, :])
        id_pieces.append(np.full(n_steps + 1, i, dtype=np.int32))
    if not xy_pieces:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.int32),
        )
    return np.concatenate(xy_pieces, axis=0), np.concatenate(id_pieces, axis=0)


def nearest_segment_indices(
    points_xyz: np.ndarray,
    segments: np.ndarray,
    radius_m: float,
    sample_step_m: float = DEFAULT_CORRIDOR_SAMPLE_STEP_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest segment index and XY distance per point (``-1`` / ``inf`` if none)."""
    n = int(np.asarray(points_xyz).shape[0]) if points_xyz.size else 0
    seg_idx = np.full(n, -1, dtype=np.int32)
    dist_out = np.full(n, np.inf, dtype=np.float64)
    if n == 0:
        return seg_idx, dist_out
    radius = float(radius_m)
    if radius <= 0 or segments.size == 0 or segments.shape[0] == 0:
        return seg_idx, dist_out
    samples_xy, sample_seg_ids = sample_segments_xy_with_ids(
        segments, sample_step_m=sample_step_m
    )
    if samples_xy.shape[0] == 0:
        return seg_idx, dist_out
    points_xy = np.asarray(points_xyz[:, :2], dtype=np.float64)
    finite = np.isfinite(points_xy).all(axis=1)
    if not np.any(finite):
        return seg_idx, dist_out
    tree = cKDTree(samples_xy)
    dist, nn = tree.query(
        points_xy[finite],
        k=1,
        distance_upper_bound=radius,
        workers=-1,
    )
    hit = np.isfinite(dist) & (dist <= radius)
    if not np.any(hit):
        return seg_idx, dist_out
    finite_idx = np.flatnonzero(finite)
    hit_idx = finite_idx[hit]
    nn_hit = np.asarray(nn, dtype=np.int64)[hit]
    seg_idx[hit_idx] = sample_seg_ids[nn_hit]
    dist_out[hit_idx] = dist[hit]
    return seg_idx, dist_out


def points_near_network_segments(
    points_xyz: np.ndarray,
    segments: np.ndarray,
    radius_m: float,
    sample_step_m: float = DEFAULT_CORRIDOR_SAMPLE_STEP_M,
) -> np.ndarray:
    """Bool mask for points whose XY lies within ``radius_m`` of any segment."""
    seg_idx, _dist = nearest_segment_indices(
        points_xyz,
        segments,
        radius_m=radius_m,
        sample_step_m=sample_step_m,
    )
    return seg_idx >= 0


def segments_hidden_by_filters(
    attr_flags: np.ndarray,
    is_path: np.ndarray,
    is_acces_impossible: np.ndarray,
    hide_prec_alti_9999: bool,
    hide_pos_sol_lt0: bool,
    hide_paths: bool,
    hide_acces_impossible: bool,
    is_unpaved: np.ndarray | None = None,
    hide_unpaved_roads: bool = False,
    is_ferry: np.ndarray | None = None,
    hide_ferry: bool = False,
    is_fictif: np.ndarray | None = None,
    hide_fictif: bool = False,
    is_not_en_service: np.ndarray | None = None,
    hide_not_en_service: bool = False,
) -> np.ndarray:
    """Per-segment bool: True if the segment should be hidden by GUI filters."""
    flags = np.asarray(attr_flags, dtype=np.int32)
    n = flags.shape[0]
    hidden = np.zeros(n, dtype=bool)
    if hide_paths:
        hidden |= np.asarray(is_path, dtype=bool)
    if hide_unpaved_roads and is_unpaved is not None:
        hidden |= np.asarray(is_unpaved, dtype=bool)
    if hide_ferry and is_ferry is not None:
        hidden |= np.asarray(is_ferry, dtype=bool)
    if hide_fictif and is_fictif is not None:
        hidden |= np.asarray(is_fictif, dtype=bool)
    if hide_not_en_service and is_not_en_service is not None:
        hidden |= np.asarray(is_not_en_service, dtype=bool)
    if hide_acces_impossible:
        hidden |= np.asarray(is_acces_impossible, dtype=bool)
    if hide_prec_alti_9999:
        hidden |= (flags == ATTR_PREC_UNKNOWN) | (flags == ATTR_BOTH)
    if hide_pos_sol_lt0:
        hidden |= (flags == ATTR_BELOW_GROUND) | (flags == ATTR_BOTH)
    return hidden


def corridor_overlay_from_assignments(
    assignments: dict[str, dict],
    type_enabled: dict[str, bool],
    color_mode: str,
    hide_prec_alti_9999: bool,
    hide_pos_sol_lt0: bool,
    hide_paths: bool,
    hide_acces_impossible: bool,
    n_points: int,
    hide_unpaved_roads: bool = False,
    hide_ferry: bool = False,
    hide_fictif: bool = False,
    hide_not_en_service: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose active corridor mask + RGB from per-type nearest-segment maps."""
    mask = np.zeros(n_points, dtype=bool)
    colors = np.zeros((n_points, 3), dtype=np.uint8)
    if n_points <= 0 or not assignments:
        return mask, colors
    best_dist = np.full(n_points, np.inf, dtype=np.float64)
    mode = str(color_mode).strip().lower()
    if mode not in NETWORK_COLOR_MODES:
        mode = "type"
    for network_type, data in assignments.items():
        if not type_enabled.get(network_type, True):
            continue
        seg_idx = np.asarray(data["seg_idx"], dtype=np.int32)
        dist = np.asarray(data["dist"], dtype=np.float64)
        attr_flags = np.asarray(data["attr_flags"], dtype=np.int32)
        is_path = np.asarray(data["is_path"], dtype=bool)
        acces = np.asarray(data["acces_impossible"], dtype=bool)
        poly_ids = np.asarray(data["polyline_ids"], dtype=np.int32)

        def _opt_bool(key: str) -> np.ndarray | None:
            value = data.get(key)
            return None if value is None else np.asarray(value, dtype=bool)

        is_unpaved = _opt_bool("is_unpaved")
        is_ferry = _opt_bool("is_ferry")
        is_fictif = _opt_bool("is_fictif")
        is_not_en_service = _opt_bool("is_not_en_service")
        if seg_idx.shape[0] != n_points:
            continue
        seg_hidden = segments_hidden_by_filters(
            attr_flags,
            is_path,
            acces,
            hide_prec_alti_9999=hide_prec_alti_9999,
            hide_pos_sol_lt0=hide_pos_sol_lt0,
            hide_paths=hide_paths,
            hide_acces_impossible=hide_acces_impossible,
            is_unpaved=is_unpaved,
            hide_unpaved_roads=hide_unpaved_roads,
            is_ferry=is_ferry,
            hide_ferry=hide_ferry,
            is_fictif=is_fictif,
            hide_fictif=hide_fictif,
            is_not_en_service=is_not_en_service,
            hide_not_en_service=hide_not_en_service,
        )
        valid = seg_idx >= 0
        if np.any(valid):
            valid_idx = np.flatnonzero(valid)
            valid[valid_idx] = ~seg_hidden[seg_idx[valid_idx]]
        better = valid & (dist < best_dist)
        if not np.any(better):
            continue
        best_dist[better] = dist[better]
        better_idx = np.flatnonzero(better)
        better_seg = seg_idx[better_idx]
        if mode == "polyline":
            # Precompute per-segment polyline colors once for this type.
            seg_colors = np.empty((poly_ids.shape[0], 3), dtype=np.uint8)
            for pid in np.unique(poly_ids):
                seg_colors[poly_ids == pid] = polyline_id_color(int(pid))
            colors[better_idx] = seg_colors[better_seg]
        else:
            type_color = attr_flag_color(0, network_type)
            colors[better_idx] = type_color
    mask = np.isfinite(best_dist)
    return mask, colors


def attr_flag_color(_flag: int, network_type: str) -> np.ndarray:
    """RGB uint8 for a network layer (always the type color)."""
    color = NETWORK_COLORS.get(network_type)
    if color is None:
        return np.array([200, 200, 200], dtype=np.uint8)
    return color


def _hsv_to_rgb_u8(h: float, s: float, v: float) -> np.ndarray:
    """Convert HSV in [0, 1] to RGB uint8."""
    h = h % 1.0
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i % 6
    if i_mod == 0:
        r, g, b = v, t, p
    elif i_mod == 1:
        r, g, b = q, v, p
    elif i_mod == 2:
        r, g, b = p, v, t
    elif i_mod == 3:
        r, g, b = p, q, v
    elif i_mod == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return np.array(
        [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))],
        dtype=np.uint8,
    )


def polyline_id_color(polyline_id: int) -> np.ndarray:
    """Distinct RGB uint8 for a polyline id (stable across sessions)."""
    # Slight S/V jitter so nearby ids are easier to tell apart.
    h = (int(polyline_id) * _POLYLINE_HUE_STEP) % 1.0
    s = 0.55 + 0.35 * (((int(polyline_id) * 3) % 5) / 4.0)
    v = 0.80 + 0.20 * (((int(polyline_id) * 7) % 3) / 2.0)
    return _hsv_to_rgb_u8(h, s, v)


def segment_colors(
    n_segments: int,
    network_type: str,
    attr_flags: np.ndarray | None = None,
    imputed_z: np.ndarray | None = None,
    polyline_ids: np.ndarray | None = None,
    color_mode: str = "type",
) -> np.ndarray:
    """Per-endpoint colors (N, 2, 3) uint8 for viser ``add_line_segments``."""
    _ = (attr_flags, imputed_z)
    if n_segments <= 0:
        return np.empty((0, 2, 3), dtype=np.uint8)
    mode = str(color_mode).strip().lower()
    if mode == "polyline":
        if polyline_ids is None or len(polyline_ids) != n_segments:
            raise ValueError(
                "polyline color_mode requires polyline_ids of length n_segments"
            )
        colors = np.empty((n_segments, 3), dtype=np.uint8)
        # Vectorize via unique ids (typically hundreds, not millions).
        ids = np.asarray(polyline_ids, dtype=np.int32)
        for pid in np.unique(ids):
            colors[ids == pid] = polyline_id_color(int(pid))
        return np.broadcast_to(colors[:, None, :], (n_segments, 2, 3)).copy()

    color = NETWORK_COLORS.get(network_type)
    if color is None:
        color = np.array([200, 200, 200], dtype=np.uint8)
    return np.broadcast_to(color, (n_segments, 2, 3)).copy()


def load_roi_networks(
    networks_root: Path | str,
    roi_dir: Path | str,
    network_types: list[str] | tuple[str, ...] = NETWORK_TYPES,
    max_segment_length_m: float = DEFAULT_MAX_SEGMENT_LENGTH_M,
    fallback_z: float = 0.0,
    build_width_ribbons: bool = True,
) -> dict[str, tuple]:
    """Load all requested network types for a ROI."""
    out: dict[str, tuple] = {}
    root = Path(networks_root)
    for network_type in network_types:
        gpkg = resolve_network_gpkg(root, roi_dir, network_type)
        if gpkg is None:
            continue
        out[network_type] = load_network_line_segments(
            gpkg,
            max_segment_length_m=max_segment_length_m,
            fallback_z=fallback_z,
            build_width_ribbons=build_width_ribbons,
        )
    return out
