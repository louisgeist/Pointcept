#!/usr/bin/env python3
"""Standalone viser viewer for a Malibu3D supplementary sample tile.

Requires numpy, plyfile, and viser. Road-network corridor display also needs
geopandas, scipy, and shapely.

Example:
  python scripts/visualize/visualize_suppmat_zone_viser.py --zone-dir .
  python scripts/visualize/visualize_suppmat_zone_viser.py --zone-dir suppmat_zones/D075_UU-S1-4_3-3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import viser
from plyfile import PlyData

try:
    import suppmat_network_utils as net_vis
except ImportError:  # pragma: no cover - allow running from repo scripts/
    _SCRIPT_DIR = Path(__file__).resolve().parent
    if str(_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPT_DIR))
    import suppmat_network_utils as net_vis

DISPLAY_MODES = {
    "RGB": "RGB",
    "Semantic": "Semantic",
    "Natural habitat": "Natural habitat",
    "Forest": "Forest",
    "Elevation": "Elevation",
    "Strength": "Strength",
    "Network corridor": "Network corridor",
}

_ELEVATION_COLOR_STOPS = np.array(
    [
        [49, 54, 149],
        [69, 117, 180],
        [116, 173, 209],
        [171, 217, 233],
        [224, 243, 248],
        [254, 224, 144],
        [253, 174, 97],
        [244, 109, 67],
        [215, 48, 39],
        [165, 0, 38],
    ],
    dtype=np.float64,
)

DEFAULT_CORRIDOR_RADIUS_M = 2.5
DEFAULT_CORRIDOR_ALPHA = 0.7


def hex_to_rgb(hex_color: str) -> np.ndarray:
    value = str(hex_color).strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    return np.array(
        [int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)],
        dtype=np.uint8,
    )


def build_discrete_palette(palette_cfg: dict) -> tuple[list[str], np.ndarray, np.ndarray, str]:
    names = list(palette_cfg["names"])
    colors_hex = list(palette_cfg["colors"])
    palette = np.stack([hex_to_rgb(c) for c in colors_hex], axis=0)
    unknown_rgb = hex_to_rgb(str(palette_cfg.get("unknown_color", "#808080")))
    unknown_name = str(palette_cfg.get("unknown_name", "Unknown"))
    return names, palette, unknown_rgb, unknown_name


def semantic_to_colors(
    labels: np.ndarray,
    palette: np.ndarray,
    unknown_rgb: np.ndarray,
) -> np.ndarray:
    labels = labels.astype(np.int64, copy=False)
    colors = np.tile(unknown_rgb, (labels.shape[0], 1))
    valid = (labels >= 0) & (labels < len(palette))
    if np.any(valid):
        colors[valid] = palette[labels[valid]]
    return colors


def scalar_to_colors(
    values: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    nan_rgb: np.ndarray | None = None,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if nan_rgb is None:
        nan_rgb = np.array([128, 128, 128], dtype=np.uint8)
    colors = np.tile(nan_rgb, (values.shape[0], 1)).astype(np.uint8, copy=False)
    valid = np.isfinite(values)
    if not np.any(valid):
        return colors, float("nan"), float("nan")
    finite = values[valid]
    if vmin is None:
        vmin = float(np.percentile(finite, percentile_low))
    if vmax is None:
        vmax = float(np.percentile(finite, percentile_high))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    normalized = np.zeros_like(values, dtype=np.float64)
    normalized[valid] = np.clip((values[valid] - vmin) / (vmax - vmin), 0.0, 1.0)
    stop_count = _ELEVATION_COLOR_STOPS.shape[0]
    scaled = normalized * (stop_count - 1)
    lower = np.floor(scaled).astype(np.int64)
    lower = np.clip(lower, 0, stop_count - 1)
    upper = np.clip(lower + 1, 0, stop_count - 1)
    blend = (scaled - lower)[..., np.newaxis]
    lower_colors = _ELEVATION_COLOR_STOPS[lower]
    upper_colors = _ELEVATION_COLOR_STOPS[upper]
    rgb = np.rint((1.0 - blend) * lower_colors + blend * upper_colors).astype(np.uint8)
    colors[valid] = rgb[valid]
    return colors, vmin, vmax


def apply_alpha_overlay(
    base_colors: np.ndarray,
    overlay_mask: np.ndarray,
    overlay_rgb: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if not np.any(overlay_mask):
        return base_colors
    out = base_colors.astype(np.float64, copy=True)
    err = np.asarray(overlay_rgb, dtype=np.float64)
    if err.ndim == 1:
        err = err.reshape(1, 3)
    a = float(np.clip(alpha, 0.0, 1.0))
    out[overlay_mask] = (1.0 - a) * out[overlay_mask] + a * err[overlay_mask]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def label_name(class_id: int, names: list[str], unknown_name: str) -> str:
    if 0 <= class_id < len(names):
        return names[class_id]
    return unknown_name


def read_ply_attributes(ply_path: Path) -> dict[str, np.ndarray]:
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"].data
    out: dict[str, np.ndarray] = {}
    for name in vertex.dtype.names or ():
        arr = np.asarray(vertex[name])
        if arr.dtype.kind in ("S", "U", "O"):
            continue
        out[name] = arr
    return out


def load_npy_sidecar(zone_dir: Path, filename: str, n_points: int) -> np.ndarray:
    path = zone_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing sidecar: {path}")
    arr = np.load(path).reshape(-1)
    if arr.shape[0] != n_points:
        raise ValueError(
            f"Length mismatch for {filename} ({arr.shape[0]}) vs PLY ({n_points})"
        )
    return arr


def discover_zone_files(zone_dir: Path) -> tuple[Path, dict]:
    meta_path = zone_dir / "zone_meta.json"
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        ply_name = str(meta.get("ply_file", ""))
        ply_path = zone_dir / ply_name
        if ply_path.is_file():
            return ply_path, meta
    ply_candidates = sorted(zone_dir.glob("*.ply"))
    if not ply_candidates:
        raise FileNotFoundError(f"No .ply file found in {zone_dir}")
    return ply_candidates[0], {}


def load_zone_bundle(zone_dir: Path, max_points: int, seed: int) -> dict:
    zone_dir = zone_dir.resolve()
    ply_path, meta = discover_zone_files(zone_dir)
    palettes_path = zone_dir / "palettes.json"
    if not palettes_path.is_file():
        raise FileNotFoundError(f"Missing palettes.json in {zone_dir}")
    with palettes_path.open("r", encoding="utf-8") as f:
        palettes = json.load(f)

    attributes = read_ply_attributes(ply_path)
    for axis in ("x", "y", "z"):
        if axis not in attributes:
            raise KeyError(f"Missing '{axis}' in PLY: {ply_path}")
    points = np.stack(
        [attributes["x"], attributes["y"], attributes["z"]], axis=1
    ).astype(np.float64)
    n_points = points.shape[0]

    if all(k in attributes for k in ("red", "green", "blue")):
        colors_rgb = np.stack(
            [attributes["red"], attributes["green"], attributes["blue"]], axis=1
        ).astype(np.uint8)
    else:
        colors_rgb = np.full((n_points, 3), 128, dtype=np.uint8)

    semantic = None
    if "semantic" in attributes:
        semantic = attributes["semantic"].astype(np.int32, copy=False)
    else:
        semantic = load_npy_sidecar(zone_dir, "segment.npy", n_points).astype(np.int32)

    elevation = load_npy_sidecar(zone_dir, "elevation.npy", n_points).astype(np.float32)
    natural_habitat = load_npy_sidecar(
        zone_dir, "natural_habitat.npy", n_points
    ).astype(np.int32)
    forest = load_npy_sidecar(zone_dir, "forest.npy", n_points).astype(np.int32)
    strength = load_npy_sidecar(zone_dir, "strength.npy", n_points).astype(np.float32)

    if max_points > 0 and n_points > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n_points, size=max_points, replace=False)
        points = points[keep]
        colors_rgb = colors_rgb[keep]
        semantic = semantic[keep]
        elevation = elevation[keep]
        natural_habitat = natural_habitat[keep]
        forest = forest[keep]
        strength = strength[keep]

    network_graphs = list(meta.get("network_graphs", []))
    if not network_graphs:
        network_graphs = sorted(
            p.name
            for p in zone_dir.glob("*.gpkg")
            if "_ROADS" in p.name or "_RAILROADS" in p.name or "_TRANSMISSION_LINES" in p.name
        )

    return {
        "zone_dir": zone_dir,
        "ply_path": ply_path,
        "meta": meta,
        "palettes": palettes,
        "points": points,
        "colors_rgb": colors_rgb,
        "semantic": semantic,
        "elevation": elevation,
        "natural_habitat": natural_habitat,
        "forest": forest,
        "strength": strength,
        "network_graphs": network_graphs,
    }


def build_display_modes(bundle: dict) -> tuple[list[str], dict[str, np.ndarray], dict[str, tuple[list[str], str]]]:
    palettes = bundle["palettes"]
    color_by_mode: dict[str, np.ndarray] = {
        DISPLAY_MODES["RGB"]: bundle["colors_rgb"],
    }
    palette_info: dict[str, tuple[list[str], str]] = {}

    sem_names, sem_palette, sem_unknown_rgb, sem_unknown_name = build_discrete_palette(
        palettes["semantic"]
    )
    color_by_mode[DISPLAY_MODES["Semantic"]] = semantic_to_colors(
        bundle["semantic"], sem_palette, sem_unknown_rgb
    )
    palette_info[DISPLAY_MODES["Semantic"]] = (sem_names, sem_unknown_name)

    for mode_label, palette_key, labels in (
        (DISPLAY_MODES["Natural habitat"], "natural_habitat", bundle["natural_habitat"]),
        (DISPLAY_MODES["Forest"], "forest", bundle["forest"]),
    ):
        names, palette, unknown_rgb, unknown_name = build_discrete_palette(
            palettes[palette_key]
        )
        color_by_mode[mode_label] = semantic_to_colors(labels, palette, unknown_rgb)
        palette_info[mode_label] = (names, unknown_name)

    elev_cfg = palettes.get("elevation", {})
    elev_colors, _, _ = scalar_to_colors(
        bundle["elevation"],
        percentile_low=float(elev_cfg.get("percentile_low", 2.0)),
        percentile_high=float(elev_cfg.get("percentile_high", 98.0)),
    )
    color_by_mode[DISPLAY_MODES["Elevation"]] = elev_colors

    strength_colors, _, _ = scalar_to_colors(bundle["strength"])
    color_by_mode[DISPLAY_MODES["Strength"]] = strength_colors

    display_modes = [
        m
        for m in (
            DISPLAY_MODES["RGB"],
            DISPLAY_MODES["Semantic"],
            DISPLAY_MODES["Natural habitat"],
            DISPLAY_MODES["Forest"],
            DISPLAY_MODES["Elevation"],
            DISPLAY_MODES["Strength"],
        )
        if m in color_by_mode
    ]
    return display_modes, color_by_mode, palette_info


def build_network_corridor_overlay(
    points: np.ndarray,
    zone_dir: Path,
    network_graphs: list[str],
    *,
    corridor_radius_m: float,
    fallback_z: float,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if not network_graphs:
        return None, None
    assignments: dict[str, dict] = {}
    type_enabled: dict[str, bool] = {}
    for graph_name in network_graphs:
        gpkg_path = zone_dir / graph_name
        if not gpkg_path.is_file():
            print(f"Warning: network graph not found: {gpkg_path}")
            continue
        network_type = "ROADS"
        if "_RAILROADS" in graph_name or graph_name.endswith("_RAILROADS.gpkg"):
            network_type = "RAILROADS"
        elif "_TRANSMISSION_LINES" in graph_name:
            network_type = "TRANSMISSION_LINES"
        elif "_ROADS" in graph_name:
            network_type = "ROADS"
        (
            segments,
            _length_m,
            _imputed_z,
            _ribbon_verts,
            _ribbon_faces,
            _ribbon_face_imputed,
            seg_attr_flags,
            _ribbon_face_attr_flags,
            seg_is_path,
            _ribbon_face_is_path,
            seg_acces_impossible,
            _ribbon_face_acces_impossible,
            seg_polyline_ids,
            _ribbon_face_polyline_ids,
            seg_is_unpaved,
            _ribbon_face_is_unpaved,
            seg_is_ferry,
            _ribbon_face_is_ferry,
            seg_is_fictif,
            _ribbon_face_is_fictif,
            seg_is_not_en_service,
            _ribbon_face_is_not_en_service,
        ) = net_vis.load_network_line_segments(
            gpkg_path,
            fallback_z=float(fallback_z),
            build_width_ribbons=False,
        )
        if segments.shape[0] == 0:
            continue
        seg_idx, seg_dist = net_vis.nearest_segment_indices(
            points,
            segments,
            radius_m=corridor_radius_m,
        )
        assignments[network_type] = {
            "seg_idx": seg_idx,
            "dist": seg_dist,
            "attr_flags": np.asarray(seg_attr_flags, dtype=np.int32),
            "is_path": np.asarray(seg_is_path, dtype=bool),
            "acces_impossible": np.asarray(seg_acces_impossible, dtype=bool),
            "is_unpaved": np.asarray(seg_is_unpaved, dtype=bool),
            "is_ferry": np.asarray(seg_is_ferry, dtype=bool),
            "is_fictif": np.asarray(seg_is_fictif, dtype=bool),
            "is_not_en_service": np.asarray(seg_is_not_en_service, dtype=bool),
            "polyline_ids": np.asarray(seg_polyline_ids, dtype=np.int32),
        }
        type_enabled[network_type] = True
    mask, rgb = net_vis.corridor_overlay_from_assignments(
        assignments,
        type_enabled=type_enabled,
        color_mode="type",
        hide_prec_alti_9999=True,
        hide_pos_sol_lt0=True,
        hide_paths=False,
        hide_acces_impossible=False,
        n_points=int(points.shape[0]),
    )
    return mask, rgb


def nearest_point_index(
    points: np.ndarray,
    ray_origin: tuple[float, float, float],
    ray_direction: tuple[float, float, float],
) -> tuple[int, float]:
    origin = np.asarray(ray_origin, dtype=np.float64)
    direction = np.asarray(ray_direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return -1, float("inf")
    direction = direction / norm
    vectors = points.astype(np.float64) - origin
    t = vectors @ direction
    t = np.maximum(t, 0.0)
    closest = origin + np.outer(t, direction)
    dist_sq = np.sum((points.astype(np.float64) - closest) ** 2, axis=1)
    idx = int(np.argmin(dist_sq))
    return idx, float(np.sqrt(dist_sq[idx]))


def fit_camera(server: viser.ViserServer, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    extent = float(np.linalg.norm(points - center, axis=1).max())
    if extent < 1e-6:
        extent = 1.0
    look_at = center
    direction = np.array([1.0, -1.0, 0.85], dtype=np.float64)
    direction /= float(np.linalg.norm(direction))
    position = center + direction * (2.2 * extent)
    server.initial_camera.look_at = look_at
    server.initial_camera.position = position
    cam_dist = float(np.linalg.norm(position - look_at))
    server.initial_camera.far = float(max(1000.0, 20.0 * cam_dist))
    server.initial_camera.near = float(max(0.01, cam_dist / 1000.0))
    return look_at, position


def run_viewer(
    zone_dir: Path,
    *,
    max_points: int,
    seed: int,
    point_size: float,
    pick_radius: float,
    initial_mode: str,
    corridor_radius_m: float,
    corridor_alpha: float,
) -> None:
    bundle = load_zone_bundle(zone_dir, max_points=max_points, seed=seed)
    points = bundle["points"]
    display_modes, color_by_mode, palette_info = build_display_modes(bundle)

    network_mask: np.ndarray | None = None
    network_rgb: np.ndarray | None = None
    if bundle["network_graphs"]:
        network_mask, network_rgb = build_network_corridor_overlay(
            points,
            bundle["zone_dir"],
            bundle["network_graphs"],
            corridor_radius_m=corridor_radius_m,
            fallback_z=float(points[:, 2].mean()),
        )
        if network_mask is not None and np.any(network_mask):
            color_by_mode[DISPLAY_MODES["Network corridor"]] = apply_alpha_overlay(
                bundle["colors_rgb"],
                network_mask,
                network_rgb,
                corridor_alpha,
            )
            display_modes.append(DISPLAY_MODES["Network corridor"])

    if initial_mode not in display_modes:
        initial_mode = display_modes[0]

    origin = points.mean(axis=0)
    display_points = (points - origin).astype(np.float32)

    def colors_for_mode(mode: str) -> np.ndarray:
        return color_by_mode[mode]

    server = viser.ViserServer()
    look_at, camera_position = fit_camera(server, display_points)
    cloud_handle = server.scene.add_point_cloud(
        name="/suppmat_zone",
        points=display_points,
        colors=colors_for_mode(initial_mode),
        point_size=float(point_size),
    )
    pick_marker: viser.IcosphereHandle | None = None

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        nonlocal pick_marker
        client.gui.add_markdown(
            "**Navigation:** left-drag orbit, right-drag pan, scroll zoom. "
            "**Inspect:** Shift+click a point."
        )
        info_md = client.gui.add_markdown("No point selected.")
        reset_btn = client.gui.add_button("Reset view")
        mode_dd = client.gui.add_dropdown(
            "Display mode",
            tuple(display_modes),
            initial_value=initial_mode,
        )
        point_size_slider = client.gui.add_slider(
            "Point size",
            min=0.01,
            max=0.2,
            step=0.01,
            initial_value=float(point_size),
        )

        @mode_dd.on_update
        def _(_event: viser.GuiEvent) -> None:
            cloud_handle.colors = colors_for_mode(mode_dd.value)

        @point_size_slider.on_update
        def _(_event: viser.GuiEvent) -> None:
            cloud_handle.point_size = float(point_size_slider.value)

        @reset_btn.on_click
        def _(_event: viser.GuiEvent) -> None:
            with client.atomic():
                client.camera.look_at = look_at
                client.camera.position = camera_position

        @client.scene.on_click(modifier="shift")
        def _(event: viser.SceneClickEvent) -> None:
            nonlocal pick_marker
            idx, dist = nearest_point_index(
                display_points, event.ray_origin, event.ray_direction
            )
            if idx < 0 or dist > float(pick_radius):
                info_md.content = "No point nearby."
                return
            lines = [
                f"XYZ ({points[idx, 0]:.2f}, {points[idx, 1]:.2f}, {points[idx, 2]:.2f})"
            ]
            r, g, b = bundle["colors_rgb"][idx]
            lines.append(f"RGB ({int(r)}, {int(g)}, {int(b)})")
            sid = int(bundle["semantic"][idx])
            names, unknown = palette_info.get(DISPLAY_MODES["Semantic"], ([], "Unknown"))
            lines.append(f"Semantic: {sid} ({label_name(sid, names, unknown)})")
            for mode_label, arr, palette_key in (
                (DISPLAY_MODES["Natural habitat"], bundle["natural_habitat"], "natural_habitat"),
                (DISPLAY_MODES["Forest"], bundle["forest"], "forest"),
            ):
                lid = int(arr[idx])
                p_names, p_unknown = palette_info.get(mode_label, ([], "Unknown"))
                lines.append(f"{mode_label}: {lid} ({label_name(lid, p_names, p_unknown)})")
            ev = float(bundle["elevation"][idx])
            lines.append(
                f"Elevation: {ev:.2f} m" if np.isfinite(ev) else "Elevation: NaN"
            )
            sv = float(bundle["strength"][idx])
            lines.append(
                f"Strength: {sv:.4f}" if np.isfinite(sv) else "Strength: NaN"
            )
            info_md.content = "\n\n".join(lines)
            if pick_marker is not None:
                pick_marker.remove()
            pick_marker = server.scene.add_icosphere(
                name="/pick_marker",
                radius=float(point_size) * 2.0,
                color=(255, 220, 0),
                position=display_points[idx],
            )

    print(f"Loaded {points.shape[0]:,} points from {bundle['ply_path']}")
    print(f"Display modes: {', '.join(display_modes)}")
    print("Open http://localhost:8080 in your browser.")
    while True:
        time.sleep(1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zone-dir",
        type=Path,
        default=Path("."),
        help="Folder with zone_meta.json, PLY, npy sidecars, palettes.json",
    )
    parser.add_argument("--max-points", type=int, default=0, help="0 = keep all points")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point-size", type=float, default=0.05)
    parser.add_argument("--pick-radius", type=float, default=0.5)
    parser.add_argument("--initial-mode", type=str, default="RGB")
    parser.add_argument("--corridor-radius-m", type=float, default=DEFAULT_CORRIDOR_RADIUS_M)
    parser.add_argument("--corridor-alpha", type=float, default=DEFAULT_CORRIDOR_ALPHA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_viewer(
        args.zone_dir,
        max_points=int(args.max_points),
        seed=int(args.seed),
        point_size=float(args.point_size),
        pick_radius=float(args.pick_radius),
        initial_mode=str(args.initial_mode),
        corridor_radius_m=float(args.corridor_radius_m),
        corridor_alpha=float(args.corridor_alpha),
    )


if __name__ == "__main__":
    main()
