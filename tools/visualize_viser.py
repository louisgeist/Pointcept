"""Browse MALiBU3D tiles in 3D with viser: the inputs and every label, one layer at a time.

Point it at one tile folder or any folder above (an ROI, a region): every tile found under it is shown,
placed at its real Lambert-93 position. For the toy region:

    pip install viser
    python tools/visualize_viser.py data/malibu3d/test/D075-2021_LIDARHD/UU-S1-4

then open http://localhost:8080 (over SSH, forward the port first: ``ssh -N -L 8080:127.0.0.1:8080 <host>``).
The "Layer" menu switches the colouring (RGB, intensity, semantic, elevation, forest, roads, the four
natural-habitat axes) and shows its legend. Colours are the paper's, from ``visualize_viser_palette.json``
next to this script. The forest and road rasters (1 m) are read at each point; roads need
``rasterize_network.py`` first, otherwise the layer is Void.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import viser

PALETTE = json.loads((Path(__file__).with_name("visualize_viser_palette.json")).read_text())
# natural_habitat.npy columns, in order (labels.json: natural_habitat.columns), and their display names
NATHAB_AXES = [("nh_habitat", "Habitat type"), ("nh_moisture", "Moisture regime"),
               ("nh_soil", "Soil chemistry"), ("nh_climate", "Bioclimatic zone")]
VOID = 255  # label of the points a task does not cover on these tiles


def find_up(start: Path, name: str) -> Path | None:
    for d in [start, *start.parents]:
        if (d / name).is_file():
            return d / name
    return None


def hex_to_rgb(colors: list[str]) -> np.ndarray:
    return np.array([[int(c.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)] for c in colors], np.uint8)


def raster_origins(root: Path) -> dict:
    """tile_id -> (x, y) of the south-west corner of its 1 m rasters, from tiles.csv."""
    path = find_up(root, "tiles.csv")
    if path is None:
        return {}
    with open(path, newline="") as f:
        return {r["tile_id"]: (float(r["forest_origin_x"]), float(r["forest_origin_y"]))
                for r in csv.DictReader(f) if r.get("forest_origin_x")}


def read_raster(tile: Path, name: str, abs_xy: np.ndarray, origin) -> np.ndarray | None:
    """Value of the 1 m raster ``name``.npy (channel 0, rows south-up) under each point; None if absent."""
    path = tile / f"{name}.npy"
    meta = json.loads((tile / "meta.json").read_text()).get(name, {}) if (tile / "meta.json").is_file() else {}
    if "origin_x" in meta:
        origin = (meta["origin_x"], meta["origin_y"])
    if origin is None:
        return None
    if not path.is_file():
        # rasterize_network.py writes no file for a tile without any road, only its meta.json entry
        return np.zeros(len(abs_xy), np.uint8) if meta.get("empty") else None
    raster = np.load(path)[0]
    ix = np.floor(abs_xy[:, 0] - origin[0]).astype(int)
    iy = np.floor(abs_xy[:, 1] - origin[1]).astype(int)
    inside = (ix >= 0) & (iy >= 0) & (ix < raster.shape[1]) & (iy < raster.shape[0])
    out = np.full(len(abs_xy), VOID, np.uint8)
    out[inside] = raster[iy[inside], ix[inside]]
    return out


def load_tiles(root: Path, voxel: float) -> dict:
    tiles = sorted(p.parent for p in root.rglob("coord.npy"))
    if not tiles:
        raise SystemExit(f"no tile (coord.npy) under {root}")
    origins = raster_origins(root)
    parts = []
    for tile in tiles:
        xyz = np.load(tile / "coord.npy").astype(np.float64) + np.load(tile / "coord_translation.npy")
        n = len(xyz)
        part = {
            "xyz": xyz,
            "rgb": np.load(tile / "color.npy"),
            "strength": np.load(tile / "strength.npy"),
            "segment": np.load(tile / "segment.npy"),
        }
        elevation = tile / "elevation.npy"
        part["elevation"] = np.load(elevation) if elevation.is_file() else np.full(n, np.nan, np.float32)
        nathab = tile / "natural_habitat.npy"
        part["natural_habitat"] = np.load(nathab) if nathab.is_file() else np.full((n, 4), VOID, np.uint8)
        for name in ("forest_2d", "network"):
            values = read_raster(tile, name, xyz[:, :2], origins.get(tile.name))
            part[name] = values if values is not None else np.full(n, VOID, np.uint8)
        parts.append(part)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    print(f"{len(tiles)} tiles, {len(data['xyz']):,} points")

    if voxel > 0:  # keep one point per voxel
        vox = np.floor((data["xyz"] - data["xyz"].min(0)) / voxel).astype(np.int64)
        dims = vox.max(0) + 1
        _, keep = np.unique((vox[:, 0] * dims[1] + vox[:, 1]) * dims[2] + vox[:, 2], return_index=True)
        data = {k: v[keep] for k, v in data.items()}
        print(f"voxel {voxel} m -> {len(keep):,} points")
    return data


def legend(names: list[str], colors: list[str]) -> str:
    # viser renders markdown as MDX: no raw HTML; `#rrggbb` in inline code is drawn as a colour chip
    return "  \n".join(f"`{c}` {n}" for n, c in zip(names, colors))


def categorical(values: np.ndarray, palette: dict) -> tuple[np.ndarray, str]:
    """Colours of class ids (VOID and unknown ids -> the Void colour) and the legend of the classes present."""
    colors = palette["colors"]  # the last class is Void
    lut = hex_to_rgb(colors + [colors[-1]] * (256 - len(colors)))
    present = sorted({min(int(i), len(colors) - 1) for i in np.unique(values)})
    return lut[values], legend([palette["names"][i] for i in present], [colors[i] for i in present])


def ramp(t: np.ndarray, palette: dict) -> np.ndarray:
    """t in [0, 1] (NaN allowed) -> uint8 RGB along the palette's piecewise-linear 64-stop ramp."""
    stops = np.asarray(palette["stops"], np.float32)
    pos = np.clip(np.nan_to_num(t), 0, 1) ** palette["gamma"] * (len(stops) - 1)
    i0 = np.minimum(pos.astype(int), len(stops) - 2)
    f = (pos - i0)[:, None]
    out = (stops[i0] * (1 - f) + stops[i0 + 1] * f).round().astype(np.uint8)
    out[np.isnan(t)] = hex_to_rgb([palette["nan_color"]])[0]
    return out


def build_layers(data: dict) -> dict:
    """Layer name -> (colours (N, 3) uint8, legend markdown)."""
    elev, inten, roads = PALETTE["elevation"], PALETTE["intensity"], PALETTE["roads"]
    lo, hi = np.percentile(data["strength"], [inten["percentile_low"], inten["percentile_high"]])
    road_rgb = (data["rgb"] * roads["off_road_rgb_dim"]).astype(np.uint8)  # roads over a dimmed RGB
    road_rgb[data["network"] == 1] = hex_to_rgb([roads["colors"][1]])[0]
    stop_hex = lambda p, i: "#%02x%02x%02x" % tuple(p["stops"][i])
    layers = {
        "RGB": (data["rgb"], ""),
        "LiDAR intensity": (ramp((data["strength"] - lo) / (hi - lo), inten),
                            legend(["low", "high"], [stop_hex(inten, 0), stop_hex(inten, -1)])),
        "Semantic": categorical(data["segment"], PALETTE["semantic"]),
        "Elevation": (ramp((data["elevation"] - elev["vmin"]) / (elev["vmax"] - elev["vmin"]), elev),
                      legend([f"{elev['vmin']:.0f} m above ground", f"{elev['vmax']:.0f} m and more", "no DTM"],
                             [stop_hex(elev, 0), stop_hex(elev, -1), elev["nan_color"]])),
        "Road network": (road_rgb, legend(["Road"], [roads["colors"][1]])),
        "Forest": categorical(data["forest_2d"], PALETTE["forest"]),
    }
    for j, (key, label) in enumerate(NATHAB_AXES):
        layers[f"Natural habitat: {label}"] = categorical(data["natural_habitat"][:, j], PALETTE[key])
    return layers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="a tile folder, or any folder above (ROI, region, split)")
    ap.add_argument("--voxel", type=float, default=0.3, help="subsampling voxel (m), 0 keeps every point")
    ap.add_argument("--point-size", type=float, default=0.3)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    data = load_tiles(args.path.resolve(), args.voxel)
    layers = build_layers(data)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    centre = (data["xyz"].min(0) + data["xyz"].max(0)) / 2  # Lambert-93 is too large for float32
    first = next(iter(layers))
    cloud = server.scene.add_point_cloud("/tiles", points=(data["xyz"] - centre).astype(np.float32),
                                         colors=layers[first][0], point_size=args.point_size)

    layer = server.gui.add_dropdown("Layer", options=tuple(layers), initial_value=first)
    legend_md = server.gui.add_markdown(layers[first][1])
    size = server.gui.add_slider("Point size (m)", min=0.05, max=2.0, step=0.05, initial_value=args.point_size)

    @layer.on_update
    def _(_) -> None:
        cloud.colors, legend_md.content = layers[layer.value]

    @size.on_update
    def _(_) -> None:
        cloud.point_size = size.value

    print(f"open http://localhost:{args.port}")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
