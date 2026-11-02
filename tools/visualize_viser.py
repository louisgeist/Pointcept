"""Browse MALiBU3D tiles in 3D with viser: the inputs and every label, one layer at a time.

Point it at one tile folder or any folder above (an ROI, a region): every tile found under it is shown,
placed at its real Lambert-93 position. For the toy region:

    pip install viser
    python tools/visualize_viser.py data/malibu3d/test/D075-2021_LIDARHD/UU-S1-4

then open http://localhost:8080 (over SSH, forward the port first: ``ssh -L 8080:localhost:8080 <host>``).
The "Layer" menu switches the colouring (RGB, intensity, semantic, elevation, forest, roads, the four
natural-habitat axes) and shows its legend. Colours and class names come from ``palettes.json``, found in
the folder or one of its parents (the dataset root after ``unzip_malibu3d.py``). The forest and road
rasters (1 m) are read at each point; roads need ``rasterize_network.py`` first, otherwise the layer is grey.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import viser
from matplotlib import colormaps

GREY = "#808080"
ROAD_COLOR = "#e41a1c"
ELEVATION_MAX_M = 30.0  # colour scale of the elevation layer: 0 m (ground) .. 30 m


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
    out = np.zeros(len(abs_xy), np.uint8)
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
        part["natural_habitat"] = np.load(nathab) if nathab.is_file() else np.full((n, 4), 255, np.uint8)
        for name in ("forest_2d", "network"):
            values = read_raster(tile, name, xyz[:, :2], origins.get(tile.name))
            part[name] = values if values is not None else np.full(n, 255, np.uint8)
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


def categorical(values: np.ndarray, palette: dict) -> tuple[np.ndarray, str]:
    """Colours of class ids and the markdown legend of the classes present."""
    lut = hex_to_rgb(palette["colors"] + [palette.get("unknown_color", GREY)] * (256 - len(palette["colors"])))
    names = palette["names"]
    legend = "<br>".join(f'<span style="color:{palette["colors"][i]}">&#9632;</span> {names[i]}'
                         for i in np.unique(values) if i < len(names))
    return lut[values], legend or "no label on these tiles"


def ramp(values: np.ndarray, vmax: float, cmap: str) -> np.ndarray:
    rgb = colormaps[cmap](np.clip(np.nan_to_num(values) / vmax, 0, 1))[:, :3]
    rgb[np.isnan(values)] = 0.5  # no value: grey
    return (rgb * 255).astype(np.uint8)


def build_layers(data: dict, palettes: dict, nathab_columns: list[str]) -> dict:
    """Layer name -> (colours (N, 3) uint8, legend markdown)."""
    gray = (np.clip(data["strength"], 0, 1) * 255).astype(np.uint8)
    roads = {"names": ["No road", "Road"], "colors": ["#d9d9d9", ROAD_COLOR]}
    layers = {
        "RGB": (data["rgb"], ""),
        "Intensity": (np.repeat(gray[:, None], 3, 1), "LiDAR intensity, dark = low"),
        "Semantic": categorical(data["segment"], palettes["segment"]),
        "Elevation": (ramp(data["elevation"], ELEVATION_MAX_M, "viridis"),
                      f"height above ground, 0 m (dark) .. {ELEVATION_MAX_M:.0f} m (yellow); grey = no DTM"),
        "Forest": categorical(data["forest_2d"], palettes["forest"]),
        "Roads": categorical(data["network"], roads),
    }
    for j, column in enumerate(nathab_columns):
        label = "Natural habitat: " + column.removeprefix("nathab_").replace("_", " ")
        layers[label] = categorical(data["natural_habitat"][:, j], palettes["natural_habitat"][column])
    return layers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="a tile folder, or any folder above (ROI, region, split)")
    ap.add_argument("--voxel", type=float, default=0.3, help="subsampling voxel (m), 0 keeps every point")
    ap.add_argument("--point-size", type=float, default=0.3)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    root = args.path.resolve()
    palettes_path, labels_path = find_up(root, "palettes.json"), find_up(root, "labels.json")
    if palettes_path is None or labels_path is None:
        raise SystemExit(f"palettes.json / labels.json not found in {root} or its parents")
    palettes = json.loads(palettes_path.read_text())
    nathab_columns = json.loads(labels_path.read_text())["natural_habitat"]["columns"]

    data = load_tiles(root, args.voxel)
    layers = build_layers(data, palettes, nathab_columns)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    centre = (data["xyz"].min(0) + data["xyz"].max(0)) / 2  # Lambert-93 is too large for float32
    first = next(iter(layers))
    cloud = server.scene.add_point_cloud("/tiles", points=(data["xyz"] - centre).astype(np.float32),
                                         colors=layers[first][0], point_size=args.point_size)

    layer = server.gui.add_dropdown("Layer", options=tuple(layers), initial_value=first)
    legend = server.gui.add_markdown(layers[first][1])
    size = server.gui.add_slider("Point size (m)", min=0.05, max=2.0, step=0.05, initial_value=args.point_size)

    @layer.on_update
    def _(_) -> None:
        cloud.colors, legend.content = layers[layer.value]

    @size.on_update
    def _(_) -> None:
        cloud.point_size = size.value

    print(f"open http://localhost:{args.port}")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
