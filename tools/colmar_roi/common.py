"""Shared helpers for the Colmar ROI viewers (bundle loading, layer definitions, colour ramps)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Elevation and intensity ramps (64 uniform stops), ranges, gamma and NaN colours come from palette_paper.json
# (copied from the Blender figure pipeline, drprojects/blendify configs/palettes/malibu3d_extra.json).

# Road network drawing (graph overlay): sky-blue tubes for the edges, one kind of darker/bigger ball for the nodes
EDGE_COLOR = "#87CEEB"
EDGE_DIAM_M = 3.6
NODE_COLOR = "#1e3a8a"
NODE_DIAM_M = 7.2
NETWORK_DIM = 0.55  # off-network points are shown as dimmed RGB in the road-network layer

# Per-point categorical columns, in the order of the shader's class attributes: (bundle key, palette key)
CAT_COLUMNS = [
    ("segment", "semantic"), ("forest", "forest"), ("nh_habitat", "nh_habitat"),
    ("nh_moisture", "nh_moisture"), ("nh_soil", "nh_soil"), ("nh_climate", "nh_climate"),
]
# Layers in display order. kind: rgb | intensity | categorical (column) | elevation | network.
# "sub" nests a layer under a sub-heading of its group (natural habitat = one task split into four ecological axes).
LAYERS = [
    {"label": "RGB", "kind": "rgb", "group": "Input"},
    {"label": "LiDAR intensity", "kind": "intensity", "group": "Input"},
    {"label": "Semantic", "kind": "categorical", "column": 0, "group": "Tasks"},
    {"label": "Elevation", "kind": "elevation", "group": "Tasks"},
    {"label": "Road network", "kind": "network", "group": "Tasks"},
    {"label": "Forest", "kind": "categorical", "column": 1, "group": "Tasks"},
    {"label": "Habitat type", "kind": "categorical", "column": 2, "group": "Tasks", "sub": "Natural habitat"},
    {"label": "Moisture regime", "kind": "categorical", "column": 3, "group": "Tasks", "sub": "Natural habitat"},
    {"label": "Soil chemistry", "kind": "categorical", "column": 4, "group": "Tasks", "sub": "Natural habitat"},
    {"label": "Bioclimatic zone", "kind": "categorical", "column": 5, "group": "Tasks", "sub": "Natural habitat"},
]
POINT_KEYS = ["segment", "forest", "nh_habitat", "nh_moisture", "nh_soil", "nh_climate", "elevation", "strength"]


@dataclass
class Bundle:
    xyz: np.ndarray  # (N, 3) float32, metres, ROI-local (origin_l93 = Lambert-93 of the local 0)
    rgb: np.ndarray
    arrays: dict[str, np.ndarray]
    meta: dict
    origin_l93: np.ndarray
    graph: dict[str, np.ndarray]  # graph_nodes (n,3), graph_seg (k,6) edge pieces, ROI-local, on the DTM

    def __len__(self) -> int:
        return len(self.xyz)

    def take(self, idx: np.ndarray) -> "Bundle":
        return Bundle(self.xyz[idx], self.rgb[idx], {k: v[idx] for k, v in self.arrays.items()},
                      self.meta, self.origin_l93, self.graph)

    def voxel_keep(self, voxel: float, seed: int = 0) -> np.ndarray:
        """Indices (sorted) keeping one random point per cubic voxel of ``voxel`` metres."""
        perm = np.random.default_rng(seed).permutation(len(self))
        vox = np.floor(self.xyz[perm] / voxel).astype(np.int64)
        dims = vox.max(0) + 1
        key = (vox[:, 0] * dims[1] + vox[:, 1]) * dims[2] + vox[:, 2]
        _, first = np.unique(key, return_index=True)
        return np.sort(perm[first])

    def subsample(self, voxel: float, seed: int = 0) -> "Bundle":
        return self if voxel <= 0 else self.take(self.voxel_keep(voxel, seed))


def load_bundle(path: str | Path) -> Bundle:
    z = np.load(path, allow_pickle=False)
    return Bundle(z["xyz"], z["rgb"], {k: z[k] for k in POINT_KEYS}, json.loads(str(z["meta"])), z["origin_l93"],
                  {k: z[k] for k in ("graph_nodes", "graph_seg", "graph_n_edges")})


PALETTE_FILE = Path(__file__).with_name("palette_paper.json")


def palette(_b: Bundle | None, key: str) -> dict:
    """Class names / colours of a task, from the paper palette (palette_paper.json)."""
    return json.loads(PALETTE_FILE.read_text())[key]


def hex_to_rgb(colors: list[str]) -> np.ndarray:
    return np.array([[int(c[i : i + 2], 16) for i in (1, 3, 5)] for c in colors], dtype=np.uint8)


def ramp(name: str) -> np.ndarray:
    """(64, 3) float RGB stops (0-255) of the "elevation" or "intensity" ramp."""
    return np.asarray(palette(None, name)["stops"], dtype=np.float32)


def elevation_map(elev: np.ndarray) -> np.ndarray:
    """Elevation (m) -> [0, 1] on the palette's absolute range (NaN kept)."""
    p = palette(None, "elevation")
    return np.clip((elev - p["vmin"]) / (p["vmax"] - p["vmin"]), 0, 1) ** p["gamma"]


def intensity_map(strength: np.ndarray) -> np.ndarray:
    """LiDAR strength -> [0, 1] on the palette's percentile range, with its gamma."""
    p = palette(None, "intensity")
    lo, hi = np.percentile(strength, [p["percentile_low"], p["percentile_high"]])
    return np.clip((strength - lo) / (hi - lo), 0, 1) ** p["gamma"]


def ramp_colors(t: np.ndarray, stops: np.ndarray, nan_color: str) -> np.ndarray:
    """Map t in [0, 1] (NaN allowed) to uint8 RGB with a piecewise-linear ramp."""
    pos = np.clip(np.nan_to_num(t, nan=0.0), 0, 1) * (len(stops) - 1)
    i0 = np.minimum(pos.astype(int), len(stops) - 2)
    f = (pos - i0)[:, None]
    out = (stops[i0] * (1 - f) + stops[i0 + 1] * f).round().astype(np.uint8)
    out[np.isnan(t)] = hex_to_rgb([nan_color])[0]
    return out


def layer_colors(b: Bundle) -> dict[str, np.ndarray]:
    """Per-point (N, 3) uint8 colours for every layer, keyed by display label (graph overlay not included)."""
    out = {}
    for layer in LAYERS:
        kind = layer["kind"]
        if kind == "rgb":
            out[layer["label"]] = b.rgb
        elif kind == "network":
            out[layer["label"]] = (b.rgb * NETWORK_DIM).astype(np.uint8)
        elif kind == "elevation":
            out[layer["label"]] = ramp_colors(elevation_map(b.arrays["elevation"]), ramp("elevation"),
                                              palette(b, "elevation")["nan_color"])
        elif kind == "intensity":
            out[layer["label"]] = ramp_colors(intensity_map(b.arrays["strength"]), ramp("intensity"),
                                              palette(b, "intensity")["nan_color"])
        else:
            key, pal_key = CAT_COLUMNS[layer["column"]]
            lut = hex_to_rgb(palette(b, pal_key)["colors"] + ["#808080"])
            out[layer["label"]] = lut[np.minimum(b.arrays[key].astype(int), len(lut) - 1)]
    return out
