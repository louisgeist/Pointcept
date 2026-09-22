#!/usr/bin/env python3
"""Visualize raw ECLAIR tile(s) directly from the .laz, with viser.

Companion to scripts/eclair/analyze_point_counts.py and the min_points guard
in configs/eclair/pretrain-sonata-v1m2-eclair.py: lets you eyeball the near-
empty outlier tiles that the min_points=2000 filter now excludes, straight
from the raw LAZ (no preprocessing step needed — reuses build_scene from
preprocess_eclair.py via a dynamic import so no torch/pointcept dependency).

Usage::

    # one tile
    python scripts/eclair/visualize_excluded_tile.py --tile pointcloud_234

    # all 5 tiles found below the min_points=2000 threshold (2026-09-22 survey)
    python scripts/eclair/visualize_excluded_tile.py

Serves one viser server per tile, one after another (Ctrl+C to move to the
next). Point counts are tiny (order of a few hundred to ~1500), so don't
expect much geometry — that's the point.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tiles found under min_points=2000 in the 2026-09-22 survey (all "rejected").
DEFAULT_EXCLUDED_TILES = [
    "pointcloud_562",
    "pointcloud_234",
    "pointcloud_233",
    "pointcloud_334",
    "pointcloud_245",
]

_preprocess_path = os.path.join(
    REPO_ROOT, "pointcept", "datasets", "preprocessing", "eclair", "preprocess_eclair.py"
)
_spec = importlib.util.spec_from_file_location("preprocess_eclair", _preprocess_path)
_preprocess_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_preprocess_mod)
build_scene = _preprocess_mod.build_scene


def visualize(laz_path: str, port: int) -> None:
    scene = build_scene(laz_path)
    coord = scene["coord"]
    color = np.clip(scene["color"], 0, 255).astype(np.uint8)
    n = coord.shape[0]
    bbox_min = coord.min(axis=0)
    bbox_max = coord.max(axis=0)

    print(f"[visualize_excluded_tile] {os.path.basename(laz_path)}")
    print(f"[visualize_excluded_tile] points: {n:,}")
    print(f"[visualize_excluded_tile] bbox min: {bbox_min}, max: {bbox_max}")
    print(f"[visualize_excluded_tile] extent (m): {bbox_max - bbox_min}")

    try:
        import viser
    except ImportError as error:
        raise ImportError("viser is required. Install with: pip install viser") from error

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")
    server.gui.add_markdown(
        f"**{os.path.basename(laz_path)}** — {n:,} points\n\n"
        f"bbox extent (m): {tuple(round(v, 1) for v in (bbox_max - bbox_min))}"
    )
    # Recenter so the (tiny, possibly far-from-origin) tile is visible on load.
    center = (bbox_min + bbox_max) / 2
    server.scene.add_point_cloud(
        "/tile",
        points=(coord - center).astype(np.float32),
        colors=color,
        point_size=0.15,
    )
    print(f"[visualize_excluded_tile] serving at http://localhost:{port}  (Ctrl+C for next tile)")
    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--raw-root",
        default="data/eclair/raw/pointclouds",
        help="Directory of raw pointcloud_*.laz files",
    )
    parser.add_argument(
        "--tile",
        action="append",
        default=None,
        help="Tile name (with or without .laz), repeatable. Defaults to the 5 known outliers.",
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    tiles = args.tile if args.tile else DEFAULT_EXCLUDED_TILES
    for tile in tiles:
        name = tile if tile.endswith(".laz") else f"{tile}.laz"
        laz_path = os.path.join(args.raw_root, name)
        if not os.path.isfile(laz_path):
            print(f"[visualize_excluded_tile] SKIP, not found: {laz_path}", file=sys.stderr)
            continue
        visualize(laz_path, args.port)


if __name__ == "__main__":
    main()
