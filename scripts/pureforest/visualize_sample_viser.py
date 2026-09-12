#!/usr/bin/env python3
"""Interactively visualize a PureForest preprocessed tile with viser.

Loads ``coord.npy`` / ``color.npy`` / ``category.npy`` from a tile directory and
serves an interactive RGB point cloud (works over SSH; Open3D needs a local
display).

Usage::

    # by absolute/relative tile path
    python scripts/pureforest/visualize_sample_viser.py \\
      --tile data/pureforest/train/Abies_alba-C9-407_1_102

    # by split + index / name substring
    python scripts/pureforest/visualize_sample_viser.py \\
      --data-root data/pureforest --split train --index 0

    python scripts/pureforest/visualize_sample_viser.py \\
      --data-root data/pureforest_toy --split train --name Castanea
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _rgb_uint8(color: np.ndarray) -> np.ndarray:
    color = np.asarray(color, dtype=np.float32)
    if color.size and color.max() <= 1.5:
        color = color * 255.0
    return np.clip(color, 0, 255).astype(np.uint8)


def _list_tiles(data_root: str, split: str) -> list[str]:
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise SystemExit(f"Split directory not found: {split_dir}")
    names = sorted(
        entry
        for entry in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, entry))
        and os.path.isfile(os.path.join(split_dir, entry, "coord.npy"))
    )
    if not names:
        raise SystemExit(f"No preprocessed tiles under {split_dir}")
    return names


def _resolve_tile_dir(args: argparse.Namespace) -> str:
    if args.tile is not None:
        tile_dir = os.path.abspath(args.tile)
        if not os.path.isdir(tile_dir):
            raise SystemExit(f"--tile is not a directory: {tile_dir}")
        return tile_dir

    names = _list_tiles(args.data_root, args.split)
    idx = args.index
    if args.name is not None:
        matches = [i for i, n in enumerate(names) if args.name in n]
        if not matches:
            raise SystemExit(
                f"No tile matching {args.name!r} in {len(names)} tiles "
                f"under {args.data_root}/{args.split}."
            )
        idx = matches[0]
    return os.path.join(args.data_root, args.split, names[idx % len(names)])


def _load_tile(tile_dir: str) -> tuple[np.ndarray, np.ndarray, int | None, str]:
    coord_path = os.path.join(tile_dir, "coord.npy")
    color_path = os.path.join(tile_dir, "color.npy")
    category_path = os.path.join(tile_dir, "category.npy")
    if not os.path.isfile(coord_path):
        raise SystemExit(f"Missing coord.npy in {tile_dir}")
    if not os.path.isfile(color_path):
        raise SystemExit(f"Missing color.npy in {tile_dir}")

    coord = np.load(coord_path).astype(np.float32)
    color = _rgb_uint8(np.load(color_path))
    if coord.shape[0] != color.shape[0]:
        raise SystemExit(
            f"Length mismatch: coord={coord.shape} color={color.shape} ({tile_dir})"
        )

    category = None
    if os.path.isfile(category_path):
        category = int(np.load(category_path).reshape(-1)[0])

    return coord, color, category, os.path.basename(os.path.normpath(tile_dir))


def _class_label(category: int | None) -> str:
    if category is None:
        return "unknown"
    # Load pureforest_classes by path to avoid importing pointcept.datasets
    # (that package __init__ pulls torch_cluster / full model stack).
    try:
        import importlib.util

        classes_path = os.path.join(
            REPO_ROOT,
            "pointcept",
            "datasets",
            "preprocessing",
            "pureforest",
            "pureforest_classes.py",
        )
        spec = importlib.util.spec_from_file_location(
            "pureforest_classes_standalone", classes_path
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return f"{category} ({mod.class_name_for_id(category)})"
    except Exception:
        return str(category)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a PureForest preprocessed tile with viser."
    )
    parser.add_argument(
        "--tile",
        default=None,
        help="Path to a preprocessed tile directory (coord.npy + color.npy).",
    )
    parser.add_argument("--data-root", default="data/pureforest")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--name",
        default=None,
        help="Substring match on patch_id (overrides --index when --tile is unset).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=500_000,
        help="Random subsample cap for display performance (default: 500000).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.08,
        help="Initial point size in scene units (default: 0.08).",
    )
    args = parser.parse_args()

    tile_dir = _resolve_tile_dir(args)
    coord, color, category, patch_id = _load_tile(tile_dir)
    n_raw = coord.shape[0]

    if n_raw > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(n_raw, args.max_points, replace=False)
        coord = coord[keep]
        color = color[keep]

    label = _class_label(category)
    print(f"[visualize_sample_viser] tile: {tile_dir}")
    print(f"[visualize_sample_viser] patch_id={patch_id}  class={label}")
    print(f"[visualize_sample_viser] points: {coord.shape[0]:,} / {n_raw:,}")

    try:
        import viser
    except ImportError as exc:
        raise SystemExit(
            "viser is required. Install with: pip install viser"
        ) from exc

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.gui.add_markdown(
        f"**PureForest** — `{patch_id}`\n\n"
        f"class: `{label}`  ·  points: `{coord.shape[0]:,}` / `{n_raw:,}`\n\n"
        f"`{tile_dir}`"
    )

    with server.gui.add_folder("Display"):
        point_size = server.gui.add_slider(
            "point size",
            min=0.01,
            max=0.5,
            step=0.01,
            initial_value=float(args.point_size),
        )

    handle = server.scene.add_point_cloud(
        "/tile",
        coord,
        color,
        point_size=point_size.value,
    )

    @point_size.on_update
    def _(_) -> None:
        handle.point_size = point_size.value

    bbox_min = coord.min(axis=0)
    bbox_max = coord.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    radius = max(float(np.linalg.norm(bbox_max - bbox_min)) / 2.0, 1.0)

    def _focus(client) -> None:
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = tuple(
            center + np.array([0.0, -radius * 1.3, radius * 1.1])
        )
        client.camera.look_at = tuple(center)

    with server.gui.add_folder("Camera"):
        focus_button = server.gui.add_button("Reset view")

    @focus_button.on_click
    def _(event) -> None:
        _focus(event.client)

    @server.on_client_connect
    def _(client) -> None:
        _focus(client)

    print(f"[visualize_sample_viser] serving at http://localhost:{args.port}")
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
