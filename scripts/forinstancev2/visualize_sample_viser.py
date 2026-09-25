#!/usr/bin/env python3
"""Interactively visualize a ForInstanceV2 preprocessed tile with viser.

Loads ``coord.npy`` / ``segment.npy`` (+ optional ``instance.npy``) from a
tile directory and serves an interactive point cloud (works over SSH; Open3D
needs a local display). ForInstanceV2 has no color/intensity on disk (see
``pointcept/datasets/preprocessing/forinstancev2/preprocess_forinstancev2.py``),
so points are colored by segment class by default (or by tree instance id
with ``--color-by instance``).

Usage::

    # by absolute/relative tile path
    python scripts/forinstancev2/visualize_sample_viser.py \\
      --tile data/forinstancev2/train/Yuchen_2023_dls_merged_230209_panoptic_train

    # by split + index / name substring
    python scripts/forinstancev2/visualize_sample_viser.py \\
      --data-root data/forinstancev2 --split train --name Yuchen

    # color by tree instance id instead of segment class
    python scripts/forinstancev2/visualize_sample_viser.py \\
      --data-root data/forinstancev2 --split train --name Yuchen --color-by instance
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

SEGMENT_NAMES = ["Ground", "Low vegetation", "Tree"]
SEGMENT_COLORS = np.array(
    [
        [139, 90, 43],  # Ground — brown
        [150, 220, 90],  # Low vegetation — light green
        [20, 110, 40],  # Tree — dark green
    ],
    dtype=np.uint8,
)


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


def _instance_colors(instance: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    uniq = np.unique(instance)
    palette = {}
    for tid in uniq:
        if tid == 0:
            palette[tid] = np.array([120, 120, 120], dtype=np.uint8)  # no-tree/ground
        else:
            palette[tid] = rng.integers(40, 255, size=3).astype(np.uint8)
    return np.stack([palette[t] for t in instance])


def _load_tile(tile_dir: str, color_by: str) -> tuple[np.ndarray, np.ndarray, str, dict]:
    coord_path = os.path.join(tile_dir, "coord.npy")
    segment_path = os.path.join(tile_dir, "segment.npy")
    instance_path = os.path.join(tile_dir, "instance.npy")
    if not os.path.isfile(coord_path):
        raise SystemExit(f"Missing coord.npy in {tile_dir}")

    coord = np.load(coord_path).astype(np.float32)
    stats = {}

    segment = np.load(segment_path).reshape(-1) if os.path.isfile(segment_path) else None
    instance = np.load(instance_path).reshape(-1) if os.path.isfile(instance_path) else None

    if color_by == "instance":
        if instance is None:
            raise SystemExit(f"--color-by instance but no instance.npy in {tile_dir}")
        color = _instance_colors(instance)
        stats["n_trees"] = int(len(np.unique(instance[instance != 0])))
    else:
        if segment is None:
            raise SystemExit(f"--color-by segment but no segment.npy in {tile_dir}")
        color = SEGMENT_COLORS[np.clip(segment, 0, len(SEGMENT_COLORS) - 1)]
        for cid, name in enumerate(SEGMENT_NAMES):
            stats[name] = int((segment == cid).sum())

    if coord.shape[0] != color.shape[0]:
        raise SystemExit(f"Length mismatch: coord={coord.shape} color={color.shape} ({tile_dir})")

    return coord, color, os.path.basename(os.path.normpath(tile_dir)), stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a ForInstanceV2 preprocessed tile with viser."
    )
    parser.add_argument("--tile", default=None, help="Path to a preprocessed tile directory.")
    parser.add_argument("--data-root", default="data/forinstancev2")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--name", default=None, help="Substring match on tile name (overrides --index)."
    )
    parser.add_argument(
        "--color-by", default="segment", choices=["segment", "instance"],
        help="Color points by semantic segment class (default) or tree instance id.",
    )
    parser.add_argument(
        "--max-points", type=int, default=1_000_000,
        help="Random subsample cap for display performance (default: 1,000,000).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.03)
    args = parser.parse_args()

    tile_dir = _resolve_tile_dir(args)
    coord, color, tile_id, stats = _load_tile(tile_dir, args.color_by)
    n_raw = coord.shape[0]

    # ForInstanceV2 stores absolute geographic coords (e.g. Lambert93, ~1e5-1e6
    # range) with an extent of only tens/hundreds of meters -- rendering that
    # directly loses float32 precision on the GPU (WebGL vertex buffers) and
    # can make the whole cloud jitter/vanish. Recenter near the origin, purely
    # for display; doesn't touch the on-disk data.
    origin = coord.min(axis=0)
    coord = coord - origin
    print(f"[visualize_sample_viser] recentered for display, origin={origin}")

    if n_raw > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(n_raw, args.max_points, replace=False)
        coord = coord[keep]
        color = color[keep]

    print(f"[visualize_sample_viser] tile: {tile_dir}")
    print(f"[visualize_sample_viser] tile_id={tile_id}  color_by={args.color_by}  stats={stats}")
    print(f"[visualize_sample_viser] points: {coord.shape[0]:,} / {n_raw:,}")

    try:
        import viser
    except ImportError as exc:
        raise SystemExit("viser is required. Install with: pip install viser") from exc

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    legend = (
        "  ·  ".join(f"{k}: `{v:,}`" for k, v in stats.items())
        if args.color_by == "segment"
        else f"n_trees: `{stats.get('n_trees', '?')}`"
    )
    server.gui.add_markdown(
        f"**ForInstanceV2** — `{tile_id}`\n\n"
        f"color_by: `{args.color_by}`  ·  points: `{coord.shape[0]:,}` / `{n_raw:,}`\n\n"
        f"{legend}\n\n"
        f"`{tile_dir}`"
    )

    with server.gui.add_folder("Display"):
        point_size = server.gui.add_slider(
            "point size", min=0.005, max=0.5, step=0.005, initial_value=float(args.point_size)
        )

    handle = server.scene.add_point_cloud("/tile", coord, color, point_size=point_size.value)

    @point_size.on_update
    def _(_) -> None:
        handle.point_size = point_size.value

    bbox_min = coord.min(axis=0)
    bbox_max = coord.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    radius = max(float(np.linalg.norm(bbox_max - bbox_min)) / 2.0, 1.0)

    def _focus(client) -> None:
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = tuple(center + np.array([0.0, -radius * 1.3, radius * 1.1]))
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
