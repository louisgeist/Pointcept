#!/usr/bin/env python3
"""Interactively visualize an OpenGF preprocessed tile with viser.

Loads ``coord.npy`` / ``segment.npy`` from a tile directory and serves an
interactive point cloud (works over SSH; Open3D needs a local display).
OpenGF has no color on disk (see
``pointcept/datasets/preprocessing/opengf/preprocess_opengf.py``), so points
are colored by segment class: 0=Ground, 1=Non-ground, 2=Outlier (Outlier
only ever appears on Test/T2 tiles and a handful of train/val tiles).

Usage::

    # by absolute/relative tile path
    python scripts/opengf/visualize_sample_viser.py \\
      --tile data/opengf/test/T2_0-2

    # by split + index / name substring
    python scripts/opengf/visualize_sample_viser.py \\
      --data-root data/opengf --split train --name S6_33

    python scripts/opengf/visualize_sample_viser.py \\
      --data-root data/opengf --split test --name T2_0-2
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

SEGMENT_NAMES = ["Ground", "Non-ground", "Outlier"]
SEGMENT_COLORS = np.array(
    [
        [139, 90, 43],  # Ground — brown
        [46, 139, 87],  # Non-ground — green
        [224, 32, 32],  # Outlier — red
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


def _load_tile(tile_dir: str) -> tuple[np.ndarray, np.ndarray, str, dict]:
    coord_path = os.path.join(tile_dir, "coord.npy")
    segment_path = os.path.join(tile_dir, "segment.npy")
    if not os.path.isfile(coord_path):
        raise SystemExit(f"Missing coord.npy in {tile_dir}")
    if not os.path.isfile(segment_path):
        raise SystemExit(f"Missing segment.npy in {tile_dir}")

    coord = np.load(coord_path).astype(np.float32)
    segment = np.load(segment_path).reshape(-1)
    color = SEGMENT_COLORS[np.clip(segment, 0, len(SEGMENT_COLORS) - 1)]

    stats = {name: int((segment == cid).sum()) for cid, name in enumerate(SEGMENT_NAMES)}

    if coord.shape[0] != color.shape[0]:
        raise SystemExit(f"Length mismatch: coord={coord.shape} color={color.shape} ({tile_dir})")

    return coord, color, os.path.basename(os.path.normpath(tile_dir)), stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize an OpenGF preprocessed tile with viser."
    )
    parser.add_argument("--tile", default=None, help="Path to a preprocessed tile directory.")
    parser.add_argument("--data-root", default="data/opengf")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--name", default=None, help="Substring match on tile name (overrides --index)."
    )
    parser.add_argument(
        "--max-points", type=int, default=1_000_000,
        help="Random subsample cap for display performance (default: 1,000,000).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.05)
    args = parser.parse_args()

    tile_dir = _resolve_tile_dir(args)
    coord, color, tile_id, stats = _load_tile(tile_dir)
    n_raw = coord.shape[0]

    if n_raw > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(n_raw, args.max_points, replace=False)
        # Keep all outlier points regardless of subsampling -- they're the
        # rarest class (<0.5% even on T2) and easy to lose to random sampling.
        segment = np.load(os.path.join(tile_dir, "segment.npy")).reshape(-1)
        outlier_idx = np.where(segment == 2)[0]
        keep = np.unique(np.concatenate([keep, outlier_idx]))
        coord = coord[keep]
        color = color[keep]

    print(f"[visualize_sample_viser] tile: {tile_dir}")
    print(f"[visualize_sample_viser] tile_id={tile_id}  stats={stats}")
    print(f"[visualize_sample_viser] points: {coord.shape[0]:,} / {n_raw:,}")

    try:
        import viser
    except ImportError as exc:
        raise SystemExit("viser is required. Install with: pip install viser") from exc

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    legend = "  ·  ".join(f"{k}: `{v:,}`" for k, v in stats.items())
    server.gui.add_markdown(
        f"**OpenGF** — `{tile_id}`\n\n"
        f"points: `{coord.shape[0]:,}` / `{n_raw:,}`\n\n"
        f"{legend}\n\n"
        f"`{tile_dir}`"
    )

    with server.gui.add_folder("Display"):
        point_size = server.gui.add_slider(
            "point size", min=0.01, max=0.5, step=0.01, initial_value=float(args.point_size)
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
