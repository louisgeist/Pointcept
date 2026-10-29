"""Live viser viewer of the Colmar city-centre ROI with a layer switcher.

    python tools/colmar_roi/viser_viewer.py --bundle outputs/colmar_roi/bundle.npz --voxel 0.4

Open http://localhost:8080 (SSH tunnel from the Mac). The "Layer" dropdown switches the colouring and
shows the legend of the categorical layers; the road-network layer adds the graph (sky-blue edges, darker
node balls).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import viser

from common import (CAT_COLUMNS, EDGE_COLOR, EDGE_DIAM_M, LAYERS, NODE_COLOR, NODE_DIAM_M, hex_to_rgb, layer_colors,
                    load_bundle, palette)


def _legend_md(b, column: int) -> str:
    key, pal_key = CAT_COLUMNS[column]
    pal = palette(b, pal_key)
    ids = sorted(set(b.arrays[key].tolist()))
    return "<br>".join(f'<span style="color:{pal["colors"][i]}">&#9632;</span> {pal["names"][i]}'
                       for i in ids if i < len(pal["names"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="outputs/colmar_roi/bundle.npz")
    ap.add_argument("--voxel", type=float, default=0.4, help="extra voxel subsampling (m) for the live view")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--point-size", type=float, default=0.35)
    args = ap.parse_args()

    b = load_bundle(args.bundle).subsample(args.voxel)
    colors = layer_colors(b)
    labels = [layer["label"] for layer in LAYERS]
    by_label = {layer["label"]: layer for layer in LAYERS}
    print(f"{len(b):,} points, layers: {labels}")

    centre = (b.xyz.min(0) + b.xyz.max(0)) / 2  # centre the cloud at the viewer origin
    server = viser.ViserServer(host="127.0.0.1", port=args.port)  # localhost only, reach it through an SSH tunnel
    server.scene.set_up_direction("+z")
    cloud = server.scene.add_point_cloud(
        "/roi", points=(b.xyz - centre).astype(np.float32), colors=colors[labels[0]],
        point_size=args.point_size, point_shape="square")

    # road graph: sky-blue edges (line segments) and one kind of node ball, only visible in the road-network layer
    g = b.graph
    lift = np.array([0, 0, 0.5 * EDGE_DIAM_M], np.float32)
    seg = (g["graph_seg"].reshape(-1, 2, 3) - centre).astype(np.float32) + lift
    overlay = [
        server.scene.add_line_segments("/network/edges", points=seg, colors=hex_to_rgb([EDGE_COLOR])[0], line_width=4.0),
        server.scene.add_point_cloud(
            "/network/nodes", points=(g["graph_nodes"] - centre).astype(np.float32) + [0, 0, 0.5 * NODE_DIAM_M],
            colors=np.tile(hex_to_rgb([NODE_COLOR]), (len(g["graph_nodes"]), 1)), point_size=NODE_DIAM_M,
            point_shape="circle"),
    ]

    with server.gui.add_folder("Layer"):
        mode = server.gui.add_dropdown("Show", options=tuple(labels), initial_value=labels[0])
        legend = server.gui.add_markdown("")
    size = server.gui.add_slider("Point size (m)", min=0.05, max=1.5, step=0.05, initial_value=args.point_size)

    def refresh() -> None:
        layer = by_label[mode.value]
        cloud.colors = colors[mode.value]
        for node in overlay:
            node.visible = layer["kind"] == "network"
        if layer["kind"] == "categorical":
            legend.content = _legend_md(b, layer["column"])
        elif layer["kind"] == "network":
            legend.content = (f'<span style="color:{EDGE_COLOR}">&#9679;</span> Road edge<br>'
                              f'<span style="color:{NODE_COLOR}">&#9679;</span> Node')
        else:
            legend.content = ""

    @mode.on_update
    def _(_) -> None:
        refresh()

    @size.on_update
    def _(_) -> None:
        cloud.point_size = size.value

    refresh()
    print(f"viser on http://localhost:{args.port}")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
