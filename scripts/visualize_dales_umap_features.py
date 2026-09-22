#!/usr/bin/env python3
"""UMAP scatter of frozen-backbone features extracted by
scripts/extract_dales_grid_probe_features.py or
scripts/extract_flair3d_grid_probe_features.py, colored by class (dataset-
agnostic -- reads whatever coord/feat/segment/class_names is in the .npz).

Fully local/offline -- only reads the small .npz feature dumps, no model or
checkpoint needed here. Meant to be hacked on directly (it's short and flat
on purpose): tweak n_neighbors/min_dist/metric, swap tab20 for another
colormap, drop a panel, add per-panel titles with the real mIoU, etc.

Usage::

    python scripts/visualize_dales_umap_features.py \\
      --features sonata_enc=stats/umap/dales/data/sonata_enc.npz \\
                 litept_dec=stats/umap/dales/data/litept_dec.npz \\
                 litept_enc=stats/umap/dales/data/litept_enc.npz \\
      --output stats/umap/dales/plots/umap.png

Paper-ready panels (no legend/title/frame, one file per backbone)::

    python scripts/visualize_dales_umap_features.py \\
      --features sonata_enc=stats/umap/dales/data/sonata_enc.npz \\
      --minimal

Save the raw post-UMAP embedding (2D coords + segment + class_names) alongside
the figure, so restyling later (colors, point size, legend, ...) doesn't
require refitting UMAP -- the expensive step::

    python scripts/visualize_dales_umap_features.py \\
      --features sonata_enc=stats/umap/dales/data/sonata_enc.npz \\
      --save-embeddings
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap

# Intuitive per-class colors instead of an arbitrary tab20 cycle -- tweak
# freely, falls back to tab20 for any class name not listed here (DALES has
# 8 classes, Flair3D+ segment v20 has 15, ECLAIR has 11, H3D has 11 -- tab20
# covers all four without repeats). "Ground"/"Vegetation"/"Buildings"/"Poles"
# (DALES+ECLAIR) and "Vehicle" (ECLAIR+H3D) are shared keys across datasets
# (same class-name strings), tuned to sit close to their nearest Flair3D+ hue
# so every dataset reads as one coherent palette rather than unrelated ones.
CLASS_COLORS = {
    # DALES classic palette (official DALES paper colors; + shared with
    # ECLAIR where the class name matches, see above)
    "Ground": "#F3D6AB",  # beige sable
    "Vegetation": "#467342",  # vert foncé
    "Cars": "#E932EF",  # magenta
    "Trucks": "#F3EE00",  # jaune
    "Power lines": "#BE9999",  # rose grisé
    "Fences": "#00E90B",  # vert vif
    "Poles": "#EF7200",  # orange
    "Buildings": "#D64236",  # rouge brique
    "Unknown": "#000874",  # bleu nuit -- DALES "unknown / ignored"
    # Flair3D+ segment v20 (finer12)
    "Building": "#A44219",
    "Greenhouse": "#C184CB",
    "Impervious surface": "#76828E",
    "Other soil": "#AB7255",
    "Herbaceous": "#AFC869",
    "Vineyard": "#7A34CE",
    "Brushwood": "#6B5854",
    "Other infrastructures": "#E708D7",
    "Swimming pool": "#00F2DB",
    "Water": "#79A9CA",
    "Deciduous": "#798F48",
    "Coniferous": "#344E41",
    "Bridge": "#023661",
    "Agricultural soil": "#D2A726",
    "Soil under vegetation": "#AF9A73",
    # ECLAIR-only classes (Ground/Vegetation/Buildings/Poles above are
    # shared with DALES). The wire/tower trio shares Flair3D's "Other
    # infrastructures" magenta family (its closest semantic bucket, since
    # Flair3D doesn't split utility network objects into sub-classes).
    "Unassigned": "#C7C7C7",
    "Noise": "black",
    "Transmission Wires": "#E708D7",  # = Flair3D "Other infrastructures"
    "Distribution Wires": "#FF8AD8",
    "Transmission Towers": "#7A1163",
    "Fence": "orange",  # singular -- ECLAIR's own name, DALES has "Fences"
    "Vehicle": "royalblue",  # singular -- ECLAIR's own name, DALES has "Cars"; shared with H3D below
    # H3D -- mostly exact Flair3D+ hue reuse; "Façade"/"Vertical Surface"/
    # "Chimney" have no direct Flair3D+ analog so get a bespoke color that
    # stays in the same family as their closest relative (building/hard-
    # surface) rather than an arbitrary tab20 pick.
    "Low Vegetation": "#AFC869",  # = Flair3D "Herbaceous"
    "Impervious Surface": "#76828E",  # = Flair3D "Impervious surface"
    "Urban Furniture": "#E708D7",  # = Flair3D "Other infrastructures"
    "Roof": "#A44219",  # = Flair3D "Building"
    "Façade": "#8C97A0",  # hard-surface family (lighter companion to Vertical Surface below) -- kept off the brick/soil/shrub hue so it doesn't collide with Roof
    "Shrub": "#4F6B3A",  # darker olive green -- distinct from Tree/Low Vegetation greens and from the brick/soil browns
    "Tree": "#798F48",  # = Flair3D "Deciduous" (same green as "Vegetation" above)
    "Soil or Gravel": "tan",  # = same as DALES/ECLAIR "Ground" (same real-world concept), kept off Roof's brick hue
    "Vertical Surface": "#4A5560",  # hard-surface family, darker than Façade/Impervious Surface
    "Chimney": "#C184CB",  # = Flair3D "Greenhouse" -- repurposed as a small-object accent
}


def parse_features_arg(items):
    """['name=path.npz', ...] -> OrderedDict-like list of (name, path)."""
    parsed = []
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            raise ValueError(f"--features entries must be name=path.npz, got {item!r}")
        parsed.append((name, path))
    return parsed


def plot_panel(ax, name, class_names, segment, embedding, point_size, title=None):
    tab20 = plt.get_cmap("tab20")
    for cls in range(len(class_names)):
        mask = segment == cls
        if not mask.any():
            continue
        color = CLASS_COLORS.get(class_names[cls], tab20(cls % 20))
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=point_size, color=color, label=f"{class_names[cls]}",
            alpha=0.7, linewidths=0,
        )
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", nargs="+", required=True, help="One or more name=path.npz (from extract_dales_grid_probe_features.py).")
    parser.add_argument("--output", default=None, help="Combined-figure mode: output file path (default: stats/umap/<dataset>/plots/<dataset>_umap_nn<n_neighbors>_md<min_dist>_<metric>_n<total_points>.<format>). Minimal mode: output directory for the individual per-panel files (default: stats/umap/<dataset>/plots), each named <dataset>_<panel>_umap_nn<n_neighbors>_md<min_dist>_<metric>_n<panel_points>.<format>. <dataset> is derived from the first --features npz's path (plots/ is a sibling of its data/ dir).")
    parser.add_argument("--format", default="png", help="Output file format (png, pdf, svg, ...) -- only used to build default filenames above; if --output names a file explicitly (combined mode), its own extension controls the format (matplotlib infers it).")
    parser.add_argument("--minimal", action="store_true", help="Paper-ready mode: no color legend, no titles (per-panel or top), no axis frame -- saves each panel as its own individual image instead of one combined multi-panel figure.")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--points-per-class", type=int, default=None, help="Subsample each class down to at most N points before fitting UMAP (default: use everything in the .npz). Handy for fast iteration on n-neighbors/min-dist.")
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--save-embeddings", action="store_true", help="Also save the raw post-UMAP data (2D embedding, segment, class_names) as one .npz per panel, so figures can be restyled later without refitting UMAP.")
    parser.add_argument("--embeddings-output", default=None, help="Directory for --save-embeddings .npz files (default: stats/umap/<dataset>/embeddings, sibling of data/ and plots/). Filename per panel: <dataset>_<panel>_umap_nn<n>_md<d>_<metric>_n<points>.npz")
    args = parser.parse_args()

    panels = parse_features_arg(args.features)
    rng = np.random.default_rng(args.seed)

    # stats/umap/<dataset>/data/*.npz -> <dataset>
    dataset_name = Path(panels[0][1]).resolve().parent.parent.name
    default_plots_dir = Path(panels[0][1]).resolve().parent.parent / "plots"
    md = str(args.min_dist).replace(".", "p")

    class_names = None
    fitted = []  # (name, embedding, segment, feat_channels, n_points)
    for name, path in panels:
        data = np.load(path, allow_pickle=True)
        feat = data["feat"].astype(np.float32)
        segment = data["segment"]
        if class_names is None:
            class_names = [str(c) for c in data["class_names"]]

        if args.points_per_class is not None:
            keep = []
            for cls in range(len(class_names)):
                idx = np.flatnonzero(segment == cls)
                if idx.size > args.points_per_class:
                    idx = rng.choice(idx, size=args.points_per_class, replace=False)
                keep.append(idx)
            keep = np.concatenate(keep)
            feat, segment = feat[keep], segment[keep]

        print(f"[umap] {name}: {feat.shape[0]:,} pts x {feat.shape[1]}ch -- fitting UMAP ...")
        reducer = umap.UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.seed,
        )
        embedding = reducer.fit_transform(feat)
        fitted.append((name, embedding, segment, feat.shape[1], feat.shape[0]))

    if args.save_embeddings:
        embeddings_dir = Path(args.embeddings_output) if args.embeddings_output is not None else default_plots_dir.parent / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        for (name, source_path), (_, embedding, segment, feat_channels, n_points) in zip(panels, fitted):
            out_path = embeddings_dir / f"{dataset_name}_{name}_umap_nn{args.n_neighbors}_md{md}_{args.metric}_n{n_points}.npz"
            np.savez(
                out_path,
                embedding=embedding.astype(np.float32),
                segment=segment,
                class_names=np.array(class_names),
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
                metric=args.metric,
                seed=args.seed,
                feat_channels=feat_channels,
                source=str(source_path),
            )
            print(f"[umap] wrote {out_path}")

    if args.minimal:
        out_dir = Path(args.output) if args.output is not None else default_plots_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, embedding, segment, _, n_points in fitted:
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_panel(ax, name, class_names, segment, embedding, args.point_size, title=None)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.tight_layout()
            out_path = out_dir / f"{dataset_name}_{name}_umap_nn{args.n_neighbors}_md{md}_{args.metric}_n{n_points}.{args.format}"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"[umap] wrote {out_path}")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), squeeze=False)
    axes = axes[0]
    for ax, (name, embedding, segment, feat_channels, n_points) in zip(axes, fitted):
        plot_panel(ax, name, class_names, segment, embedding, args.point_size, title=f"{name}  ({feat_channels}ch, {n_points:,} pts)")

    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), markerscale=3, frameon=False)
    fig.suptitle(f"n_neighbors={args.n_neighbors}   min_dist={args.min_dist}   metric={args.metric}")
    fig.tight_layout()

    if args.output is None:
        default_plots_dir.mkdir(parents=True, exist_ok=True)
        n_points_for_name = fitted[0][4]
        args.output = str(default_plots_dir / f"{dataset_name}_umap_nn{args.n_neighbors}_md{md}_{args.metric}_n{n_points_for_name}.{args.format}")

    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"[umap] wrote {args.output}")


if __name__ == "__main__":
    main()
