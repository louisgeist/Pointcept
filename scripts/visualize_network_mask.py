#!/usr/bin/env python3
"""Visualize Flair3D network GT masks, optional pred probs/binary, and mean RGB.

Layout without ``--logits``::

    [ ROADS ] [ RAILROADS ] [ TRANSMISSION_LINES ]   # GT
    [-------- mean-pooled RGB (same 1 m grid) --------]

Layout with ``--logits``::

    [ GT ROADS ] [ GT RAIL ] [ GT TL ]
    [ Prob ROADS ] [ Prob RAIL ] [ Prob TL ]
    [ Bin@thr ROADS ] [ Bin RAIL ] [ Bin TL ]
    [-------- mean-pooled RGB --------]

Example (GT only)::

    python scripts/visualize_network_mask.py \\
      --tile data/flair3d_plus/train/D067-2021_LIDARHD/AF-S1-22/D067-2021_AF-S1-22_1-1 \\
      --out /tmp/network_mask_rgb.png

Example (GT + predictions)::

    python scripts/visualize_network_mask.py \\
      --tile data/flair3d_plus/train/D067-2021_LIDARHD/AF-S1-22/D067-2021_AF-S1-22_1-1 \\
      --logits exp/default/result/D067-2021_AF-S1-22_1-1_logits_network.npy \\
      --threshold 0.2 \\
      --prob-autoscale \\
      --out /tmp/network_gt_pred.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_FG_COLORS = {
    "ROADS": "#ff5555",
    "RAILROADS": "#55ff55",
    "TRANSMISSION_LINES": "#5599ff",
}
_LOGITS_SUFFIX = "_logits_network.npy"


def _load_raster_utils():
    """Import network_xy_raster_utils without pulling pointcept.datasets (torch)."""
    path = os.path.join(
        REPO_ROOT,
        "pointcept",
        "datasets",
        "preprocessing",
        "flair3d_plus",
        "network_xy_raster_utils.py",
    )
    spec = importlib.util.spec_from_file_location("network_xy_raster_utils", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.GridSpec, mod.mean_rgb_raster


def _load_network(tile: Path) -> tuple[np.ndarray, dict, list[str]]:
    meta = json.loads((tile / "meta.json").read_text())
    if "network" not in meta:
        raise KeyError(f"No meta.network in {tile / 'meta.json'}")
    net_meta = meta["network"]
    channel_order = list(net_meta["channel_order"])
    h, w = int(net_meta["height"]), int(net_meta["width"])
    net_path = tile / "network.npy"
    if net_path.is_file() and not net_meta.get("empty", False):
        network = np.load(net_path)
    else:
        network = np.zeros((len(channel_order), h, w), dtype=np.uint8)
    if network.shape != (len(channel_order), h, w):
        raise ValueError(
            f"network.npy shape {network.shape} != "
            f"({len(channel_order)}, {h}, {w}) from meta"
        )
    return network, net_meta, channel_order


def _load_logits(path: Path, n_channels: int, h: int, w: int) -> np.ndarray:
    logits = np.load(path)
    if logits.ndim != 3:
        raise ValueError(f"logits must be (C, H, W), got shape {logits.shape}")
    if logits.shape != (n_channels, h, w):
        raise ValueError(
            f"logits shape {logits.shape} != expected ({n_channels}, {h}, {w}) "
            f"from tile meta.network"
        )
    return logits.astype(np.float32, copy=False)


def _patch_id_from_logits(path: Path) -> str | None:
    name = path.name
    if name.endswith(_LOGITS_SUFFIX):
        return name[: -len(_LOGITS_SUFFIX)]
    return None


def _imshow_binary(ax, mask: np.ndarray, fg_hex: str) -> None:
    cmap = ListedColormap(["#111111", fg_hex])
    ax.imshow(
        np.flipud(mask.astype(np.uint8)),
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )


def _imshow_prob(
    ax,
    prob: np.ndarray,
    *,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
):
    """Show soft probs; NaN (unobserved) rendered black.

    Pass ``vmin=None, vmax=None`` to let matplotlib autoscale on finite values
    for that panel (prefer a shared range via explicit vmin/vmax when comparing
    channels).
    """
    show = np.flipud(prob.astype(np.float32, copy=True))
    masked = np.ma.masked_invalid(show)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#111111")
    return ax.imshow(
        masked, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
    )


def _prob_display_range(
    logits: np.ndarray, *, autoscale: bool
) -> tuple[float | None, float | None]:
    """Return (vmin, vmax) for pred-prob panels.

    Default: fixed [0, 1]. With ``autoscale=True``: shared min/max over all
    finite values across channels (so the colorbar stays comparable).
    """
    if not autoscale:
        return 0.0, 1.0
    finite = logits[np.isfinite(logits)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmin == vmax:
        # Avoid a collapsed colormap when all finite values are equal.
        eps = 1e-6 if vmin == 0.0 else abs(vmin) * 1e-6
        return vmin - eps, vmax + eps
    return vmin, vmax


def _binarize(logits: np.ndarray, threshold: float) -> np.ndarray:
    """Foreground where prob >= threshold; NaN / unobserved -> background."""
    valid = np.isfinite(logits)
    return (valid & (logits >= threshold)).astype(np.uint8)


def render(
    tile: Path,
    out: Path,
    *,
    logits_path: Path | None = None,
    threshold: float = 0.2,
    prob_autoscale: bool = False,
    dpi: int = 150,
) -> Path:
    GridSpec, mean_rgb_raster = _load_raster_utils()

    network, net_meta, channel_order = _load_network(tile)
    h, w = int(net_meta["height"]), int(net_meta["width"])
    pixel_m = float(net_meta["pixel_m"])
    grid = GridSpec(
        origin_x=float(net_meta["origin_x"]),
        origin_y=float(net_meta["origin_y"]),
        width=w,
        height=h,
        pixel_m=pixel_m,
    )

    logits = None
    binary = None
    if logits_path is not None:
        logits = _load_logits(logits_path, len(channel_order), h, w)
        binary = _binarize(logits, threshold)

    coord = np.load(tile / "coord.npy")
    color = np.load(tile / "color.npy")
    transl = np.load(tile / "coord_translation.npy")
    abs_xy = coord[:, :2].astype(np.float64) + transl[:2].astype(np.float64)
    mean_rgb, count = mean_rgb_raster(abs_xy, color, grid)
    rgb_show = np.flipud(np.clip(np.round(mean_rgb), 0, 255).astype(np.uint8))

    n_mask_rows = 3 if logits is not None else 1
    n_rows = n_mask_rows + 1  # + RGB
    fig_h = 4.0 * n_rows
    # Extra right column reserved for the prob colorbar when logits are shown.
    n_cols = 4 if logits is not None else 3
    width_ratios = [1.0, 1.0, 1.0, 0.06] if logits is not None else [1.0, 1.0, 1.0]
    fig = plt.figure(figsize=(12.5 if logits is not None else 12, fig_h), facecolor="white")
    gs = fig.add_gridspec(
        n_rows, n_cols, height_ratios=[1.0] * n_mask_rows + [1.2], width_ratios=width_ratios
    )

    # Row 0: GT
    for i, name in enumerate(channel_order):
        ax = fig.add_subplot(gs[0, i])
        mask = network[i].astype(bool)
        _imshow_binary(ax, mask, _FG_COLORS.get(name, "#ffffff"))
        ax.set_title(f"GT {name}\npositives={int(mask.sum())}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    last_im = None
    if logits is not None:
        vmin, vmax = _prob_display_range(logits, autoscale=prob_autoscale)
        scale_note = (
            f"autoscale [{vmin:.4g}, {vmax:.4g}]"
            if prob_autoscale
            else "[0, 1]"
        )
        # Row 1: soft probs
        for i, name in enumerate(channel_order):
            ax = fig.add_subplot(gs[1, i])
            last_im = _imshow_prob(ax, logits[i], vmin=vmin, vmax=vmax)
            finite = np.isfinite(logits[i])
            ax.set_title(
                f"Pred prob {name}\n"
                f"finite={int(finite.sum())}, "
                f"max={float(np.nanmax(logits[i])) if np.any(finite) else float('nan'):.3f}",
                fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        # Row 2: binarized
        for i, name in enumerate(channel_order):
            ax = fig.add_subplot(gs[2, i])
            assert binary is not None
            bin_mask = binary[i].astype(bool)
            _imshow_binary(ax, bin_mask, _FG_COLORS.get(name, "#ffffff"))
            ax.set_title(
                f"Pred bin @{threshold:g} {name}\npositives={int(bin_mask.sum())}",
                fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        if last_im is not None:
            cax = fig.add_subplot(gs[1, 3])
            fig.colorbar(last_im, cax=cax, label=f"P(fg) {scale_note}")

    # RGB row spanning the three image columns
    ax_rgb = fig.add_subplot(gs[n_mask_rows, :3])
    ax_rgb.imshow(rgb_show, interpolation="nearest")
    ax_rgb.set_title(
        f"Mean-pooled RGB (pixel_m={pixel_m}, grid={w}x{h}, "
        f"occupied={int((count > 0).sum())}/{h * w})",
        fontsize=12,
    )
    ax_rgb.set_xticks([])
    ax_rgb.set_yticks([])
    ax_rgb.set_aspect("equal")

    title = f"Network mask: {tile.name}"
    if logits_path is not None:
        title += f"  |  logits: {logits_path.name}"
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1.0, 0.97])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot network GT masks (and optional pred probs / binarized) "
            "above mean-pooled LiDAR RGB."
        )
    )
    parser.add_argument(
        "--tile",
        type=Path,
        required=True,
        help="Path to a preprocessed Flair3D+ tile directory "
        "(must contain meta.json, coord.npy, color.npy, coord_translation.npy).",
    )
    parser.add_argument(
        "--logits",
        type=Path,
        default=None,
        help="Optional path to `{tile}_logits_network.npy` (C, H, W) soft probs; "
        "NaN = unobserved. Adds prob + binarized rows.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Foreground threshold for binarized pred row (default: 0.2).",
    )
    parser.add_argument(
        "--prob-autoscale",
        action="store_true",
        help="Color-scale pred probs over shared finite min/max across channels "
        "instead of fixed [0, 1] (useful when probs are peaked near 0).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: /tmp/<tile_name>_network_mask_rgb.png).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    tile = args.tile.resolve()
    if not tile.is_dir():
        raise SystemExit(f"Tile directory not found: {tile}")
    for name in ("meta.json", "coord.npy", "color.npy", "coord_translation.npy"):
        if not (tile / name).is_file():
            raise SystemExit(f"Missing {name} in {tile}")

    logits_path = None
    if args.logits is not None:
        logits_path = args.logits.resolve()
        if not logits_path.is_file():
            raise SystemExit(f"Logits file not found: {logits_path}")
        patch_id = _patch_id_from_logits(logits_path)
        if patch_id is not None and patch_id != tile.name:
            print(
                f"warning: logits patch_id={patch_id!r} != tile name={tile.name!r}",
                file=sys.stderr,
            )
        if not (0.0 <= args.threshold <= 1.0):
            raise SystemExit(f"--threshold must be in [0, 1], got {args.threshold}")

    out = args.out
    if out is None:
        suffix = "_gt_pred" if logits_path is not None else "_network_mask_rgb"
        out = Path("/tmp") / f"{tile.name}{suffix}.png"

    written = render(
        tile,
        out,
        logits_path=logits_path,
        threshold=args.threshold,
        prob_autoscale=args.prob_autoscale,
        dpi=args.dpi,
    )
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
