"""Export the Colmar ROI bundle as a standalone WebGL2 viewer: size-budgeted overview + zoom spots.

    python tools/colmar_roi/export_html.py --budget-mb 5

Everything is sized from one download budget (gzip MB, ``--budget-mb``):

* **Overview**: the whole ROI, voxel-subsampled with the coarsest voxel whose file fits the budget.
* **Zoom spots**: around a chosen point (xy), the N nearest points of the full 0.1 m grid, i.e. a vertical
  cylinder holding exactly N points; N is the largest count whose file fits the budget, the same for every
  spot. The viewer flies to a spot on click, downloads its file and shows it in place of the overview there.

Output (serve it over http(s)):

    index.html            viewer, metadata and road graph embedded
    overview.bin.gz       level 0
    spots/{id}.bin.gz     one per zoom spot

Optional ``--levels 1.5,0.4,0.1`` additionally turns on automatic per-tile level-of-detail streaming
(tiles/{level}_{tile}.bin.gz, loaded by camera distance, ``--lod-dist``). With one level and no spot the
viewer is a single self-contained ``index.html``.

Binary layout (planar, little-endian, before gzip), N points:
    pos   uint16 x N*3 (interleaved xyz, dequantised with meta.pos_scale / meta.pos_offset, global)
    rgb   uint8  x N*3
    cls0  uint8  x N*4 (semantic, forest, habitat type, moisture regime)
    cls1  uint8  x N*4 (soil chemistry, bioclimatic zone, 0, 0)
    cont  uint8  x N*2 (elevation 0..254, 255 = NaN; LiDAR intensity 0..255)
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from common import (CAT_COLUMNS, EDGE_COLOR, EDGE_DIAM_M, LAYERS, NODE_COLOR, NODE_DIAM_M, Bundle, elevation_map,
                    load_bundle, palette, ramp)

TEMPLATE = Path(__file__).with_name("viewer_template.html")
TEMPLATE_THREE = Path(__file__).with_name("viewer_template_three.html")  # same viewer on three.js (needs vendor/)
# local ROI coordinates (m); adjust with --spot "Name=x,y"
DEFAULT_SPOTS = ["Bridge=104.5,276", "Church=753,141"]
SPOT_SIZE_M = 0.4  # default zoom point size (m): native LiDAR spacing is ~0.25 m, not the 0.1 m voxel
OVERVIEW_SIZE_PER_VOXEL = 0.85  # default overview point size = 0.85 x its voxel (1.4 m for Colmar's ~1.66 m voxel)


class Quantiser:
    """Global quantisation parameters, shared by every level, tile and spot.

    Elevation and intensity are stored already mapped to [0, 1] through the palette's range / gamma
    (palette_paper.json), so the viewer only has to look the colour up in the ramp.
    """

    def __init__(self, full: Bundle):
        self.lo = full.xyz.min(0).astype(np.float64)
        self.hi = full.xyz.max(0).astype(np.float64)
        self.scale = (self.hi - self.lo) / 65535.0
        p = palette(None, "intensity")
        self.s_bounds = np.percentile(full.arrays["strength"], [p["percentile_low"], p["percentile_high"]])

    def encode(self, b: Bundle, idx: np.ndarray) -> bytes:
        q = np.round((b.xyz[idx].astype(np.float64) - self.lo) / self.scale).astype(np.uint16)
        elev_t = elevation_map(b.arrays["elevation"][idx])
        e8 = np.round(np.nan_to_num(elev_t) * 254).astype(np.uint8)
        e8[np.isnan(elev_t)] = 255
        gamma = palette(None, "intensity")["gamma"]
        s_lo, s_hi = self.s_bounds
        s8 = np.round(np.clip((b.arrays["strength"][idx] - s_lo) / (s_hi - s_lo), 0, 1) ** gamma * 255).astype(np.uint8)
        cols = [b.arrays[key][idx].astype(np.uint8) for key, _ in CAT_COLUMNS]
        zero = np.zeros(len(idx), np.uint8)
        cls0 = np.stack(cols[:4], axis=1)
        cls1 = np.stack(cols[4:] + [zero] * (8 - len(cols)), axis=1)
        return (q.tobytes() + b.rgb[idx].astype(np.uint8).tobytes() + cls0.tobytes() + cls1.tobytes()
                + np.stack([e8, s8], axis=1).tobytes())


def _gz(data: bytes, level: int = 9) -> bytes:
    return gzip.compress(data, compresslevel=level)


def fit_overview_voxel(full: Bundle, qz: Quantiser, budget: float) -> float:
    """Smallest voxel (m) whose level-0 file fits ``budget`` bytes (size ~ voxel^-2)."""
    voxel = 1.4
    for _ in range(6):
        keep = full.voxel_keep(voxel)
        size = len(_gz(qz.encode(full, keep)))
        print(f"  overview voxel {voxel:.3f} m -> {len(keep):,} pts, {size / 1e6:.2f} MB")
        if 0.97 * budget <= size <= budget:
            return voxel
        voxel = round(voxel * (size / budget * 1.01) ** 0.5, 3)
    return voxel


def spot_indices(tree: cKDTree, xy: tuple[float, float], n: int) -> tuple[np.ndarray, float]:
    d, idx = tree.query(xy, k=n)
    return np.sort(idx), float(d.max())


def fit_spot_count(full: Bundle, qz: Quantiser, tree: cKDTree, spots: list, budget: float) -> int:
    """Largest N such that the N-nearest-points file of every spot fits ``budget`` bytes."""
    n = int(budget / 8.0)
    for _ in range(4):
        sizes = [len(_gz(qz.encode(full, spot_indices(tree, xy, n)[0]))) for _, _, xy, _ in spots]
        print(f"  spot N={n:,}: " + ", ".join(f"{s / 1e6:.2f} MB" for s in sizes))
        worst = max(sizes)
        if 0.97 * budget <= worst <= budget:
            return n
        n = int(n * budget / worst * 0.995)
    return n


def _three_name(page: str) -> str:
    return page.replace(".html", "_three.html")


def export_zone(args, out: Path, bundle: str, title: str, spots_arg: list, page: str = "index.html",
                prefix: str = "", zone_id: str = "colmar", zones: list | None = None, cam: dict | None = None,
                overview_size: float | None = None, disabled: list | None = None) -> None:
    """Write one viewer page ``out/page`` and its binary files under ``out/prefix``."""
    budget = args.budget_mb * 1e6
    full = load_bundle(bundle)
    qz = Quantiser(full)
    centre = (qz.lo + qz.hi) / 2
    data = out / prefix  # binary files of this zone; URLs in the meta are relative to the page (prefix + name)
    data.mkdir(parents=True, exist_ok=True)
    cam = dict(cam or {})  # zones.json "cam": yaw_deg, pitch, dist_mul, target_z (default: median z, i.e. mid-slope)
    if "yaw_deg" in cam:
        cam["yaw"] = float(np.radians(cam.pop("yaw_deg")))
    if cam and "target_z" not in cam:
        cam["target_z"] = float(np.median(full.xyz[::97, 2]) - centre[2])

    # ---- zoom spots (cylinders of exactly N points of the full grid)
    spots_in = []
    for s in spots_arg:
        name, xy = s.split("=")
        vals = [float(v) for v in xy.split(",")]  # x,y[,view distance m[,view yaw deg[,view pitch rad]]]
        spots_in.append((name.strip().lower().replace(" ", "_"), name.strip(), tuple(vals[:2]), vals[2:]))
    spots_meta = []
    if spots_in:
        print("fitting spots to the budget")
        tree = cKDTree(full.xyz[:, :2])
        n_spot = fit_spot_count(full, qz, tree, spots_in, budget)
        (data / "spots").mkdir(exist_ok=True)
        for sid, label, xy, view in spots_in:
            idx, radius = spot_indices(tree, xy, n_spot)
            blob = _gz(qz.encode(full, idx))
            (data / "spots" / f"{sid}.bin.gz").write_bytes(blob)
            zc = float(full.xyz[idx, 2].mean())
            spots_meta.append({"id": sid, "label": label, "n": int(n_spot), "bytes": len(blob), "radius": radius,
                               "centre": [xy[0] - centre[0], xy[1] - centre[1], zc - centre[2]],
                               "url": f"{prefix}spots/{sid}.bin.gz"})
            if view:
                spots_meta[-1]["dist"] = view[0]
            if len(view) > 1:
                spots_meta[-1]["yaw"] = float(np.radians(view[1]))
            if len(view) > 2:
                spots_meta[-1]["pitch"] = view[2]
            print(f"spot {label}: {n_spot:,} pts, radius {radius:.0f} m, {len(blob) / 1e6:.2f} MB")

    # ---- overview / levels
    if args.levels == "auto":
        print("fitting the overview to the budget")
        voxels = [fit_overview_voxel(full, qz, budget)]
    else:
        voxels = [float(v) for v in args.levels.split(",")]
        assert voxels == sorted(voxels, reverse=True), "levels must go from coarse to fine"
    lod_dist = [float(v) for v in args.lod_dist.split(",")]
    assert len(lod_dist) >= len(voxels) - 1, "need one --lod-dist per level above 0"
    ov_size = overview_size or args.overview_size or round(OVERVIEW_SIZE_PER_VOXEL * voxels[0], 2)
    sizes = [ov_size] + [max(1.25 * v, args.spot_size) for v in voxels[1:]]

    nx = int(np.ceil(qz.hi[0] / args.tile)) + 1

    def tile_ids(b: Bundle) -> np.ndarray:
        return np.floor(b.xyz[:, 0] / args.tile).astype(int) + nx * np.floor(b.xyz[:, 1] / args.tile).astype(int)

    embed = len(voxels) == 1 and not spots_meta
    levels_meta, tile_n, first0, data_b64, total = [], {}, {}, "", 0
    for lvl, vox in enumerate(voxels):
        b = full.subsample(vox) if vox > 0.1 + 1e-9 else full
        tid = tile_ids(b)
        order = np.argsort(tid, kind="stable")
        ids, starts, counts = np.unique(tid[order], return_index=True, return_counts=True)
        for i, s, c in zip(ids.tolist(), starts.tolist(), counts.tolist()):
            tile_n.setdefault(i, [0] * len(voxels))[lvl] = c
            if lvl == 0:
                first0[i] = s
        if lvl == 0:
            blob = _gz(qz.encode(b, order))
            entry = {"voxel": vox, "size": sizes[0], "n": len(b)}
            if embed:
                data_b64 = base64.b64encode(blob).decode("ascii")
            else:
                (data / "overview.bin.gz").write_bytes(blob)
                entry["url"] = prefix + "overview.bin.gz"
            levels_meta.append(entry)
            total += len(blob)
            print(f"overview: voxel {vox} m, {len(b):,} pts, {len(blob) / 1e6:.2f} MB gz")
        else:
            (data / "tiles").mkdir(exist_ok=True)
            tsz = []
            for i, s, c in zip(ids.tolist(), starts.tolist(), counts.tolist()):
                blob = _gz(qz.encode(b, order[s : s + c]))
                (data / "tiles" / f"{lvl}_{i}.bin.gz").write_bytes(blob)
                tsz.append(len(blob))
            levels_meta.append({"voxel": vox, "size": sizes[lvl], "n": len(b)})
            total += sum(tsz)
            print(f"level {lvl}: voxel {vox} m, {len(b):,} pts, {len(tsz)} tiles, {sum(tsz) / 1e6:.1f} MB gz total")

    z_lo, z_hi = qz.lo[2] - centre[2], qz.hi[2] - centre[2]
    tiles = []
    for i in sorted(tile_n):
        bmin = [(i % nx) * args.tile - centre[0], (i // nx) * args.tile - centre[1], z_lo]
        tiles.append({"id": i, "n": tile_n[i], "first0": first0.get(i, 0),
                      "bmin": bmin, "bmax": [bmin[0] + args.tile, bmin[1] + args.tile, z_hi]})

    # ---- road graph, in centred coordinates, lifted by one radius so the tubes / balls sit on the ground
    g = full.graph
    c32 = centre.astype(np.float32)
    nodes = g["graph_nodes"] - c32
    nodes[:, 2] += 0.5 * NODE_DIAM_M
    segs = g["graph_seg"] - np.tile(c32, 2)
    segs[:, [2, 5]] += 0.5 * EDGE_DIAM_M
    network = {
        "nodes_b64": base64.b64encode(nodes.astype("<f4").tobytes()).decode(),
        "segs_b64": base64.b64encode(segs.astype("<f4").tobytes()).decode(),
        "edge": {"color": EDGE_COLOR, "diam": EDGE_DIAM_M, "count": int(g["graph_n_edges"])},
        "node": {"color": NODE_COLOR, "diam": NODE_DIAM_M, "count": int(len(nodes))},
    }

    n_all = len(full)
    columns = []
    for key, pal_key in CAT_COLUMNS:
        ids, counts = np.unique(full.arrays[key], return_counts=True)
        pal = palette(full, pal_key)
        columns.append({"names": pal["names"], "colors": pal["colors"],
                        "shares": {str(int(i)): float(c) / n_all for i, c in zip(ids, counts)}})
    meta = {
        "title": title, "credit": args.credit, "zones": zones, "page": page, "zone": zone_id, "cam": cam, "disabled": disabled or [], "layers": LAYERS, "n_full": len(full), "voxel": full.meta["voxel"], "columns": columns, "network": network,
        "levels": levels_meta, "tiles": tiles, "lod_dist": lod_dist, "budget": args.lod_budget,
        "tile_url": prefix + "tiles/{l}_{i}.bin.gz", "spots": spots_meta, "spot_size": args.spot_size,
        "pos_scale": qz.scale.tolist(), "pos_offset": (qz.lo - centre).tolist(),
        "radius": float(np.linalg.norm(qz.hi - qz.lo) / 2), "extent": (qz.hi - qz.lo).tolist(),
        "elevation_range": [palette(None, "elevation")["vmin"], palette(None, "elevation")["vmax"]],
        "elevation_ramp": (ramp("elevation") / 255).round(5).tolist(),
        "intensity_ramp": (ramp("intensity") / 255).round(5).tolist(),
        "void": palette(None, "elevation")["nan_color"],
    }
    for template, name, suffix in [(TEMPLATE, page, "")] + ([(TEMPLATE_THREE, _three_name(page), " (three.js)")] if args.three else []):
        html = (template.read_text(encoding="utf-8").replace("__TITLE__", title + suffix)
                .replace("__META__", json.dumps(meta, ensure_ascii=False)).replace("__DATA__", data_b64))
        (out / name).write_text(html, encoding="utf-8")
    print(f"wrote {out / page} ({(out / page).stat().st_size / 1e6:.2f} MB); "
          f"first screen {levels_meta[0]['n']:,} pts; spots {len(spots_meta)}; level files total {total / 1e6:.1f} MB")



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bundle", default="outputs/colmar_roi/bundle.npz")
    ap.add_argument("--out-dir", default="outputs/colmar_roi/web")
    ap.add_argument("--budget-mb", type=float, default=5.0, help="gzip size of the overview and of each spot file")
    ap.add_argument("--spot", action="append", help='zoom spot "Name=x,y" in local ROI metres (repeatable)')
    ap.add_argument("--levels", default="auto",
                    help='"auto" (one budget-sized overview) or voxel sizes coarse to fine, e.g. 1.5,0.4,0.1 (enables tile LOD)')
    ap.add_argument("--lod-dist", default="250,90", help="LOD: eye-to-tile distances (m) below which level 1, 2, ... load")
    ap.add_argument("--tile", type=float, default=64.0, help="tile edge (m)")
    ap.add_argument("--lod-budget", type=float, default=3.5e6, help="LOD: max points in GPU memory for levels >= 1")
    ap.add_argument("--overview-size", type=float, help="point size (m) of the overview (level 0); default 0.85 x its voxel")
    ap.add_argument("--spot-size", type=float, default=SPOT_SIZE_M, help="point size (m) of the zoom spots; keep it close to the overview size")
    ap.add_argument("--three", action="store_true", help="also write index_three.html, the three.js version of the viewer (+ vendor/)")
    ap.add_argument("--zones", help="JSON list of zones (id, page, label, bundle, spots): one page per zone with a zone "
                    "switcher, data under <out-dir>/<id>/; overrides --bundle / --spot / --title")
    ap.add_argument("--title", default="Colmar city centre")
    ap.add_argument("--credit", default="LiDAR HD © IGN")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if args.zones:
        cfg = json.loads(Path(args.zones).read_text())
        zones = [{"id": z["id"], "label": z["label"], "page": z["page"], "roi": z.get("roi", "")} for z in cfg]
        for z in cfg:
            print(f"==== zone {z['id']} ({z['bundle']})")
            export_zone(args, out, z["bundle"], z["label"], z.get("spots", []), z["page"], z["id"] + "/", z["id"], zones, z.get("cam"),
                        z.get("overview_size"), z.get("disabled_layers"))
    else:
        export_zone(args, out, args.bundle, args.title, args.spot if args.spot is not None else DEFAULT_SPOTS)
    shutil.copy(Path(__file__).with_name("embed_example.html"), out / "embed_example.html")
    if args.three:
        shutil.copytree(Path(__file__).with_name("vendor"), out / "vendor", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
