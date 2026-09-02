# Malibu3D sample tile (single subtile)

A self-contained **Malibu3D** subtile (~100 m × 100 m) for supplementary visualization: LiDAR PLY, Pointcept `.npy` sidecars, ROADS GeoPackage, color palettes, and a standalone [viser](https://github.com/nerfstudio-project/viser) viewer.

The bundle is included under [`suppmat_zones/D075_UU-S1-4_3-3/`](suppmat_zones/D075_UU-S1-4_3-3/).

## Visualize

From the repo root:

```bash
pip install -r requirements-suppmat-vis.txt
python scripts/visualize/visualize_suppmat_zone_viser.py \
  --zone-dir suppmat_zones/D075_UU-S1-4_3-3
```

Or from the bundle folder:

```bash
pip install -r requirements-suppmat-vis.txt
python visualize_suppmat_zone_viser.py --zone-dir .
```

Open http://localhost:8080. **Shift+click** inspects a point.

### Display modes

- **RGB** — orthophoto colors
- **Semantic** — land-cover (v20 palette)
- **Natural habitat** — nathab class
- **Forest** — forest cover (Not Forest / Forest / Void)
- **Elevation** — DEM elevation (m)
- **Strength** — normalized LiDAR intensity `[0, 1]`
- **Network corridor** — RGB + ROADS overlay (2.5 m XY corridor)

## Bundle contents

| File | Role |
|------|------|
| `D075_UU-S1-4_3-3.ply` | LiDAR (XYZ Lambert-93, RGB, semantic) |
| `coord.npy`, `color.npy`, `segment.npy` | Pointcept scene arrays |
| `elevation.npy`, `natural_habitat.npy`, `forest.npy`, `strength.npy` | Per-point GT sidecars |
| `coord_translation.npy`, `meta.json` | Scene metadata |
| `D075_UU-S1-4_ROADS.gpkg` | BDTOPO roads (ROI-level; corridor only hits subtile points) |
| `palettes.json` | Discrete LUTs (semantic v20, forest, natural_habitat) + elevation ramp |
| `zone_meta.json` | Manifest (paths, CRS, label definitions) |
| `visualize_suppmat_zone_viser.py` | Standalone viewer |
| `suppmat_network_utils.py` | Network corridor helpers |
| `requirements-suppmat-vis.txt` | Python deps for the viewer |

## Note on coordinates

The PLY uses absolute Lambert-93 coordinates. The network GPKG is at ROI extent; only points near road centerlines within the subtile are tinted in **Network corridor** mode.
