# Suppmat zone bundle (single subtile)

Package one Flair3D subtile (~100 m × 100 m) into a self-contained folder for
Pointcept supplementary material: enhanced PLY, Pointcept `.npy` sidecars, ROADS
GeoPackage, color palettes, and a standalone viser viewer.

## 1. Package (on hecate, Flair3D-build repo)

```bash
python scripts/package_suppmat_zone.py \
  ply_path=/data/geist/Flair3D-build/data/flair3d_label_enhanced/LIDARHD/D075-2021_LIDARHD/UU-S1-4/D075-2021_LIDARHD_UU-S1-4_3-3.ply \
  output_dir=suppmat_zones/D075_UU-S1-4_3-3 \
  hydra.job.chdir=false
```

Default output when `output_dir` is omitted: `suppmat_zones/{D075_UU-S1-4_3-3}/`.

### Bundle contents

| File | Role |
|------|------|
| `D075_UU-S1-4_3-3.ply` | Enhanced LiDAR (XYZ Lambert-93, RGB, semantic) |
| `coord.npy`, `color.npy`, `segment.npy` | Pointcept scene arrays |
| `elevation.npy`, `natural_habitat.npy`, `forest.npy`, `strength.npy` | Per-point GT sidecars |
| `coord_translation.npy`, `meta.json` | Scene metadata |
| `D075_UU-S1-4_ROADS.gpkg` | BDTOPO roads (ROI-level; corridor only hits subtile points) |
| `palettes.json` | Discrete LUTs (semantic v20, forest, natural_habitat) + elevation ramp |
| `zone_meta.json` | Manifest (paths, CRS, label definitions) |
| `visualize_suppmat_zone_viser.py` | Standalone viewer |
| `suppmat_network_utils.py` | Vendored network corridor helpers |
| `requirements-suppmat-vis.txt` | Python deps for the viewer |

Copy the whole folder into Pointcept-suppmat.

## 2. Visualize (standalone)

From the bundle folder (no Flair3D-build checkout required):

```bash
pip install -r requirements-suppmat-vis.txt
python visualize_suppmat_zone_viser.py --zone-dir .
```

Or from the repo before packaging:

```bash
python scripts/visualize/visualize_suppmat_zone_viser.py \
  --zone-dir suppmat_zones/D075_UU-S1-4_3-3
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

## Config

Hydra config: [`conf/config_package_suppmat_zone.yaml`](conf/config_package_suppmat_zone.yaml).

Useful overrides:

| Override | Effect |
|----------|--------|
| `pointcept_data_root=...` | Root for resolving Pointcept scene `.npy` |
| `networks_root=...` | BDTOPO source for `{dept}_{roi}_ROADS.gpkg` (default `/data/geist/datasets/flair-networks`) |
| `copy_viewer_scripts=false` | Skip copying viewer scripts into the bundle |

## Note on coordinates

The enhanced PLY uses absolute Lambert-93 coordinates. The network GPKG is at ROI
extent; only points near road centerlines within the subtile are tinted in
**Network corridor** mode.
