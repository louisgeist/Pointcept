# Colmar city-centre ROI viewer

ROI `D068-2021_LIDARHD/UU-S1-12` (test split, 50 sub-tiles, ~0.5 km², 9.3 M points on the 0.1 m delivery grid), plus two more
test ROIs of D068 in the same site (`zones.json`): "Rural area" (`UA-S1-10`, near Mulhouse, 17.5 M points) and "Mountain forest" (Vosges,
`FF-S1-16`, 43.8 M points; no road network, layer greyed out via `disabled_layers`). Data: MALiBU3D v2 (cm-accurate coordinates), copied from Jean Zay
`$SCRATCH/data_pointcept/flair3d_plus_v2/test/D068-2021_LIDARHD/<roi>` to `outputs/colmar_roi/roi_v2/<roi>`.
Layers — **Input**: RGB, LiDAR intensity. **Tasks**: semantic v20, elevation (m above DTM), road network
(BDTOPO graph: sky-blue tubes for the edges, dark-blue balls for the nodes, draped on the road surface incl. bridge decks), forest, and the four
natural habitat, one task split into four ecological axes (habitat type, moisture regime, soil chemistry, bioclimatic zone).
Colours: `palette_paper.json` (paper palette given by Louis; elevation = coolwarm 0–30 m and intensity ramps copied from the Blender pipeline `drprojects/blendify`, `configs/palettes/malibu3d_extra.json`). Changing it only needs a re-export, not a bundle rebuild.

```bash
python tools/colmar_roi/build_bundle.py                           # -> outputs/colmar_roi/bundle.npz (0.1 m voxel + road graph)
python tools/colmar_roi/viser_viewer.py --voxel 0.4 --port 8890   # live viser viewer (localhost only)
python tools/colmar_roi/export_html.py --budget-mb 5              # -> outputs/colmar_roi/web/ (static site folder)
# other zones: build_bundle.py --roi-dir outputs/colmar_roi/roi_v2/<roi> --graph <D068_<roi>_ROADS_graph.gpkg> --out outputs/colmar_roi/bundle_<roi>.npz
python tools/colmar_roi/export_html.py --zones tools/colmar_roi/zones.json --three   # all zones, one page each + zone switcher
```

With `--zones`, each zone gets its page (`index.html` = Colmar, `farmland.html`, `forest.html`) and its files under `<id>/`; the
"Zone" section of the panel switches page and keeps the layer, theme and panel state. Per zone, `cam` sets the default view
(`yaw_deg`, `pitch`, `dist_mul`, `target_z`) and a spot can carry its own view: `"Name=x,y,dist,yaw_deg,pitch"`.
The overview point size defaults to 0.85 x its voxel (1.4 m for Colmar, 2.9 m for the forest at a 3.4 m voxel).

## Web export (`export_html.py`)

One download budget (`--budget-mb`, gzip) sizes everything:
- `overview.bin.gz`: whole ROI at the coarsest voxel that fits (1.66 m, 486 k points, 4.9 MB). Only this is loaded at start.
- `spots/*.bin.gz`: around each zoom spot, the N nearest points of the full 0.1 m grid (a cylinder of exactly N = 541 274
  points, same for every spot, ~5 MB; data v2 with cm-accurate coordinates). The viewer flies there on click, downloads the file and shows it in place of the
  overview inside the cylinder. Spots: `--spot "Name=x,y"` in local ROI metres (defaults: Bridge, Church).
- `--levels 1.5,0.4,0.1` additionally enables automatic per-tile LOD streaming by camera distance (`--lod-dist`).

Serve over http(s) (`file://` does not work unless there is a single level and no spot):
`cd outputs/colmar_roi/web && python3 -m http.server 8893 --bind 127.0.0.1`.
Page hash options: `#layer=1..10&theme=dark&collapsed=1&spot=church&cam=yaw,pitch,dist,x,y,z` (for `<iframe>` embedding).

`embed_example.html` (copied next to `index.html`) shows how to embed the viewer in another page with an iframe and click-to-load.

`export_html.py --three` also writes `index_three.html`, the same viewer on three.js r160.1 (pinned in `vendor/`, MIT): identical output on the checked views; the per-tile LOD streaming (`--levels` with several levels) is not ported. UI code is duplicated between `viewer_template.html` and `viewer_template_three.html`, so UI changes must be made twice.
