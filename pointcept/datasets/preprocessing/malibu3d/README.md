# MALiBU3D road network tools

The Hugging Face release ships the MALiBU3D tiles ready for training (see `scripts/unzip_malibu3d.py`), with the
layout read by `MALiBU3DDataset`:

```text
data/malibu3d/<split>/<DEPT_YEAR>_LIDARHD/<ROI>/<TILE_ID>/
├── coord.npy             float32 xyz, relative to coord_translation
├── coord_translation.npy float64 xyz offset (absolute Lambert-93 = coord + coord_translation)
├── color.npy             uint8 RGB
├── strength.npy          float32 intensity, in [0, 1]
├── segment.npy           uint8 land-cover class (15 classes + void = 15)
├── natural_habitat.npy   uint8 (N, 4) ecological axes (tiles with CarHab coverage)
├── elevation.npy         float32 height above the DTM (tiles with a DTM)
├── forest_2d.npy         uint8 (1, H, W) 1 m forest mask
└── network.npy           uint8 (3, H, W) 1 m network masks     (rasterize_network.py, below)
```

with `TILE_ID = <DEPT_YEAR>_<ROI>_<I-J>`, e.g. `D010-2019_AA-S1-1_1-1`. The 1 m rasters are south-up (row 0 is
the southern edge) on a grid snapped to whole Lambert-93 meters. Commands are run from the repository root and
driven by `tiles.csv` (read through `tile_catalog.py`).

The road masks of the `network` target are not shipped: they are rasterized from the road graphs of each ROI.

## `rasterize_network.py`: road graphs -> `network.npy`

The `network` target is stored as graphs (`<DEPT>_<ROI>_<NETWORK>_graph.gpkg`, layers `nodes` / `edges` /
`metadata`, EPSG:2154), one per ROI and network type. This script turns them into the `(3, H, W)` binary masks
used for training, channels `ROADS, RAILROADS, TRANSMISSION_LINES`. Each ROI graph is densified once
(`--sample_step_m`, 0.25 m, along each edge; with `--line_width_m` 0.2 > 0, two samples at ± 0.1 m across the
edge instead of the centerline), the touched 1 m cells are collected, then sliced to each tile's grid.

```bash
python pointcept/datasets/preprocessing/malibu3d/rasterize_network.py \
    --data_root data/malibu3d \
    --split_manifest_csv data/malibu3d/tiles.csv \
    --num_workers 16
```

- Graphs are read from each ROI directory (layout of the Hugging Face release), or from `--network_graphs_root`
  when given.
- A tile with `has_roads_graph` (`ROADS`) = True whose ROADS graph is missing is an error. RAILROADS and
  TRANSMISSION_LINES have no flag and are rasterized when their graph exists.
- A tile without any positive pixel gets no `network.npy`, only its grid in `meta.json["network"]` with
  `"empty": true`; `MALiBU3DDataset` then uses an all-zero mask.
- A tile without `coord.npy` is an error, unless it is listed in `--missing_tiles_file`. Other options:
  `--splits`, `--pixel_m`, `--force_reload_bounds` (recompute the tile bounding box instead of reading it from
  `meta.json`), `--max_rois` (debug).

## Network graph and APLS modules

The network task is evaluated with APLS on graphs, not only on masks. These modules are imported by
`tools/eval_network_apls.py`, `pointcept/utils/network_apls.py` and the tests; they have no command line.

| Module | Role |
|---|---|
| `network_xy_raster_utils.py` | 1 m Lambert-93 grids (`GridSpec`), segment densification, masks, pixel graphs, RDP simplification, endpoint repair |
| `network_label_utils.py` | Read the ground-truth `*_graph.gpkg` (segments or full topology), write predicted graphs to GeoPackage |
| `network_graph_pipeline.py` | Predicted mask -> graph: opening/closing, small-object removal, skeletonization, pixel graph, endpoint fix, RDP, node merge, small-component removal |
| `network_prediction_stitch.py` | Stitch the per-tile predicted probability rasters (`<tile_id>_logits_network.npy`) of a ROI onto one ROI raster |
| `apls_metric.py` | APLS between a ground-truth and a predicted graph (fixed protocol, `APLS_PROTOCOL`) and dataset aggregation |

## Other modules

- `tile_catalog.py`: reads `tiles.csv` into `Tile` records.
