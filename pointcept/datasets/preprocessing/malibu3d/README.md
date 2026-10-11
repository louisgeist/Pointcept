# MALiBU3D preprocessing

Scripts that build the MALiBU3D Pointcept scene folders and the network (road) graph tools. Commands are run
from the repository root. Every script is driven by the tile table: the MALiBU3D `tiles.csv` (one row per
LiDAR tile) or an older `scene_split_manifest.csv`, both read through `tile_catalog.py`. Disk contents are only
checked against it, never globbed.

Output layout, shared by all scripts and by `MALiBU3DDataset`:

```text
data/malibu3d/<split>/<DEPT_YEAR>_LIDARHD/<ROI>/<TILE_ID>/
├── coord.npy             float32 xyz, relative to coord_translation
├── coord_translation.npy float64 xyz offset (absolute Lambert-93 = coord + coord_translation)
├── color.npy             uint8 RGB
├── strength.npy          float32 intensity / 60000, clipped to [0, 1]
├── segment.npy           uint8 land-cover class (15 classes + void = 15)
├── natural_habitat.npy   uint8 (N, 4) ecological axes (tiles with CarHab coverage)
├── elevation.npy         float32 height above the DTM (tiles with a DTM)
├── forest_2d.npy         uint8 (1, H, W) 1 m forest mask       (rasterize_forest.py)
├── network.npy           uint8 (3, H, W) 1 m network masks     (rasterize_network.py)
└── meta.json
```

with `TILE_ID = <DEPT_YEAR>_<ROI>_<I-J>`, e.g. `D010-2019_AA-S1-1_1-1`. The 1 m rasters are south-up (row 0 is
the southern edge) on a grid snapped to whole Lambert-93 meters.

## 1. `unzip_raw_archives.py`: raw archives -> `raw/`

Extracts the per-modality zip archives of `--source_root/<MODALITY>/*.zip` (modalities `DEM_ELEV`, `FOREST`,
`LIDARHD`, `NATURAL_HABITAT`) into `--target_root`, and copies the side-car files found at the source root
(class lists, date-gap GeoPackage, ...).

```bash
python pointcept/datasets/preprocessing/malibu3d/unzip_raw_archives.py \
    --source_root /path/to/archives \
    --target_root data/malibu3d/raw \
    --workers 8
```

- `--workers N`: threads per modality (one archive per thread). `DEM_ELEV` is always extracted sequentially.
- Without `--overwrite`, files that already exist are skipped, so an interrupted run can be resumed.

Resulting layout (same tile name in every modality, only the modality token and extension change):

```text
raw/
├── LIDARHD/<DEPT_YEAR>_LIDARHD/<ROI>/<DEPT_YEAR>_LIDARHD_<ROI>_<I-J>.ply
├── FOREST/<DEPT_YEAR>_FOREST/<ROI>/<DEPT_YEAR>_FOREST_<ROI>_<I-J>.tif
├── NATURAL_HABITAT/<DEPT_YEAR>_NATURAL_HABITAT/<ROI>/<DEPT_YEAR>_NATURAL_HABITAT_<ROI>_<I-J>.tif
└── DEM_ELEV/<DEPT_YEAR>_DEM_ELEV/<ROI>/<DEPT_YEAR>_DEM_ELEV_<ROI>_<I-J>.tif
```

`DEM_ELEV` archives that unpack directly to `<ROI>/` are moved under `<DEPT_YEAR>_DEM_ELEV/`.

## 2. `preprocess_malibu3d.py`: PLY + rasters -> scene folders

Reads labeled PLY files (fields `x y z red green blue intensity semantic`) from `--ply_root`, with the layout
of `raw/LIDARHD` above, and samples the `NATURAL_HABITAT` and `DEM_ELEV` GeoTIFFs of `--dataset_root` at
each point (nearest pixel).

```bash
python pointcept/datasets/preprocessing/malibu3d/preprocess_malibu3d.py \
    --ply_root data/malibu3d/ply \
    --dataset_root data/malibu3d/raw \
    --output_root data/malibu3d \
    --split_manifest_csv data/malibu3d/tiles.csv \
    --num_workers 16
```

| Option | Default | Effect |
|---|---|---|
| `--split` | `train val test` | Splits to process |
| `--segment_definition` | `v20` | Land-cover label definition (`malibu3d_label_remap.py`) |
| `--natural_habitat_definition` | `default` | CarHab definition used when sampling, before conversion to the 4 axes |
| `--force` | off | Reprocess tiles whose `coord.npy` already exists |
| `--log_file`, `--missing_scenes_file`, `--missing_ply_preflight_file` | under `--output_root` | Report paths |

Per tile (columns of `tiles.csv`; older names in parentheses):

- `has_natural_habitat` (`NATURAL_HABITAT`) = True: the CarHab raster is sampled and stored as
  `natural_habitat.npy`, `uint8 (N, 4)`: habitat type, moisture regime, soil chemistry, bioclimatic zone
  (`nathab_axes.py`).
- `has_elevation` (`DEM_ELEV`) = True: `elevation.npy` = z minus the DTM (band 2 of the DEM_ELEV GeoTIFF);
  NaN outside the raster.
- `date_gap_days` is copied to `meta.json`, along with the label definitions and the axes layout.

Required columns: `split, dept_year, roi, scene_i_j, tile_id, has_natural_habitat, has_elevation,
date_gap_days`.

Reports under `--output_root`:

- `missing_ply_preflight.txt`, written before any tile is processed: tiles whose PLY file is missing;
- `missing_scenes.txt`, written at the end: missing PLY files, missing rasters (flag True but no GeoTIFF) and
  failed tiles;
- `preprocess_malibu3d.log`.

Resume: `coord.npy` is written last, so a tile that has it is complete and is skipped. Use `--force` after
changing a label definition, the modality flags or the PLY labels: the skip only checks `coord.npy`.

## 3. `rasterize_forest.py`: FOREST GeoTIFF -> `forest_2d.npy`

Reads the window of each tile's FOREST GeoTIFF that covers the tile's point bounding box and writes it on the
tile's 1 m grid: a direct read when the GeoTIFF already has that resolution and is aligned with the grid,
otherwise a majority-vote resampling. Values: 0 = not forest, 1 = forest, `--ignore_index` (2) = no data.
The grid is stored in `meta.json["forest_2d"]`.

```bash
python pointcept/datasets/preprocessing/malibu3d/rasterize_forest.py \
    --data_root data/malibu3d \
    --source_dataset_root data/malibu3d/raw \
    --split_manifest_csv data/malibu3d/tiles.csv \
    --num_workers 8
```

A tile without `coord.npy` is an error, unless it is listed in the known-missing file (`--missing_tiles_file`,
default `data/malibu3d/missing_coord_tiles.details.csv` when it exists). A missing GeoTIFF is reported and makes
the run fail at the end. Other options: `--splits`, `--pixel_m` (1.0), `--force_reload_bounds` (recompute the
bounding box instead of reading it from `meta.json`).

## 4. `rasterize_network.py`: road graphs -> `network.npy`

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
- Missing `coord.npy` and `--missing_tiles_file` behave as in `rasterize_forest.py`. Other options:
  `--splits`, `--pixel_m`, `--force_reload_bounds`, `--max_rois` (debug).

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

- `tile_catalog.py`: reads `tiles.csv` or `scene_split_manifest.csv` into `Tile` records (column mapping in its
  docstring).
- `malibu3d_label_remap.py`: label definitions (raw id -> train id LUTs, class names, ignore index) of
  `segment` and `natural_habitat`.
- `nathab_axes.py`: layout of `natural_habitat.npy` and the CarHab -> ecological axes LUTs.
