# Data preparation

Commands are run from the repository root with `export PYTHONPATH=./`. Every dataset ends up under
`data/<dataset>/`, the path read by the configs.

## MALiBU3D

Download the dataset from [Hugging Face](https://huggingface.co/datasets/LouisGeist/MALiBU3D) (790 GB of zipped
regions, 1.6 TB once extracted) and extract it:

```bash
hf download LouisGeist/MALiBU3D --repo-type dataset --local-dir data/malibu3d_zip
python pointcept/datasets/preprocessing/malibu3d/unzip_malibu3d.py \
    --zip-root data/malibu3d_zip --output-root data/malibu3d --workers 8
```

`unzip_malibu3d.py` resumes an interrupted extraction (a region whose files are all present with the right size is
skipped), copies `tiles.csv`, `labels.json`, `palettes.json` and `val_subset_2000tiles.csv` next to the tiles, and
puts the `toy/` region of the Hub in the same layout (its tiles are listed in `tiles_toy.csv`). A single region can
be downloaded with `--include "data/train/D075-2021_LIDARHD/*"` and the same metadata files.

Layout read by `MALiBU3DDataset`:

```text
data/malibu3d/<split>/<DEPT_YEAR>_LIDARHD/<ROI>/<TILE_ID>/
├── coord.npy             float32 (N, 3) xyz, relative to coord_translation
├── coord_translation.npy float64 (3,) offset (absolute Lambert-93 = coord + coord_translation)
├── color.npy             uint8 (N, 3) RGB
├── strength.npy          float32 (N,) intensity, in [0, 1]
├── segment.npy           uint8 (N,) land-cover class (15 classes, void = 15)
├── natural_habitat.npy   uint8 (N, 4) ecological axes (tiles with CarHab coverage)
├── elevation.npy         float32 (N,) height above the DTM (tiles with a DTM)
├── forest_2d.npy         uint8 (1, H, W) 1 m forest mask
└── network.npy           uint8 (1, H, W) 1 m road mask     (rasterize_network.py, below)
```

with `TILE_ID = <DEPT_YEAR>_<ROI>_<I-J>`, e.g. `D010-2019_AA-S1-1_1-1`. The 1 m rasters are south-up (row 0 is
the southern edge) on a grid snapped to whole Lambert-93 meters. `tiles.csv` lists every tile with its split and
the availability of each label (`has_natural_habitat`, `has_elevation`, `has_roads_graph`); it is read through
`pointcept/datasets/tile_catalog.py`. The [dataset card](https://huggingface.co/datasets/LouisGeist/MALiBU3D)
describes all columns.

### Road masks

The road network target is shipped as graphs (`<DEPT>_<ROI>_ROADS_graph.gpkg` in each ROI directory, layers
`nodes` / `edges` / `metadata`, EPSG:2154), which the APLS evaluation compares to the predicted graphs. Training
uses 1 m binary masks rasterized once from these graphs:

```bash
python pointcept/datasets/preprocessing/malibu3d/rasterize_network.py \
    --data_root data/malibu3d --split_manifest_csv data/malibu3d/tiles.csv --num_workers 16
```

Each ROI graph is densified once (`--sample_step_m`, 0.25 m, along each edge; with `--line_width_m` 0.2 > 0, two
samples at ± 0.1 m across the edge instead of the centerline), the touched 1 m cells are collected, then sliced to
each tile's grid.

- Graphs are read from each ROI directory, or from `--network_graphs_root` when given.
- A tile with `has_roads_graph = True` whose graph is missing, or a tile without `coord.npy`, is an error.
- A tile without any road pixel gets no `network.npy`, only its grid in `meta.json["network"]` with
  `"empty": true`; `MALiBU3DDataset` then uses an all-zero mask.
- Other options: `--splits`, `--pixel_m`, `--force_reload_bounds` (recompute the tile bounding box instead of
  reading it from `meta.json`), `--max_rois` (debug).

The mask -> graph post-processing and the APLS metric live in `pointcept/evaluation/network/`.

### Validation subset

The training configs validate on `val_subset_2000tiles.csv`: a fixed subset of 2,000 validation tiles,
representative of the label distribution of the whole validation split. Test metrics always use the full test
split. To validate on the whole validation split, add `--options data.val.stratified_subset_manifest=False`.

## Downstream datasets

| Dataset | Task | Metric | Source |
|---|---|---|---|
| DALES | semantic segmentation, 8 classes | mIoU | [University of Dayton](https://www.udayton.edu/engineering/centers/vision_lab/research/was_data_analysis_and_processing/dale.php) (`DALESObjects` PLY release) |
| H3D | semantic segmentation, 11 classes | macro-F1 | [IFP, University of Stuttgart](https://ifpwww.ifp.uni-stuttgart.de/benchmark/hessigheim/default.aspx) (epoch March 2018) |
| ECLAIR | semantic segmentation, 11 classes | mIoU | [SharperShape/eclair-dataset](https://github.com/SharperShape/eclair-dataset) |
| OpenGF | ground filtering | IoU of the ground class (Test II, without outliers) | [Nathan-UW/OpenGF](https://github.com/Nathan-UW/OpenGF) |
| PureForest | tile classification, 13 tree species | mIoU | [IGNF/PureForest](https://huggingface.co/datasets/IGNF/PureForest) |

Preprocessing writes Pointcept scene folders under `data/<dataset>/{train,val,test}/`. `laspy` and `lazrs` read
the LAZ files of H3D, ECLAIR, OpenGF and PureForest.

```bash
# DALES: extract the archive so that data/dales/raw/DALESObjects/{train,test}/*.ply exists
python pointcept/datasets/preprocessing/dales/preprocess_dales.py \
    --dataset_root data/dales/raw/DALESObjects --output_root data/dales --num_workers 8

# H3D: data/h3d/Epoch_March2018/LiDAR/Mar18_{train,val}.laz and Mar18_test_GroundTruth.las
python pointcept/datasets/preprocessing/h3d/preprocess_h3d.py \
    --dataset_root data/h3d/Epoch_March2018 --output_root data/h3d --chunk-size 100 --num_workers 8

# ECLAIR: data/eclair/raw/{labels.json, pointclouds/*.laz}
python pointcept/datasets/preprocessing/eclair/preprocess_eclair.py \
    --dataset_root data/eclair/raw --output_root data/eclair --num_workers 8

# OpenGF: data/opengf/raw/{Training,Validation,Test}/...
python pointcept/datasets/preprocessing/opengf/preprocess_opengf.py \
    --dataset_root data/opengf/raw --output_root data/opengf --num_workers 8
```

DALES has no validation split: its configs validate on the test split, as is common practice on this dataset.

### PureForest

Download (or clone) the Hugging Face dataset to `data/pureforest/PureForest` (it contains `data/lidar-*.zip` and
`metadata/`), unzip the LiDAR archives (`-n` skips files already extracted) and link the metadata next to them:

```bash
REPO=data/pureforest
mkdir -p "${REPO}/extracted"
for z in "${REPO}/PureForest/data"/lidar-*.zip; do
  unzip -n "$z" -d "${REPO}/extracted"
done
ln -sfn ../PureForest/metadata "${REPO}/extracted/metadata"

python pointcept/datasets/preprocessing/pureforest/preprocess_pureforest.py \
  --dataset_root data/pureforest/extracted --output_root data/pureforest --num_workers 8
```

Each row of `metadata/PureForest-patches.csv` (`patch_id`, `split`, `class_index`) is one tile, read from
`lidar/{split}/{TRAIN|VAL|TEST}-{patch_id}.laz`; LAZ files that are not in the table are ignored. The output is

```text
data/pureforest/{train,val,test}/<patch_id>/
├── coord.npy      # float32: XY centered on the mean, Z shifted to its minimum, divided by 25 m
├── color.npy      # float32 RGB in [0, 255]
└── category.npy   # int32 class id
```

and `PureForestDataset` multiplies `coord` back by 25 m at load time. Re-running the command resumes (complete tiles
are skipped); `missing_laz_preflight.txt` lists the table rows whose LAZ file is absent, and
`preprocess_errors.log` the tiles that failed. Other options: `--manifest_csv`, `--splits`, `--overwrite`, and
`--toy` (the first 2 tiles of each class in each split, about 78 tiles; use a separate `--output_root`).

**MALiBU3D leakage.** Two PureForest test forests (BD Forêt polygons `Quercus_pubescens-C0-730` and
`Quercus_robur-C0-389`, 2,170 of the 52,935 test tiles) overlap MALiBU3D train/val tiles. The configs probing
MALiBU3D backbones set `exclude_malibu3d_leakage_tiles=True` on `PureForestDataset` to drop every tile of these
forests (`MALIBU3D_LEAKAGE_FORESTS` in `pointcept/datasets/pureforest.py`).
