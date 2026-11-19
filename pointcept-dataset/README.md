# MALiBU3D land cover for Pointcept

15-class semantic segmentation on [MALiBU3D](https://huggingface.co/datasets/LouisGeist/MALiBU3D), as a single dataset file for a vanilla [Pointcept](https://github.com/Pointcept/Pointcept) checkout. No multitask code, no road rasters.

## Copy into Pointcept

From your Pointcept repository root, with this `pointcept-dataset/` folder available:

```bash
cp pointcept-dataset/pointcept/datasets/malibu3d.py pointcept/datasets/malibu3d.py
mkdir -p configs/malibu3d
cp pointcept-dataset/configs/malibu3d/semseg-pt-v3m1-0-base.py configs/malibu3d/
```

In `pointcept/datasets/__init__.py`:

```python
from .malibu3d import MALiBU3DDataset
```

## Data

Extract the release so tiles look like this (`TILE_ID` is `<DEPT_YEAR>_<ROI>_<I-J>`):

```text
data/malibu3d/<split>/<DEPT_YEAR>_LIDARHD/<ROI>/<TILE_ID>/
├── coord.npy      float32 (N, 3) xyz, meters, relative to the tile origin
├── color.npy      uint8 (N, 3) RGB
├── strength.npy   float32 (N,) intensity in [0, 1]
└── segment.npy    uint8 (N,) class id, Void = 15
data/malibu3d/tiles.csv
```

`coord.npy` is not shifted by `coord_translation.npy`. Class ids `0..14` are the names in `MALIBU3D_CLASS_NAMES`; `15` is Void and the loss `ignore_index`. Nothing is remapped to `-1`.

Some tiles are entirely Void: 520 in train, 56 in val (4 of them in the 2,000-tile subset) and 255 in test, for example glaciers (COSIA snow) or tidal flats where LiDAR HD and COSIA disagree. On such a batch, upstream `LovaszLoss` crashes (`output with shape [] doesn't match the broadcast shape [0, 15]`) and upstream `CrossEntropyLoss` returns NaN. Validation runs one tile per batch, so the first validation hits it. `malibu3d.py` therefore registers `MALiBU3DCrossEntropyLoss` and `MALiBU3DLovaszLoss`. They give the same values as the upstream losses and return 0 on an all-Void batch. The config uses them; keep them in your own configs.

The training repo's `pointcept/datasets/preprocessing/malibu3d/unzip_malibu3d.py` builds this layout from the Hugging Face zips. Land cover does not need `rasterize_network.py`.

`MALiBU3DDataset` reads `tiles.csv` (or the `csv_manifest` you pass). Train drops tiles with `n_points < 1000`. Val and test stay complete. A directory listed in the CSV and missing on disk is an error: point `csv_manifest` at a CSV of only the tiles you extracted (`tiles_toy.csv` for the toy region).

To validate on the paper's 2,000-tile subset instead of the full val split, set `subset_manifest="data/malibu3d/val_subset_2000tiles.csv"` on `data.val` only.

## Train

`PT-v3m1`, `DefaultSegmentorV2`, grid 0.1 m, features `coord * 0.01` + RGB + intensity. The schedule is 20 epochs with 4 validations. It is a starting recipe, not the multitask numbers from the paper.

```bash
sh scripts/train.sh -g 4 -d malibu3d -c semseg-pt-v3m1-0-base -n malibu3d-landcover
```

`-c` is the config name under `configs/malibu3d/`. Batch 12 is sized for 4 GPUs; lower `batch_size` in the config if you run out of memory.
