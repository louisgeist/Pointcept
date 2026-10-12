# PureForest preprocessing

Tile-level tree-species classification (13 classes) on IGN
[PureForest](https://huggingface.co/datasets/IGNF/PureForest). Commands are run from the repository root.

## 1. Download and extract

Download (or clone) the Hugging Face dataset to `data/pureforest/PureForest` (it contains
`data/lidar-*.zip` and `metadata/`), then unzip the LiDAR archives (`-n` skips files already extracted)
and link the metadata next to them:

```bash
REPO=data/pureforest
mkdir -p "${REPO}/extracted"
for z in "${REPO}/PureForest/data"/lidar-*.zip; do
  unzip -n "$z" -d "${REPO}/extracted"
done
ln -sfn ../PureForest/metadata "${REPO}/extracted/metadata"
```

To extract in parallel, replace the loop with
`ls "${REPO}/PureForest/data"/lidar-*.zip | xargs -P 8 -I {} unzip -n {} -d "${REPO}/extracted"`.

Expected input of the preprocessing script:

```text
data/pureforest/extracted/          # --dataset_root
├── lidar/{train,val,test}/*.laz
└── metadata/PureForest-patches.csv
```

## 2. Preprocess

Requires `laspy` and `lazrs` (`pip install laspy lazrs`).

```bash
python pointcept/datasets/preprocessing/pureforest/preprocess_pureforest.py \
  --dataset_root data/pureforest/extracted \
  --output_root data/pureforest \
  --num_workers 8
```

The script is manifest-driven: each row of `PureForest-patches.csv` (`patch_id`, `split`, `class_index`)
is one tile, read from `lidar/{split}/{TRAIN|VAL|TEST}-{patch_id}.laz`, e.g.
`lidar/train/TRAIN-Pinus_halepensis-C8-3_1_244.laz`. LAZ files that are not in the manifest are ignored.

Output:

```text
data/pureforest/{train,val,test}/<patch_id>/
├── coord.npy      # float32: XY centered on the mean, Z shifted to its minimum, divided by 25 m
├── color.npy      # float32 RGB in [0, 255]
└── category.npy   # int32 class id
```

`PureForestDataset` multiplies `coord` back by 25 m at load time.

Options:

| Option | Default | Effect |
|---|---|---|
| `--manifest_csv` | `<dataset_root>/metadata/PureForest-patches.csv` | Tile list |
| `--splits` | `train,val,test` | Comma-separated splits to convert |
| `--num_workers` | `1` | Worker processes |
| `--overwrite` | off | Reconvert tiles that are already complete |
| `--toy` | off | Convert only the first 2 tiles of each class in each split (about 78 tiles) |

Re-running the command resumes: tiles whose `coord.npy` and `color.npy` are complete are skipped (writes are
atomic, `coord.npy` last). Reports written under `--output_root`:

- `missing_laz_preflight.txt`: manifest rows whose LAZ file is absent (written before conversion);
- `preprocess_errors.log`: tiles that failed, with their traceback (only when there are failures).

Use a separate output root for the toy subset:

```bash
python pointcept/datasets/preprocessing/pureforest/preprocess_pureforest.py \
  --toy \
  --dataset_root data/pureforest/extracted \
  --output_root data/pureforest_toy
```

## 3. MALiBU3D leakage exclusion

`malibu3d_leakage_excluded_test_tiles.txt` lists the PureForest test patches whose forest (BD Forêt
polygon) overlaps a MALiBU3D train/val tile. Configs evaluating MALiBU3D backbones set
`exclude_malibu3d_leakage_tiles=True` on `PureForestDataset` to drop them.

## 4. Training and probing

- Supervised training from scratch: `configs/pureforest/cls-litept-b-v1m0-pureforest.py`.

```bash
python tools/train.py --config-file configs/pureforest/cls-litept-b-v1m0-pureforest.py
```

- Linear probe of a frozen backbone (PureForest column of the paper): `configs/pureforest/probe-*.py`,
  with the checkpoint path in `weight`. `tools/probe_pureforest.py` pools the frozen encoder multi-scale
  features of every tile once, then fits a logistic regression for each pooling and L2 strength of
  `probe` in the config, selects the best on validation and reports test.

```bash
export PYTHONPATH=$PWD
python tools/probe_pureforest.py --config-file configs/pureforest/probe-litept-b-v1m0-pureforest.py
# -> exp/pureforest/probe-litept-b-v1m0-pureforest/{train,val,test}.npz, metrics.json
```

The pooled features are reused when the command is run again (`--force-extract` recomputes them).
`--point-budget` caps the points per extraction batch (PureForest tiles vary about 10x in size).
