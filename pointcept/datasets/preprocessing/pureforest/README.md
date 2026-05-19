# PureForest preprocessing

Tile-level tree species classification (13 classes) from IGNF [PureForest](https://huggingface.co/datasets/IGNF/PureForest).

Preprocessing is **manifest-driven**: every row in `PureForest-patches.csv` defines a tile to convert. Expected LAZ path:

```text
lidar/{split}/{TRAIN|VAL|TEST}-{patch_id}.laz
```

(e.g. manifest `patch_id=Pinus_halepensis-C8-3_1_244`, split `train` → `lidar/train/TRAIN-Pinus_halepensis-C8-3_1_244.laz`).

## On-disk layout

```text
data/pureforest/
├── PureForest/              # HF download (zips + metadata)
│   └── data/lidar-<species>.zip
├── extracted/               # unzip target (--dataset_root)
│   ├── lidar/{train,val,test}/*.laz
│   └── metadata/            # symlink to ../PureForest/metadata
└── {train,val,test}/        # preprocess output (--output_root)
    └── <patch_id>/
        ├── coord.npy
        ├── color.npy
        └── category.npy
```

LAZ files on disk that are **not** in the manifest are ignored.

## Extract archives
From the Pointcept root (`cd /path/to/your/Pointcept`):

Mono-worker
```bash
REPO=data/pureforest
mkdir -p "${REPO}/extracted"
for z in "${REPO}/PureForest/data"/lidar-*.zip; do
  unzip -n "$z" -d "${REPO}/extracted"
done
ln -sfn "${REPO}/PureForest/metadata" "${REPO}/extracted/metadata"
```

Parallel (TODO@Geist : check the command works)
```bash
parallel -j 8 'unzip -n {} -d '"${REPO}/extracted"'' ::: "${REPO}/PureForest/data"/lidar-*.zip
```

## Preprocess

Requires `laspy` and `lazrs`:

```bash
pip install laspy lazrs
```

```bash
python pointcept/datasets/preprocessing/pureforest/preprocess_pureforest.py \
  --dataset_root data/pureforest/extracted \
  --output_root data/pureforest \
  --num_workers 8
```

Before conversion, the script writes `missing_laz_preflight.txt` under `--output_root` listing manifest rows whose LAZ file is absent.

## Toy subset

`--toy` selects **2 tiles per class per split** on-the-fly from `PureForest-patches.csv` (~78 scenes). No intermediate manifest file is written. Use a dedicated output root so it does not mix with the full dataset:

```bash
python pointcept/datasets/preprocessing/pureforest/preprocess_pureforest.py \
  --toy \
  --dataset_root data/pureforest/extracted \
  --output_root data/pureforest_toy \
  --num_workers 4
```

`missing_laz_preflight.txt` under `--output_root` lists which selected tiles are still missing on disk.

Smoke-test training:

```bash
python tools/train.py --config-file configs/pureforest/cls-spunet-v1m0-pureforest-toy.py
```

## Train

```bash
python tools/train.py --config-file configs/pureforest/cls-spunet-v1m0-pureforest.py
```