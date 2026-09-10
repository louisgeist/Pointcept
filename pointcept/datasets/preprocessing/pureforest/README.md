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

## Link Hugging Face download

Run from the Pointcept repository root. Point `data/pureforest/PureForest` at your local Hugging Face clone (must contain `data/lidar-*.zip` and `metadata/`):

```bash
mkdir -p data/pureforest
ln -sfn /path/to/PureForest data/pureforest/PureForest

# Example:
# ln -sfn /data/geist/datasets/PureForest data/pureforest/PureForest

ls data/pureforest/PureForest/metadata/PureForest-patches.csv
```

## Extract LAZ archives

From the repository root:

**Sequential** (`-n` skips files already extracted):

```bash
REPO=data/pureforest
mkdir -p "${REPO}/extracted"
for z in "${REPO}/PureForest/data"/lidar-*.zip; do
  echo ">>> $(basename "$z")"
  unzip -n "$z" -d "${REPO}/extracted"
done
ln -sfn ../PureForest/metadata "${REPO}/extracted/metadata"
```

**Parallel** (8 jobs at once; use one of the options below):

With GNU `parallel` (if installed):

```bash
REPO=data/pureforest
mkdir -p "${REPO}/extracted"

module load parallel/20210922 # on Jean-Zay

parallel -j 8 'unzip -n {} -d '"${REPO}/extracted"'' ::: "${REPO}/PureForest/data"/lidar-*.zip
ln -sfn ../PureForest/metadata "${REPO}/extracted/metadata"
```

Without `parallel` (POSIX `xargs`, usually available on clusters):

```bash
REPO=data/pureforest
mkdir -p "${REPO}/extracted"
ls "${REPO}/PureForest/data"/lidar-*.zip | xargs -P 8 -I {} unzip -n {} -d "${REPO}/extracted"
ln -sfn ../PureForest/metadata "${REPO}/extracted/metadata"
```

If neither works, use the **Sequential** loop above (slower but sufficient).

```bash
find "${REPO}/extracted/lidar" -name '*.laz' | wc -l   # expect ~135569
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

## Offline pooled embeddings + sklearn linear probe

Faster alternative to `GridProbeClassifier` lr sweeps for frozen-backbone probes:
one GPU pass dumps per-tile **mean** and **max** pools; then a CPU sklearn
`LogisticRegression` grid over `C` (L2 ≈ weight decay) tries mean / max /
concat / sum.

Requires `scikit-learn` (`conda install scikit-learn` or recreate the env from
`environment.yml`).

Common setup (repo root):

```bash
export PYTHONPATH="$PWD"
BS=24
SPLITS="train val test"
OUT=stats/pureforest/embeddings
```

Optional `--point-max 5000` if VRAM OOMs on full tiles (val has no SphereCrop).
Writes `{split}.npz` (`names`, `category`, `mean_feat`, `max_feat`) + `meta.json`.

### Extract — encoder multiscale (local)

Malibu3D / Sonata weights under `ckpt/malibu3d/` (see `ckpt/README.md`).

```bash
# Sonata outdoor SSL (1232ch)
python scripts/extract_pureforest_pooled_embeddings.py \
  --config configs/pureforest/cls-sonata-v1m2-pureforest-lin-grid-enc.py \
  --weight ckpt/malibu3d/sonata_outdoor/epoch_120.pth \
  --output-dir ${OUT}/sonata_outdoor_ms \
  --splits ${SPLITS} --batch-size ${BS}

# LitePT-B Malibu3D multitask (1386ch)
python scripts/extract_pureforest_pooled_embeddings.py \
  --config configs/pureforest/cls-litept-b-v1m0-pureforest-lin-grid-enc.py \
  --weight ckpt/malibu3d/litept_b_multitask/model_best.pth \
  --output-dir ${OUT}/litept_b_malibu3d_ms \
  --splits ${SPLITS} --batch-size ${BS}

# PTv3 Malibu3D multitask (992ch)
python scripts/extract_pureforest_pooled_embeddings.py \
  --config configs/pureforest/cls-ptv3-v1m0-pureforest-lin-grid-enc.py \
  --weight ckpt/malibu3d/ptv3_multitask/model_best.pth \
  --output-dir ${OUT}/ptv3_malibu3d_ms \
  --splits ${SPLITS} --batch-size ${BS}

# SpUNet Malibu3D multitask (512ch)
python scripts/extract_pureforest_pooled_embeddings.py \
  --config configs/pureforest/cls-spunet-v1m0-pureforest-lin-grid-enc.py \
  --weight ckpt/malibu3d/spunet_multitask/model_best.pth \
  --output-dir ${OUT}/spunet_malibu3d_ms \
  --splits ${SPLITS} --batch-size ${BS}

# KPConvX Malibu3D multitask (928ch)
python scripts/extract_pureforest_pooled_embeddings.py \
  --config configs/pureforest/cls-kpconvx-v1m0-pureforest-lin-grid-enc.py \
  --weight ckpt/malibu3d/kpconvx_multitask/model_best.pth \
  --output-dir ${OUT}/kpconvx_malibu3d_ms \
  --splits ${SPLITS} --batch-size ${BS}
```

### LitePT-B préentraîné ECLAIR (job 1330042)

```bash
python scripts/extract_pureforest_pooled_embeddings.py \
  --config configs/pureforest/cls-litept-b-v1m0-pureforest-lin-grid-enc.py \
  --weight ckpt/1330042/model_best.pth \
  --output-dir ${OUT}/litept_b_preECLAIR_ms \
  --splits ${SPLITS} --batch-size ${BS}
```

On Jean-Zay you can point `--weight` directly at the lustre path without copying.

Toy smoke (override data root):

```bash
python scripts/extract_pureforest_pooled_embeddings.py \
  --config configs/pureforest/cls-sonata-v1m2-pureforest-lin-grid-enc.py \
  --weight ckpt/malibu3d/sonata_outdoor/epoch_120.pth \
  --data-root data/pureforest_toy \
  --output-dir ${OUT}/sonata_outdoor_toy \
  --splits ${SPLITS} --batch-size 2
```

### Probe (sklearn)

```bash
python scripts/probe_pureforest_sklearn.py \
  --embeddings-dir ${OUT}/sonata_outdoor_ms \
  --output-dir stats/pureforest/sklearn_probe/sonata_outdoor_ms
```

Repeat with each `${OUT}/<tag>`. Selects best `(agg, C)` on val
(`--select-metric mIoU` by default), reports test once, writes `metrics.json`
and `best_test_predictions.npz`. Options: `--class-weight balanced`,
`--Cs 0.01 0.1 1 10 100`, `--aggs mean concat`.

### Slurm (Jean-Zay H100)

Extract all MS-encoder backbones (Malibu3D + LitePT preECLAIR) with
`batch_size=24` by default:

```bash
sbatch scripts/pureforest/sbatch_extract_pooled_embeddings_h100.sh

# Optional overrides:
BATCH_SIZE=32 POINT_MAX=5000 SKIP_EXISTING=1 \
  sbatch scripts/pureforest/sbatch_extract_pooled_embeddings_h100.sh
```

Outputs: `stats/pureforest/embeddings/<tag>/{train,val,test}.npz`.
Logs: `logs/slurm/$SLURM_JOB_ID/`.
