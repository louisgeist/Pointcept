# Downstream evaluation of frozen backbones

The backbones trained on MALiBU3D are evaluated on five other aerial LiDAR datasets by linear probing of their
frozen features (Tab. 3 of the paper), next to a LitePT-B trained from scratch on each dataset. Commands are run
from the repository root with `export PYTHONPATH=./`.

## Datasets

| Dataset | Task | Metric | Source |
|---|---|---|---|
| DALES | semantic segmentation, 8 classes | mIoU | [University of Dayton](https://www.udayton.edu/engineering/centers/vision_lab/research/was_data_analysis_and_processing/dale.php) (`DALESObjects` PLY release) |
| H3D | semantic segmentation, 11 classes | macro-F1 | [IFP, University of Stuttgart](https://ifpwww.ifp.uni-stuttgart.de/benchmark/hessigheim/default.aspx) (epoch March 2018) |
| ECLAIR | semantic segmentation, 11 classes | mIoU | [SharperShape/eclair-dataset](https://github.com/SharperShape/eclair-dataset) |
| OpenGF | ground filtering | IoU of the ground class (Test II, without outliers) | [Nathan-UW/OpenGF](https://github.com/Nathan-UW/OpenGF) |
| PureForest | tile classification, 13 tree species | mIoU | [IGNF/PureForest](https://huggingface.co/datasets/IGNF/PureForest) |

Preprocessing writes Pointcept scene folders under `data/<dataset>/{train,val,test}/` (the path read by the
configs). `laspy` and `lazrs` read the LAZ files of H3D, ECLAIR, OpenGF and PureForest.

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

PureForest: see [its preprocessing README](../pointcept/datasets/preprocessing/pureforest/README.md). Its test
tiles whose forest overlaps a MALiBU3D train/val tile are excluded from the evaluation of MALiBU3D backbones
(`exclude_malibu3d_leakage_tiles=True`).

DALES has no validation split: its configs validate on the test split, as is common practice on this dataset.

## Linear probing protocol

The paper probes the encoder multi-scale features of the frozen backbone:
`configs/<dataset>/<backbone>-<version>-<dataset>-lin-grid-enc.py`, with `<backbone>-<version>` one of
`litept-b-v1m0`, `ptv3-v1m0`, `spunet-v1m0`, `kpconvx-v1m0` and `sonata-v1m2`. Each config trains 12 linear heads (CE + Lovasz, one per
learning rate from 1e-4 to 5e-1) on one shared backbone forward pass. The head with the best validation
metric (macro-F1 on H3D, mIoU elsewhere) is then retrained with 10 seeds, and the test metric is reported as mean
± std. `tools/sweep_then_seeds.py` runs both steps (sweep, then seeds):

```bash
python tools/sweep_then_seeds.py \
    --grid-config configs/h3d/litept-b-v1m0-h3d-lin-grid-enc.py \
    --weight ckpt/litept-b/model.pth \
    --save-root exp/h3d/litept-b_probe
# -> exp/h3d/litept-b_probe/.../seed_ensemble_results.json (test metrics, mean +/- std over the seeds)
```

Each step is an ordinary `tools/train.py` run, so a single grid can also be launched with
`python tools/train.py --config-file <grid config> --num-gpus 1`. The script skips a step whose result file
already exists and resumes an interrupted one from its last checkpoint.

The configs load the released checkpoints from `ckpt/<name>/model.pth` (see [checkpoints.md](checkpoints.md)).
Other backbones of the same architecture (e.g. the ablation backbones of Tab. 4) are probed with the same
configs and `--weight` (or `--options weight=...` with `tools/train.py`).

`semseg-litept-b-v1m0-<dataset>.py` trains LitePT-B from scratch (`tools/train.py`). The `ablations/` folder of
each dataset holds the probes of the appendix ablations:

- `*-lin-grid-dec*`, `*-lin-grid-enc-dec*`: other feature taps of the same backbones (decoder, decoder
  hypercolumn, encoder + decoder), for the feature-source ablation.
- `*-norgb*`: probes with the RGB input masked (no-RGB ablation).
- `sonata-v1m1-*`: probes of the official indoor Sonata release (`ckpt/sonata/pretrain-sonata-v1m1-0-base.pth`).
- ECLAIR only: `pretrain-sonata-v1m2-eclair.py` (Sonata pretrained on ECLAIR, Tab. 4) and its linear probe
  `sonata-v1m2-eclair-lin.py`.

## PureForest

PureForest is probed on frozen features pooled per tile: `tools/probe_pureforest.py` extracts the mean- and
max-pooled encoder features of every tile once, then fits a multinomial logistic regression (L-BFGS) for each
pooling and L2 strength of the config, selects the best pair on validation and reports test mIoU.

```bash
python tools/probe_pureforest.py --config-file configs/pureforest/probe-litept-b-v1m0-pureforest.py
# -> exp/pureforest/probe-litept-b-v1m0-pureforest/{train,val,test}.npz, metrics.json
```

The pooled features are reused when the command is run again (`--force-extract` recomputes them); another
checkpoint is probed with `--options weight=...`. LitePT-B from scratch:
`configs/pureforest/cls-litept-b-v1m0-pureforest.py`. `configs/pureforest/ablations/` holds the no-RGB probe and
the probe of the official indoor Sonata.
