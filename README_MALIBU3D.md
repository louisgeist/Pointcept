# Malibu3D — Multimodal Aerial LiDAR Benchmark

Anonymous supplementary code for **Malibu3D+**: multimodal pretraining on [FLAIR-HUB](https://huggingface.co/datasets/IGNF/FLAIR-HUB) with extended geospatial modalities, and linear-probe transfer to DALES, H3D, and ECLAIR.

Built on [Pointcept](README.md). This document covers installation, data preparation, training, evaluation, and result reproduction.

---

## Installation

```bash
conda env create -f environment.yml
conda activate pointcept-torch2.5.0-cu12.4
pip install -r requirements-malibu3d.txt  # optional extras if not in conda
```

Compile CUDA extensions (included in `environment.yml` pip section): `pointops`, `pointops2`, `pointgroup_ops`, `pointrope`, `pointseg`.

---

## Checkpoints

Frozen-backbone weights are included under [`ckpt/`](ckpt/). See [ckpt/README.md](ckpt/README.md) for the layout. Path constants live in [`configs/_base_/paths.py`](configs/_base_/paths.py).

---

## Data preparation

### Malibu3D+ (FLAIR-HUB extensions)

1. Download FLAIR-HUB base tiles from [IGNF/FLAIR-HUB](https://huggingface.co/datasets/IGNF/FLAIR-HUB).
2. Download extended modalities (LiDAR HD, forest mask, land use, natural habitat) — see [README_malibu3dplus.md](README_malibu3dplus.md).
3. Preprocess into Pointcept format:

```bash
python pointcept/datasets/preprocessing/malibu3d_plus/preprocess_malibu3d_v2.py \
  --ply_root data/malibu3d_plus/raw \
  --out_root data/malibu3d_plus
```

Details: [`pointcept/datasets/preprocessing/malibu3d_plus/README.md`](pointcept/datasets/preprocessing/malibu3d_plus/README.md).

### DALES / H3D / ECLAIR

Place preprocessed datasets under `data/dales`, `data/h3d`, and `data/eclair` following the Pointcept layout expected by configs in `configs/dales/`, `configs/h3d/`, and `configs/eclair/`.

---

## Training

### Malibu3D+ multitask pretrain

```bash
# LitePT-B multitask (primary baseline)
sh scripts/train.sh -g 1 -d malibu3d_default \
  -c malibu3d_default/multi-litept-b-v1m0-malibu3d \
  -n malibu3d_litept_b_multitask

# PT-v3 multitask
sh scripts/train.sh -g 1 -d malibu3d_default \
  -c malibu3d_default/multi-ptv3-v1m0-malibu3d \
  -n malibu3d_ptv3_multitask

# SpUNet multitask
sh scripts/train.sh -g 1 -d malibu3d_default \
  -c malibu3d_default/multi-spunet-v1m0-malibu3d \
  -n malibu3d_spunet_multitask

# KPConvX multitask
sh scripts/train.sh -g 1 -d malibu3d_default \
  -c malibu3d_default/multi-kpconvx-v1m0-malibu3d \
  -n malibu3d_kpconvx_multitask
```

### Sonata self-supervised pretrain on Malibu3D+

```bash
sh scripts/train.sh -g 1 -d malibu3d_default \
  -c malibu3d_default/pretrain-sonata-v1m2-malibu3d \
  -n malibu3d_sonata_ssl
```

---

## Evaluation (linear probing)

### Single probe (manual)

```bash
sh scripts/train.sh -g 1 -d h3d \
  -c h3d/litept-b-v1m0-h3d-lin_enc \
  -w ckpt/malibu3d/litept_b_multitask/model_best.pth \
  -n h3d_lpt_enc_probe
```

### Grid search + seed ensemble (recommended)

See [README_grid_then_seed.md](README_grid_then_seed.md) for the full pipeline via `tools/grid_then_seeds.py`.

```bash
python tools/grid_then_seeds.py \
  --grid-config configs/h3d/litept-b-v1m0-h3d-lin_enc.py \
  --weight ckpt/malibu3d/litept_b_multitask/model_best.pth \
  --save-root exp/grid_then_seeds/h3d_lpt_enc
```

Metrics are written to `grid_then_seeds_summary.csv` under `--save-root`.

---

## Results table (reproduction commands)

Run each command below after downloading checkpoints and preparing data. Metrics appear in `grid_then_seeds_summary.csv` (column `test_*_mean` / `test_*_std`).

| Benchmark | Backbone | Tap | Metric | Reproduction command |
|-----------|----------|-----|--------|---------------------|
| H3D | LitePT-B | encoder | macro-F1 | `python tools/grid_then_seeds.py --grid-config configs/h3d/litept-b-v1m0-h3d-lin_enc.py --weight ckpt/malibu3d/litept_b_multitask/model_best.pth --save-root exp/results/h3d_lpt_enc` |
| H3D | PT-v3 | encoder | macro-F1 | `python tools/grid_then_seeds.py --grid-config configs/h3d/ptv3-v1m0-h3d-lin-grid-enc.py --weight ckpt/malibu3d/ptv3_multitask/model_best.pth --save-root exp/results/h3d_ptv3_enc` |
| H3D | SpUNet | encoder | macro-F1 | `python tools/grid_then_seeds.py --grid-config configs/h3d/spunet-v1m0-h3d-lin-grid-enc.py --weight ckpt/malibu3d/spunet_multitask/model_best.pth --save-root exp/results/h3d_spunet_enc` |
| H3D | KPConvX | encoder | macro-F1 | `python tools/grid_then_seeds.py --grid-config configs/h3d/kpconvx-v1m0-h3d-lin-grid-enc.py --weight ckpt/malibu3d/kpconvx_multitask/model_best.pth --save-root exp/results/h3d_kpconvx_enc` |
| H3D | Sonata-v1m2 | default | macro-F1 | `python tools/grid_then_seeds.py --grid-config configs/h3d/sonata-v1m2-h3d-lin-grid.py --weight ckpt/malibu3d/sonata_outdoor/epoch_120.pth --save-root exp/results/h3d_sonata` |
| DALES | LitePT-B | encoder | mIoU | `python tools/grid_then_seeds.py --grid-config configs/dales/litept-b-v1m0-dales-lin-grid-enc.py --weight ckpt/malibu3d/litept_b_multitask/model_best.pth --save-root exp/results/dales_lpt_enc` |
| DALES | PT-v3 | encoder | mIoU | `python tools/grid_then_seeds.py --grid-config configs/dales/ptv3-v1m0-dales-lin-grid-enc.py --weight ckpt/malibu3d/ptv3_multitask/model_best.pth --save-root exp/results/dales_ptv3_enc` |
| DALES | SpUNet | encoder | mIoU | `python tools/grid_then_seeds.py --grid-config configs/dales/spunet-v1m0-dales-lin-grid-enc.py --weight ckpt/malibu3d/spunet_multitask/model_best.pth --save-root exp/results/dales_spunet_enc` |
| ECLAIR | LitePT-B | encoder | mIoU | `python tools/grid_then_seeds.py --grid-config configs/eclair/litept-b-v1m0-eclair-lin_enc.py --weight ckpt/malibu3d/litept_b_multitask/model_best.pth --save-root exp/results/eclair_lpt_enc` |
| ECLAIR | PT-v3 | encoder | mIoU | `python tools/grid_then_seeds.py --grid-config configs/eclair/ptv3-v1m0-eclair-lin-grid-enc.py --weight ckpt/malibu3d/ptv3_multitask/model_best.pth --save-root exp/results/eclair_ptv3_enc` |
| ECLAIR | SpUNet | encoder | mIoU | `python tools/grid_then_seeds.py --grid-config configs/eclair/spunet-v1m0-eclair-lin-grid-enc.py --weight ckpt/malibu3d/spunet_multitask/model_best.pth --save-root exp/results/eclair_spunet_enc` |

Feature-source ablation configs (decoder vs encoder taps) are listed in [README_grid_then_seed.md](README_grid_then_seed.md).

---

## Config reference

| Directory | Purpose |
|-----------|---------|
| [`configs/malibu3d_default/`](configs/malibu3d_default/) | Malibu3D+ multitask pretrain + Sonata SSL |
| [`configs/malibu3d/`](configs/malibu3d/) | Legacy Malibu3D configs |
| [`configs/dales/`](configs/dales/) | DALES linear-probe grid configs |
| [`configs/h3d/`](configs/h3d/) | H3D linear-probe grid configs |
| [`configs/eclair/`](configs/eclair/) | ECLAIR linear-probe grid configs |
| [`configs/pureforest/`](configs/pureforest/) | PureForest baseline |

---

## Citation

```bibtex
@misc{malibu3d2026,
  title={Malibu3D: Multimodal Aerial LiDAR Representation Learning},
  year={2026},
  note={Anonymous submission}
}
```

Also cite Pointcept, FLAIR-HUB, and backbone papers as appropriate (see [README.md](README.md)).
