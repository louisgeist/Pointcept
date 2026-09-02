# Malibu3D — Aerial LiDAR Benchmark

Anonymous supplementary code for **Malibu3D**: multitask pretraining on [FLAIR-HUB](https://huggingface.co/datasets/IGNF/FLAIR-HUB) with extra geospatial labels, and linear-probe transfer to DALES, H3D, and ECLAIR.

Built on [Pointcept](README.md). This document covers installation, sample-tile visualization, data preparation, training, evaluation, and result reproduction.

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

## Visualization

You can visualize the provided sample tile with [viser](https://github.com/nerfstudio-project/viser):

```bash
pip install -r requirements-suppmat-vis.txt
python scripts/visualize/visualize_suppmat_zone_viser.py \
  --zone-dir suppmat_zones/D075_UU-S1-4_3-3
```

Open http://localhost:8080. The viewer shows the tile; use the GUI to switch display modes:

| Mode | Content |
|------|---------|
| RGB | Orthophoto colors |
| Strength | LiDAR intensity |
| Semantic | Land cover (LC) |
| Natural habitat | Habitat class |
| Forest | Forest / not forest |
| Elevation | DEM elevation (m) |
| Network corridor | RGB + road network overlay |

**Shift+click** a point to inspect its label.

---

## Data preparation

### Malibu3D (FLAIR-HUB extensions)

Malibu3D aligns extra geospatial labels on [FLAIR-HUB](https://huggingface.co/datasets/IGNF/FLAIR-HUB) patches (same ROI / patch ids):

| Layer | Source | Notes |
|-------|--------|-------|
| LIDARHD | [LiDAR HD](https://geoservices.ign.fr/lidarhd) | PLY point clouds; RGB and CoSIA projected from aerial rasters; partial coverage |
| FOREST | [Masque Forêt](https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_MASQUE-FORET) | Binary FAO forest mask, 20 cm/px, full coverage |
| LAND_USE | [OCS GE](https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_OCS-GE) Usage | 20 functional land-use classes, 20 cm/px |
| NATURAL_HABITAT | [CarHab](https://cartes.gouv.fr/rechercher-une-donnee/dataset/INPN-CARHAB_HABITATS) | 44 habitat classes, 20 cm/px; 55 / 74 couples |
| DEM_ELEV | FLAIR-HUB | DSM (20 cm) and DTM (1 m); height ≈ DSM − DTM |

1. Download FLAIR-HUB base tiles from [IGNF/FLAIR-HUB](https://huggingface.co/datasets/IGNF/FLAIR-HUB).
2. Download the extra labels above (same spatial extent and naming as FLAIR-HUB; LiDAR files use `.ply`).
3. Preprocess into Pointcept format:

```bash
python pointcept/datasets/preprocessing/malibu3d/preprocess_malibu3d_v2.py \
  --ply_root data/malibu3d/raw \
  --out_root data/malibu3d
```

Details: [`pointcept/datasets/preprocessing/malibu3d/README.md`](pointcept/datasets/preprocessing/malibu3d/README.md).

### DALES / H3D / ECLAIR

Place preprocessed datasets under `data/dales`, `data/h3d`, and `data/eclair` following the Pointcept layout expected by configs in `configs/dales/`, `configs/h3d/`, and `configs/eclair/`.

---

## Training

### Malibu3D multitask pretrain

```bash
# LitePT-B multitask (primary baseline)
sh scripts/train.sh -g 1 -d malibu3d \
  -c malibu3d/multi-litept-b-v1m0-malibu3d \
  -n malibu3d_litept_b_multitask

# PT-v3 multitask
sh scripts/train.sh -g 1 -d malibu3d \
  -c malibu3d/multi-ptv3-v1m0-malibu3d \
  -n malibu3d_ptv3_multitask

# SpUNet multitask
sh scripts/train.sh -g 1 -d malibu3d \
  -c malibu3d/multi-spunet-v1m0-malibu3d \
  -n malibu3d_spunet_multitask

# KPConvX multitask
sh scripts/train.sh -g 1 -d malibu3d \
  -c malibu3d/multi-kpconvx-v1m0-malibu3d \
  -n malibu3d_kpconvx_multitask
```

### Sonata self-supervised pretrain on Malibu3D

```bash
sh scripts/train.sh -g 1 -d malibu3d \
  -c malibu3d/pretrain-sonata-v1m2-malibu3d \
  -n malibu3d_sonata_ssl
```

---

## Evaluation (linear probing)

Recommended path: [`tools/grid_then_seeds.py`](tools/grid_then_seeds.py) — **lr sweep (grid) → pick the validation winner → 10-init seed ensemble → aggregate**.

Checkpoint selection uses `GridProbeEvaluator.select_metric` in each grid config: **H3D → `macro_f1`**, **DALES and ECLAIR → `mIoU`**. `--weight` overrides the checkpoint already set in the config.

```bash
python tools/grid_then_seeds.py \
  --grid-config configs/h3d/litept-b-v1m0-h3d-lin_enc.py \
  --weight ckpt/malibu3d/litept_b_multitask/model_best.pth \
  --save-root exp/grid_then_seeds/h3d_lpt_enc
```

Under `--save-root`:

| File | Content |
|------|---------|
| `grid/grid_search_results.json` | leaderboard + winner |
| `seed_ensemble_config.py` | generated 10-init config |
| `seeds/seed_ensemble_results.json` | test mean ± std |
| `grid_then_seeds_summary.csv` | one-row summary (`test_*_mean` / `test_*_std`) |

Reuse an existing grid (seeds only):

```bash
python tools/grid_then_seeds.py --skip-grid \
  --grid-dir exp/grid_then_seeds/h3d_lpt_enc/grid \
  --grid-config configs/h3d/litept-b-v1m0-h3d-lin_enc.py \
  --save-root exp/grid_then_seeds/h3d_lpt_enc
```

Single probe without the grid/ensemble driver:

```bash
sh scripts/train.sh -g 1 -d h3d \
  -c h3d/litept-b-v1m0-h3d-lin_enc \
  -w ckpt/malibu3d/litept_b_multitask/model_best.pth \
  -n h3d_lpt_enc_probe
```

---

## Config reference

| Directory | Purpose |
|-----------|---------|
| [`configs/malibu3d/`](configs/malibu3d/) | Malibu3D multitask pretrain + Sonata SSL |
| [`configs/dales/`](configs/dales/) | DALES linear-probe grid configs |
| [`configs/h3d/`](configs/h3d/) | H3D linear-probe grid configs |
| [`configs/eclair/`](configs/eclair/) | ECLAIR linear-probe grid configs |
| [`configs/pureforest/`](configs/pureforest/) | PureForest baseline |

Encoder vs decoder (and encoder+decoder) feature-source ablations live next to these configs (`*-lin-grid-dec.py`, `*-lin_dec.py`, `*-enc-dec.py`).
