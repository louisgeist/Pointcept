# MALiBU3D

**A Multitask Aerial LiDAR Benchmark for Large-Scale 3D Scene Understanding**

[Paper (TODO)](https://arxiv.org/abs/XXXX.XXXXX) ·
[Dataset](https://huggingface.co/datasets/LouisGeist/MALiBU3D) ·
[Checkpoints](https://huggingface.co/LouisGeist/MALiBU3D-backbones)

MALiBU3D is a large-scale multitask aerial LiDAR benchmark: 59 billion points over 2,221 km² of metropolitan France,
at the overlap of the national [LiDAR HD](https://geoservices.ign.fr/lidarhd) programme and
[FLAIR-HUB](https://huggingface.co/datasets/IGNF/FLAIR-HUB). Every point has its intensity and aerial RGB, and the
supervision covers five tasks:

| Task | Target | Metric |
|---|---|---|
| Land cover | 15 classes per point | mIoU |
| Road network | 1 m road raster, evaluated as a graph | APLS |
| Natural habitat | distribution per 100 m tile on 4 ecological axes | total variation (TV) |
| Forest cover | 1 m forest raster | IoU of the forest class |
| Elevation | height above the terrain per point | MAE (m) |

This repository holds the code of the paper: data loading, the five baselines (Sonata, LitePT-B, PTv3, SpUNet,
KPConvX) trained on all tasks at once, their evaluation, and the cross-domain linear probing of the frozen
backbones on DALES, H3D, ECLAIR, OpenGF and PureForest. It is built on
[Pointcept](https://github.com/Pointcept/Pointcept) (forked at commit
[`81ab8f5`](https://github.com/Pointcept/Pointcept/commit/81ab8f5c4781bf63ed76d6e97e2d7241e4ceffc8)), with the
models and datasets the benchmark does not use removed.

## Installation

```bash
conda env create -f environment.yml   # PyTorch 2.5.0, CUDA 12.4 (toolkit included)
conda activate malibu3d

# CUDA extensions, built against the installed PyTorch
pip install --no-build-isolation ./libs/pointops ./libs/pointrope   # KPConvX, LitePT
pip install --no-build-isolation flash-attn==2.7.3                  # PTv3, Sonata, LitePT

# KPConvX C++ helpers (in place)
(cd pointcept/models/kpconvx/cpp_wrappers/cpp_subsampling && python setup.py build_ext --inplace)
(cd pointcept/models/kpconvx/cpp_wrappers/cpp_neighbors && python setup.py build_ext --inplace)
```

flash-attention downloads a prebuilt wheel when one matches PyTorch and CUDA, and otherwise compiles (long; limit
the parallel jobs with `MAX_JOBS=8`). Run every command below from the repository root with `export PYTHONPATH=./`;
`pytest tests/` checks the installation.

## Data

Download the dataset from Hugging Face (790 GB of zipped regions) and extract it into `data/malibu3d`, the path read by the configs:

```bash
hf download LouisGeist/MALiBU3D --repo-type dataset --local-dir data/malibu3d_zip
python scripts/unzip_malibu3d.py --zip-root data/malibu3d_zip --output-root data/malibu3d --workers 8
```

The [dataset card](https://huggingface.co/datasets/LouisGeist/MALiBU3D) describes the files of each tile and
`tiles.csv`, the table of all tiles read by `MALiBU3DDataset`. A single region can be downloaded with
`--include "data/train/D075-2021_LIDARHD/*"` and the same table.

The road masks are not shipped: rasterize them once from the road graphs of each region (`network.npy` per tile):

```bash
python pointcept/datasets/preprocessing/malibu3d/rasterize_network.py \
    --data_root data/malibu3d --split_manifest_csv data/malibu3d/tiles.csv --num_workers 16
```

**Validation subset.** The training configs validate on `val_subset_2000tiles.csv` (shipped with the dataset): a
fixed subset of 2,000 validation tiles, representative of the label distribution of the whole validation split.
Test metrics always use the full test split. To validate on the whole validation split, add
`--options data.val.stratified_subset_manifest=False`.

## Training and evaluation on MALiBU3D

Configs live in `configs/malibu3d/`. Train with `scripts/train.sh` (runs in `exp/malibu3d/<name>`), or with
`tools/train.py` to override config entries with `--options key=value`:

```bash
sh scripts/train.sh -g 1 -d malibu3d -c malibu3d/multi-litept-b-v1m0-malibu3d -n litept-b
python tools/train.py --config-file configs/malibu3d/multi-litept-b-v1m0-malibu3d.py --num-gpus 1 \
    --options save_path=exp/malibu3d/litept-b
```

At the end of training, the best checkpoint is evaluated on the test split, including APLS on the predicted road
graphs (`network_apls_eval` in the configs; `tools/eval_network_apls.py` recomputes it from the saved
predictions). `scripts/test.sh -d malibu3d -n <name>` tests a finished run again.

| Config | Model | Paper |
|---|---|---|
| `multi-sonata-ft-v1m0-malibu3d.py` | Sonata fine-tuned on the five tasks (1 GPU) | Tab. 2 |
| `multi-litept-b-v1m0-malibu3d.py` | LitePT-B (1 GPU) | Tab. 2 |
| `multi-ptv3-v1m0-malibu3d.py` | PTv3 (4 GPUs) | Tab. 2 |
| `multi-spunet-v1m0-malibu3d.py` | SpUNet (1 GPU) | Tab. 2 |
| `multi-kpconvx-v1m0-malibu3d.py` | KPConvX (6 GPUs) | Tab. 2 |
| `pretrain-sonata-v1m2-malibu3d.py` | Sonata self-supervised pretraining (24 GPUs), the starting point of Sonata-FT | Tab. 2 |
| `ablations/*.py` | LitePT-B without loss balancing, with GradNorm, natural habitat trained per point, land cover only, tested without RGB | Tab. 4 |
| `segment/`, `elevation/`, `network/` | single-task baselines | |
| `probe/sonata-v1m2-malibu3d-lin*.py` | linear probes of a frozen Sonata on land cover | |

The multitask runs use 200k iterations, a global batch of 12 tiles (24 for KPConvX) and AMP; the GPU count is the one
of the paper (`num_gpu` in each config). The losses of the five tasks are balanced with GradNormLite.

The Sonata pretraining has no online validation. To follow it, `PeriodicProbeHook` can launch a short linear probe
of every `probe_save_freq`-th checkpoint in the background, locally or as a cluster job; results gather in
`<run>/probes/summary.csv`:

```bash
# probe_job.sh: a Slurm script running tools/train.py with configs/malibu3d/probe/sonata-v1m2-malibu3d-lin.py
# and --options weight=$1 save_path=$2
# (plus the multi-GPU launch options: the config is set for 24 GPUs)
python tools/train.py --config-file configs/malibu3d/pretrain-sonata-v1m2-malibu3d.py \
    --options save_path=exp/malibu3d/sonata_pretrain probe_save_freq=10 \
    probe_command="sbatch probe_job.sh {weight} {probe_dir}"
```

The placeholders and a local example are described in the config, next to `probe_command`.

## Pretrained checkpoints

The six backbones of Tab. 2 (five multitask models with their heads, and the self-supervised Sonata) are on
[Hugging Face](https://huggingface.co/LouisGeist/MALiBU3D-backbones). See [docs/checkpoints.md](docs/checkpoints.md)
to download them, run them on a tile, probe or fine-tune them.

## Cross-domain evaluation

The frozen backbones are probed on DALES, H3D, ECLAIR, OpenGF and PureForest, next to a LitePT-B trained from
scratch on each dataset (Tab. 3). Preprocessing, configs and the probing protocol: [docs/probing.md](docs/probing.md).

## Results

MALiBU3D test split (Tab. 2):

| Model | Land cover mIoU ↑ | APLS ↑ | Natural habitat TV ↓ | Forest IoU ↑ | Elevation MAE (m) ↓ |
|---|---|---|---|---|---|
| Sonata (SSL + fine-tuning) | 71.8 | 45.7 | 25.1 | 87.2 | 0.97 |
| LitePT-B | 70.9 | 43.0 | 25.4 | 86.7 | 1.06 |
| PTv3 | 67.9 | 43.0 | 27.2 | 86.5 | 0.99 |
| SpUNet | 67.5 | 39.6 | 27.3 | 86.6 | 0.96 |
| KPConvX | 67.4 | 39.8 | 29.2 | 84.7 | 1.03 |

Linear probes of the frozen backbones (Tab. 3; macro-F1 on H3D, ground IoU on OpenGF, mIoU elsewhere):

| Backbone | DALES | H3D | ECLAIR | OpenGF | PureForest |
|---|---|---|---|---|---|
| Sonata (MALiBU3D SSL) | 76.0 | 71.4 | 75.8 | 90.6 | 75.3 |
| KPConvX | 74.7 | 75.1 | 75.7 | 86.9 | 71.7 |
| LitePT-B | 74.6 | 75.0 | 73.0 | 87.3 | 68.6 |
| PTv3 | 70.2 | 69.9 | 69.4 | 87.1 | 67.6 |
| SpUNet | 66.0 | 61.6 | 67.8 | 82.5 | 63.5 |
| *LitePT-B from scratch* | *83.7* | *81.5* | *78.7* | *87.1* | *78.3* |

## Citation

```bibtex
@article{geist2026malibu3d,
  title   = {{MALiBU3D}: A Multitask Aerial {LiDAR} Benchmark for Large-Scale {3D} Scene Understanding},
  author  = {Geist, Louis and Vincent, Elliot and Chebbi, Mohamed Ali and Garioud, Anatol and Gonthier, Nicolas
             and Giordano, S{\'e}bastien and David, Nicolas and Wegner, Jan Dirk and Landrieu, Loic and Robert, Damien},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License and acknowledgements

The code is released under the [MIT license](LICENSE), as Pointcept. `pointcept/models/kpconvx/` comes from
[ml-kpconvx](https://github.com/apple-aiml-research/ml-kpconvx/tree/main/Pointcept-wrapper) (MIT, © Apple Inc.,
[license](pointcept/models/kpconvx/LICENSE)). The dataset has its own license, given on its
[Hugging Face page](https://huggingface.co/datasets/LouisGeist/MALiBU3D).

This work builds on [Pointcept](https://github.com/Pointcept/Pointcept) and on the models
[Point Transformer V3](https://github.com/Pointcept/PointTransformerV3), [Sonata](https://github.com/facebookresearch/sonata),
[LitePT](https://github.com/prs-eth/LitePT) and [KPConvX](https://github.com/apple-aiml-research/ml-kpconvx); please cite
them as well when you use them. The LiDAR HD point clouds and the FLAIR-HUB annotations are produced by
[IGN](https://www.ign.fr).
