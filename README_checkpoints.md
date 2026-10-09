# MALiBU3D checkpoints: download, forward, probing, fine-tuning

Released backbones for MALiBU3D, hosted on Hugging Face at
[`LouisGeist/MALiBU3D-backbones`](https://huggingface.co/LouisGeist/MALiBU3D-backbones) (model card with the
mIoU table: see that page). Every checkpoint is rebuilt from a **default config of this repository**:

| HF folder | Params | Config (`configs/flair3d_default/`) | Trained on |
|---|---|---|---|
| `sonata-ft/` | 124.8 M | `multi-sonata-ft-v1m0-flair3d.py` | 1 GPU, batch 12 |
| `litept-b/` | 45.1 M | `multi-litept-b-v1m0-flair3d.py` | 1 GPU, batch 12 |
| `ptv3/` | 46.2 M | `multi-ptv3-v1m0-flair3d.py` | 4 GPUs, batch 12 (3/GPU), SyncBN |
| `spunet/` | 42.3 M | `multi-spunet-v1m0-flair3d.py` | 1 GPU, batch 12 |
| `kpconvx/` | 13.6 M | `multi-kpconvx-v1m0-flair3d.py` | 6 GPUs, batch 24 (4/GPU), SyncBN |
| `sonata-pretrained/` | 108.5 M | `probe/sonata-v1m2-flair3d-lin-grid.py` (backbone only) | SSL pre-training, `pretrain-sonata-v1m2-flair3d.py` |

The five multitask models (all but `sonata-pretrained`) hold a backbone and the heads of the eight targets
(`segment`, `forest_2d`, `elevation`, four `nathab_*` axes, `network`). `sonata-pretrained` is the Sonata
student backbone without any head. All supervised runs: 200k iterations, AMP.

## 1. Download

The repo is private for now: log in once (`hf auth login`, token with read access).

One model, into `ckpt/` (the path expected by `multi-sonata-ft-v1m0-flair3d.py` for `sonata-pretrained`):

```bash
hf download LouisGeist/MALiBU3D-backbones litept-b/model.pth --local-dir ckpt
# -> ckpt/litept-b/model.pth
```

Everything, then check the hashes:

```bash
hf download LouisGeist/MALiBU3D-backbones --local-dir ckpt/malibu3d-backbones
cd ckpt/malibu3d-backbones && sha256sum -c SHA256SUMS
```

From Python (cached under `~/.cache/huggingface`):

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("LouisGeist/MALiBU3D-backbones", "litept-b/model.pth")
```

Each `model.pth` contains only tensors and numbers (`torch.load(path, weights_only=True)` is safe):
`{"state_dict": ..., "epoch": int, "best_metric_value": float}` (no optimizer state; `sonata-pretrained` has no
`best_metric_value`). Keys have no `module.`
prefix; the repository's `CheckpointLoader` accepts both forms.

## 2. Forward pass on a tile

Environment: see the *Environment setup* section of `CLAUDE.md` / `environment.yml` (spconv, and flash-attn for
PTv3 and Sonata). A GPU is required; use `cuda:0`.

`scripts/hf_release/forward_example.py` builds the model from its config, loads the weights (downloaded from the
Hub unless `--weight` is given), runs the config's validation pipeline on one preprocessed tile and prints the
result per task:

```bash
export PYTHONPATH=$PWD
python scripts/hf_release/forward_example.py --model litept-b \
    --tile D067-2021_AU-S1-21_3-6 --split val \
    --csv-manifest data/flair3d_plus/raw/scene_split_manifest_D067.csv --max-points 30000 \
    --save out_litept_b.npz
```

```
litept-b: 30000 points after voxelisation, 45.1 M params
segment                    classes predicted: 8 | acc=0.936 mIoU(present classes)=0.713
nathab_habitat_type        classes predicted: 4
...
elevation                  mean=0.83 std=1.96 min=-0.30 max=10.67
```

- `--tile` is the tile directory name as listed in the tile table (`tiles.csv`, or a regional manifest passed with
  `--csv-manifest`); the tile must be preprocessed on disk under `data/flair3d_plus/` (see
  `README_flair3dplus.md` and the dataset card).
- `--model` is one of `sonata-ft litept-b ptv3 spunet kpconvx sonata-pretrained`. `sonata-pretrained` has no head:
  the script prints the backbone features (`(N, 1232)`).
- `--max-points N` keeps the N voxels nearest the tile centre (fits any GPU); `0` runs the whole tile.
- `--save` writes the voxel coordinates and the predictions (`<task>_pred`, or `feat`) to a `.npz`.

The same thing in a few lines:

```python
import torch
from huggingface_hub import hf_hub_download
from pointcept.models import build_model
from pointcept.utils.config import Config

cfg = Config.fromfile("configs/flair3d_default/multi-litept-b-v1m0-flair3d.py")
model = build_model(cfg.model).cuda().eval()
sd = torch.load(hf_hub_download("LouisGeist/MALiBU3D-backbones", "litept-b/model.pth"), weights_only=True)["state_dict"]
model.load_state_dict(sd)          # strict: every key matches
# inp: batch dict from the config's val pipeline (coord, grid_coord, feat, offset, ...); see forward_example.py
out = model(inp)                   # out["seg_logits"], out["seg_logits_by_task"], out["reg_pred_by_task"]
```

Regression test of the released weights (needs the local D067 tiles): `pytest tests/test_checkpoint_forward.py`.

## 3. Frozen-backbone probing (cross-domain)

Use the file as the frozen backbone of a GridProbe config: only `weight=` changes. Example on H3D with
Sonata, then the full grid → seed-ensemble pipeline (see `README_grid_then_seed.md`):

```bash
export PYTHONPATH=$PWD
python tools/train.py --config-file configs/h3d/sonata-v1m2-h3d-lin-grid.py --num-gpus 1 \
    --options weight=ckpt/sonata-pretrained/model.pth

./submit_grid_then_seeds_h100.sh configs/h3d/sonata-v1m2-h3d-lin-grid.py ckpt/sonata-pretrained/model.pth h3d_sonata
```

`sonata-pretrained` keeps its `backbone.*` keys, so the `CheckpointLoader(keywords="module.student.backbone",
replacement="module.backbone")` already present in the Sonata configs loads it unchanged (`embedding.mask_token`,
only used by masked SSL, is ignored by the probe models).

## 4. Fine-tuning and training

Multitask fine-tuning of Sonata from the released SSL weights (`weight = ckpt/sonata-pretrained/model.pth`):

```bash
hf download LouisGeist/MALiBU3D-backbones sonata-pretrained/model.pth --local-dir ckpt
sh scripts/train.sh -g 1 -d flair3d -c flair3d_default/multi-sonata-ft-v1m0-flair3d -n sonata_ft
```

The other configs reproduce the supervised runs; use the GPU count of the table above, e.g.
`sh scripts/train.sh -g 4 -d flair3d -c flair3d_default/multi-ptv3-v1m0-flair3d -n ptv3_multi`. They read the
full national tile table, so they only run where the whole dataset is preprocessed.
