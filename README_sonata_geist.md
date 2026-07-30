# Sonata on Flair3D+ (Geist)

Self-supervised pretraining with **Sonata-v1m2** on Flair3D+, plus **periodic linear
probing** on semantic segmentation (v20) for representation quality tracking.

Official Sonata docs: [`pointcept/models/sonata/README.md`](pointcept/models/sonata/README.md).

## Overview

Sonata SSL runs with `evaluate=False` (no online val). Representation quality is
measured by separate short **linear-probe** jobs on frozen backbone features.

On Jean-Zay, the **pretrain job** submits probes itself via `LinProbeSbatchHook`
(after each `epoch_N.pth`). No separate watcher terminal is required.

```text
Job A: sbatch pretrain Sonata
        → CheckpointSaver writes epoch_N.pth
        → LinProbeSbatchHook: sbatch probe (non-blocking)
Job B_i: short *-lin probe
        → metrics.json (best_val_mIoU)
        → append PRETRAIN_DIR/lin_probe_results.csv  (live)
wandb sync (later, login node)  →  optional W&B curves
```

## Prerequisites

1. Flair3D+ preprocessed under `data/flair3d_plus` with **`segment=v20`**
   (check a tile `meta.json` → `label_definitions.segment`).
2. Manifests:
   - `data/flair3d_plus/raw/scene_split_manifest.csv`
   - `data/flair3d_plus/missing_ply_preflight.txt`
   - `data/flair3d_plus/too_small_tiles.csv`
3. Stratified val sidecar (if missing):
```bash
python scripts/build_stratified_subset.py \
  --data_root data/flair3d_plus \
  --csv_manifest data/flair3d_plus/raw/scene_split_manifest.csv \
  --split val --max_sample 2000 --keys segment \
  --output data/flair3d_plus/manifests/val_dev_subset_2000.csv
```

## Configs

| Role | Path |
|------|------|
| Pretrain | [`configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py`](configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py) |
| Linear probe | [`configs/flair3d_default/segment/sonata-v1m2-flair3d-lin.py`](configs/flair3d_default/segment/sonata-v1m2-flair3d-lin.py) |

### Pretrain defaults

- Split: **`train` only** (no val/test leakage into SSL)
- Features: **coord + color + strength** (`in_channels=7`)
- Schedule: iter-limited — `total_iters=100_000` (placeholder), `iter_per_epoch=1000`
- Hardware template: **8× A100**, `batch_size_per_gpu=2` (`batch_size=16`)
- Checkpoints: `CheckpointSaver(save_freq=5)` → `epoch_5.pth`, `epoch_10.pth`, …
- `LinProbeSbatchHook` (after saver): submits `scripts/sonata/sbatch_lin_probe.sh` per epoch ckpt
¬- W&B project: `flair3d_sonata`
- Scene-level `RandomDropColor` / `RandomDropStrength` in `global_shared_transform` only

### Linear probe defaults

- `DefaultSegmentorV2` + frozen `PT-v3m2` (`enc_mode=True`, `backbone_out_channels=1232`)
- Remap weights: `module.student.backbone` → `module.backbone`
- Labels: segment **v20** (15 classes, `ignore_index=15`)
- Train: full train split; Val: stratified `val_dev_subset_2000.csv`; **no test**
- Schedule: **`total_iters=1000`**, **`iter_per_epoch=100`** (10 mini-epochs)
- Hardware template: **1× A100**, `batch_size_per_gpu=4`
- No `RandomDropColor` / `RandomDropStrength`
- Hook `MetricsJsonWriter` writes `save_path/metrics.json` at end of run
- End of sbatch: [`scripts/sonata/append_lin_probe_result.py`](scripts/sonata/append_lin_probe_result.py) appends CSV on the pretrain dir

## Jean-Zay workflow

All launchers live under [`scripts/sonata/`](scripts/sonata/):

- [`scripts/sonata/sbatch_pretrain.sh`](scripts/sonata/sbatch_pretrain.sh) — 8× A100, `WANDB_MODE=offline` (+ hook submits probes)
- [`scripts/sonata/sbatch_lin_probe.sh`](scripts/sonata/sbatch_lin_probe.sh) — 1× A100, short walltime
- [`scripts/sonata/periodic_lin_probe.py`](scripts/sonata/periodic_lin_probe.py) — **optional** watcher (local / replay only)
- [`scripts/sonata/append_lin_probe_result.py`](scripts/sonata/append_lin_probe_result.py) — CSV append at end of probe job

```bash
# 1) Pretrain only — probes are submitted automatically by LinProbeSbatchHook
sbatch scripts/sonata/sbatch_pretrain.sh sonata_pretrain_flair3dplus
# PRETRAIN_DIR=logs/slurm/$SLURM_JOB_ID

# 2) Live mIoU (written by each probe job when it finishes)
tail -f logs/slurm/<PRETRAIN_JOB_ID>/lin_probe_results.csv

# 3) Later, from a login node with network: sync offline W&B
wandb sync logs/slurm/<PRETRAIN_JOB_ID>/wandb
# Probe job ids / wandb dirs: column probe_job_dir in the CSV
```

### Manual single probe

```bash
WEIGHT=/path/to/epoch_10.pth EXP_NAME=sonata_lin_ep10 \
  PRETRAIN_JOB_DIR=logs/slurm/<PRETRAIN_JOB_ID> \
  PRETRAIN_EPOCH=10 PRETRAIN_ITERS=10000 \
  sbatch scripts/sonata/sbatch_lin_probe.sh
```

### Local smoke (no Slurm)

`LinProbeSbatchHook` is a no-op without `sbatch`. Use the watcher in local mode:

```bash
sh scripts/train.sh -g 8 -d flair3d_default -c pretrain-sonata-v1m2-flair3d \
  -n sonata_pretrain_flair3dplus

python scripts/sonata/periodic_lin_probe.py \
  --pretrain_job_dir exp/flair3d_default/sonata_pretrain_flair3dplus \
  --mode local --gpus 1 --once
```

## Batch size / VRAM on Jean-Zay

Calibrate **1 GPU** first (VRAM per rank), then set
`batch_size = batch_size_per_gpu * num_gpu` for multi-GPU pretrain.
Keep a **~10–20 %** margin below the confirmed max.

| Usage | Recommendation |
|-------|----------------|
| Quick OOM smoke (1–2 manual tries) | Interactive `salloc` (1× H100, 1–2 h) |
| Binary search + soak | [`sbatch_find_max_batch_size.sh`](sbatch_find_max_batch_size.sh) + [`scripts/find_max_batch_size.py`](scripts/find_max_batch_size.py) |

**Align `--mix-prob` with the real config** (`find_max_batch_size` defaults to `1.0`):

| Phase | Real config | Probe flag |
|-------|-------------|------------|
| Pretrain Sonata | `mix_prob=0` (no Mix3D; MultiView SSL) | `--mix-prob 0` |
| Linear probe | `mix_prob=0.8` | `--mix-prob 0.8` |

Probe overlays **replace `hooks` entirely**, so `LinProbeSbatchHook` from the
pretrain config is never run during VRAM search (no spurious `sbatch` probes).

### A — Pretrain (1× H100)

```bash
# Interactive smoke
salloc -A uhn@h100 -C h100 --gres=gpu:1 --cpus-per-task=24 --time=02:00:00
# then on the node:
python scripts/find_max_batch_size.py \
  --config-file configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py \
  --mode train --min-bs 1 --max-bs 8 \
  --probe-steps 32 --soak-steps 200 \
  --mix-prob 0 --num-gpus 1
```

Or via sbatch:

```bash
CONFIG=configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py \
  MODE=train MIX_PROB=0 \
  MIN_BS_TRAIN=1 MAX_BS_TRAIN=8 \
  PROBE_STEPS=32 SOAK_STEPS_TRAIN=200 \
  sbatch sbatch_find_max_batch_size.sh
```

Write the confirmed `batch_size_per_gpu` into
[`configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py`](configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py)
and set `batch_size = batch_size_per_gpu * 8`.

### B — Linear probe (1× H100)

```bash
python scripts/find_max_batch_size.py \
  --config-file configs/flair3d_default/segment/sonata-v1m2-flair3d-lin.py \
  --mode train --min-bs 1 --max-bs 8 \
  --probe-steps 32 --soak-steps 200 \
  --mix-prob 0.8 --num-gpus 1
# optional: --mode val for batch_size_val
```

Or via sbatch:

```bash
CONFIG=configs/flair3d_default/segment/sonata-v1m2-flair3d-lin.py \
  MODE=train MIX_PROB=0.8 \
  MIN_BS_TRAIN=1 MAX_BS_TRAIN=8 \
  PROBE_STEPS=32 SOAK_STEPS_TRAIN=200 \
  sbatch sbatch_find_max_batch_size.sh
```

Update
[`configs/flair3d_default/segment/sonata-v1m2-flair3d-lin.py`](configs/flair3d_default/segment/sonata-v1m2-flair3d-lin.py)
with the confirmed sizes.

Pretrain MultiView (2 global + 4 local) is much heavier than a segmentor — do
**not** reuse a LitePT batch size for Sonata SSL.

## Monitoring

### Live CSV

Each finished probe appends a row to:

`$PRETRAIN_DIR/lin_probe_results.csv`

Columns: `pretrain_epoch`, `pretrain_iters`, `ckpt`, `probe_job_dir`,
`best_val_mIoU`, `status`, `timestamp`.

```bash
tail -f $PRETRAIN_DIR/lin_probe_results.csv
column -t -s, $PRETRAIN_DIR/lin_probe_results.csv
```

State / dedup: `$PRETRAIN_DIR/lin_probe_state.json` (hook marks `submitted`; probe marks `ok`/`failed`).

### `metrics.json`

Each probe job writes (via `MetricsJsonWriter`):

```json
{
  "best_val_mIoU": 0.42,
  "metric_name": "mIoU",
  "best_metric_value": 0.42,
  "epoch": 10
}
```

### Wandb

- Pretrain and probes use `WANDB_MODE=offline` on Jean-Zay.
- Shared project: `flair3d_sonata`.
- Sync from a login node when you want the UI; **live tracking = CSV**.

## Notes / caveats

- Probe jobs must not steal pretrain GPUs: separate Slurm jobs (queue).
- Hook submit is fire-and-forget; Slurm queues probes if GPUs are busy.
- Checkpoint names are `epoch_{N}.pth` (N = trainer epoch = N × `iter_per_epoch` iters).
- If a short probe (1000 iters) is still too slow on 1 GPU, consider
  `data.train.max_sample=...` as an escape hatch (not the default).
- Decoder probing (`*-dec`) and full fine-tuning (`*-ft`) are out of scope for now.
