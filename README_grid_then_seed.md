# Grid → seed-ensemble runbook (H3D + noRGB ablation) — H100

Chained one-job pipeline: **lr sweep (grid) → pick winner by validation metric →
generate + run the 10-init seed-ensemble → aggregate**.
Wrapper: [`sbatch_grid_then_seeds_h100.sh`](sbatch_grid_then_seeds_h100.sh) →
[`tools/grid_then_seeds.py`](tools/grid_then_seeds.py).

```
sbatch sbatch_grid_then_seeds_h100.sh <grid_config> <weight.pth> [exp_name]
```

- Checkpoint selection metric is **`GridProbeEvaluator.select_metric`** baked into each
  grid config and propagated verbatim into the generated seed config:
  **H3D grid/seed → `macro_f1`**, **ECLAIR (and DALES) → `mIoU`**.
- `<weight>` is already written in every config (`weight = …`); the wrapper still
  requires it as arg 2 — pass the same path (or run `python tools/grid_then_seeds.py
  --grid-config <cfg>` directly, which falls back to the config's `weight=`).
- H100 job = 1 GPU, 24 CPU, 5 h wall + auto-requeue/resume (`POINTCEPT_SLURM_REQUEUE=1`),
  so a job that needs more than 5 h just re-queues and resumes the unfinished phase.

## Frozen-backbone checkpoints

```bash
cd /lustre/fswork/projects/rech/unv/usi32yh/Pointcept
B=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm

W_LPT=$B/873542/model/model_best.pth        # LitePT-B  Flair3D+ multitask (baseline / "classique")
W_PTV3=$B/1095469/model/model_best.pth      # PT-v3     Flair3D+ multitask
W_SPUNET=$B/1052217/model/model_best.pth    # SpUNet    Flair3D+ multitask
W_KPX=$B/1159986/model/model_best.pth       # KPConvX   Flair3D+ multitask
W_SONATA=/lustre/fsn1/projects/rech/unv/usi32yh/logs/pointcept_logs/slurm/862680/model/epoch_120.pth  # Sonata SSL
W_NOGNL=$B/1288597/model/model_best.pth     # LitePT-B  multitask WITHOUT GradNormLite
W_MONOLC=$B/1293025/model/model_best.pth    # LitePT-B  mono-task land-cover (segment v20) only
W_PRECLAIR=$B/1330042/model/model_best.pth  # LitePT-B  supervised ECLAIR semseg from scratch
```

## H3D — grid + seed (best checkpoint = validation **macro-F1**)

### One probe per backbone (encoder multiscale tap)

```bash
sbatch sbatch_grid_then_seeds_h100.sh configs/h3d/litept-b-v1m0-h3d-lin_enc.py      $W_LPT     h3d_lpt_enc
sbatch sbatch_grid_then_seeds_h100.sh configs/h3d/ptv3-v1m0-h3d-lin-grid-enc.py     $W_PTV3    h3d_ptv3_enc
sbatch sbatch_grid_then_seeds_h100.sh configs/h3d/spunet-v1m0-h3d-lin-grid-enc.py   $W_SPUNET  h3d_spunet_enc
sbatch sbatch_grid_then_seeds_h100.sh configs/h3d/kpconvx-v1m0-h3d-lin-grid-enc.py  $W_KPX     h3d_kpconvx_enc
sbatch sbatch_grid_then_seeds_h100.sh configs/h3d/sonata-v1m2-h3d-lin-grid.py       $W_SONATA  h3d_sonata
```

### LitePT-B pretraining ablations (backbone changes, probe recipe identical)

```bash
sbatch sbatch_grid_then_seeds_h100.sh configs/experiment/w110/1/abla_grid_on_h3d/litept-b-v1m0-h3d-lin_enc_1.py          $W_NOGNL     h3d_lpt_noGNL
sbatch sbatch_grid_then_seeds_h100.sh configs/experiment/w110/1/abla_grid_on_h3d/litept-b-v1m0-h3d-lin_enc_2.py          $W_MONOLC    h3d_lpt_monoLC
sbatch sbatch_grid_then_seeds_h100.sh configs/experiment/w110/2/abla_grid_from_preECLAIR/litept-b-v1m0-h3d-lin_enc_1.py  $W_PRECLAIR  h3d_lpt_preECLAIR
```

### Extra tap-point variants (optional — same `select_metric="macro_f1"`)

```bash
sbatch sbatch_grid_then_seeds_h100.sh configs/h3d/litept-b-v1m0-h3d-lin_dec.py        $W_LPT     h3d_lpt_dec
sbatch sbatch_grid_then_seeds_h100.sh configs/h3d/litept-b-v1m0-h3d-lin_dec_ss.py     $W_LPT     h3d_lpt_decSS
sbatch sbatch_grid_then_seeds_h100.sh configs/h3d/ptv3-v1m0-h3d-lin-grid-dec.py       $W_PTV3    h3d_ptv3_dec
sbatch sbatch_grid_then_seeds_h100.sh configs/h3d/spunet-v1m0-h3d-lin-grid-dec.py     $W_SPUNET  h3d_spunet_dec
sbatch sbatch_grid_then_seeds_h100.sh configs/h3d/spunet-v1m0-h3d-lin-grid-enc-dec.py $W_SPUNET  h3d_spunet_encdec
```

## noRGB ablation — LitePT-B, grid + seed

Probes are trained **and** evaluated with colour removed on every forward
(`RandomDropColor(1.0, 1.0, keep_mask=True)` → the frozen backbone's learned
`color_mask_value`); coord/strength untouched. Same frozen backbone as the classic
LitePT-B (`873542`).

```bash
# H3D  → select_metric = macro_f1
sbatch sbatch_grid_then_seeds_h100.sh configs/experiment/w110/4/grid_h3d_norgb/litept-b-v1m0-h3d-lin_enc-norgb_1.py        $W_LPT  h3d_lpt_norgb

# ECLAIR → select_metric = mIoU
sbatch sbatch_grid_then_seeds_h100.sh configs/experiment/w110/4/grid_eclair_norgb/litept-b-v1m0-eclair-lin_enc-norgb_1.py  $W_LPT  eclair_lpt_norgb
```

## Outputs (per job, under `logs/slurm/<jobid>/`)

| File | Content |
|---|---|
| `grid/grid_search_results.json` | leaderboard + winner (`select_metric`, `best_val_mIoU`, `best_val_macro_f1`, `test_*`) |
| `grid/grid_probe_miou_history.csv` | per-epoch per-probe `mIoU`/`mIoU_best`/`f1_macro`/`f1_macro_best` |
| `seed_ensemble_config.py` | the 10-init config generated from the winner's full `probe_config` |
| `seeds/seed_ensemble_results.json` | test `mIoU`/`mAcc`/`allAcc`/`f1_macro` mean ± std ± min/max, `select_metric`, per-probe rows |
| `grid_then_seeds_summary.csv` | one row: `winner_select_metric`, `winner_val_mIoU`, `winner_val_f1_macro`, `test_*_mean/std`, … |

W&B: two runs (grid + seeds) sharing group `gts-<jobid>`.

## Notes

- **Grid already run?** Reuse it, seeds only:
  ```bash
  EXTRA_ARGS="--skip-grid --grid-dir logs/slurm/<gridjob>" \
    sbatch sbatch_grid_then_seeds_h100.sh <grid_config> <weight.pth> <exp_name>
  ```
- **Regenerate the seed config without a GPU:**
  ```bash
  python tools/grid_then_seeds.py --make-config-only --grid-config <cfg> --grid-dir <grid_dir> --save-root <out>
  ```
- The static seed configs (`configs/experiment/w110/2/grid_seed_h3d/*`,
  `configs/experiment/w110/4/seed_*_norgb/*`) bake a fixed winner lr by hand — only useful
  if you are **not** using this chained pipeline. `grid_then_seeds.py` always refits to the
  real (now macro-F1-selected) winner.
- A100 variant: same commands with `sbatch_grid_then_seeds.sh` (no `_h100`).
