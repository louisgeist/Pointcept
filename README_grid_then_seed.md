# Grid → seed-ensemble runbook (H3D / DALES / ECLAIR + noRGB) — H100

Chained one-job pipeline: **lr sweep (grid) → pick winner by validation metric →
generate + run the 10-init seed-ensemble → aggregate**.
Wrapper: [`submit_grid_then_seeds_h100.sh`](submit_grid_then_seeds_h100.sh) →
[`sbatch_grid_then_seeds_h100.sh`](sbatch_grid_then_seeds_h100.sh) →
[`tools/grid_then_seeds.py`](tools/grid_then_seeds.py).

```
./submit_grid_then_seeds_h100.sh <grid_config> <weight.pth> [exp_name]
```

- Slurm walltime is set automatically from the grid config path (override with
  `SLURM_TIME=HH:MM:SS` if needed):
  **H3D → 4 h**, **DALES → 8 h**, **ECLAIR → 12 h**.
  `submit_*` auto-picks native Slurm (`/usr/bin/sbatch` + `--time` on the CLI)
  when available; on Jean-Zay with the IMAGINE wrapper in `PATH`, it falls back
  to injecting `#SBATCH --time=…` into a temporary copy of the batch script.
  Force native: `SBATCH_CMD=/usr/bin/sbatch ./submit_grid_then_seeds_h100.sh …`
- Checkpoint selection metric is **`GridProbeEvaluator.select_metric`** baked into each
  grid config and propagated verbatim into the generated seed config:
  **H3D grid/seed → `macro_f1`**, **ECLAIR (and DALES) → `mIoU`**.
- `<weight>` is already written in every config (`weight = …`); the wrapper still
  requires it as arg 2 — pass the same path (or run `python tools/grid_then_seeds.py
  --grid-config <cfg>` directly, which falls back to the config's `weight=`).
- H100 job = 1 GPU, 24 CPU + auto-requeue/resume (`POINTCEPT_SLURM_REQUEUE=1`),
  so a job that exceeds its walltime re-queues and resumes the unfinished phase.

## Frozen-backbone checkpoints

```bash
cd /lustre/fswork/projects/rech/unv/usi32yh/Pointcept
B=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm

W_LPT=$B/873542/model/model_best.pth        # LitePT-B  Flair3D+ multitask (baseline / "classique")
W_PTV3=$B/1095469/model/model_best.pth      # PT-v3     Flair3D+ multitask
W_SPUNET=$B/1052217/model/model_best.pth    # SpUNet    Flair3D+ multitask
W_KPX=$B/1159986/model/model_best.pth       # KPConvX   Flair3D+ multitask
W_SONATA=/lustre/fsn1/projects/rech/unv/usi32yh/logs/pointcept_logs/slurm/862680/model/epoch_120.pth  # Sonata SSL (Flair3D+ fork, outdoor)
W_SONATA_INDOOR=ckpt/sonata/pretrain-sonata-v1m1-0-base.pth  # Meta/HF official indoor release
W_NOGNL=$B/1288597/model/model_best.pth     # LitePT-B  multitask WITHOUT GradNormLite
W_MONOLC=$B/1293025/model/model_best.pth    # LitePT-B  mono-task land-cover (segment v20) only
W_PRECLAIR=$B/1330042/model/model_best.pth  # LitePT-B  supervised ECLAIR semseg from scratch

SB=./submit_grid_then_seeds_h100.sh   # auto --time: H3D 4h / DALES 8h / ECLAIR 12h
```

## H3D — grid + seed (best checkpoint = validation **macro-F1**)

### One probe per backbone (encoder multiscale tap)

```bash
SB=./submit_grid_then_seeds_h100.sh
$SB configs/h3d/litept-b-v1m0-h3d-lin_enc.py      $W_LPT     h3d_lpt_enc
$SB configs/h3d/ptv3-v1m0-h3d-lin-grid-enc.py     $W_PTV3    h3d_ptv3_enc
$SB configs/h3d/spunet-v1m0-h3d-lin-grid-enc.py   $W_SPUNET  h3d_spunet_enc
$SB configs/h3d/kpconvx-v1m0-h3d-lin-grid-enc.py  $W_KPX     h3d_kpconvx_enc
$SB configs/h3d/sonata-v1m2-h3d-lin-grid.py       $W_SONATA  h3d_sonata
```

### Sonata-v1m1 indoor (official HF release — suppmat baseline)

Cross-domain aerial probes on DALES / H3D / ECLAIR using the **official indoor**
Sonata checkpoint ([`pretrain-sonata-v1m1-0-base.pth`](https://huggingface.co/facebook/sonata)).
There is **no official outdoor Sonata release**; the Flair3D fork above (`W_SONATA`) is a
separate line. Indoor configs use `in_channels=9` (`coord+color+normal`), zero-fill
missing normals (and color on DALES), `stride=(2,2,2,2)`, **`grid_size=0.02`**
(native indoor pretrain), and `FixedScaleCoord(1/N)` before `GridSample` with
`N ∈ {10, 25, 50}` for the scale ablation. **Do not** reuse the lr winner from
`W_SONATA` (Flair3D fork) or from another coord scale.

Download once (Jean-Zay or local):

```bash
mkdir -p ckpt/sonata
huggingface-cli download facebook/sonata pretrain-sonata-v1m1-0-base.pth \
  --local-dir ckpt/sonata
```

```bash
SB=./submit_grid_then_seeds_h100.sh
# coord /25
$SB configs/dales/sonata-v1m1-dales-lin-grid.py            $W_SONATA_INDOOR  dales_sonata_indoor_s25
$SB configs/h3d/sonata-v1m1-h3d-lin-grid.py              $W_SONATA_INDOOR  h3d_sonata_indoor_s25
$SB configs/eclair/sonata-v1m1-eclair-lin-grid.py         $W_SONATA_INDOOR  eclair_sonata_indoor_s25

# coord /10
$SB configs/dales/sonata-v1m1-dales-lin-grid-scale10.py  $W_SONATA_INDOOR  dales_sonata_indoor_s10
$SB configs/h3d/sonata-v1m1-h3d-lin-grid-scale10.py      $W_SONATA_INDOOR  h3d_sonata_indoor_s10
$SB configs/eclair/sonata-v1m1-eclair-lin-grid-scale10.py $W_SONATA_INDOOR  eclair_sonata_indoor_s10

# coord /50
$SB configs/dales/sonata-v1m1-dales-lin-grid-scale50.py  $W_SONATA_INDOOR  dales_sonata_indoor_s50
$SB configs/h3d/sonata-v1m1-h3d-lin-grid-scale50.py      $W_SONATA_INDOOR  h3d_sonata_indoor_s50
$SB configs/eclair/sonata-v1m1-eclair-lin-grid-scale50.py $W_SONATA_INDOOR  eclair_sonata_indoor_s50
```

### LitePT-B pretraining ablations (backbone changes, probe recipe identical)

```bash
SB=./submit_grid_then_seeds_h100.sh
$SB configs/experiment/w110/1/abla_grid_on_h3d/litept-b-v1m0-h3d-lin_enc_1.py          $W_NOGNL     h3d_lpt_noGNL
$SB configs/experiment/w110/1/abla_grid_on_h3d/litept-b-v1m0-h3d-lin_enc_2.py          $W_MONOLC    h3d_lpt_monoLC
$SB configs/experiment/w110/2/abla_grid_from_preECLAIR/litept-b-v1m0-h3d-lin_enc_1.py  $W_PRECLAIR  h3d_lpt_preECLAIR
```

### Extra tap-point variants (optional — same `select_metric="macro_f1"`)

```bash
SB=./submit_grid_then_seeds_h100.sh
$SB configs/h3d/litept-b-v1m0-h3d-lin_dec.py        $W_LPT     h3d_lpt_dec
$SB configs/h3d/litept-b-v1m0-h3d-lin_dec_ss.py     $W_LPT     h3d_lpt_decSS
$SB configs/h3d/ptv3-v1m0-h3d-lin-grid-dec.py       $W_PTV3    h3d_ptv3_dec
$SB configs/h3d/spunet-v1m0-h3d-lin-grid-dec.py     $W_SPUNET  h3d_spunet_dec
$SB configs/h3d/spunet-v1m0-h3d-lin-grid-dec-hc.py  $W_SPUNET  h3d_spunet_decHC
$SB configs/h3d/spunet-v1m0-h3d-lin-grid-enc-dec.py $W_SPUNET  h3d_spunet_encdec
```

## Feature-source ablation (DALES + ECLAIR) — fill the paper table

Subset ablation of probe feature taps (SS = last stage; decoder multi-scale =
hypercolumn; encoder multi-scale). Read **`test_mIoU_mean` ± `test_mIoU_std`**
from `logs/slurm/<jobid>/grid_then_seeds_summary.csv` (H3D uses
`test_f1_macro_*` instead — already filled except SpUNet dec-hc above).

| Table row | # ch. | DALES config | ECLAIR config |
|---|---:|---|---|
| LitePT Decoder (last stage) | 72 | `litept-b-v1m0-dales-lin-grid-dec-ss.py` | `litept-b-v1m0-eclair-lin_dec_ss.py` |
| LitePT Decoder (multi-scale) | 1404 | `litept-b-v1m0-dales-lin-grid-dec.py` | `litept-b-v1m0-eclair-lin_dec.py` |
| LitePT Encoder (multi-scale) | 1386 | `litept-b-v1m0-dales-lin-grid-enc.py` | `litept-b-v1m0-eclair-lin_enc.py` |
| PTv3 Decoder (multi-scale) | 1024 | `ptv3-v1m0-dales-lin-grid.py` | `ptv3-v1m0-eclair-lin-grid-dec.py` |
| PTv3 Encoder (multi-scale) | 992 | `ptv3-v1m0-dales-lin-grid-enc.py` | `ptv3-v1m0-eclair-lin-grid-enc.py` |
| SpUNet Decoder (last stage) | 96 | `spunet-v1m0-dales-lin-grid-dec.py` | `spunet-v1m0-eclair-lin-grid-dec.py` |
| SpUNet Encoder (multi-scale) | 512 | `spunet-v1m0-dales-lin-grid-enc.py` | `spunet-v1m0-eclair-lin-grid-enc.py` |
| SpUNet Decoder (multi-scale) | 832 | `spunet-v1m0-dales-lin-grid-dec-hc.py` | `spunet-v1m0-eclair-lin-grid-dec-hc.py` |
| SpUNet Enc+dec (multi-scale) | 1088 | `spunet-v1m0-dales-lin-grid-enc-dec.py` | `spunet-v1m0-eclair-lin-grid-enc-dec.py` |

Paths under `configs/dales/` and `configs/eclair/` respectively.

### DALES — 9 jobs (`select_metric=mIoU`)

```bash
SB=./submit_grid_then_seeds_h100.sh
$SB configs/dales/litept-b-v1m0-dales-lin-grid-dec-ss.py  $W_LPT    dales_lpt_decSS
$SB configs/dales/litept-b-v1m0-dales-lin-grid-dec.py     $W_LPT    dales_lpt_dec
$SB configs/dales/litept-b-v1m0-dales-lin-grid-enc.py     $W_LPT    dales_lpt_enc
$SB configs/dales/ptv3-v1m0-dales-lin-grid.py             $W_PTV3   dales_ptv3_dec
$SB configs/dales/ptv3-v1m0-dales-lin-grid-enc.py         $W_PTV3   dales_ptv3_enc
$SB configs/dales/spunet-v1m0-dales-lin-grid-dec.py       $W_SPUNET dales_spunet_dec
$SB configs/dales/spunet-v1m0-dales-lin-grid-enc.py       $W_SPUNET dales_spunet_enc
$SB configs/dales/spunet-v1m0-dales-lin-grid-dec-hc.py    $W_SPUNET dales_spunet_decHC
$SB configs/dales/spunet-v1m0-dales-lin-grid-enc-dec.py   $W_SPUNET dales_spunet_encdec
```

### ECLAIR — 9 jobs (`select_metric=mIoU`)

```bash
SB=./submit_grid_then_seeds_h100.sh
$SB configs/eclair/litept-b-v1m0-eclair-lin_dec_ss.py     $W_LPT    eclair_lpt_decSS
$SB configs/eclair/litept-b-v1m0-eclair-lin_dec.py        $W_LPT    eclair_lpt_dec
$SB configs/eclair/litept-b-v1m0-eclair-lin_enc.py        $W_LPT    eclair_lpt_enc
$SB configs/eclair/ptv3-v1m0-eclair-lin-grid-dec.py       $W_PTV3   eclair_ptv3_dec
$SB configs/eclair/ptv3-v1m0-eclair-lin-grid-enc.py       $W_PTV3   eclair_ptv3_enc
$SB configs/eclair/spunet-v1m0-eclair-lin-grid-dec.py     $W_SPUNET eclair_spunet_dec
$SB configs/eclair/spunet-v1m0-eclair-lin-grid-enc.py     $W_SPUNET eclair_spunet_enc
$SB configs/eclair/spunet-v1m0-eclair-lin-grid-dec-hc.py  $W_SPUNET eclair_spunet_decHC
$SB configs/eclair/spunet-v1m0-eclair-lin-grid-enc-dec.py $W_SPUNET eclair_spunet_encdec
```

## noRGB ablation — LitePT-B, grid + seed

Probes are trained **and** evaluated with colour removed on every forward
(`RandomDropColor(1.0, 1.0, keep_mask=True)` → the frozen backbone's learned
`color_mask_value`); coord/strength untouched. Same frozen backbone as the classic
LitePT-B (`873542`).

```bash
SB=./submit_grid_then_seeds_h100.sh
# H3D  → select_metric = macro_f1
$SB configs/experiment/w110/4/grid_h3d_norgb/litept-b-v1m0-h3d-lin_enc-norgb_1.py        $W_LPT  h3d_lpt_norgb

# ECLAIR → select_metric = mIoU
$SB configs/experiment/w110/4/grid_eclair_norgb/litept-b-v1m0-eclair-lin_enc-norgb_1.py  $W_LPT  eclair_lpt_norgb
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
    ./submit_grid_then_seeds_h100.sh <grid_config> <weight.pth> <exp_name>
  ```
- **Regenerate the seed config without a GPU:**
  ```bash
  python tools/grid_then_seeds.py --make-config-only --grid-config <cfg> --grid-dir <grid_dir> --save-root <out>
  ```
- The static seed configs (`configs/experiment/w110/2/grid_seed_h3d/*`,
  `configs/experiment/w110/4/seed_*_norgb/*`) bake a fixed winner lr by hand — only useful
  if you are **not** using this chained pipeline. `grid_then_seeds.py` always refits to the
  real (now macro-F1-selected) winner.
- A100 variant: same commands with [`submit_grid_then_seeds.sh`](submit_grid_then_seeds.sh) (no `_h100`).
