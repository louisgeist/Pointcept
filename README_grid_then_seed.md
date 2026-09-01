# Grid → seed-ensemble pipeline (H3D / DALES / ECLAIR)

Chained pipeline: **lr sweep (grid) → pick winner by validation metric → generate and run a 10-init seed ensemble → aggregate**.

Entry point: [`tools/grid_then_seeds.py`](tools/grid_then_seeds.py).

```bash
python tools/grid_then_seeds.py \
  --grid-config configs/h3d/litept-b-v1m0-h3d-lin_enc.py \
  --weight ckpt/malibu3d/litept_b_multitask/model_best.pth \
  --save-root exp/grid_then_seeds/h3d_lpt_enc
```

- Checkpoint selection metric is **`GridProbeEvaluator.select_metric`** in each grid config:
  **H3D → `macro_f1`**, **DALES and ECLAIR → `mIoU`**.
- `weight` is already set in each config; the CLI argument overrides it when provided.
- Checkpoints are bundled under `ckpt/` (see [ckpt/README.md](ckpt/README.md)).

## Frozen-backbone checkpoints

```bash
W_LPT=ckpt/malibu3d/litept_b_multitask/model_best.pth
W_PTV3=ckpt/malibu3d/ptv3_multitask/model_best.pth
W_SPUNET=ckpt/malibu3d/spunet_multitask/model_best.pth
W_KPX=ckpt/malibu3d/kpconvx_multitask/model_best.pth
W_SONATA=ckpt/malibu3d/sonata_outdoor/epoch_120.pth
```

## H3D — grid + seed (validation **macro-F1**)

```bash
python tools/grid_then_seeds.py --grid-config configs/h3d/litept-b-v1m0-h3d-lin_enc.py \
  --weight $W_LPT --save-root exp/grid_then_seeds/h3d_lpt_enc

python tools/grid_then_seeds.py --grid-config configs/h3d/ptv3-v1m0-h3d-lin-grid-enc.py \
  --weight $W_PTV3 --save-root exp/grid_then_seeds/h3d_ptv3_enc

python tools/grid_then_seeds.py --grid-config configs/h3d/spunet-v1m0-h3d-lin-grid-enc.py \
  --weight $W_SPUNET --save-root exp/grid_then_seeds/h3d_spunet_enc

python tools/grid_then_seeds.py --grid-config configs/h3d/kpconvx-v1m0-h3d-lin-grid-enc.py \
  --weight $W_KPX --save-root exp/grid_then_seeds/h3d_kpconvx_enc

python tools/grid_then_seeds.py --grid-config configs/h3d/sonata-v1m2-h3d-lin-grid.py \
  --weight $W_SONATA --save-root exp/grid_then_seeds/h3d_sonata
```

## DALES / ECLAIR — feature-source ablation (validation **mIoU**)

See the config table in [README_MALIBU3D.md](README_MALIBU3D.md). Example (DALES, LitePT encoder tap):

```bash
python tools/grid_then_seeds.py \
  --grid-config configs/dales/litept-b-v1m0-dales-lin-grid-enc.py \
  --weight $W_LPT --save-root exp/grid_then_seeds/dales_lpt_enc
```

## Outputs

Under `--save-root`:

| File | Content |
|------|---------|
| `grid/grid_search_results.json` | leaderboard + winner |
| `grid/grid_probe_miou_history.csv` | per-epoch probe metrics |
| `seed_ensemble_config.py` | generated 10-init config |
| `seeds/seed_ensemble_results.json` | test mean ± std |
| `grid_then_seeds_summary.csv` | one-row summary |

## Notes

- **Grid already run?** Reuse it, seeds only:
  ```bash
  python tools/grid_then_seeds.py --skip-grid --grid-dir exp/grid_then_seeds/h3d_lpt_enc/grid \
    --grid-config configs/h3d/litept-b-v1m0-h3d-lin_enc.py --save-root exp/grid_then_seeds/h3d_lpt_enc
  ```
- **Regenerate seed config without GPU:**
  ```bash
  python tools/grid_then_seeds.py --make-config-only \
    --grid-config configs/h3d/litept-b-v1m0-h3d-lin_enc.py \
    --grid-dir exp/grid_then_seeds/h3d_lpt_enc/grid \
    --save-root exp/grid_then_seeds/h3d_lpt_enc
  ```
