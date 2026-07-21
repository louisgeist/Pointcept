# Saturate VRAM / find max batch size (Jean Zay)

How to find safe `batch_size`, `batch_size_val`, and `batch_size_test` without running a full train/val/test epoch.

## Why not a full epoch?

Peak **activation** memory for a given batch shows up on the first `forward` (+ `backward` in train). Late OOMs you may have seen usually come from:

- denser tiles / crops after a while
- **Mix3D** (`mix_prob`, often `0.8`) merging scene pairs → larger per-scene tensors
- CUDA fragmentation over many steps
- first val / checkpoint / sparse-conv buffers

So: a **short probe underestimates** risk; a **full dataset epoch overpays** (I/O + metrics) without targeting the worst case.

**Recommended:** 2 phases — fast binary search, then a longer soak on the candidate.

## Tools

| File | Role |
|------|------|
| [`scripts/find_max_batch_size.py`](scripts/find_max_batch_size.py) | Dichotomie + soak (subprocess `tools/train.py` / `tools/test.py`) |
| [`sbatch_find_max_batch_size.sh`](sbatch_find_max_batch_size.sh) | Slurm launcher (H100 / `pointcept_124`) |

Older manual grid: [`configs/experiment/w105/1/saturate_vram/`](configs/experiment/w105/1/saturate_vram/) (`bs4`…`bs20`, `iter_per_epoch=100`). Prefer the script above.

## Method (2 phases)

1. **Probe (dichotomie)** — short runs to find a candidate BS  
2. **Soak** — longer run on that candidate to catch late OOMs  
3. Keep ~**15–20%** VRAM headroom (or use `BS-1`) for production

### Train

- Use real AMP / `point_max` / model from the target config
- Set **`mix_prob=1`** for the probe (worst-case Mix3D every step). Keep your real `mix_prob` (e.g. `0.8`) in training configs
- Prefer **even** batch sizes (Mix3D pairs scenes)
- Defaults: `--probe-steps 64`, `--soak-steps 500`

### Val / test

- No Mix3D in the usual eval path → search `batch_size_val` / `batch_size_test` separately (often larger than train)
- Cap samples with `--max-sample` (default `128`); do **not** need the full split
- Val ≈ test for VRAM when the forward path is similar (`test_single_fragment=True`, etc.)

Each trial is a **new Python process** (clean CUDA context).

## Local / interactive commands

```bash
# Train
python scripts/find_max_batch_size.py \
  --config-file configs/experiment/w105/2/10h/litept-v1m0-flair3d_13.py \
  --mode train --min-bs 2 --max-bs 32 --probe-steps 64 --soak-steps 500

# Val
python scripts/find_max_batch_size.py \
  --config-file configs/experiment/w105/2/10h/litept-v1m0-flair3d_13.py \
  --mode val --min-bs 1 --max-bs 16 --max-sample 128 --soak-steps 0

# Test
python scripts/find_max_batch_size.py \
  --config-file configs/experiment/w105/2/10h/litept-v1m0-flair3d_13.py \
  --mode test --min-bs 1 --max-bs 16 --max-sample 128 --soak-steps 0
```

### CLI knobs

| Flag | Meaning |
|------|---------|
| `--probe-steps` | Train steps **per dichotomie trial** (phase 1) |
| `--soak-steps` | Train steps for **confirmation** on the best BS (phase 2); `0` = skip. For val/test, used as a larger `max_sample` floor |
| `--mix-prob` | Train only; default `1.0` for worst-case Mix3D |
| `--even-bs` / `--no-even-bs` | Even BS in train (default on) |
| `--work-dir` / `--csv` | Where to write temp configs, logs, results |

Artifacts default under `exp/batch_size_search/<config>_<mode>_<timestamp>/`.

## Jean Zay (Slurm)

Same defaults as the commands above (train + val + test, 1×H100, 6h):

```bash
sbatch sbatch_find_max_batch_size.sh
```

Results: `logs/slurm/$SLURM_JOB_ID/batch_size_search/{train,val,test}/results.csv`  
Summary also appended to `logs/slurm/$SLURM_JOB_ID/job_info.log`.

### 🧩 Useful overrides

```bash
# Train only
MODE=train sbatch sbatch_find_max_batch_size.sh

# Train + val
MODES="train val" sbatch sbatch_find_max_batch_size.sh

# Other config / bounds
CONFIG=configs/experiment/w105/2/10h/litept-v1m0-flair3d_12.py \
  MAX_BS_TRAIN=24 MAX_BS_EVAL=12 \
  sbatch sbatch_find_max_batch_size.sh
```

| Env var | Default | Role |
|---------|---------|------|
| `CONFIG` | `.../litept-v1m0-flair3d_13.py` | Config path |
| `MODE` / `MODES` | `train val test` | Which searches to run |
| `MIN_BS_TRAIN` / `MAX_BS_TRAIN` | `2` / `32` | Train search range |
| `MIN_BS_EVAL` / `MAX_BS_EVAL` | `1` / `16` | Val/test search range |
| `PROBE_STEPS` | `64` | Dichotomie train steps |
| `SOAK_STEPS_TRAIN` | `500` | Train soak |
| `MIX_PROB` | `1.0` | Probe Mix3D prob |
| `MAX_SAMPLE` | `128` | Val/test sample cap |
| `SOAK_STEPS_EVAL` | `0` | Eval soak (`max_sample` floor) |
| `NUM_WORKER` | `8` | DataLoader workers |
| `EXTRA_OPTIONS` | empty | Extra Pointcept `key=value` options |

Env setup matches other H100 jobs (`arch/h100`, `cuda/12.4.1`, `pointcept_124`, H100 `pointops` on `PYTHONPATH`).

## Applying results

After a confirmed BS:

- Train: `batch_size = confirmed_bs * num_gpu` (probed with `mix_prob=1`; keep real `mix_prob` in the experiment config)
- Val: `batch_size_val = …`
- Test: `batch_size_test = …`

Configs that share the same backbone / heads / `point_max` / AMP usually share the same max BS — no need to re-sweep every sibling experiment file.

## Pitfalls

- Do **not** probe train with `mix_prob=0` (too optimistic)
- Do **not** take the BS that OOMs at the edge with no margin
- Script strips `PreciseEvaluator` from probe hooks so a full test is not launched after a short train
- Test mode builds a 1-step seed checkpoint first, then runs `tools/test.py`
