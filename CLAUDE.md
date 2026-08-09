# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a fork of **Pointcept** (github.com/Pointcept/Pointcept), a registry/config-driven codebase for point
cloud perception research (semantic/instance segmentation, pre-training, classification). The upstream project
supports many backbones (PTv3, PTv2, LitePT, SparseUNet/SpUNet, KPConvX, OctFormer, Swin3D, ...) and datasets
(ScanNet, S3DIS, SemanticKITTI, nuScenes, Waymo, ...).

The fork's focus is **Flair3D / Flair3D+** (French national LiDAR HD + FLAIR-HUB aerial imagery, extended with
new geospatial modalities — see `README_flair3dplus.md`), plus a Sensaturban dataset integration. Most active
development lives in `pointcept/datasets/flair3d*.py`, `pointcept/datasets/preprocessing/flair3d_plus/`, and
`configs/flair3d_default/` / `configs/experiment/`.

`README_geist.md` is the maintainer's running scratchpad (French/English mixed) with copy-pasteable preprocessing
and training commands for the exact datasets currently in use — check it first for concrete invocations before
re-deriving something from the general Pointcept README. `README_sonata_geist.md` covers Sonata self-supervised
pretraining on Flair3D+ specifically (see below) — check it before re-deriving Sonata/linear-probe invocations.

## Environment setup

```bash
conda env create -f environment.yml --verbose   # env name: pointcept-torch2.5.0-cu12.4
conda activate pointcept-torch2.5.0-cu12.4
```

torch-scatter/-sparse/-cluster/-spline-conv/-geometric wheels must match the exact torch+cuda build:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
# same pattern for torch-sparse, torch-cluster, torch-spline-conv, torch-geometric
```

PTv1/PTv2 custom ops (and other `libs/*`) build via `python setup.py install` from inside each `libs/<name>` dir;
set `TORCH_CUDA_ARCH_LIST` for the target GPU when building in Docker or for a different arch than the build host.

Always `export PYTHONPATH=./` (or `$PWD`) before running training/testing/preprocessing scripts directly, or use
`scripts/train.sh` / `scripts/test.sh`, which set this up.

## Common commands

**Train** (script form, recommended — creates `exp/<dataset>/<exp_name>`, wipes it first):
```bash
sh scripts/train.sh -g <NUM_GPU> -d <dataset_name> -c <config_name_or_path> -n <exp_name>
sh scripts/train.sh -g 1 -d flair3d -c ptv3_nonormal_subtile -n ptv3_nonormal_subtile
# resume: add -r true
```
`-c` accepts either a path relative to `configs/` (e.g. `experiment/w101/8/nh_multilabel/litept-v1m0-flair3d_1`)
or a bare name resolved under `configs/<dataset>/`.

**Train directly** (needed for `--options` overrides or when not using the wrapper script):
```bash
export PYTHONPATH="$PWD"
python tools/train.py --config-file configs/flair3d_default/spunet_toy.py --num-gpus 1 \
  --options epoch=10 eval_epoch=10 max_sample_train=2 max_sample_val=2
```

**Test / precise evaluation:**
```bash
sh scripts/test.sh -d <dataset_name> -n <exp_name> -w model_best
# or: python tools/test.py --config-file <CONFIG> --options save_path=<EXP_DIR> weight=<CHECKPOINT>
```
Note: `PreciseEvaluator` hook already runs precise testing automatically at the end of training.

**Tests:** `pytest tests/` (unittest-style test files under `tests/`, e.g.
`pytest tests/test_flair3d_label_remap.py`). There's also a standalone smoke test for the iter-limited sampler:
`python tools/test_iter_limited_sampler.py`.

**Slurm:** `sbatch_flair3d.sh`, `sbatch_s3dis.sh`, `sbatch_s3dis_multigpu.sh`,
`sbatch_find_max_batch_size.sh` — training under Slurm sets `JOB_DIR` so `scripts/train.sh` places the experiment
under `logs/slurm/%j/` instead of `exp/`. `scripts/slurm_requeue_watchdog.sh` / `slurm_requeue_trap.sh` handle
requeue-on-preemption.

## Config system

Configs are plain Python dicts loaded via `pointcept.utils.config.Config` (mmcv-style): a config file can set
`_base_ = ["../_base_/default_runtime.py", ...]` to inherit and override. `configs/_base_/default_runtime.py`
defines the shared defaults (hooks, trainer/tester type, schedule params, wandb, etc.) that most configs build on.
Objects (models, datasets, hooks) are constructed by name through registries (`pointcept/utils/registry.py`,
mirroring mmcv's `Registry`/`build_from_cfg`) — e.g. `MODELS`/`MODULES` in `pointcept/models/builder.py`,
similarly for datasets. `--options key=value` on the CLI patches the merged config dict before building
(`Config.merge_from_dict`), including dotted keys like `data.train.max_sample=30`.

Per the Cursor rule in `.cursor/rules/experiment-config-generation.mdc` (applies to `configs/experiment/**`;
mirrored here — update both files together if these conventions change):
- New experiment configs must be **fully standalone** (no `_base_=["lite_ft_1"]`-style sibling inheritance) —
  copy a reference config and edit only what's needed.
- Path convention: `configs/experiment/w<NN>/<day>/...` where `<day>` is 1=Mon…5=Fri (a date-based folder, not
  `grp_exp`).
- Set `grp_exp`/`num_exp` explicitly; `num_exp` must match the `<model>_<num_exp>.py` filename suffix.
- If the config loads pretrained weights / uses `CheckpointLoader`, treat and label it as fine-tuning (`FT`), not
  `scratch`.

## Training schedule: two mutually exclusive modes

Resolved in `pointcept/engines/defaults.py` (`default_config_parser`); documented in detail (with resume caveats)
in `README_geist.md` and in comments in `configs/_base_/default_runtime.py`.

- **Classic mode** (`total_iters = None`, default): `epoch` = total dataset passes, `eval_epoch` = number of
  trainer epochs (validates each), `loop = epoch // eval_epoch` is derived.
- **Iter-limited mode** (`total_iters` set): each trainer epoch runs a fixed `iter_per_epoch` (default 1000)
  random batch steps via `IterLimitedSampler`, instead of a full dataset pass — use this for large datasets.
  `num_epochs = total_iters // iter_per_epoch` is derived, `eval_every` controls validation cadence (default 5),
  `total_iters` must be divisible by `iter_per_epoch`. Not supported with `PartialSampledTrainer` or
  `MultiDatasetTrainer` (use `DefaultTrainer`). The sampler's unseen-pool state is **not checkpointed** — resume
  restarts the shuffle (training is still correct, just not bit-identical).

## Flair3D label/segment definitions — read before touching training configs

Enforced by `.cursor/rules/flair3d-segment-label-definition.mdc` (mirrored here — update both files together):
`label_definitions["segment"]` in a train
config **must match** the preprocessing flag baked into `segment.npy` on disk (check each tile's `meta.json` →
`label_definitions.segment`). A mismatch causes CUDA `nll_loss` asserts at train time, not a clean error.

- Real experiments (`configs/experiment/**`, `configs/flair3d_default/**` production configs): use
  `segment="v20"`. v20/v19 share the same train taxonomy (finer12, 15 classes + void); v20 differs only in an
  upstream `other_infrastructure_filter` fix. Keep `v19` only in historical experiment configs that already used
  it — don't introduce new v19 configs.
- Toy/local profiling configs (`configs/flair3d_default/*toy*`) may use `segment="v18"` if local tiles (e.g.
  D067) are still v18 — never copy that into a real experiment config.
- If unsure whether on-disk data matches, check with:
  `python -c "import json; print(json.load(open('<tile>/meta.json'))['label_definitions'])"`

## Flair3D+ multi-task architecture

Semantic/scene targets (`segment`, `forest`, `land_use`, `natural_habitat`, `climatic_domain`,
`natural_habitat_multilabel`, `network`, elevation regression) are wired through
`pointcept/datasets/flair3d_config_utils.py`:
- `init_task_configs` / `init_task_criteria` build the per-task config + loss wiring from a tuple of target keys;
  `get_semantic_config`, `get_classification_config`, `get_multilabel_classification_config`,
  `get_elevation_config`, `get_network_config`, `get_pixel_semantic_config` supply the per-target-type pieces.
- Single-task configs set `target_key` on `Flair3DDataset` (train/val/test); multi-task configs
  (`multi-*-v1m0-flair3d.py`, root of `flair3d_default/`) pass multiple target keys through `init_task_configs`.
- Preprocessing (`pointcept/datasets/preprocessing/flair3d_plus/preprocess_flair3d_v2.py`) writes label npys with
  `--{task}_definition` flags (segment/land_use/natural_habitat/forest); `Flair3DLabelRemap` in the data pipeline
  allows **on-the-fly** remapping at train time (e.g. `natural_habitat` `by_moisture`/`by_climatic_domain`)
  without re-preprocessing, provided the on-disk storage definition is LUT-compatible with the target remap —
  see `label_definitions` + `storage_definitions` usage in configs under `configs/experiment/w101/5/nathab_moisture/`.
- Network labels (roads/railroads/transmission lines) are graph-derived and rasterized into per-tile binary
  masks by `pointcept/datasets/preprocessing/flair3d_plus/rasterize_network.py` (hard-fails on missing
  `*_graph.gpkg` for tiles flagged `True` in the split manifest).
- Forest 2D grid labels (`forest_2d`) are rasterized directly from the FOREST GeoTIFF (the same source used by
  the per-point `forest` task) into per-tile masks by
  `pointcept/datasets/preprocessing/flair3d_plus/rasterize_forest.py` at a configurable `--pixel_m` (default
  0.5m); this must be run once before any `forest_2d` training, since no tile has `forest_2d.npy` until then.
- Stratified validation subsets (to validate on a fixed ~2k-tile sample instead of the full split) are
  precomputed once via `scripts/build_stratified_subset.py` and referenced from a config via
  `data.val.stratified_subset_manifest=<csv>`.

## Sonata SSL pretraining on Flair3D+

Full details, prerequisites, and copy-pasteable Jean-Zay commands: `README_sonata_geist.md`. Summary:

- Pretrain config: `configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py` (Sonata-v1m2 MultiView SSL, `train`
  split only, `evaluate=False` — no online val). Linear-probe config:
  `configs/flair3d_default/segment/sonata-v1m2-flair3d-lin.py` (frozen `PT-v3m2` backbone + linear head on
  segment v20, `total_iters=1000`/`iter_per_epoch=100`).
- Representation quality is tracked out-of-band via short linear-probe jobs on frozen checkpoints, not online
  validation. Two hooks in `pointcept/engines/hooks/misc.py` drive this: `LinProbeSbatchHook` (after each
  `CheckpointSaver` epoch, submits a probe job via `sbatch` — no-op without Slurm) and `MetricsJsonWriter` (each
  probe run writes `save_path/metrics.json` with `best_val_mIoU`).
- Jean-Zay launchers live under `scripts/sonata/`: `sbatch_pretrain.sh`, `sbatch_lin_probe.sh`,
  `append_lin_probe_result.py` (appends probe results to `$PRETRAIN_DIR/lin_probe_results.csv`), and
  `periodic_lin_probe.py` (optional local/replay watcher when not using `LinProbeSbatchHook`).
- Batch-size calibration for both configs goes through `scripts/find_max_batch_size.py` /
  `sbatch_find_max_batch_size.sh` — align `--mix-prob` with the config (`0` for Sonata pretrain, `0.8` for the
  linear probe); probe overlays replace `hooks` entirely so `LinProbeSbatchHook` never fires during VRAM search.

## Big-picture module layout

- `pointcept/engines/` — training/testing loops (`train.py`, `test.py`, `defaults.py` for config parsing,
  `launch.py` for multi-GPU/distributed launch) and `hooks/` (checkpointing, evaluators incl. `SemSegEvaluator`
  / `PreciseEvaluator`, multitask metric logging). `evaluator.py` handles cross-rank confusion-matrix/histogram
  sync for distributed eval.
- `pointcept/models/` — one subpackage per architecture (`point_transformer_v3/`, `litept/`, `sparse_unet/`,
  `kpconvx/`, etc.), plus `default.py` for the generic task wrapper (segmentation/classification/regression
  heads, multi-task loss combination, masked-feature learning) and `builder.py` for the `MODELS`/`MODULES`
  registries.
- `pointcept/datasets/` — one file per dataset, all built through `datasets/builder.py`'s registry and sharing
  `defaults.py` base classes (`DefaultDataset`, etc.); `preprocessing/<dataset>/` holds raw-data → `.npy`
  conversion scripts per dataset (e.g. `preprocessing/flair3d_plus/`, `preprocessing/pureforest/`).
- `configs/_base_/` — shared runtime defaults; `configs/<dataset>/` — per-dataset base configs; `configs/experiment/`
  — dated, standalone experiment configs (`w<week>/<weekday>/...`); `configs/flair3d_default/` — one
  mono-task config set per semantic target (`segment/`, `forest/`, `land_use/`, `natural_habitat/`) across 4
  backbones (LitePT, SpUNet, PTv3, KPConvX), regenerable via `python tools/gen_flair3d_mono_configs.py`.
- `tools/` — CLI entry points (`train.py`, `test.py`, plus dataset-specific ones like
  `test_s3dis_6fold.py`, `create_waymo_semseg_submission.py`).
- `scripts/` — shell wrappers (`train.sh`, `test.sh`) and one-off Python analysis/preprocessing-adjacent scripts
  for Flair3D (climatic domain stats, manifest building, dataset visualization with `viser`, etc.) — largely
  maintainer tooling, not part of the core library.
