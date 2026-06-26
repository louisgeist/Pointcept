

Attention à bien donner les bonnes wheels pour les package torch-scatter and co :
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
```

### S3DIS check
#### PP
Le fichier à fix manuellement: 
`/data/geist/datasets/s3dis/Stanford3dDataset_v1.2/Area_5/office_19/Annotations/ceiling_1.txt`

```bash
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py \
  --splits Area_1 Area_2 Area_3 Area_4 Area_5 Area_6 \
  --dataset_root /data/geist/datasets/s3dis/Stanford3dDataset_v1.2 \
  --output_root /data/geist/Pointcept/data/s3dis \
  --align_angle
```

#### PP on JZ
Copy the fixed file
```bash
scp -r -J passerelle lgeist@hecate:/data/geist/datasets/s3dis/Stanford3dDataset_v1.2/Area_5/office_19/Annotations/ceiling_1.txt usi32yh@jean-zay.idris.fr:/lustre/fsn1/projects/rech/unv/usi32yh/data/s3dis/Stanford3dDataset_v1.2/Area_5/office_19/Annotations/ceiling_1.txt
```


```bash
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py   --splits Area_1 Area_2 Area_3 Area_4 Area_5 Area_6   --dataset_root /lustre/fsn1/projects/rech/unv/usi32yh/data/s3dis/Stanford3dDataset_v1.2   --output_root data/s3dis   --align_angle
```

#### Train
train S3DIS without the normal features:
```bash
sh scripts/train.sh -g 1 -d s3dis -c ptv3_nonormal -n ptv3_nonormal
```

### Flair3D+

#### Preprocessing

Label remaps are defined in ``pointcept/datasets/preprocessing/flair3d_plus/flair3d_label_remap.py``.
Use ``--{task}_definition`` flags to override defaults (land_use=filtered,
natural_habitat=by_habitat_x_domain, segment/forest=default). Re-run with ``--force`` when
definitions change.

**On-the-fly label remapping (e.g. natural_habitat `by_moisture`):** preprocess once with
``--natural_habitat_definition default`` so ``natural_habitat.npy`` keeps the 44 fine CarHab ids.
At training time, add ``Flair3DLabelRemap`` to the data pipeline and set
``init_task_configs(..., definitions={"natural_habitat": "by_moisture"})`` with
``storage_definitions={"natural_habitat": "default"}``. No re-preprocessing is needed to try
other LUT-compatible mappings in parallel jobs.

```bash
python pointcept/datasets/preprocessing/flair3d_plus/preprocess_flair3d_v2.py \
 --ply_root /lustre/fsn1/projects/rech/unv/usi32yh/data_flair3d_build/flair3d_label_enhanced \
 --dataset_root /lustre/fswork/projects/rech/unv/usi32yh/Pointcept/data/flair3d_plus/raw \
 --output_root $WORK/Pointcept/data/flair3d_plus \
 --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest.csv \
 --natural_habitat_definition default \
 --num_workers 24 \
 --force
```

On hecate:

```bash
python pointcept/datasets/preprocessing/flair3d_plus/preprocess_flair3d_v2.py \
 --ply_root /data/geist/Flair3D-build/data/flair3d_label_enhanced \
 --dataset_root /data/geist/Pointcept/data/flair3d_plus/raw \
 --output_root data/flair3d_plus \
 --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest_D067.csv \
 --natural_habitat_definition default \
 --num_workers 24 \
 --force
```

Training configs must match on-disk definitions for tasks without on-the-fly remap, or set
``label_definitions`` + ``Flair3DLabelRemap`` when remapping at load time (see
``configs/experiment/w101/5/nathab_moisture/litept-v1m0-flair3d_1.py`` and ``_2`` / ``_3`` for v2/v3).

**Tile climatic domain audit (`by_climatic_domain`):** maps CarHab ids 0–35 to Temperate /
Mediterranean / Alpine; ids 36–43 (mineral, aquatic, cultivated, built, N/A, roads) → void.
Run on Jean-Zay to check whether tiles are strictly pure (single domain among eligible points):

```bash
python scripts/analyze_flair3d_tile_climatic_domain.py \
  --data_root $WORK/Pointcept/data/flair3d_plus \
  --csv_manifest data/flair3d_plus/raw/scene_split_manifest.csv \
  --splits train,val,test \
  --missing_tiles_manifest data/flair3d_plus/missing_ply_preflight.txt \
  --too_small_tiles_manifest data/flair3d_plus/too_small_tiles.csv \
  --num_workers 24 \
  --output_dir stats/flair3d/tile_climatic_domain
```

Outputs: ``summary.json``, ``tiles.csv``, ``mixed_tiles.csv`` under ``--output_dir``.
Requires NH preprocessed with ``--natural_habitat_definition default``.


#### Train Flair3D

On hecate:
```bash
sh scripts/train.sh -g 1 -d flair3d -c ptv3_nonormal_subtile -n ptv3_nonormal_subtile
```
ou regarder script debug mode

Or directly with Python (from repo root):

```bash
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python tools/train.py --config-file configs/pureforest/kpconvx-toy.py --num-gpus 1
```

Mini-dataset smoke test (10 epochs, eval every 10, 2 train + 2 val samples).

### Training schedule (`epoch` vs `total_iters`)

Two mutually exclusive modes are resolved in
[`pointcept/engines/defaults.py`](pointcept/engines/defaults.py) (`default_config_parser`).

#### Classic mode (`total_iters = None`)

| Parameter | Role |
|-----------|------|
| `epoch` | Total dataset passes over the full run |
| `eval_epoch` | Number of trainer epochs (each ends with validation if `evaluate=True`) |
| `loop` | Derived: `epoch // eval_epoch` — dataset repeats per trainer epoch (`data.train.loop`) |

Scheduler steps: `len(train_loader) * eval_epoch`.

#### Iter-limited mode (`total_iters` set)

Use this on large datasets: each trainer epoch runs a fixed number of random batch
steps instead of a full dataset pass.

| Parameter | Role |
|-----------|------|
| `total_iters` | Total optimizer steps (scheduler budget) |
| `iter_per_epoch` | Batch steps per epoch (default **1000**); uses `IterLimitedSampler` |
| `num_epochs` | Derived: `total_iters // iter_per_epoch` (`max_epoch` in the trainer) |
| `eval_every` | Run validation every N epochs (default **5**); also on the last epoch |
| `loop` | Forced to **1** |

`IterLimitedSampler` draws without replacement when the per-epoch index budget fits in the
dataset (`iter_per_epoch × batch_size` per GPU, or `× world_size` globally in DDP);
otherwise it falls back to sampling with replacement.

`total_iters` must be divisible by `iter_per_epoch`. `epoch` and `eval_epoch` are **classic-mode only**.

Example (100k steps, 1000 steps/epoch → 100 epochs, validate every 5 epochs → 20 validations):

```python
total_iters = 100_000
iter_per_epoch = 1000
eval_every = 5
warmup_steps = 2500
```

CLI override:

```bash
python tools/train.py --config-file configs/flair3d_default/spunet_toy.py \
  --options total_iters=1000000 eval_every=5 iter_per_epoch=1000
```

Smoke test: `python tools/test_iter_limited_sampler.py`

**Not supported** with `PartialSampledTrainer` or `MultiDatasetTrainer` (use `DefaultTrainer`).

With the script (recommended; `EXTRA_OPTIONS` is passed as `--options` to the Python command):

```bash
export EXTRA_OPTIONS="epoch=10 eval_epoch=10 max_sample_train=2 max_sample_val=2"
sh scripts/train.sh -g 1 -d flair3d -c ptv3_nonormal_subtile -n ptv3_nonormal_subtile
```

#### Flair3D+ mono-task (one semantic target per run)

Configs under [`configs/flair3d_default/`](configs/flair3d_default/) — one folder per target, four backbones each (LitePT, SpUNet, PTv3, KPConvX). All mono runs use ``lr=1e-3`` and ``scene_split_manifest.csv``.

```text
configs/flair3d_default/
├── segment/       # litept|spunet|ptv3|kpconvx-v1m0-flair3d.py (self-contained each)
├── forest/
├── land_use/
└── natural_habitat/
```

Each file inherits only ``default_runtime``; task wiring uses ``init_task_configs`` / ``init_task_criteria``. Regenerate with ``python tools/gen_flair3d_mono_configs.py`` if needed.

Example:

```bash
python tools/train.py --config-file configs/flair3d_default/land_use/litept-v1m0-flair3d.py --num-gpus 1
```

```bash
python tools/train.py --config-file configs/experiment/w96/6/flair_lp/segment-litept-v1m0-flair3d.py --num-gpus 1
```

Multi-target training (all semantic tasks + elevation) remains in ``multi-*-v1m0-flair3d.py`` at the root of ``flair3d_default/``.

#### Flair3D+ multi-target (segment, forest, land_use, natural_habitat, elevation)

Class names and ``num_classes`` / ``ignore_index`` per semantic target are defined in
[`pointcept/datasets/flair3d_config_utils.py`](pointcept/datasets/flair3d_config_utils.py).

- **Semantic targets**: set ``target_key`` on ``Flair3DDataset`` (train/val/test) to one of
  ``segment``, ``forest``, ``land_use``, ``natural_habitat``. The corresponding ``*.npy`` is
  copied into ``segment`` for the existing GridSample / loss pipeline. Example config:
  [`configs/flair3d_plus/litept_target_forest.py`](configs/flair3d_plus/litept_target_forest.py).
- 
- To confirm ...|**Checkpoint transfer** between tasks: use ``strict=False`` on ``load_state_dict``, or
  ``CheckpointLoader`` with ``exclude_keys`` for the old head (``seg_head`` / ``reg_head``).

- **W&B**: root config fields ``target_key`` and ``task`` (``semseg`` or ``regression``) are added
  as run tags when present.
- **Regression metrics** (multitask val/test): MAE and RMSE, logged to TensorBoard/W&B
  under ``val/reg/<task>/`` and ``test/reg/<task>/``.

```bash
python tools/train.py --config-file configs/flair3d_plus/litept_target_forest.py --num-gpus 1
```

Train directement une config dans experiment:
```bash
cd /lustre/fswork/projects/rech/unv/usi32yh/Pointcept
python -m tools.train \
  --config-file configs/experiment/w88/1/check_scheduler/1_ptv3_harmonized-transforms.py \
  --num-gpus 1 \
  --num-machines 1 \
  --machine-rank 0 \
  --dist-url auto \
  --options save_path=logs/local/test
```

### Other datasets

#### Dales
Preprocessing:
```
python pointcept/datasets/preprocessing/dales/preprocess_dales.py \
  --dataset_root data/dales/raw \
  --output_root data/dales \
  --num_workers 8 \
  --chunking 4
```

# Brouillon
python -m tools.train \
  --config-file configs/experiment/w90/5/dales2/ptv3_2b.py\
  --num-gpus 1 \
  --num-machines 1 \
  --machine-rank 0 \
  --dist-url auto \
  --options epoch=1 eval_epoch=1 data.train.max_sample=30 data.test.max_sample=30 data.val.max_sample=30
  
  
  data.train.max_sample=30


  python -m tools.train \
  --config-file configs/experiment/w90/5/dales2/ptv3_2b.py\
  --num-gpus 2 \
  --num-machines 1 \
  --machine-rank 0 \
  --dist-url auto \
  --options epoch=1 eval_epoch=1 data.train.max_sample=300 data.test.max_sample=30 data.val.max_sample=30