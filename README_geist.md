

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
Use ``--{task}_definition`` flags to override defaults (segment=v20, land_use=default,
natural_habitat=by_habitat_x_domain, forest=default). ``segment=v20`` matches Flair3D-build
label v20 (same finer12 taxonomy as v19; upstream other-infra filter only).
``climatic_domain.npy`` is written by default when ``--natural_habitat_definition default``;
pass ``--no-write-climatic-domain-category`` to skip. Re-run with ``--force`` when
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
 --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest_D073.csv \
 --natural_habitat_definition default \
 --num_workers 24 \
 --force
```

Training configs must match on-disk definitions for tasks without on-the-fly remap, or set
``label_definitions`` + ``Flair3DLabelRemap`` when remapping at load time (see
``configs/experiment/w101/5/nathab_moisture/litept-v1m0-flair3d_1.py`` and ``_2`` / ``_3`` for v2/v3).

**Tile climatic domain fractions (`by_climatic_domain`):** maps CarHab ids 0–35 to Temperate /
Mediterranean / Alpine; ids 36–43 (mineral, aquatic, cultivated, built, N/A, roads) → void.
Exports per-tile counts and fractions over all points for downstream analysis:

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

Output: ``tile_domain_fractions.csv`` under ``--output_dir``.
Requires NH preprocessed with ``--natural_habitat_definition default``.

Analyze the exported CSV locally:

```bash
python scripts/analyze_flair3d_tile_domain_fractions_csv.py \
  --input stats/flair3d/tile_climatic_domain/tile_domain_fractions.csv \
  --output_dir stats/flair3d/tile_climatic_domain/analysis
```

Outputs: ``summary.json``, ``summary_1km.json``, classified CSVs, and bar plots
(``subtile_*.png`` for subtiles, ``tile_1km_*.png`` for 1 km² tiles; percentages only).

**Tile climatic-domain classification labels (`climatic_domain.npy`):** one scalar per subtile
(0=Temperate, 1=Mediterranean, 2=Alpine, -1=mixed/void/missing), assigned at 1 km² granularity
when exactly one climatic domain is present among NH points. Written automatically at the end of
preprocessing when ``--natural_habitat_definition default`` (default-on; disable with
``--no-write-climatic-domain-category``).

If scene preprocessing finished but the climatic-domain pass failed with
``ModuleNotFoundError: No module named 'pointcept'``, re-run the same preprocess command with
``PYTHONPATH`` set to the repo root and **without** ``--force`` (existing scenes are skipped;
only ``climatic_domain.npy`` is written):

```bash
export PYTHONPATH="$WORK/Pointcept:$PYTHONPATH"   # or export PYTHONPATH="$PWD" from repo root
python pointcept/datasets/preprocessing/flair3d_plus/preprocess_flair3d_v2.py \
  --ply_root /lustre/fsn1/projects/rech/unv/usi32yh/data_flair3d_build/flair3d_label_enhanced \
  --dataset_root /lustre/fswork/projects/rech/unv/usi32yh/Pointcept/data/flair3d_plus/raw \
  --output_root $WORK/Pointcept/data/flair3d_plus \
  --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest.csv \
  --natural_habitat_definition default \
  --num_workers 32
```

Or on an existing output root:

```bash
python scripts/assign_flair3d_climatic_domain_labels.py \
  --output_root data/flair3d_plus \
  --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest.csv \
  --splits train,val,test
```

**Add network labels** (from Flair3D-build exported graphs):

First enrich the split manifest with ``ROADS`` / ``RAILROADS`` / ``TRANSMISSION_LINES``
via Flair3D-build (see its ``README_network.md`` §3): ``True`` only when usable
segments remain after export filters (statuses ``ok`` / ``skipped_exists``).

```bash
# From Flair3D-build — export GPKGs + enrich CSV (or enrich only if graphs exist)
python scripts/export_network_graphs.py network=v4          # Hecate
python scripts/export_network_graphs.py network=v4_jz       # Jean Zay
# python scripts/export_network_graphs.py network=v4_jz enrich_manifest_only=true
```

The networks are represented as graphs. To train for this task, we convert them to binary masks
with ``rasterize_network.py``: densify each ROI once (``sample_step_m=0.25`` along edges,
``line_width_m=0.2`` → samples at ``± width/2`` on the segment normal, no centerline point),
then slice 1 m cells into patches (hard-fails if a ``True`` flag has no ``*_graph.gpkg``,
or if a manifest ``LIDARHD=True`` patch is missing ``coord.npy`` on disk). Empty tiles store
``meta.network`` only (no ``network.npy``).

On Hecate (D067)
```bash
python pointcept/datasets/preprocessing/flair3d_plus/rasterize_network.py \
  --data_root data/flair3d_plus \
  --network_graphs_root /data/geist/Flair3D-build/data/network_graphs \
  --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest_D067.csv \
  --num_workers 12
```

On Jean Zay :
```bash
python pointcept/datasets/preprocessing/flair3d_plus/rasterize_network.py \
  --data_root data/flair3d_plus \
  --network_graphs_root /lustre/fsn1/projects/rech/unv/usi32yh/data_flair3d_build/network_graphs \
  --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest.csv \
  --num_workers 16
```

**Add forest_2d labels** (2D grid variant of the per-point `forest` task): unlike network,
FOREST is already a raster, so ``rasterize_forest.py`` just resamples the window of the source
FOREST GeoTIFF covering each tile's point-cloud bounding box (majority vote) onto the target
``pixel_m`` grid and writes it out ``(1, H, W)`` south-up, same layout as ``network.npy``.
FOREST coverage is complete for every manifest patch (no "expected but absent" case like
network), so this must be run once before any `forest_2d` training — no tile has
`forest_2d.npy` until this has run:

```bash
python pointcept/datasets/preprocessing/flair3d_plus/rasterize_forest.py \
    --data_root data/flair3d_plus \
    --source_dataset_root data/flair3d_plus/raw \
    --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest.csv \
    --pixel_m 0.5
```

**Visualize network masks** (GT binary panels + mean-pooled LiDAR RGB on the same 1 m grid).
Mutually exclusive modes: ``--tile`` (one subtile) or ``--roi`` (all subtiles stitched).
With predictions (``--logits`` / ``--result-dir``), also shows soft probs, a binarized row
(``--threshold``, default 0.2; NaN cells = unobserved → background), and predicted-graph
panels. Pred colormap is fixed ``[0, 1]`` by default; ``--prob-autoscale`` stretches it to
the shared finite min/max across channels:

```bash
# GT only (subtile)
python scripts/visualize_network_mask.py \
  --tile data/flair3d_plus/train/D067-2021_LIDARHD/AF-S1-22/D067-2021_AF-S1-22_1-1 \

# GT + pred probs + binarized @ 0.5 (subtile)
python scripts/visualize_network_mask.py \
  --tile data/flair3d_plus/train/D067-2021_LIDARHD/AF-S1-22/D067-2021_AF-S1-22_1-1 \
  --logits exp/flair3d/spunet_network_overfit_long/result/D067-2021_AF-S1-22_1-1_logits_network.npy \
  --threshold 0.5 \
  --prob-autoscale

# + predicted graph row (mask -> graph pipeline, same as APLS eval) with GT graph overlay
python scripts/visualize_network_mask.py \
  --tile data/flair3d_plus/train/D067-2021_LIDARHD/AF-S1-22/D067-2021_AF-S1-22_1-1 \
  --logits exp/flair3d/spunet_network_overfit_long/result/D067-2021_AF-S1-22_1-1_logits_network.npy \
  --threshold 0.5 \
  --network-graphs-root /data/geist/Flair3D-build/data/network_graphs

# Full ROI (stitched GT + RGB + logits; official per-ROI APLS when --network-graphs-root is set)
python scripts/visualize_network_mask.py \
  --roi data/flair3d_plus/train/D067-2021_LIDARHD/AF-S1-22 \
  --result-dir exp/flair3d/spunet_network_overfit_ROI/result \
  --threshold 0.5 \
  --prob-autoscale \
  --network-graphs-root /data/geist/Flair3D-build/data/network_graphs \
  --out /tmp/AF-S1-22_network_roi.png
```

Graph row colors: predicted edges/nodes = yellow/orange, GT overlay (with `--network-graphs-root`)
= cyan/white. In ``--tile`` mode the graph is built **per subtile** for quick visual QA only —
real APLS numbers (`tools/eval_network_apls.py`) stitch all subtiles of a ROI first, so a
subtile-local graph here can show extra truncated/disconnected branches near tile edges that
won't appear in the real per-ROI prediction. With `--network-graphs-root` in tile mode, a
per-channel APLS score (GT clipped to the subtile) is a **local sanity-check only**. In
``--roi`` mode, stitching + APLS match the official per-ROI metric.

**Network test predictions** (`{tile}_logits_network.npy`): shape `(r, H, W)` with
`r=3` (ROADS / RAILROADS / TRANSMISSION_LINES), same grid as `network.npy` /
`meta.network`. Values are soft foreground probabilities in `[0, 1]`. Cells with
**no LiDAR point** in that 1 m Lambert pixel are stored as **`NaN`** (unobserved —
not background). Use `np.isnan` / `np.nan_to_num` before thresholding.

**Network cell binning (precision):** absolute Lambert XY is reconstructed in
**float64** from local `coord` (float32) + `coord_translation` (float64) via
`ExtractAbsXY`, then converted once to integer cell indices
(`network_cell` absolute, `network_pix` relative to the tile origin) by
`NetworkRasterToPointLabels`. Do not carry absolute Lambert as float32 — ULP at
Y ≈ 6–7×10⁶ is ~0.5–1 m and breaks 1 m cell assignment.

**Per-subtile natural habitat multi-label (`natural_habitat_multilabel.npy`):** length-15 int8
multi-hot per subtile (temperate, mediterranean, alpine, humid, mesic, dry, forest, open,
acidic, basic, cultivated, built, road, mineral, aquatic). Each label is set when its point
fraction is >= 1% of all subtile points (`coord.npy`). Computed **per subtile** (no 1 km²
aggregation). Opt-in at preprocess time (`--write-natural-habitat-multilabel`); off by default.

```bash
python scripts/assign_flair3d_natural_habitat_multilabel.py \
  --output_root data/flair3d_plus \
  --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest.csv \
  --splits train,val,test \
  --threshold 0.01
```

Requires NH preprocessed with ``--natural_habitat_definition default``.

Train multi-task segment + forest + elevation + NH multilabel (LitePT):

```bash
sh scripts/train.sh -g 1 -d flair3d -c experiment/w101/8/nh_multilabel/litept-v1m0-flair3d_1 -n litept_nh_multilabel_mt
```

Val/test logs multi-label metrics: ``macro_f1``, ``micro_f1``, ``subset_acc``, ``hamming_acc``, per-label F1 under ``val/natural_habitat_multilabel/``.

Train mono-task classification (LitePT):

```bash
sh scripts/train.sh -g 1 -d flair3d -c experiment/w101/5/climatic_domain_cls/litept-v1m0-flair3d_1 -n litept_climatic_domain_cls
```

Train multi-task natural_habitat + climatic_domain:

```bash
sh scripts/train.sh -g 1 -d flair3d -c experiment/w105/5/saturate_vram/bs20 -n checkflair3d_bs20
```


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
python tools/train.py --config-file configs/flair3d_default/spunet_nh_multilabel_toy.py --num-gpus 1
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

`IterLimitedSampler` keeps a **persistent unseen pool** across trainer epochs:
each epoch draws the next `iter_per_epoch × batch_size` indices (per GPU, or
`× world_size` globally in DDP) without replacement. When the remaining unseen
indices are fewer than the epoch budget, the leftover pool is discarded and a
fresh full shuffle is created.

When the per-epoch index budget exceeds the dataset size, the sampler falls back
to sampling with replacement within the epoch.

**Resume caveat:** the unseen pool is not checkpointed; on resume a new shuffle is
started (training continues normally, but sampling order is not bit-identical to
an uninterrupted run).

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

#### Stratified split

Use a **precomputed CSV sidecar** to validate on a fixed subset (~2k tiles on ~34k val) instead of the full split. Selection stratifies on **segment** (point-level histogram L1) and **natural_habitat_multilabel** (15-label presence L1, inverse-frequency weighted for rare labels such as `mineral`, `aquatic`, `built`).

Generate the sidecar once (warm random + greedy, fixed `--seed`):

```bash
python scripts/build_stratified_subset.py \
  --data_root data/flair3d_plus \
  --csv_manifest data/flair3d_plus/raw/scene_split_manifest.csv \
  --split val \
  --max_sample 2000 \
  --warm-random 1000 \
  --seed 0 \
  --keys segment natural_habitat_multilabel \
  --output data/flair3d_plus/manifests/val_dev_subset_2000.csv
```

Definie `data.val` (and optionally `data.test`):

```python
val=dict(
    ...
    stratified_subset_manifest="data/flair3d_plus/manifests/val_dev_subset_2000.csv",
)
```

Behavior:

- **`stratified_subset_manifest` set** → filter val/test to the fixed sidecar (no recompute at train time)..

Generated files (same output prefix):

- Sidecar: `val_dev_subset_2000.csv` (`split`, `patch_id`, diagnostic columns)
- Segment distribution: `val_dev_subset_2000.distribution_segment.csv` (`stage=full|subset`, point fractions)
- NH distribution: `val_dev_subset_2000.distribution_nh.csv` (`stage=full|subset`, scene presence per label)
- Metadata: `val_dev_subset_2000.csv.meta.json`


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

Train directement une config dans experiment (sur JeanZay, JZ):
```bash
cdpt
python -m tools.train \
  --config-file configs/experiment/w105/2/10h/litept-v1m0-flair3d_12.py \
  --num-gpus 1 \
  --num-machines 1 \
  --machine-rank 0 \
  --dist-url auto \
  --options save_path=outputs/vram
```

### Sonata pretrain + periodic linear probe (Flair3D+)

See [`README_sonata_geist.md`](README_sonata_geist.md).

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