# Forest 2D task — design spec

Date: 2026-08-09
Status: approved for planning

## Context

The `forest` target is currently a per-point (3D) binary semantic segmentation task: each
LiDAR point gets a Forest/Not-Forest label sampled (nearest-pixel) from the IGN "Masque
Forêt" GeoTIFF at preprocessing time (`forest.npy`, per-point, via
`preprocess_flair3d_v2.py`).

We want a second, independent variant of the same underlying label: a genuinely 2D task
that mean-pools per-point backbone features onto a 2D grid and applies a linear head to
produce a binary forest/non-forest raster prediction — analogous to how the existing
`network` (roads/rail) task works, except:
- mean-pooling instead of max-pooling (forest is an area-coverage class, not a thin
  curvilinear structure — max-pooling is the right choice for catching a single strong
  "there's a road here" signal, mean-pooling is the right choice for a coverage class where
  we want the aggregate signal across all points in the cell),
- no APLS/graph-based downstream evaluation (that's specific to network's graph-structured
  target),
- no dilated ("relaxed", buffer-tolerant) precision/recall/F1 — that metric exists because a
  1px lateral offset destroys IoU on a 1px-wide line while still being a good prediction;
  it's not a meaningful diagnostic for a blobby area-coverage mask like forest.

It must be possible to choose between the 3D (`forest`) and 2D (`forest_2d`) variants (or
run both at once) simply by including one or both keys in a multi-task config's
`target_keys`, the same way every other Flair3D+ task is selected.

## Non-goals

- No changes to the existing per-point `forest` task or its preprocessing.
- No TIFF reads at train/eval time — the 2D grid is precomputed once at preprocessing time,
  like `network`.
- No APLS-style downstream graph evaluation script for `forest_2d`.
- No dilated precision/recall/F1 for `forest_2d`.

## Resolution

`pixel_m = 0.5`. Rationale: LiDAR HD has a nominal average density of ~10 pts/m². At 0.5m
(0.25 m² cells) that's ~2.5 pts/pixel on average — enough for mean-pooling to have real
denoising value while still resolving forest boundaries meaningfully finer than the existing
1m `network` grid. At the native tiff resolution (0.2m, ~0.4 pts/pixel) most cells would
contain 0 or 1 point, defeating the point of pooling and effectively duplicating the
per-point 3D task.

## Design

### 1. Task registration (`pointcept/datasets/flair3d_config_utils.py`)

Add to `FLAIR3D_PIXEL_SEMANTIC_TASKS`:

```python
"forest_2d": {
    "task_type": "pixel_semantic",
    "num_networks": 1,
    "num_classes": 2,
    "ignore_index": 2,
    "channel_names": ["FOREST"],
    "names": ["Not Forest", "Forest", "Void"],
    "pooling": "mean",
    "enable_dilated_prf": False,
},
```

`network`'s entry is unchanged (`pooling` absent → defaults to `"max"`, `enable_dilated_prf`
absent → defaults to `True`), so existing behavior for `network` is bit-for-bit unchanged.

`get_pixel_semantic_config("forest_2d")` (already generic) serves this. Using the task is
just: include `"forest_2d"` (instead of, or alongside, `"forest"`) in a config's
`target_keys` tuple passed to `init_task_configs`/`init_task_criteria`/
`init_multitask_collect_keys` — no other config-level plumbing needed.

`init_multitask_collect_keys` currently hardcodes the extra grid-meta Collect keys to the
literal `network_*` names whenever *any* pixel_semantic target is present. Generalize it to
loop over every pixel_semantic key actually present in `target_keys` and emit
`{key}_cell, {key}_pix, {key}_origin_x, {key}_origin_y, {key}_pixel_m, {key}_height,
{key}_width}` per key. For `target_keys=("network",)` this produces the exact same output as
today (pure refactor, no config file needs to change); for `target_keys=("network",
"forest_2d")` it produces both independent key sets, so the two tasks can coexist in one
multi-task run without collisions.

### 2. Preprocessing (new standalone script, additive over existing tiles)

New script `pointcept/datasets/preprocessing/flair3d_plus/rasterize_forest.py`, modeled on
`rasterize_network.py`:

- For each already-preprocessed tile: compute absolute XY bounds from `coord.npy` +
  `coord_translation.npy` (reuse `_abs_xy_bounds_from_coord` / `grid_from_xy_bounds` from
  `network_xy_raster_utils.py`, `pixel_m=0.5`).
- Resolve the source FOREST GeoTIFF path via the existing `build_modality_patch_path`
  helper (`preprocess_flair3d_v2.py`).
- Read the corresponding window from the native 0.2m tiff and resample directly to the 0.5m
  target grid via a decimated `rasterio` read using `Resampling.mode` (majority vote) — this
  handles the non-integer 0.5/0.2 = 2.5x downsampling ratio correctly (a manual block-reshape
  would not divide evenly).
- Write `forest_2d.npy` — shape `(1, H, W)` uint8 — and `meta.json["forest_2d"]` with
  `origin_x, origin_y, pixel_m=0.5, width, height, crs="EPSG:2154", channel_order=["FOREST"]`.
- FOREST coverage is complete across all 74 (département, year) couples (unlike `network`,
  which can be legitimately absent for an ROI), so there is no "missing source" case to
  handle — every tile gets a `forest_2d.npy`.
- Standalone/additive: runs against tiles that have already been through the main LiDAR
  preprocessing pipeline; does not require re-running `preprocess_flair3d_v2.py`.

### 3. Data loading & transform (generalize existing network-specific code, not duplicate)

- `Flair3DDataset._load_network_label` (`flair3d.py`) gains a `target_key` parameter
  (default `"network"`, preserving current behavior exactly) and is called once per
  pixel_semantic task present for a given sample. Add `"forest_2d"` to
  `FLAIR3D_SPECIFIC_ASSETS`.
- `NetworkRasterToPointLabels` (`transform.py`) gains a `target_key: str = "network"`
  constructor argument; every hardcoded `"network"`-prefixed field read/write inside becomes
  `f"{target_key}..."`. Class name and registration are kept as-is (renaming would require
  touching the 14 existing configs that reference `dict(type="NetworkRasterToPointLabels")`
  literally, for no functional benefit — the default parameter preserves their behavior
  unchanged). New `forest_2d` pipelines use
  `dict(type="NetworkRasterToPointLabels", target_key="forest_2d")`.
- Pipeline shape for `forest_2d` mirrors `network`'s exactly, in train/val/test alike:
  `ExtractAbsXY` → (crop/augment) → `NetworkRasterToPointLabels(target_key="forest_2d")` →
  `ToTensor` → `Collect`. No special-casing at test time — same `MultiTaskTester`, no
  APLS-style stitching/export step.

### 4. Model (`pointcept/models/default.py`, `MultiTaskSegmentorV2`)

- `_pixel_pool_and_gather` / `_compute_pixel_logits` currently read
  `input_dict["network_cell"]` / `input_dict["network_pix"]` /
  `input_dict.get("network_height")` / `input_dict.get("network_width")` unconditionally,
  regardless of which pixel_semantic task is being processed. Generalize to
  `input_dict[f"{task_name}_cell"]` etc.
- Add configurable pooling: `torch_scatter.scatter_mean` when
  `task_config.get("pooling", "max") == "mean"`, else the current `scatter_max` behavior
  (default preserved for `network` and any task that doesn't set `pooling`).
- Head construction (`nn.Linear(backbone_out_channels, num_classes * num_networks)`), loss
  wiring (CE + Lovasz per channel), and dense-grid reconstruction for eval logits are already
  fully generic per pixel_semantic task — no changes needed there.

### 5. Evaluation metrics (`pointcept/engines/hooks/evaluator.py`)

Per pixel_semantic task, today: plain IoU/mIoU/Acc/allAcc (from the confusion histogram,
generic, unconditional), exact precision/recall/F1 (derived from the same histogram, generic,
unconditional), and dilated ("relaxed") precision/recall/F1 (separate accumulator, currently
unconditional for every pixel_semantic task). Add a per-task opt-out:

- In the val-loop accumulation block (~evaluator.py:1041-1075): skip the
  `dilated_precision_recall_counts` computation and the `val_dilated_prf/...` storage writes
  entirely when `task_config.get("enable_dilated_prf", True)` is `False`.
- In the sync/log block (~evaluator.py:1298-1361): skip `local_dilated_prf_totals` /
  `sync_dilated_prf_totals` and the dilated fields/log line under the same flag; keep exact
  precision/recall/F1 and plain IoU/Acc unconditionally (those are cheap, already-generic,
  and meaningful for `forest_2d`).
- This flag is a static, config-derived value — identical on every rank — so skipping the
  `sync_dilated_prf_totals` all_reduce call for a disabled task is safe (no distributed
  deadlock risk).
- `network`'s entry does not set `enable_dilated_prf`, so it keeps computing dilated P/R/F1
  exactly as today.

No APLS-equivalent script is added for `forest_2d` — plain IoU/Acc/exact-P-R-F1 is the full
metric surface for this task.

### 5b. Test-set precision/recall/F1 (new — extends `MultiTaskTester`, `pointcept/engines/test.py`)

**Gap found during design review**: everything in section 5 above is the *online validation*
evaluator (`MultiTaskEvaluator`, runs periodically during training on the `val` split). The
final "precise test" pass (`PreciseEvaluator` → `MultiTaskTester.test()`, run once at the end
of training on the `test` split, see `pointcept/engines/hooks/misc.py:745`) currently computes
**no aggregate metric at all** for any `pixel_semantic` task — it only merges per-tile
test-time fragments into a dense probability raster and saves it to disk
(`{patch_id}_logits_{task_name}.npy`, `test.py:1269-1352`). The final aggregation loop
(`test.py:~1585-1690`) only handles `semantic_tasks` / `regression_tasks` /
`multilabel_tasks` / `tile_distribution_tasks`. For `network`, the only consumer of those
saved rasters is the opt-in `NetworkAPLSEvaluator` (graph-based APLS, not P/R/F1). This means
today there is **no test-set number to cite** for any pixel_semantic task, forest_2d
included — only training-time validation numbers exist.

Add a generic (applies to every `pixel_semantic` task, so `network` gets it too, purely
additive to its existing APLS metric) precision/recall/F1 computation, foreground-class-only
("class 1", i.e. `names.index("Foreground")`, matching how the existing exact/dilated P-R-F1
already restrict to `fg_idx`):

**Key simplification found while re-reading `test.py`**: the dense whole-tile GT raster for a
pixel_semantic task is *already* sitting in `targets_by_task[task_name]` at full `(r, H, W)`
resolution — no fragment-merging or model dependency needed to get it. Proof:
`Flair3DDataset.prepare_test_data` (`flair3d.py:551-599`) calls `self.get_data(idx)` (which
already produces the channel-selected dense raster, `flair3d.py:396`) *before* running any
per-fragment transform, then explicitly does, for every pixel_semantic target key
(`flair3d.py:564-567`):
```python
if key in FLAIR3D_PIXEL_SEMANTIC_TARGETS:
    # Keep raster in data_dict so fragments still carry it for the
    # pixel_semantic head; also expose GT at scene level for metrics.
    result_dict[key] = deepcopy(data_dict[key])
```
`NetworkRasterToPointLabels` (which turns the raster into per-point labels) only runs later,
per-fragment, inside `post_transform` (`flair3d.py:596-597`) — so this `result_dict[key]` copy
is untouched, full-resolution, dense. In `test.py`, `targets_by_task = batch_targets[b_idx]`
(`test.py:1015-1041`, `1393`) is exactly this `result_dict`, so `targets_by_task[task_name]` is
the dense `(r, H, W)` GT array, shape- and grid-aligned with the merged prediction raster
(`pixel_logits_np[task_name]`, also `(r, H, W)`, `test.py:1327-1352`) because both ultimately
derive from the same on-disk raster/meta. This holds **regardless of the npy-cache fast path**
(`test.py:1139-1162`) — `batch_pixel_logits_np` is populated identically whether predictions
come from a fresh forward pass or from cache, and `targets_by_task` never depends on the model
at all. (An earlier draft of this section proposed fetching and fragment-merging
`output_dict["pixel_seg_target_dense_by_task"]` from the model — that approach is strictly
worse: more code, and silently broken under the cache-hit fast path where the model never
runs. Discarded in favor of the above.)

- Per scene, in the existing "compute metrics for each scene" loop (`test.py:1391+`, right
  where `sem_metrics_scene` is built): pull `pixel_logits_np = batch_pixel_logits_np[b_idx]`
  (not currently destructured there — `pred_cls_np`/`pred_reg_np` are, `pixel_logits_np` isn't
  yet). For each pixel_semantic task and each of its `num_networks` channels `c`:
  - `prob = pixel_logits_np[task_name][c]` (dense probs, NaN = no point in any test fragment
    observed that cell)
  - `target = targets_by_task[task_name][c]` (dense GT label ids, straight from disk)
  - `valid = isfinite(prob) & (target != ignore_index)` (excludes cells nobody observed, and
    Void cells)
  - `pred_fg = (prob > 0.5) & valid`, `gt_fg = (target == fg_idx) & valid`
  - accumulate scene-level `tp / fp / fn` (plain ints)
  - add to the existing per-scene record: `record[data_name] = dict(semantic=...,
    tile_distribution=..., pixel_semantic=pixel_prf_metrics_scene)` (extends the dict built at
    `test.py:1549`).
- This rides along the existing `comm.gather(record, dst=0)` (`test.py:1585`) — no new
  distributed-sync code needed. On rank 0, sum `tp/fp/fn` across every scene in the test split,
  mirroring how semantic-task confusion histograms are already summed in that same merge loop
  (`test.py:1594-1608`).
- Compute global `precision, recall, f1 = precision_recall_f1(tp, tp + fp, tp, tp + fn)`
  (reuse `pointcept.utils.dilated_metrics.precision_recall_f1`, already used for exactly this
  math in `evaluator.py`) per pixel_semantic task/channel; log via `logger.info` plus
  `log_dict[metric_tag("test", "precision"/"recall"/"f1", task=task_name)]`, the same
  convention already used for `mIoU`/`macro_f1`/etc. (`test.py:1667-1684`), so it surfaces in
  wandb/tensorboard exactly like every other final test metric.
- For `forest_2d` (`num_networks=1`) this yields exactly one precision/recall/F1 triplet for
  the Forest class. `network` (`num_networks=2`) gets one triplet per channel (ROADS,
  RAILROADS), purely additive — nothing existing changes for it.
- No `default.py` changes needed for this section — `pixel_seg_target_dense_by_task` remains
  unused for this feature (only the online-validation dilated-P/R computation, which
  `forest_2d` disables, still uses it).

**Precise answer to "which pixels":** the test-set P/R/F1 population is *every dense-grid
pixel, across every tile in the test split, observed by at least one point in at least one
test-time fragment (`prob` finite), whose GT label is not Void*. This is the same "occupied,
non-void cell" restriction as validation (section 5) — the only difference is that validation
computes it on a flat per-crop pooled tensor summed over the `val` split, while this computes
it on the fragment-merged, whole-tile dense grid (prediction side only — GT is always
fully dense) summed over the `test` split.

### 6. Config scope

Only the task-registry entry is added now (`FLAIR3D_PIXEL_SEMANTIC_TASKS["forest_2d"]`).
No new mono-task config directory (`configs/flair3d_default/forest_2d/`) is created in this
pass — `forest_2d` is used directly inside existing/new multi-task experiment configs (e.g.
`configs/experiment/w<NN>/<day>/...`) by adding it to `target_keys`, following the existing
`configs/experiment/**` convention (standalone, no sibling inheritance).

### 7. Tests

Following the repo convention of one focused test file per piece of logic:
- A test for `NetworkRasterToPointLabels`'s `target_key` generalization: verify that
  `network` (default) and `forest_2d` (explicit `target_key`) each produce independent,
  non-colliding field names on the same `data_dict`, and that omitting `target_key` preserves
  today's exact output for a `"network"`-named raster.
- A test for the new `pooling="mean"` branch in `_pixel_pool_and_gather`: verify mean-pool
  output differs correctly from max-pool on a small synthetic per-pixel feature/label
  fixture, and that omitting `pooling` preserves today's max-pool behavior.
- A test for the section 5b test-set precision/recall/F1 aggregation: on synthetic per-scene
  dense prediction/target grids with known observed/unobserved/void cells, verify tp/fp/fn
  match a hand-computed expectation and that unobserved (`dense_cnt == 0`) and Void cells are
  correctly excluded.

## Open items resolved during design review

- Target key / asset name: `forest_2d` (confirmed).
- Grid resolution: `0.5m` (confirmed).
- Config scope: task-registry entry only, no mono-task config directory yet (confirmed).
- Test pipeline: identical structure to train/val, no APLS-equivalent (confirmed).
- Validation metrics: exact P/R/F1 + plain IoU/Acc only, no dilated P/R/F1 (confirmed).
- Test-set metrics: no aggregate metric currently exists at test time for any pixel_semantic
  task (gap found during design review) — add a generic test-set precision/recall/F1
  (foreground class only) to `MultiTaskTester`, applying to `forest_2d` and, additively, to
  `network` (confirmed, section 5b).
