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

## Open items resolved during design review

- Target key / asset name: `forest_2d` (confirmed).
- Grid resolution: `0.5m` (confirmed).
- Config scope: task-registry entry only, no mono-task config directory yet (confirmed).
- Test pipeline: identical structure to train/val, no APLS-equivalent (confirmed).
- Metrics: exact P/R/F1 + plain IoU/Acc only, no dilated P/R/F1 (confirmed).
