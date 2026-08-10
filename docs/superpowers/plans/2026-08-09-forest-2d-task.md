# Forest 2D Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `forest_2d` pixel-semantic task — mean-pooled 2D grid forest/non-forest
segmentation, at 0.5m resolution — as an independently-selectable alternative to the
existing per-point (3D) `forest` task, following the design in
`docs/superpowers/specs/2026-08-09-forest-2d-task-design.md`.

**Architecture:** `forest_2d` is registered exactly like the existing `network` task
(`task_type: "pixel_semantic"` in `FLAIR3D_PIXEL_SEMANTIC_TASKS`), but everything that was
previously hardcoded to the literal string `"network"` (the raster→point-label transform, the
dataset loader, the model's cell/pix lookup, the collect-key builder) is generalized to be
parametrized by task name, so `network` and `forest_2d` can coexist in one multi-task run.
`forest_2d`'s own raster (`forest_2d.npy`) is produced once by a new standalone preprocessing
script, mirroring `rasterize_network.py` but reading directly from the existing FOREST
GeoTIFF instead of a vector graph.

**Tech Stack:** Python, PyTorch, `torch_scatter`, `rasterio`, pytest/unittest (repo uses
`unittest.TestCase` classes run via `pytest`).

## Global Constraints

- `forest_2d`'s registry `pooling` field is `"mean"`; `network`'s is unset (defaults to
  `"max"`, unchanged behavior).
- `forest_2d`'s `enable_dilated_prf` field is `False`; `network`'s is unset (defaults to
  `True`, unchanged behavior).
- Grid resolution for `forest_2d` is `pixel_m = 0.5` everywhere (preprocessing script default
  and registry — the registry itself does not store `pixel_m`, it is baked into
  `forest_2d.npy`'s `meta.json` entry at preprocessing time, read back at train time exactly
  like `network`'s `pixel_m`).
- `forest_2d` is **not** added to `Flair3DDataset.FLAIR3D_OPTIONAL_TARGETS` — FOREST source
  coverage is complete for all tiles, so a missing `forest_2d.npy` must hard-fail, not
  silently substitute zeros.
- No changes to the existing per-point `forest` task, its preprocessing, or its config files.
- No APLS-style downstream graph evaluation is added for `forest_2d`.
- Every change to shared code (`transform.py`, `flair3d.py`, `default.py`,
  `flair3d_config_utils.py`, `evaluator.py`, `test.py`) must leave `network`'s behavior
  bit-for-bit unchanged when no `forest_2d`-style second task is present — verified by tests
  that exercise the default/`target_key="network"` path.

---

## Task 1: Register `forest_2d` and generalize multi-task Collect-key generation

**Files:**
- Modify: `pointcept/datasets/flair3d_config_utils.py:169-181` (registry), `:415-450`
  (`init_multitask_collect_keys`)
- Test: `tests/test_pixel_semantic_collect_keys.py`

**Interfaces:**
- Produces: `FLAIR3D_PIXEL_SEMANTIC_TASKS["forest_2d"]` dict; generalized
  `init_multitask_collect_keys(target_keys, *, collect_prefix_keys=())` that emits
  `{key}_cell`/`{key}_pix`/`{key}_origin_x`/`{key}_origin_y`/`{key}_pixel_m`/`{key}_height`/
  `{key}_width` for **every** pixel_semantic key present in `target_keys`, not just a
  hardcoded `network_*` set.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pixel_semantic_collect_keys.py`:

```python
"""
Tests for FLAIR3D_PIXEL_SEMANTIC_TASKS registration of forest_2d and the
per-task-name generalization of init_multitask_collect_keys (previously
hardcoded to the literal "network_*" key set).

Run with: PYTHONPATH=./ pytest tests/test_pixel_semantic_collect_keys.py
"""

import unittest

from pointcept.datasets.flair3d_config_utils import (
    FLAIR3D_PIXEL_SEMANTIC_TASKS,
    get_pixel_semantic_config,
    init_multitask_collect_keys,
)


class TestForestTwoDRegistration(unittest.TestCase):
    def test_forest_2d_registered_with_expected_fields(self):
        cfg = get_pixel_semantic_config("forest_2d")
        self.assertEqual(cfg["task_type"], "pixel_semantic")
        self.assertEqual(cfg["num_networks"], 1)
        self.assertEqual(cfg["num_classes"], 2)
        self.assertEqual(cfg["ignore_index"], 2)
        self.assertEqual(cfg["channel_names"], ["FOREST"])
        self.assertEqual(cfg["names"], ["Not Forest", "Forest", "Void"])
        self.assertEqual(cfg["pooling"], "mean")
        self.assertFalse(cfg["enable_dilated_prf"])

    def test_get_pixel_semantic_config_returns_independent_copies(self):
        a = get_pixel_semantic_config("forest_2d")
        b = get_pixel_semantic_config("forest_2d")
        a["num_classes"] = 999
        self.assertEqual(b["num_classes"], 2)

    def test_network_defaults_unchanged(self):
        cfg = FLAIR3D_PIXEL_SEMANTIC_TASKS["network"]
        self.assertNotIn("pooling", cfg)
        self.assertNotIn("enable_dilated_prf", cfg)


class TestInitMultitaskCollectKeys(unittest.TestCase):
    def test_network_only_matches_historical_key_set(self):
        train_keys, val_keys, _ = init_multitask_collect_keys(("network",))
        expected_extra = (
            "network_cell",
            "network_pix",
            "network_origin_x",
            "network_origin_y",
            "network_pixel_m",
            "network_height",
            "network_width",
        )
        for key in expected_extra:
            self.assertIn(key, train_keys)
            self.assertIn(key, val_keys)
        self.assertIn("network", train_keys)

    def test_forest_2d_only_gets_its_own_keys_not_network(self):
        train_keys, val_keys, _ = init_multitask_collect_keys(("forest_2d",))
        self.assertIn("forest_2d_cell", train_keys)
        self.assertIn("forest_2d_pix", train_keys)
        self.assertIn("forest_2d_origin_x", train_keys)
        self.assertIn("forest_2d_origin_y", train_keys)
        self.assertIn("forest_2d_pixel_m", train_keys)
        self.assertIn("forest_2d_height", train_keys)
        self.assertIn("forest_2d_width", train_keys)
        self.assertNotIn("network_cell", train_keys)
        self.assertNotIn("network_pix", train_keys)

    def test_network_and_forest_2d_coexist_without_collision(self):
        train_keys, val_keys, _ = init_multitask_collect_keys(("network", "forest_2d"))
        for prefix in ("network", "forest_2d"):
            for suffix in (
                "cell",
                "pix",
                "origin_x",
                "origin_y",
                "pixel_m",
                "height",
                "width",
            ):
                self.assertIn(f"{prefix}_{suffix}", train_keys)
        # Both raw target names present, each exactly once.
        self.assertEqual(train_keys.count("network"), 1)
        self.assertEqual(train_keys.count("forest_2d"), 1)

    def test_forest_2d_is_scene_level_in_val_keys(self):
        # pixel_semantic targets are "scene-level": val Collect keys must NOT
        # include an origin_forest_2d entry (unlike point-wise targets).
        _, val_keys, _ = init_multitask_collect_keys(("forest_2d",))
        self.assertIn("forest_2d", val_keys)
        self.assertNotIn("origin_forest_2d", val_keys)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./ pytest tests/test_pixel_semantic_collect_keys.py -v`
Expected: FAIL — `forest_2d` is not a known target_key yet (`KeyError` from
`get_pixel_semantic_config`), and the collect-key generalization tests fail because
`init_multitask_collect_keys` currently hardcodes `network_*` regardless of `target_keys`.

- [ ] **Step 3: Add the `forest_2d` registry entry**

In `pointcept/datasets/flair3d_config_utils.py`, replace the `FLAIR3D_PIXEL_SEMANTIC_TASKS`
dict (currently lines 169-181):

```python
FLAIR3D_PIXEL_SEMANTIC_TASKS: Dict[str, Dict[str, Any]] = {
    "network": {
        "task_type": "pixel_semantic",
        "num_networks": 2,
        "num_classes": 2,
        "ignore_index": 2,
        "channel_names": list(NETWORK_CHANNEL_NAMES),
        "names": ["Background", "Foreground", "Void"],
        # Buffer ("relaxed") precision/recall/F1 tolerance, in pixels (1 px = 1 m on
        # this task's Lambert grid). See MultiTaskEvaluator's pixel_semantic handling.
        "buffer_radius_px": 3, # for dilated recall/precision/f1
    },
    "forest_2d": {
        "task_type": "pixel_semantic",
        "num_networks": 1,
        "num_classes": 2,
        "ignore_index": 2,
        "channel_names": ["FOREST"],
        "names": ["Not Forest", "Forest", "Void"],
        # Area-coverage class: mean-pool point features per cell (unlike network's
        # default max-pool, which suits catching a single strong "line here" signal).
        "pooling": "mean",
        # Buffer/dilated tolerance is a diagnostic for thin curvilinear masks
        # (road/rail); not meaningful for a blobby area-coverage mask like forest.
        "enable_dilated_prf": False,
    },
}
```

- [ ] **Step 4: Generalize `init_multitask_collect_keys`**

Replace the function (currently lines 415-450):

```python
def init_multitask_collect_keys(
    target_keys: Tuple[str, ...],
    *,
    collect_prefix_keys: Tuple[str, ...] = (),
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Build train/val Collect keys and index_valid_keys for Flair3D+ configs.

    train: coord + collect_prefix_keys + target_keys (+ per pixel_semantic task's cell
    ints / grid meta)
    val: coord + collect_prefix_keys + (task, origin_task) per point-wise target + inverse

    For every pixel_semantic target in ``target_keys`` (e.g. ``network``, ``forest_2d``),
    Collect includes ``{key}_cell`` / ``{key}_pix`` (int64 Lambert indices from
    ``NetworkRasterToPointLabels(target_key=key)``), plus ``{key}_origin_*``,
    ``{key}_pixel_m``, and ``{key}_height`` / ``{key}_width`` for dense test. Each
    pixel_semantic task gets its own independent key set, so e.g. ``network`` and
    ``forest_2d`` can coexist in the same multi-task run without collisions.
    """
    _validate_target_keys(target_keys)
    base = ("coord",) + collect_prefix_keys
    extra: Tuple[str, ...] = ()
    for key in target_keys:
        if _is_pixel_semantic_target(key):
            extra += (
                f"{key}_cell",
                f"{key}_pix",
                f"{key}_origin_x",
                f"{key}_origin_y",
                f"{key}_pixel_m",
                f"{key}_height",
                f"{key}_width",
            )
    train_keys = base + extra + target_keys
    val_target_keys: Tuple[str, ...] = ()
    for key in target_keys:
        if _is_scene_level_target(key):
            val_target_keys += (key,)
        else:
            val_target_keys += (key, f"origin_{key}")
    val_keys = base + extra + val_target_keys + ("inverse",)
    return train_keys, val_keys, FLAIR3D_MULTITASK_INDEX_VALID_KEYS
```

(Only the body of the `if`/`extra` construction changed — it now loops over every
pixel_semantic key in `target_keys` instead of checking `any(...)` and hardcoding `network_*`
once. For `target_keys=("network",)` this produces the exact same `extra` tuple as before.)

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=./ pytest tests/test_pixel_semantic_collect_keys.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pointcept/datasets/flair3d_config_utils.py tests/test_pixel_semantic_collect_keys.py
git commit -m "Register forest_2d pixel-semantic task and generalize collect-key generation

FLAIR3D_PIXEL_SEMANTIC_TASKS gains a forest_2d entry (mean-pool,
dilated P/R/F1 disabled). init_multitask_collect_keys now emits
per-task-name cell/pix/grid-meta Collect keys instead of a hardcoded
network_* set, so multiple pixel_semantic tasks can coexist."
```

---

## Task 2: Add a per-task opt-out for dilated ("relaxed") precision/recall/F1

**Files:**
- Modify: `pointcept/utils/dilated_metrics.py` (add `dilated_prf_enabled`)
- Modify: `pointcept/engines/hooks/evaluator.py:47-50` (import), `:1041-1076` (val-loop
  accumulation), `:1296-1362` (sync/log), `:1479-1499` (per-channel metric tag logging)
- Test: `tests/test_dilated_prf_opt_out.py`

**Interfaces:**
- Produces: `dilated_prf_enabled(task_config: dict) -> bool` in
  `pointcept.utils.dilated_metrics`, `True` unless the task config sets
  `enable_dilated_prf=False`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dilated_prf_opt_out.py`:

```python
"""
Test for dilated_prf_enabled, the per-task opt-out used by MultiTaskEvaluator to
skip buffer ("relaxed") precision/recall/F1 for area-coverage pixel_semantic tasks
(e.g. forest_2d) while keeping it for thin curvilinear ones (e.g. network).

Run with: PYTHONPATH=./ pytest tests/test_dilated_prf_opt_out.py
"""

import unittest

from pointcept.utils.dilated_metrics import dilated_prf_enabled


class TestDilatedPrfEnabled(unittest.TestCase):
    def test_defaults_to_true_when_key_absent(self):
        self.assertTrue(dilated_prf_enabled({}))
        self.assertTrue(dilated_prf_enabled({"task_type": "pixel_semantic"}))

    def test_false_when_explicitly_disabled(self):
        self.assertFalse(dilated_prf_enabled({"enable_dilated_prf": False}))

    def test_true_when_explicitly_enabled(self):
        self.assertTrue(dilated_prf_enabled({"enable_dilated_prf": True}))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./ pytest tests/test_dilated_prf_opt_out.py -v`
Expected: FAIL with `ImportError: cannot import name 'dilated_prf_enabled'`

- [ ] **Step 3: Add `dilated_prf_enabled` to `pointcept/utils/dilated_metrics.py`**

Append at the end of the file:

```python


def dilated_prf_enabled(task_config) -> bool:
    """Whether a pixel_semantic task's config wants dilated ("relaxed") P/R/F1.

    Defaults to True (existing behavior for tasks like ``network`` that don't set
    this key). Area-coverage tasks like ``forest_2d`` set ``enable_dilated_prf=False``
    since the buffer tolerance this module implements is a diagnostic for thin
    curvilinear masks (road/rail), not blobby area-coverage masks.
    """
    return bool(task_config.get("enable_dilated_prf", True))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=./ pytest tests/test_dilated_prf_opt_out.py -v`
Expected: PASS

- [ ] **Step 5: Wire the opt-out into `MultiTaskEvaluator` (val-loop accumulation)**

In `pointcept/engines/hooks/evaluator.py`, add `dilated_prf_enabled` to the existing import
(currently lines 47-50):

```python
from pointcept.utils.dilated_metrics import (
    dilated_precision_recall_counts,
    dilated_prf_enabled,
    precision_recall_f1,
)
```

Then, in the val-loop accumulation block (currently lines 1041-1075), replace:

```python
                    # ---- Buffer ("relaxed") precision/recall hit accumulation ----
                    # Uses the dense per-scene grids (reconstructed on the GT Lambert
                    # grid) so the tolerance dilation has real 2D neighborhoods to work
                    # with; the (P, r) pooled arrays above have no spatial layout.
                    dense_pred_list = pixel_dense_by_task.get(task_name) or []
                    dense_gt_list = pixel_dense_target_by_task.get(task_name) or []
                    radius_px = int(task_config.get("buffer_radius_px", 3))
                    for dense_pred, dense_gt in zip(dense_pred_list, dense_gt_list):
                        if dense_pred is None or dense_gt is None:
                            continue
                        dense_pred_np = dense_pred.detach().cpu().numpy()
                        dense_gt_np = dense_gt.detach().cpu().numpy()
                        for c in range(num_networks):
                            ch_name = (
                                channel_names[c] if c < len(channel_names) else f"ch{c}"
                            )
                            pred_prob = dense_pred_np[c]
                            gt = dense_gt_np[c]
                            # Unobserved cells: pred_prob is NaN, gt is -1 (see
                            # models/default.py _compute_pixel_logits). Void GT cells
                            # (ignore_index) are excluded like plain IoU excludes them.
                            observed = ~np.isnan(pred_prob) & (gt != -1)
                            valid = observed & (gt != ignore_index)
                            pred_fg = (pred_prob > 0.5) & valid
                            gt_fg = (gt == 1) & valid
                            p_num, p_denom, r_num, r_denom = (
                                dilated_precision_recall_counts(
                                    pred_fg, gt_fg, valid, radius_px=radius_px
                                )
                            )
                            base = f"val_dilated_prf/{task_name}/{ch_name}"
                            self.trainer.storage.put_scalar(f"{base}/p_num", p_num)
                            self.trainer.storage.put_scalar(f"{base}/p_denom", p_denom)
                            self.trainer.storage.put_scalar(f"{base}/r_num", r_num)
                            self.trainer.storage.put_scalar(f"{base}/r_denom", r_denom)
```

with:

```python
                    # ---- Buffer ("relaxed") precision/recall hit accumulation ----
                    # Uses the dense per-scene grids (reconstructed on the GT Lambert
                    # grid) so the tolerance dilation has real 2D neighborhoods to work
                    # with; the (P, r) pooled arrays above have no spatial layout.
                    # Skipped entirely for tasks that opt out (e.g. forest_2d): the
                    # buffer tolerance is a diagnostic for thin curvilinear masks
                    # (road/rail), not meaningful for a blobby area-coverage mask.
                    if dilated_prf_enabled(task_config):
                        dense_pred_list = pixel_dense_by_task.get(task_name) or []
                        dense_gt_list = pixel_dense_target_by_task.get(task_name) or []
                        radius_px = int(task_config.get("buffer_radius_px", 3))
                        for dense_pred, dense_gt in zip(dense_pred_list, dense_gt_list):
                            if dense_pred is None or dense_gt is None:
                                continue
                            dense_pred_np = dense_pred.detach().cpu().numpy()
                            dense_gt_np = dense_gt.detach().cpu().numpy()
                            for c in range(num_networks):
                                ch_name = (
                                    channel_names[c] if c < len(channel_names) else f"ch{c}"
                                )
                                pred_prob = dense_pred_np[c]
                                gt = dense_gt_np[c]
                                # Unobserved cells: pred_prob is NaN, gt is -1 (see
                                # models/default.py _compute_pixel_logits). Void GT cells
                                # (ignore_index) are excluded like plain IoU excludes them.
                                observed = ~np.isnan(pred_prob) & (gt != -1)
                                valid = observed & (gt != ignore_index)
                                pred_fg = (pred_prob > 0.5) & valid
                                gt_fg = (gt == 1) & valid
                                p_num, p_denom, r_num, r_denom = (
                                    dilated_precision_recall_counts(
                                        pred_fg, gt_fg, valid, radius_px=radius_px
                                    )
                                )
                                base = f"val_dilated_prf/{task_name}/{ch_name}"
                                self.trainer.storage.put_scalar(f"{base}/p_num", p_num)
                                self.trainer.storage.put_scalar(f"{base}/p_denom", p_denom)
                                self.trainer.storage.put_scalar(f"{base}/r_num", r_num)
                                self.trainer.storage.put_scalar(f"{base}/r_denom", r_denom)
```

- [ ] **Step 6: Wire the opt-out into the sync/log block, and fix the downstream metric-tag
  loop that would otherwise crash on a missing dilated value**

In the sync/log block (currently lines 1296-1362), replace:

```python
                p_num, p_denom, r_num, r_denom = local_dilated_prf_totals(
                    self.trainer.storage, task_name, ch_name
                )
                p_num, p_denom, r_num, r_denom = sync_dilated_prf_totals(
                    p_num, p_denom, r_num, r_denom
                )
                dilated_precision, dilated_recall, dilated_f1 = precision_recall_f1(
                    p_num, p_denom, r_num, r_denom
                )

                per_task_channel_metrics[task_name][ch_name] = dict(
                    precision=exact_precision,
                    recall=exact_recall,
                    f1=exact_f1,
                    dilated_precision=dilated_precision,
                    dilated_recall=dilated_recall,
                    dilated_f1=dilated_f1,
                )
                if comm.is_main_process():
                    self.trainer.logger.info(
                        "[task={}] Channel_{}-{} Result: precision/recall/f1 "
                        "{:.4f}/{:.4f}/{:.4f} | dilated(r={}px) precision/recall/f1 "
                        "{:.4f}/{:.4f}/{:.4f}".format(
                            task_name,
                            c,
                            ch_name,
                            exact_precision,
                            exact_recall,
                            exact_f1,
                            radius_px,
                            dilated_precision,
                            dilated_recall,
                            dilated_f1,
                        )
                    )
```

with:

```python
                dilated_enabled = dilated_prf_enabled(task_config)
                channel_metrics = dict(
                    precision=exact_precision,
                    recall=exact_recall,
                    f1=exact_f1,
                )
                if dilated_enabled:
                    p_num, p_denom, r_num, r_denom = local_dilated_prf_totals(
                        self.trainer.storage, task_name, ch_name
                    )
                    p_num, p_denom, r_num, r_denom = sync_dilated_prf_totals(
                        p_num, p_denom, r_num, r_denom
                    )
                    dilated_precision, dilated_recall, dilated_f1 = precision_recall_f1(
                        p_num, p_denom, r_num, r_denom
                    )
                    channel_metrics.update(
                        dilated_precision=dilated_precision,
                        dilated_recall=dilated_recall,
                        dilated_f1=dilated_f1,
                    )
                per_task_channel_metrics[task_name][ch_name] = channel_metrics
                if comm.is_main_process():
                    if dilated_enabled:
                        self.trainer.logger.info(
                            "[task={}] Channel_{}-{} Result: precision/recall/f1 "
                            "{:.4f}/{:.4f}/{:.4f} | dilated(r={}px) precision/recall/f1 "
                            "{:.4f}/{:.4f}/{:.4f}".format(
                                task_name,
                                c,
                                ch_name,
                                exact_precision,
                                exact_recall,
                                exact_f1,
                                radius_px,
                                dilated_precision,
                                dilated_recall,
                                dilated_f1,
                            )
                        )
                    else:
                        self.trainer.logger.info(
                            "[task={}] Channel_{}-{} Result: precision/recall/f1 "
                            "{:.4f}/{:.4f}/{:.4f}".format(
                                task_name,
                                c,
                                ch_name,
                                exact_precision,
                                exact_recall,
                                exact_f1,
                            )
                        )
```

**Why Step 6 also touches the downstream tag-writing loop:** `per_task_channel_metrics` used
to always contain `dilated_precision`/`dilated_recall`/`dilated_f1` keys; a later loop
(currently lines 1479-1499) does `for metric_name in ("precision", "recall", "f1",
"dilated_precision", "dilated_recall", "dilated_f1"): value = float(metric[metric_name])` —
this would raise `KeyError` (or previously, when a `None` sentinel was used instead, `TypeError:
float() argument must be a string or a number, not 'NoneType'`) once a task's dict stops
carrying those keys. Fix it to iterate over whatever keys are actually present. Replace
(currently lines 1479-1499):

```python
                for task_name, channel_metrics in per_task_channel_metrics.items():
                    for ch_name, metric in channel_metrics.items():
                        ch_slug = class_name_slug(ch_name)
                        for metric_name in (
                            "precision",
                            "recall",
                            "f1",
                            "dilated_precision",
                            "dilated_recall",
                            "dilated_f1",
                        ):
                            tag = metric_tag(
                                "val", f"{ch_slug}/{metric_name}", task=task_name
                            )
                            value = float(metric[metric_name])
                            if self.trainer.writer is not None:
                                self.trainer.writer.add_scalar(
                                    tag, value, current_epoch
                                )
                            if wandb_log is not None:
                                wandb_log[tag] = value
```

with:

```python
                for task_name, channel_metrics in per_task_channel_metrics.items():
                    for ch_name, metric in channel_metrics.items():
                        ch_slug = class_name_slug(ch_name)
                        for metric_name, value in metric.items():
                            tag = metric_tag(
                                "val", f"{ch_slug}/{metric_name}", task=task_name
                            )
                            value = float(value)
                            if self.trainer.writer is not None:
                                self.trainer.writer.add_scalar(
                                    tag, value, current_epoch
                                )
                            if wandb_log is not None:
                                wandb_log[tag] = value
```

- [ ] **Step 7: Run the full test suite for this area to check nothing broke**

Run: `PYTHONPATH=./ pytest tests/test_dilated_prf_opt_out.py tests/test_pixel_semantic_collect_keys.py -v`
Expected: PASS (this task doesn't add new evaluator-level tests — `MultiTaskEvaluator.eval()`
is only exercisable through a real training run with a trainer/storage double, which is out of
scope here; correctness of the two edited blocks is covered by (a) the `dilated_prf_enabled`
unit test, and (b) manual reasoning: the diffs only wrap existing, previously-unconditional
logic in an `if`, and the exact-P/R/F1 path — which is what `forest_2d` uses — is provably
untouched by the diff).

- [ ] **Step 8: Commit**

```bash
git add pointcept/utils/dilated_metrics.py pointcept/engines/hooks/evaluator.py tests/test_dilated_prf_opt_out.py
git commit -m "Add per-task opt-out for dilated precision/recall/F1

forest_2d disables the buffer-tolerance metric (meant for thin
curvilinear masks like roads, not area-coverage classes) via
enable_dilated_prf=False. network's behavior is unchanged."
```

---

## Task 3: Generalize `NetworkRasterToPointLabels` with a `target_key` parameter

**Files:**
- Modify: `pointcept/datasets/transform.py:437-529`
- Test: `tests/test_pixel_semantic_raster_to_points.py`

**Interfaces:**
- Produces: `NetworkRasterToPointLabels(target_key="network", ignore_index=2,
  keep_grid_meta=True)` — same registered transform name, new `target_key` constructor arg
  (default preserves current behavior exactly).
- Consumes: `data_dict[target_key]` (dense `(r, H, W)` raster), `data_dict["abs_xy"]` (from
  `ExtractAbsXY`), `data_dict[f"{target_key}_origin_x"]` / `_origin_y` / `_pixel_m`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pixel_semantic_raster_to_points.py`:

```python
"""
Tests for NetworkRasterToPointLabels's target_key generalization: the transform
used to be hardcoded to the literal "network" field name; it must now support an
arbitrary pixel_semantic target_key (e.g. "forest_2d") while preserving its exact
historical behavior when called without target_key (defaults to "network").

Also verifies the multi-task ordering requirement: since a real multi-task pipeline
runs this transform once per pixel_semantic task on the *same* data_dict, the first
call must not consume "abs_xy" so the second call can still use it.

Run with: PYTHONPATH=./ pytest tests/test_pixel_semantic_raster_to_points.py
"""

import unittest

import numpy as np

from pointcept.datasets.transform import NetworkRasterToPointLabels


def _make_data_dict(raster_key, raster, origin_x, origin_y, pixel_m, abs_xy):
    return {
        raster_key: raster,
        f"{raster_key}_origin_x": np.asarray([origin_x], dtype=np.float64),
        f"{raster_key}_origin_y": np.asarray([origin_y], dtype=np.float64),
        f"{raster_key}_pixel_m": np.asarray([pixel_m], dtype=np.float64),
        "abs_xy": abs_xy,
        "coord": np.zeros((abs_xy.shape[0], 3), dtype=np.float32),
        "index_valid_keys": ["coord", "abs_xy"],
    }


class TestDefaultTargetKeyIsNetwork(unittest.TestCase):
    def test_default_target_key_behaves_like_historical_network(self):
        # 2x2 grid, origin (0, 0), pixel_m=1. Points sit in each of the 4 cells.
        raster = np.array([[[0, 1], [1, 0]]], dtype=np.uint8)  # (r=1, H=2, W=2)
        abs_xy = np.array(
            [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]], dtype=np.float64
        )
        data_dict = _make_data_dict("network", raster, 0.0, 0.0, 1.0, abs_xy)

        out = NetworkRasterToPointLabels()(data_dict)

        np.testing.assert_array_equal(out["network"], [[0], [1], [1], [0]])
        np.testing.assert_array_equal(out["network_cell"], [[0, 0], [0, 1], [1, 0], [1, 1]])
        np.testing.assert_array_equal(out["network_pix"], [[0, 0], [0, 1], [1, 0], [1, 1]])
        np.testing.assert_array_equal(out["network_height"], [2])
        np.testing.assert_array_equal(out["network_width"], [2])
        self.assertIn("network", out["index_valid_keys"])
        self.assertIn("network_cell", out["index_valid_keys"])
        self.assertIn("network_pix", out["index_valid_keys"])


class TestExplicitTargetKey(unittest.TestCase):
    def test_forest_2d_target_key_produces_independent_fields(self):
        raster = np.array([[[1, 0], [0, 1]]], dtype=np.uint8)
        abs_xy = np.array([[0.25, 0.25], [1.25, 1.25]], dtype=np.float64)
        data_dict = _make_data_dict("forest_2d", raster, 0.0, 0.0, 0.5, abs_xy)

        out = NetworkRasterToPointLabels(target_key="forest_2d")(data_dict)

        np.testing.assert_array_equal(out["forest_2d"], [[1], [1]])
        self.assertNotIn("network_cell", out)
        self.assertNotIn("network_pix", out)
        self.assertIn("forest_2d_cell", out)
        self.assertIn("forest_2d_pix", out)


class TestTwoPixelSemanticTasksInSequence(unittest.TestCase):
    def test_network_then_forest_2d_both_succeed_on_same_data_dict(self):
        # Regression test: the original implementation popped "abs_xy" after use,
        # which would silently no-op the *second* NetworkRasterToPointLabels call
        # in a multi-task pipeline (network then forest_2d), leaving forest_2d's
        # raster un-converted (still dense) all the way to Collect.
        abs_xy = np.array([[0.5, 0.5]], dtype=np.float64)
        network_raster = np.array([[[1]]], dtype=np.uint8)  # (1, 1, 1)
        forest_raster = np.array([[[0]]], dtype=np.uint8)  # (1, 1, 1)
        data_dict = {
            "network": network_raster,
            "network_origin_x": np.asarray([0.0]),
            "network_origin_y": np.asarray([0.0]),
            "network_pixel_m": np.asarray([1.0]),
            "forest_2d": forest_raster,
            "forest_2d_origin_x": np.asarray([0.0]),
            "forest_2d_origin_y": np.asarray([0.0]),
            "forest_2d_pixel_m": np.asarray([1.0]),
            "abs_xy": abs_xy,
            "coord": np.zeros((1, 3), dtype=np.float32),
            "index_valid_keys": ["coord", "abs_xy"],
        }

        data_dict = NetworkRasterToPointLabels(target_key="network")(data_dict)
        data_dict = NetworkRasterToPointLabels(target_key="forest_2d")(data_dict)

        np.testing.assert_array_equal(data_dict["network"], [[1]])
        np.testing.assert_array_equal(data_dict["forest_2d"], [[0]])
        self.assertIn("forest_2d_cell", data_dict)
        self.assertIn("forest_2d_pix", data_dict)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./ pytest tests/test_pixel_semantic_raster_to_points.py -v`
Expected: FAIL — `NetworkRasterToPointLabels()` doesn't accept `target_key`, and (once that's
added mechanically) the two-tasks-in-sequence test would still fail because of the `abs_xy`
pop.

- [ ] **Step 3: Rewrite `NetworkRasterToPointLabels`**

Replace the class (currently `pointcept/datasets/transform.py:437-529`):

```python
@TRANSFORMS.register_module()
class NetworkRasterToPointLabels(object):
    """Lookup binary pixel-semantic rasters onto points; emit int cell indices.

    Converts ``data_dict[target_key]`` from ``(r, H, W)`` to point-wise ``(N, r)`` so
    Mix3D carries labels with points. Also writes:

    - ``{target_key}_cell`` ``(N, 2)`` int64 — absolute Lambert cells ``(iy, ix)``
      from ``floor(abs_xy / pixel_m)`` (Mix3D-safe pooling keys)
    - ``{target_key}_pix`` ``(N, 2)`` int64 — relative ``(iy, ix)`` on the GT grid
      for dense test scatter

    Binning uses ``abs_xy`` float64 (from ``ExtractAbsXY``) and float64 origins read
    from ``{target_key}_origin_x`` / ``{target_key}_origin_y`` / ``{target_key}_pixel_m``.

    ``target_key`` defaults to ``"network"`` so existing configs that instantiate this
    transform without arguments (``dict(type="NetworkRasterToPointLabels")``) keep their
    exact current behavior. Pass ``target_key="forest_2d"`` (or any other registered
    pixel_semantic task name) to bin a different, independently-gridded raster, so e.g.
    ``network`` and ``forest_2d`` can run as two separate instances of this transform in
    the same pipeline without their cell/pix/grid-meta keys colliding.

    Run after GridSample / SphereCrop and before ToTensor / Collect. When two instances
    run on the same data_dict (one per pixel_semantic task), order between them does not
    matter.
    """

    def __init__(self, target_key="network", ignore_index=2, keep_grid_meta=True):
        self.target_key = str(target_key)
        self.ignore_index = int(ignore_index)
        self.keep_grid_meta = bool(keep_grid_meta)

    def __call__(self, data_dict):
        key = self.target_key
        if key not in data_dict or "abs_xy" not in data_dict:
            return data_dict
        raster = np.asarray(data_dict[key])
        if raster.ndim != 3:
            # Already point-wise (N, r) — leave unchanged.
            return data_dict

        abs_xy = np.asarray(data_dict["abs_xy"], dtype=np.float64)
        n = int(abs_xy.shape[0])
        if "coord" in data_dict and int(data_dict["coord"].shape[0]) != n:
            raise ValueError(
                f"abs_xy length {n} != coord length {data_dict['coord'].shape[0]}; "
                "ensure abs_xy is in index_valid_keys before GridSample "
                "(ExtractAbsXY registers it automatically)."
            )
        r, height, width = (
            int(raster.shape[0]),
            int(raster.shape[1]),
            int(raster.shape[2]),
        )

        origin_x = float(
            np.asarray(data_dict.get(f"{key}_origin_x", 0.0), dtype=np.float64).reshape(
                -1
            )[0]
        )
        origin_y = float(
            np.asarray(data_dict.get(f"{key}_origin_y", 0.0), dtype=np.float64).reshape(
                -1
            )[0]
        )
        pixel_m = float(
            np.asarray(data_dict.get(f"{key}_pixel_m", 1.0), dtype=np.float64).reshape(
                -1
            )[0]
        )
        step = max(pixel_m, 1e-6)

        ix_rel = np.floor((abs_xy[:, 0] - origin_x) / step).astype(np.int64)
        iy_rel = np.floor((abs_xy[:, 1] - origin_y) / step).astype(np.int64)
        ix_abs = np.floor(abs_xy[:, 0] / step).astype(np.int64)
        iy_abs = np.floor(abs_xy[:, 1] / step).astype(np.int64)
        in_grid = (ix_rel >= 0) & (iy_rel >= 0) & (ix_rel < width) & (iy_rel < height)

        labels = np.full((n, r), self.ignore_index, dtype=np.int64)
        if in_grid.any():
            labels[in_grid] = raster[:, iy_rel[in_grid], ix_rel[in_grid]].T.astype(
                np.int64
            )

        data_dict[key] = labels
        # (iy, ix) convention matches dense scatter dense[:, iy, ix].
        data_dict[f"{key}_cell"] = np.stack([iy_abs, ix_abs], axis=1)
        data_dict[f"{key}_pix"] = np.stack([iy_rel, ix_rel], axis=1)
        if self.keep_grid_meta:
            data_dict[f"{key}_height"] = np.asarray([height], dtype=np.int64)
            data_dict[f"{key}_width"] = np.asarray([width], dtype=np.int64)

        # NOTE: unlike the original single-task implementation, abs_xy is
        # intentionally *not* popped here. A multi-task pipeline runs this
        # transform once per pixel_semantic target (e.g. "network" then
        # "forest_2d" on the same data_dict) and each instance needs abs_xy to
        # still be present. Leaving it behind is harmless: Collect only gathers
        # explicitly listed keys, so an un-Collected abs_xy is simply discarded
        # with the rest of the per-sample dict.

        if "index_valid_keys" in data_dict:
            keys = [k for k in data_dict["index_valid_keys"] if k != "abs_xy"]
            for k in (key, f"{key}_cell", f"{key}_pix"):
                if k not in keys:
                    keys.append(k)
            data_dict["index_valid_keys"] = keys
        return data_dict
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=./ pytest tests/test_pixel_semantic_raster_to_points.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pointcept/datasets/transform.py tests/test_pixel_semantic_raster_to_points.py
git commit -m "Generalize NetworkRasterToPointLabels with a target_key parameter

Defaults to \"network\" (unchanged behavior for the 14 existing
configs). Also stops popping abs_xy, which is required for a
multi-task pipeline to run this transform once per pixel_semantic
task (e.g. network then forest_2d) on the same data_dict."
```

---

## Task 4: Generalize `Flair3DDataset` pixel-semantic asset loading

**Files:**
- Modify: `pointcept/datasets/flair3d.py:28-37` (`FLAIR3D_SPECIFIC_ASSETS`), `:324-425`
  (rename/generalize `_load_network_label`/`_select_network_channels`), `:530-531` (call site
  in `get_data`)
- Test: `tests/test_pixel_semantic_dataset_loading.py`

**Interfaces:**
- Produces: `Flair3DDataset._load_pixel_semantic_label(self, data_dict, scene,
  target_key="network")` (renamed from `_load_network_label`), calling
  `Flair3DDataset._select_pixel_semantic_channels(raster, *, r, channel_names, channel_order,
  scene, target_key)` (renamed from `_select_network_channels`).
- Consumes: `get_pixel_semantic_config(target_key)`, `self.optional_target_keys`,
  `self._is_optional_target`, `self._missing_target_array` (all already generic).

- [ ] **Step 1: Write the failing test**

Create `tests/test_pixel_semantic_dataset_loading.py`:

```python
"""
Tests for Flair3DDataset's pixel-semantic asset loading, generalized from a
network-only _load_network_label to a target_key-parametrized
_load_pixel_semantic_label so both "network" and "forest_2d" (or any other
FLAIR3D_PIXEL_SEMANTIC_TASKS entry) load through the same code path.

Constructs a bare Flair3DDataset instance via object.__new__ (bypassing __init__,
which needs real on-disk manifests) and sets only the attributes the method under
test actually reads (self.optional_target_keys), following the same lightweight
pattern as tests/test_tile_distribution_pooling.py's direct model construction.

Run with: PYTHONPATH=./ pytest tests/test_pixel_semantic_dataset_loading.py
"""

import json
import os
import tempfile
import unittest

import numpy as np

from pointcept.datasets.flair3d import Flair3DDataset


def _bare_dataset(optional_target_keys=()):
    ds = object.__new__(Flair3DDataset)
    ds.optional_target_keys = tuple(optional_target_keys)
    return ds


class TestLoadPixelSemanticLabelForestTwoD(unittest.TestCase):
    def test_loads_npy_already_in_data_dict(self):
        ds = _bare_dataset()
        with tempfile.TemporaryDirectory() as scene:
            meta = {
                "forest_2d": {
                    "origin_x": 10.0,
                    "origin_y": 20.0,
                    "pixel_m": 0.5,
                    "width": 4,
                    "height": 3,
                    "channel_order": ["FOREST"],
                }
            }
            with open(os.path.join(scene, "meta.json"), "w") as f:
                json.dump(meta, f)
            raster = np.ones((1, 3, 4), dtype=np.uint8)
            data_dict = {"forest_2d": raster}

            out = ds._load_pixel_semantic_label(data_dict, scene, target_key="forest_2d")

            np.testing.assert_array_equal(out["forest_2d"], raster)
            self.assertEqual(out["forest_2d_origin_x"], [10.0])
            self.assertEqual(out["forest_2d_origin_y"], [20.0])
            self.assertEqual(out["forest_2d_pixel_m"], [0.5])

    def test_meta_only_empty_tile_synthesizes_zeros(self):
        ds = _bare_dataset()
        with tempfile.TemporaryDirectory() as scene:
            meta = {
                "forest_2d": {
                    "origin_x": 0.0,
                    "origin_y": 0.0,
                    "pixel_m": 0.5,
                    "width": 4,
                    "height": 3,
                }
            }
            with open(os.path.join(scene, "meta.json"), "w") as f:
                json.dump(meta, f)
            data_dict = {}

            out = ds._load_pixel_semantic_label(data_dict, scene, target_key="forest_2d")

            self.assertEqual(out["forest_2d"].shape, (1, 3, 4))
            self.assertTrue((out["forest_2d"] == 0).all())

    def test_missing_and_not_optional_raises(self):
        ds = _bare_dataset(optional_target_keys=())
        with tempfile.TemporaryDirectory() as scene:
            data_dict = {}
            with self.assertRaises(FileNotFoundError):
                ds._load_pixel_semantic_label(data_dict, scene, target_key="forest_2d")

    def test_missing_and_optional_uses_fill_value(self):
        ds = _bare_dataset(optional_target_keys=("forest_2d",))
        with tempfile.TemporaryDirectory() as scene:
            data_dict = {}
            out = ds._load_pixel_semantic_label(data_dict, scene, target_key="forest_2d")
            self.assertEqual(out["forest_2d"].shape[0], 1)


class TestNetworkStillWorksUnchanged(unittest.TestCase):
    def test_default_target_key_loads_network(self):
        ds = _bare_dataset()
        with tempfile.TemporaryDirectory() as scene:
            meta = {
                "network": {
                    "origin_x": 1.0,
                    "origin_y": 2.0,
                    "pixel_m": 1.0,
                    "width": 2,
                    "height": 2,
                    "channel_order": ["ROADS", "RAILROADS", "TRANSMISSION_LINES"],
                }
            }
            with open(os.path.join(scene, "meta.json"), "w") as f:
                json.dump(meta, f)
            raster = np.zeros((3, 2, 2), dtype=np.uint8)
            raster[0] = 1  # ROADS channel
            data_dict = {"network": raster}

            out = ds._load_pixel_semantic_label(data_dict, scene)

            self.assertEqual(out["network"].shape, (2, 2, 2))  # sliced to r=2
            np.testing.assert_array_equal(out["network"][0], np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./ pytest tests/test_pixel_semantic_dataset_loading.py -v`
Expected: FAIL with `AttributeError: 'Flair3DDataset' object has no attribute
'_load_pixel_semantic_label'`

- [ ] **Step 3: Add `"forest_2d"` to `FLAIR3D_SPECIFIC_ASSETS`**

In `pointcept/datasets/flair3d.py`, replace (currently lines 28-37):

```python
FLAIR3D_SPECIFIC_ASSETS = (
    "forest",
    "land_use",
    "natural_habitat",
    "elevation",
    "climatic_domain",
    "natural_habitat_multilabel",
    "coord_translation",
    "network",
)
```

with:

```python
FLAIR3D_SPECIFIC_ASSETS = (
    "forest",
    "land_use",
    "natural_habitat",
    "elevation",
    "climatic_domain",
    "natural_habitat_multilabel",
    "coord_translation",
    "network",
    "forest_2d",
)
```

(`forest_2d` is deliberately **not** added to `FLAIR3D_OPTIONAL_TARGETS`, currently lines
107-114 — FOREST source coverage is complete for every tile, so a missing `forest_2d.npy`
must hard-fail like a missing `segment.npy` would, not silently substitute zeros.)

- [ ] **Step 4: Rename and generalize `_load_network_label` / `_select_network_channels`**

Replace the two methods (currently `pointcept/datasets/flair3d.py:324-425`):

```python
    def _load_pixel_semantic_label(self, data_dict, scene, target_key="network"):
        """Load ``{target_key}.npy`` and grid meta for a pixel semantic task.

        Training heads use ``num_networks`` channels from
        ``get_pixel_semantic_config(target_key)``. On-disk rasters may have more
        channels than the task trains on (e.g. historical ``network.npy`` with
        TRANSMISSION_LINES as channel 2); those are sliced via
        ``meta.{target_key}.channel_order`` when present, else the first ``r``
        channels.

        Empty tiles may omit ``{target_key}.npy`` and only store
        ``meta.{target_key}`` (``empty: true`` + width/height); those are
        synthesized as zeros.
        """
        import json

        from pointcept.datasets.flair3d_config_utils import (
            NETWORK_CHANNEL_NAMES,
            get_pixel_semantic_config,
        )

        cfg = get_pixel_semantic_config(target_key)
        r = int(cfg["num_networks"])
        default_channel_names = (
            NETWORK_CHANNEL_NAMES if target_key == "network" else (target_key,)
        )
        channel_names = list(cfg.get("channel_names") or default_channel_names)

        origin_x = 0.0
        origin_y = 0.0
        pixel_m = 1.0
        raster_meta = {}
        meta_path = os.path.join(scene, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            maybe = meta.get(target_key) or {}
            if isinstance(maybe, dict):
                raster_meta = maybe
                origin_x = float(raster_meta.get("origin_x", 0.0))
                origin_y = float(raster_meta.get("origin_y", 0.0))
                pixel_m = float(raster_meta.get("pixel_m", 1.0))

        if target_key in data_dict:
            raster = np.asarray(data_dict[target_key])
            if raster.ndim != 3:
                raise ValueError(
                    f"{target_key}.npy expected shape (C, H, W), got {raster.shape} "
                    f"under scene: {scene}"
                )
            raster = self._select_pixel_semantic_channels(
                raster,
                r=r,
                channel_names=channel_names,
                channel_order=raster_meta.get("channel_order"),
                scene=scene,
                target_key=target_key,
            )
            raster = raster.astype(np.uint8, copy=False)
        elif raster_meta:
            # Preprocess wrote meta only (empty mask) or optional missing fill path.
            h = int(raster_meta.get("height", 1))
            w = int(raster_meta.get("width", 1))
            raster = np.zeros((r, max(h, 1), max(w, 1)), dtype=np.uint8)
        elif self._is_optional_target(target_key):
            raster = self._missing_target_array(target_key, 0)
        else:
            raise FileNotFoundError(
                f"target key '{target_key}' but {target_key}.npy missing under scene: {scene}"
            )

        # Align tiny optional fill (1,1) to meta grid when present.
        if raster.shape[1] == 1 and raster.shape[2] == 1 and raster_meta:
            h = int(raster_meta.get("height", 1))
            w = int(raster_meta.get("width", 1))
            if h > 1 or w > 1:
                raster = np.zeros((r, h, w), dtype=np.uint8)

        data_dict[target_key] = raster
        # Keep origins in float64 for precise cell binning in NetworkRasterToPointLabels.
        data_dict[f"{target_key}_origin_x"] = np.asarray([origin_x], dtype=np.float64)
        data_dict[f"{target_key}_origin_y"] = np.asarray([origin_y], dtype=np.float64)
        data_dict[f"{target_key}_pixel_m"] = np.asarray([pixel_m], dtype=np.float64)
        return data_dict

    @staticmethod
    def _select_pixel_semantic_channels(
        raster, *, r, channel_names, channel_order, scene, target_key
    ):
        """Reduce on-disk ``(C, H, W)`` to the ``r`` training channels."""
        c = int(raster.shape[0])
        if c == r:
            return raster
        if c < r:
            raise ValueError(
                f"{target_key}.npy has {c} channels but task expects {r} "
                f"({channel_names}) under scene: {scene}"
            )
        if isinstance(channel_order, (list, tuple)) and len(channel_order) == c:
            name_to_idx = {str(name): i for i, name in enumerate(channel_order)}
            missing = [name for name in channel_names if name not in name_to_idx]
            if missing:
                raise ValueError(
                    f"{target_key}.npy channel_order {list(channel_order)} missing "
                    f"{missing} under scene: {scene}"
                )
            indices = [name_to_idx[name] for name in channel_names]
            return raster[indices]
        # Historical preprocess order: ROADS, RAILROADS, TRANSMISSION_LINES, ...
        return raster[:r]
```

- [ ] **Step 5: Update the call site in `get_data`**

In `pointcept/datasets/flair3d.py`, replace (currently lines 530-531):

```python
        if "network" in pixel_semantic_keys:
            data_dict = self._load_network_label(data_dict, scene)
```

with:

```python
        for tk in pixel_semantic_keys:
            data_dict = self._load_pixel_semantic_label(data_dict, scene, target_key=tk)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `PYTHONPATH=./ pytest tests/test_pixel_semantic_dataset_loading.py -v`
Expected: PASS

- [ ] **Step 7: Confirm no other reference to the old method names remains**

Run: `grep -rn "_load_network_label\|_select_network_channels" pointcept/ tests/`
Expected: no output (already confirmed no external references exist before this plan was
written; this step just guards against having missed one while editing).

- [ ] **Step 8: Commit**

```bash
git add pointcept/datasets/flair3d.py tests/test_pixel_semantic_dataset_loading.py
git commit -m "Generalize Flair3DDataset pixel-semantic loading for forest_2d

Rename _load_network_label/_select_network_channels to their
target_key-parametrized equivalents (_load_pixel_semantic_label /
_select_pixel_semantic_channels); get_data now loops over every
pixel_semantic key present instead of special-casing \"network\".
forest_2d is added to FLAIR3D_SPECIFIC_ASSETS but deliberately not to
FLAIR3D_OPTIONAL_TARGETS (FOREST coverage is complete; missing data
must hard-fail)."
```

---

## Task 5: Generalize `MultiTaskSegmentorV2` pixel pooling (per-task keys + mean pooling)

**Files:**
- Modify: `pointcept/models/default.py:481-679`
- Test: `tests/test_pixel_semantic_pooling.py`

**Interfaces:**
- Produces: `MultiTaskSegmentorV2._pixel_pool_and_gather(self, feat, cell, pix, offset,
  point_labels, num_networks, height=None, width=None, ignore_index=2, pooling="max")`
  (renamed params `network_cell`→`cell`, `network_pix`→`pix`; new `pooling` kwarg).
  `_compute_pixel_logits` now looks up `input_dict[f"{task_name}_cell"]` etc. instead of the
  hardcoded `"network_cell"`, and passes `pooling=task_config.get("pooling", "max")`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pixel_semantic_pooling.py`:

```python
"""
Tests for MultiTaskSegmentorV2._pixel_pool_and_gather's configurable pooling mode
(max, the historical/default behavior used by "network"; mean, used by "forest_2d").

Constructs a bare MultiTaskSegmentorV2 via object.__new__ since
_pixel_pool_and_gather does not read any instance state (pure function of its
arguments) -- same lightweight-construction pattern as
tests/test_pixel_semantic_dataset_loading.py.

Run with: PYTHONPATH=./ pytest tests/test_pixel_semantic_pooling.py
"""

import unittest

import torch

from pointcept.models.default import MultiTaskSegmentorV2


def _bare_model():
    return object.__new__(MultiTaskSegmentorV2)


class TestPixelPoolAndGather(unittest.TestCase):
    def setUp(self):
        self.model = _bare_model()
        # One scene, 3 points: points 0 and 1 share cell (0, 0); point 2 is alone
        # in cell (1, 1).
        self.feat = torch.tensor([[1.0], [3.0], [10.0]])
        self.cell = torch.tensor([[0, 0], [0, 0], [1, 1]])
        self.pix = torch.tensor([[0, 0], [0, 0], [1, 1]])
        self.offset = torch.tensor([3])
        self.point_labels = torch.tensor([[1], [1], [0]])

    def test_default_pooling_is_max_unchanged(self):
        pooled, targets, _ = self.model._pixel_pool_and_gather(
            self.feat, self.cell, self.pix, self.offset, self.point_labels,
            num_networks=1,
        )
        # Cell (0,0) max(1,3)=3; cell (1,1) is 10. Order follows torch.unique.
        pooled_sorted = sorted(pooled.flatten().tolist())
        self.assertEqual(pooled_sorted, [3.0, 10.0])

    def test_mean_pooling_averages_within_cell(self):
        pooled, targets, _ = self.model._pixel_pool_and_gather(
            self.feat, self.cell, self.pix, self.offset, self.point_labels,
            num_networks=1, pooling="mean",
        )
        pooled_sorted = sorted(pooled.flatten().tolist())
        # Cell (0,0) mean(1,3)=2; cell (1,1) is 10.
        self.assertEqual(pooled_sorted, [2.0, 10.0])

    def test_explicit_max_matches_default(self):
        pooled_default, _, _ = self.model._pixel_pool_and_gather(
            self.feat, self.cell, self.pix, self.offset, self.point_labels,
            num_networks=1,
        )
        pooled_explicit, _, _ = self.model._pixel_pool_and_gather(
            self.feat, self.cell, self.pix, self.offset, self.point_labels,
            num_networks=1, pooling="max",
        )
        torch.testing.assert_close(pooled_default, pooled_explicit)

    def test_invalid_pooling_raises(self):
        with self.assertRaises(ValueError):
            self.model._pixel_pool_and_gather(
                self.feat, self.cell, self.pix, self.offset, self.point_labels,
                num_networks=1, pooling="sum",
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./ pytest tests/test_pixel_semantic_pooling.py -v`
Expected: FAIL — `_pixel_pool_and_gather` does not accept a `pooling` keyword yet.

- [ ] **Step 3: Rewrite `_pixel_pool_and_gather`**

Replace the method (currently `pointcept/models/default.py:481-597`):

```python
    def _pixel_pool_and_gather(
        self,
        feat,
        cell,
        pix,
        offset,
        point_labels,
        num_networks,
        height=None,
        width=None,
        ignore_index=2,
        pooling="max",
    ):
        """Pool point features into Lambert pixels; gather point-wise GT.

        Point labels are ``(N, r)`` (from ``NetworkRasterToPointLabels``).
        ``cell`` ``(N, 2)`` absolute ``(iy, ix)`` — Mix3D-safe unique keys.
        ``pix`` ``(N, 2)`` relative ``(iy, ix)`` — dense scatter on GT grid.
        ``pooling`` is ``"max"`` (default, e.g. ``network`` — catches a single
        strong "there's a line here" signal) or ``"mean"`` (e.g. ``forest_2d`` —
        an area-coverage class benefits from averaging the signal across every
        point in a cell rather than just the strongest one).
        Dense meta is only filled when per-scene ``H,W`` length matches ``offset``
        (skip under Mix3D).

        Returns
        -------
        pooled_feat : (P, C)
        targets : (P, r) long
        meta_parts : list of None or (n_pix, iy_u, ix_u, height, width, in_grid)
        """
        if pooling not in ("max", "mean"):
            raise ValueError(f"pooling must be 'max' or 'mean', got {pooling!r}")
        device = feat.device
        if not torch.is_tensor(point_labels):
            point_labels = torch.as_tensor(point_labels, device=device)
        else:
            point_labels = point_labels.to(device=device)
        if point_labels.ndim != 2 or point_labels.shape[0] != feat.shape[0]:
            raise ValueError(
                f"point-wise network labels expected (N={feat.shape[0]}, r), "
                f"got {tuple(point_labels.shape)}"
            )
        if point_labels.shape[1] != num_networks:
            raise ValueError(
                f"network labels expected r={num_networks} channels, "
                f"got {point_labels.shape[1]}"
            )
        if not torch.is_tensor(cell):
            cell = torch.as_tensor(cell, device=device)
        else:
            cell = cell.to(device=device)
        if not torch.is_tensor(pix):
            pix = torch.as_tensor(pix, device=device)
        else:
            pix = pix.to(device=device)
        if cell.ndim != 2 or cell.shape[0] != feat.shape[0] or cell.shape[1] < 2:
            raise ValueError(
                f"cell expected (N={feat.shape[0]}, 2), got {tuple(cell.shape)}"
            )
        if pix.ndim != 2 or pix.shape[0] != feat.shape[0] or pix.shape[1] < 2:
            raise ValueError(
                f"pix expected (N={feat.shape[0]}, 2), got {tuple(pix.shape)}"
            )

        heights = widths = None
        can_dense = False
        if height is not None and width is not None:
            heights = self._as_scalar_tensor(height, device, dtype=torch.float32)
            widths = self._as_scalar_tensor(width, device, dtype=torch.float32)
            if heights.numel() == 1 and offset.numel() > 1:
                heights = heights.expand(offset.numel())
                widths = widths.expand(offset.numel())
            can_dense = (
                heights.numel() == offset.numel() and widths.numel() == offset.numel()
            )

        batch_idx = offset2batch(offset)
        pooled_parts = []
        target_parts = []
        meta_parts = []
        n_scenes = int(offset.numel())
        for b in range(n_scenes):
            point_mask = batch_idx == b
            if not point_mask.any():
                meta_parts.append(None)
                continue
            cells_b = cell[point_mask][:, :2].long()
            pix_b = pix[point_mask][:, :2].long()
            f = feat[point_mask]
            labels_pt = point_labels[point_mask]

            _, inv = torch.unique(cells_b, dim=0, return_inverse=True)
            if pooling == "mean":
                pooled = torch_scatter.scatter_mean(f, inv, dim=0)
            else:
                pooled = torch_scatter.scatter_max(f, inv, dim=0)[0]
            idx = torch.arange(inv.numel(), device=device)
            first = torch_scatter.scatter_min(idx, inv, dim=0)[0]
            labels = labels_pt[first].long()

            pooled_parts.append(pooled)
            target_parts.append(labels)

            n_pix = int(pooled.shape[0])
            if can_dense:
                h = int(heights[b].item())
                w = int(widths[b].item())
                iy_u = pix_b[first, 0]
                ix_u = pix_b[first, 1]
                in_grid = (ix_u >= 0) & (iy_u >= 0) & (ix_u < w) & (iy_u < h)
                meta_parts.append((n_pix, iy_u, ix_u, h, w, in_grid))
            else:
                meta_parts.append((n_pix, None, None, None, None, None))

        if not pooled_parts:
            c = feat.shape[1]
            return (
                feat.new_zeros((0, c)),
                feat.new_zeros((0, num_networks), dtype=torch.long),
                meta_parts,
            )
        return (
            torch.cat(pooled_parts, dim=0),
            torch.cat(target_parts, dim=0),
            meta_parts,
        )
```

- [ ] **Step 4: Update `_compute_pixel_logits` to use per-task keys and pooling**

In `pointcept/models/default.py`, replace (currently within `_compute_pixel_logits`, the
lookup block just before `raw = head(pooled)`):

```python
            if task_name not in input_dict:
                continue
            if "network_cell" not in input_dict or "network_pix" not in input_dict:
                raise KeyError(
                    f"pixel_semantic task '{task_name}' requires 'network_cell' and "
                    "'network_pix' in input_dict (add NetworkRasterToPointLabels)."
                )
            pooled, targets, meta_parts = self._pixel_pool_and_gather(
                feat,
                input_dict["network_cell"],
                input_dict["network_pix"],
                input_dict["offset"],
                input_dict[task_name],
                num_networks,
                height=input_dict.get("network_height"),
                width=input_dict.get("network_width"),
                ignore_index=int(task_config.get("ignore_index", 2)),
            )
```

with:

```python
            if task_name not in input_dict:
                continue
            cell_key = f"{task_name}_cell"
            pix_key = f"{task_name}_pix"
            if cell_key not in input_dict or pix_key not in input_dict:
                raise KeyError(
                    f"pixel_semantic task '{task_name}' requires '{cell_key}' and "
                    f"'{pix_key}' in input_dict (add "
                    f"NetworkRasterToPointLabels(target_key='{task_name}'))."
                )
            pooled, targets, meta_parts = self._pixel_pool_and_gather(
                feat,
                input_dict[cell_key],
                input_dict[pix_key],
                input_dict["offset"],
                input_dict[task_name],
                num_networks,
                height=input_dict.get(f"{task_name}_height"),
                width=input_dict.get(f"{task_name}_width"),
                ignore_index=int(task_config.get("ignore_index", 2)),
                pooling=str(task_config.get("pooling", "max")),
            )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=./ pytest tests/test_pixel_semantic_pooling.py -v`
Expected: PASS

- [ ] **Step 6: Run the tile_distribution regression suite (touches the same file)**

Run: `PYTHONPATH=./ pytest tests/test_tile_distribution_pooling.py -v`
Expected: PASS (unaffected — different code path in the same file).

- [ ] **Step 7: Commit**

```bash
git add pointcept/models/default.py tests/test_pixel_semantic_pooling.py
git commit -m "Generalize MultiTaskSegmentorV2 pixel pooling for forest_2d

_pixel_pool_and_gather/_compute_pixel_logits now look up
{task_name}_cell/_pix/_height/_width instead of a hardcoded
\"network_*\" key, and support pooling=\"mean\" (forest_2d) alongside
the existing pooling=\"max\" default (network, unchanged)."
```

---

## Task 6: Add test-set precision/recall/F1 for pixel_semantic tasks

**Files:**
- Modify: `pointcept/utils/misc.py` (add `binary_prf_counts`)
- Modify: `pointcept/engines/test.py` (imports; per-scene loop; `record` payload; merge loop;
  final logging)
- Test: `tests/test_pixel_semantic_test_prf.py`

**Interfaces:**
- Produces: `binary_prf_counts(prob, target, ignore_index, fg_idx) -> (tp, fp, fn)` in
  `pointcept.utils.misc` (pure numpy function, no torch/model dependency).
- Consumes (in `test.py`): `targets_by_task[task_name]` (already a dense `(r, H, W)` GT array
  for pixel_semantic tasks — see spec section 5b for why this needs no fragment-merging),
  `pixel_logits_np[task_name]` (already a dense `(r, H, W)` merged-prediction array, populated
  identically whether from a fresh forward pass or the npy cache).

- [ ] **Step 1: Write the failing test for `binary_prf_counts`**

Create `tests/test_pixel_semantic_test_prf.py`:

```python
"""
Tests for binary_prf_counts (pointcept.utils.misc), the pure tp/fp/fn helper used
by MultiTaskTester to compute test-set precision/recall/F1 for pixel_semantic
tasks (e.g. forest_2d) from a scene's dense prediction/target grids.

Run with: PYTHONPATH=./ pytest tests/test_pixel_semantic_test_prf.py
"""

import unittest

import numpy as np

from pointcept.utils.misc import binary_prf_counts


class TestBinaryPrfCounts(unittest.TestCase):
    def test_simple_confusion_counts(self):
        # 2x2 grid. fg_idx=1, ignore_index=2.
        # (0,0): pred fg, gt fg -> TP
        # (0,1): pred fg, gt not-fg -> FP
        # (1,0): pred not-fg, gt fg -> FN
        # (1,1): pred not-fg, gt not-fg -> TN (not counted)
        prob = np.array([[0.9, 0.8], [0.1, 0.2]])
        target = np.array([[1, 0], [1, 0]])
        tp, fp, fn = binary_prf_counts(prob, target, ignore_index=2, fg_idx=1)
        self.assertEqual((tp, fp, fn), (1, 1, 1))

    def test_unobserved_pixels_excluded(self):
        prob = np.array([[0.9, np.nan]])
        target = np.array([[1, 1]])
        tp, fp, fn = binary_prf_counts(prob, target, ignore_index=2, fg_idx=1)
        self.assertEqual((tp, fp, fn), (1, 0, 0))

    def test_void_pixels_excluded(self):
        prob = np.array([[0.9, 0.9]])
        target = np.array([[1, 2]])  # second pixel is Void (ignore_index=2)
        tp, fp, fn = binary_prf_counts(prob, target, ignore_index=2, fg_idx=1)
        self.assertEqual((tp, fp, fn), (1, 0, 0))

    def test_all_excluded_gives_zero_counts(self):
        prob = np.full((2, 2), np.nan)
        target = np.full((2, 2), 2)
        tp, fp, fn = binary_prf_counts(prob, target, ignore_index=2, fg_idx=1)
        self.assertEqual((tp, fp, fn), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./ pytest tests/test_pixel_semantic_test_prf.py -v`
Expected: FAIL with `ImportError: cannot import name 'binary_prf_counts'`

- [ ] **Step 3: Add `binary_prf_counts` to `pointcept/utils/misc.py`**

Insert after `mean_acc_from_hist` (currently ends at line 198, right before `def
make_dirs(dir_name):`):

```python
def binary_prf_counts(prob, target, ignore_index, fg_idx):
    """True/false positive/negative counts for one binary foreground channel.

    ``prob`` (float, NaN = unobserved) and ``target`` (int label ids) are same-shape
    arrays (typically a dense (H, W) grid). A cell is scored only if ``prob`` is
    finite (something was actually predicted there) and ``target != ignore_index``
    (not a Void/unlabeled cell).

    Returns ``(tp, fp, fn)`` as plain Python ints.
    """
    prob = np.asarray(prob)
    target = np.asarray(target)
    valid = np.isfinite(prob) & (target != ignore_index)
    pred_fg = (prob > 0.5) & valid
    gt_fg = (target == fg_idx) & valid
    tp = int(np.count_nonzero(pred_fg & gt_fg))
    fp = int(np.count_nonzero(pred_fg & ~gt_fg))
    fn = int(np.count_nonzero(~pred_fg & gt_fg))
    return tp, fp, fn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=./ pytest tests/test_pixel_semantic_test_prf.py -v`
Expected: PASS

- [ ] **Step 5: Wire `binary_prf_counts` into `MultiTaskTester` — imports**

In `pointcept/engines/test.py`, add `binary_prf_counts` to the existing
`from pointcept.utils.misc import (...)` block (currently lines 27-39):

```python
from pointcept.utils.misc import (
    AverageMeter,
    abs_freq_error_rows,
    accumulate_regression_errors,
    binary_prf_counts,
    f1_scores_from_hist,
    intersection_and_union,
    intersection_and_union_gpu,
    kl_divergence_rows,
    make_dirs,
    mean_acc_from_hist,
    mean_iou_from_hist,
    pool_axis_distribution_from_probs,
)
```

And add a new import for `precision_recall_f1` right after it:

```python
from pointcept.utils.dilated_metrics import precision_recall_f1
```

- [ ] **Step 6: Compute per-scene tp/fp/fn in the "compute metrics for each scene" loop**

In `pointcept/engines/test.py`, in `_process_one_multitask_batch`, find the block:

```python
            # Compute metrics for each scene in the batch
            for b_idx in range(len(batch)):
                data_name = batch_data_names[b_idx]
                targets_by_task = batch_targets[b_idx]
                scene_extra = batch_scene_extra[b_idx]
                pred_cls_np = batch_pred_cls_np[b_idx]
                pred_reg_np = batch_pred_reg_np[b_idx]
                
                sem_metrics_scene = {}
                for task_name in semantic_tasks:
```

Replace it with (adds `pixel_logits_np = batch_pixel_logits_np[b_idx]` and a new
`pixel_prf_metrics_scene` block computed right after the existing `sem_metrics_scene` loop —
do **not** touch the body of the `semantic_tasks` loop itself):

```python
            # Compute metrics for each scene in the batch
            for b_idx in range(len(batch)):
                data_name = batch_data_names[b_idx]
                targets_by_task = batch_targets[b_idx]
                scene_extra = batch_scene_extra[b_idx]
                pred_cls_np = batch_pred_cls_np[b_idx]
                pred_reg_np = batch_pred_reg_np[b_idx]
                pixel_logits_np = batch_pixel_logits_np[b_idx]

                sem_metrics_scene = {}
                for task_name in semantic_tasks:
```

then, immediately after the `sem_metrics_scene` for-loop ends (i.e. right before the existing
line `pred_scene_cls_np = batch_pred_scene_cls_np[b_idx]`), insert:

```python
                # Test-set precision/recall/F1 for pixel_semantic tasks (foreground
                # class only). targets_by_task[task_name] is already the dense
                # (r, H, W) GT raster at full tile resolution (Flair3DDataset.
                # prepare_test_data snapshots it before any per-fragment transform
                # runs), so no fragment-merging is needed on the GT side --
                # pixel_logits_np[task_name] (the merged dense prediction, from
                # either a fresh forward pass or the npy cache) is the only thing
                # that needed merging, and that already happened above.
                pixel_prf_metrics_scene = {}
                for task_name in pixel_semantic_tasks:
                    if task_name not in targets_by_task:
                        continue
                    if task_name not in pixel_logits_np:
                        continue
                    tc = task_configs[task_name]
                    ignore_index = int(tc["ignore_index"])
                    num_networks = int(tc.get("num_networks", 1))
                    names = list(tc["names"])
                    fg_idx = names.index("Foreground") if "Foreground" in names else 1
                    channel_names = tc.get("channel_names") or [
                        f"ch{c}" for c in range(num_networks)
                    ]
                    target_arr = np.asarray(targets_by_task[task_name])
                    pred_arr = np.asarray(pixel_logits_np[task_name])
                    channel_stats = {}
                    for c in range(num_networks):
                        ch_name = (
                            channel_names[c] if c < len(channel_names) else f"ch{c}"
                        )
                        tp, fp, fn = binary_prf_counts(
                            pred_arr[c], target_arr[c], ignore_index, fg_idx
                        )
                        channel_stats[ch_name] = dict(tp=tp, fp=fp, fn=fn)
                    pixel_prf_metrics_scene[task_name] = channel_stats

```

- [ ] **Step 7: Extend the per-scene `record` payload**

Replace (currently):

```python
                record[data_name] = dict(
                    semantic=sem_metrics_scene, tile_distribution=td_metrics_scene
                )
```

with:

```python
                record[data_name] = dict(
                    semantic=sem_metrics_scene,
                    tile_distribution=td_metrics_scene,
                    pixel_semantic=pixel_prf_metrics_scene,
                )
```

- [ ] **Step 8: Accumulate tp/fp/fn across the whole test split at merge time**

Replace (currently, right after `record_sync = comm.gather(record, dst=0)`):

```python
            per_task_sem = {t: None for t in semantic_tasks}
            for _, payload in merged.items():
                for task_name, meters in payload["semantic"].items():
                    if task_name not in per_task_sem:
                        continue
                    if per_task_sem[task_name] is None:
                        per_task_sem[task_name] = {
                            "intersection": meters["intersection"].copy(),
                            "union": meters["union"].copy(),
                            "target": meters["target"].copy(),
                        }
                    else:
                        per_task_sem[task_name]["intersection"] += meters["intersection"]
                        per_task_sem[task_name]["union"] += meters["union"]
                        per_task_sem[task_name]["target"] += meters["target"]
                for task_name, meters in payload.get("tile_distribution", {}).items():
                    if task_name not in td_sums_global:
                        continue
                    td_sums_global[task_name]["kl_weighted"] += meters["kl_weighted"]
                    td_sums_global[task_name]["weight"] += meters["weight"]
                    td_sums_global[task_name]["abs_weighted"] += np.asarray(
                        meters["abs_weighted"], dtype=np.float64
                    )
```

with:

```python
            per_task_sem = {t: None for t in semantic_tasks}
            per_task_pixel_prf = {t: {} for t in pixel_semantic_tasks}
            for _, payload in merged.items():
                for task_name, meters in payload["semantic"].items():
                    if task_name not in per_task_sem:
                        continue
                    if per_task_sem[task_name] is None:
                        per_task_sem[task_name] = {
                            "intersection": meters["intersection"].copy(),
                            "union": meters["union"].copy(),
                            "target": meters["target"].copy(),
                        }
                    else:
                        per_task_sem[task_name]["intersection"] += meters["intersection"]
                        per_task_sem[task_name]["union"] += meters["union"]
                        per_task_sem[task_name]["target"] += meters["target"]
                for task_name, meters in payload.get("tile_distribution", {}).items():
                    if task_name not in td_sums_global:
                        continue
                    td_sums_global[task_name]["kl_weighted"] += meters["kl_weighted"]
                    td_sums_global[task_name]["weight"] += meters["weight"]
                    td_sums_global[task_name]["abs_weighted"] += np.asarray(
                        meters["abs_weighted"], dtype=np.float64
                    )
                for task_name, channel_stats in payload.get("pixel_semantic", {}).items():
                    if task_name not in per_task_pixel_prf:
                        continue
                    for ch_name, counts in channel_stats.items():
                        acc = per_task_pixel_prf[task_name].setdefault(
                            ch_name, {"tp": 0, "fp": 0, "fn": 0}
                        )
                        acc["tp"] += counts["tp"]
                        acc["fp"] += counts["fp"]
                        acc["fn"] += counts["fn"]
```

- [ ] **Step 9: Log and expose the final test-set precision/recall/F1**

Find the block that ends the semantic per-class IoU tag-writing loop (currently):

```python
                if self.write_cls_iou:
                    task_config = task_configs[task_name]
                    for class_idx in range(int(task_config["num_classes"])):
                        if class_idx == task_config["ignore_index"]:
                            continue
                        slug = class_name_slug(metric["names"][class_idx])
                        log_dict[
                            iou_class_tag("test", slug, task=task_name)
                        ] = float(metric["iou_class"][class_idx])

            for task_name in regression_tasks:
```

Insert a new block between them (so it reads: `...float(metric["iou_class"][class_idx])`,
then the new pixel_semantic block, then `for task_name in regression_tasks:`):

```python
                if self.write_cls_iou:
                    task_config = task_configs[task_name]
                    for class_idx in range(int(task_config["num_classes"])):
                        if class_idx == task_config["ignore_index"]:
                            continue
                        slug = class_name_slug(metric["names"][class_idx])
                        log_dict[
                            iou_class_tag("test", slug, task=task_name)
                        ] = float(metric["iou_class"][class_idx])

            for task_name in pixel_semantic_tasks:
                channel_stats = per_task_pixel_prf.get(task_name) or {}
                for ch_name, counts in channel_stats.items():
                    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
                    if tp + fp + fn == 0:
                        logger.warning(
                            "[task=%s] Channel %s: no test-set pixel_semantic "
                            "samples accumulated; skipping P/R/F1.",
                            task_name,
                            ch_name,
                        )
                        continue
                    precision, recall, f1 = precision_recall_f1(tp, tp + fp, tp, tp + fn)
                    logger.info(
                        "[task={}] Channel {} Test result: precision/recall/f1 "
                        "{:.4f}/{:.4f}/{:.4f} (tp={}, fp={}, fn={}).".format(
                            task_name, ch_name, precision, recall, f1, tp, fp, fn
                        )
                    )
                    ch_slug = class_name_slug(ch_name)
                    log_dict[
                        metric_tag("test", f"{ch_slug}/precision", task=task_name)
                    ] = float(precision)
                    log_dict[
                        metric_tag("test", f"{ch_slug}/recall", task=task_name)
                    ] = float(recall)
                    log_dict[
                        metric_tag("test", f"{ch_slug}/f1", task=task_name)
                    ] = float(f1)

            for task_name in regression_tasks:
```

- [ ] **Step 10: Run the full test file plus a syntax/import sanity check on `test.py`**

Run: `PYTHONPATH=./ pytest tests/test_pixel_semantic_test_prf.py -v`
Expected: PASS

Run: `PYTHONPATH=./ python -c "import pointcept.engines.test"`
Expected: no error (catches typos/indentation mistakes in the edits above — `test.py` has no
existing pytest coverage of `MultiTaskTester.test()` itself, since it requires a real trainer/
dataset; this import check plus careful review against the exact anchors above is the
available verification).

- [ ] **Step 11: Commit**

```bash
git add pointcept/utils/misc.py pointcept/engines/test.py tests/test_pixel_semantic_test_prf.py
git commit -m "Add test-set precision/recall/F1 for pixel_semantic tasks

MultiTaskTester previously computed no aggregate metric at all for
pixel_semantic tasks (only network's opt-in APLS). Add a generic,
foreground-class-only P/R/F1 (forest_2d and, additively, network) by
reusing the dense GT raster Flair3DDataset.prepare_test_data already
keeps at full tile resolution -- no model/fragment-merging needed on
the GT side, and it works identically under the npy-cache fast path."
```

---

## Task 7: Shared preprocessing utils + new `rasterize_forest.py` script

**Files:**
- Modify: `pointcept/datasets/preprocessing/flair3d_plus/network_xy_raster_utils.py` (add
  `abs_xy_bounds_from_coord`, `load_known_missing_tiles`, `default_missing_coord_details_csv`)
- Modify: `pointcept/datasets/preprocessing/flair3d_plus/rasterize_network.py` (use the three
  functions above instead of its own private copies — pure relocation, no behavior change)
- Create: `pointcept/datasets/preprocessing/flair3d_plus/rasterize_forest.py`
- Test: `tests/test_network_xy_raster_shared_utils.py`, `tests/test_rasterize_forest.py`

**Interfaces:**
- Produces (in `network_xy_raster_utils.py`): `abs_xy_bounds_from_coord(patch_dir) ->
  tuple[float, float, float, float]`, `load_known_missing_tiles(path) -> set[tuple[str,
  str]]`, `default_missing_coord_details_csv() -> Path`.
- Produces (in `rasterize_forest.py`): a CLI script that writes `forest_2d.npy` (`(1, H, W)`
  uint8) + `meta.json["forest_2d"]` per patch directory, plus a `process_patch(patch_dir,
  forest_tiff_path, *, pixel_m, ignore_index, force_reload_bounds=False) -> dict` function
  importable for testing.

**Design notes carried over from the spec, updated during pre-flight review (read before
implementing):** the original design duplicated three small helpers (bounds-from-coord,
missing-tiles-CSV parsing) from `rasterize_network.py`'s private, single-use functions rather
than importing them. A pre-flight review flagged that as the kind of thing a quality reviewer
treats as a defect, so this task now **extracts** them into `network_xy_raster_utils.py`
(already the shared module both scripts import from for grid math) as public functions, and
updates `rasterize_network.py` to import them too — removing its own private copies. This is a
pure relocation (identical bodies, no behavior change) of code that currently has **no test
coverage at all**; this task adds it as part of the move.
- **Row-orientation flip is mandatory and easy to get backwards**: `rasterio` reads a
  north-up GeoTIFF with row 0 = north (top). Every other grid array in this codebase
  (`network.npy`, `NetworkRasterToPointLabels`, `network_xy_raster_utils.mask_from_absolute_
  cells`) uses a **south-up** convention (row 0 = south / minimum-y, row index increases with
  northing — see `mask_from_absolute_cells`: `mask[local_iy, local_ix] = True` where
  `local_iy = y_abs/pixel_m - iy0`). A raw `rasterio` window read must be flipped
  vertically (`np.flipud`) before being saved, or every pixel binned against it at train time
  will be north/south-mirrored — a silent, severe correctness bug, not a crash. The test in
  Step 1 below exists specifically to catch a regression here.
- The window read combines a decimated read (`out_shape` sized to the target grid) with
  `resampling=Resampling.mode` (majority vote — the target 0.5m grid is a non-integer 2.5x
  downsample of the native 0.2m source, so a manual block-reshape would not divide evenly) and
  `boundless=True` (a patch's bounding box can, in principle, extend slightly past the source
  tiff's own extent at a département boundary).

### Part A: extract the shared helpers (touches `rasterize_network.py`)

- [ ] **Step 1: Write the failing test for the shared helpers**

Create `tests/test_network_xy_raster_shared_utils.py`:

```python
"""
Tests for preprocessing utilities shared between rasterize_network.py and
rasterize_forest.py: abs_xy_bounds_from_coord, load_known_missing_tiles, and
default_missing_coord_details_csv. Extracted from rasterize_network.py
(previously private, single-use helpers with no test coverage) into
network_xy_raster_utils.py so rasterize_forest.py can reuse them without
duplication.

Run with: PYTHONPATH=./ pytest tests/test_network_xy_raster_shared_utils.py
"""

import csv
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pointcept.datasets.preprocessing.flair3d_plus.network_xy_raster_utils import (
    abs_xy_bounds_from_coord,
    default_missing_coord_details_csv,
    load_known_missing_tiles,
)


class TestAbsXyBoundsFromCoord(unittest.TestCase):
    def test_computes_bounds_with_translation(self):
        with tempfile.TemporaryDirectory() as patch_dir:
            coord = np.array(
                [[0.0, 0.0, 0.0], [10.0, 5.0, 0.0], [3.0, -2.0, 0.0]], dtype=np.float32
            )
            np.save(os.path.join(patch_dir, "coord.npy"), coord)
            np.save(
                os.path.join(patch_dir, "coord_translation.npy"),
                np.array([1000.0, 2000.0, 0.0], dtype=np.float64),
            )
            xmin, ymin, xmax, ymax = abs_xy_bounds_from_coord(Path(patch_dir))
            self.assertEqual((xmin, ymin, xmax, ymax), (1000.0, 1998.0, 1010.0, 2005.0))

    def test_missing_translation_file_raises(self):
        with tempfile.TemporaryDirectory() as patch_dir:
            np.save(
                os.path.join(patch_dir, "coord.npy"),
                np.zeros((1, 3), dtype=np.float32),
            )
            with self.assertRaises(FileNotFoundError):
                abs_xy_bounds_from_coord(Path(patch_dir))


class TestLoadKnownMissingTiles(unittest.TestCase):
    def test_none_path_returns_empty_set(self):
        self.assertEqual(load_known_missing_tiles(None), set())

    def test_reads_details_csv_reason_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "missing_coord_tiles.details.csv")
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["split", "patch_id", "reason"])
                writer.writeheader()
                writer.writerow(
                    {"split": "Train", "patch_id": "A-1", "reason": "missing_coord_file"}
                )
                writer.writerow(
                    {"split": "Train", "patch_id": "B-2", "reason": "other_reason"}
                )
            result = load_known_missing_tiles(Path(path))
            self.assertEqual(result, {("train", "A-1")})

    def test_reads_plain_text_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "missing_ply_preflight.txt")
            with open(path, "w") as f:
                f.write("# comment\nVal,C-3,some note\n")
            result = load_known_missing_tiles(Path(path))
            self.assertEqual(result, {("val", "C-3")})


class TestDefaultMissingCoordDetailsCsv(unittest.TestCase):
    def test_points_under_data_flair3d_plus(self):
        path = default_missing_coord_details_csv()
        parts = path.parts
        self.assertIn("data", parts)
        self.assertIn("flair3d_plus", parts)
        self.assertEqual(path.name, "missing_coord_tiles.details.csv")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./ pytest tests/test_network_xy_raster_shared_utils.py -v`
Expected: FAIL with `ImportError` — none of the three functions exist yet in
`network_xy_raster_utils.py`.

- [ ] **Step 3: Add the three functions to `network_xy_raster_utils.py`**

Add `import csv` to the existing import block (currently `from dataclasses import dataclass`,
`from pathlib import Path`, `import numpy as np`, `from PIL import Image`), then append these
three functions anywhere after the existing imports/constants (e.g. right after the `GridSpec`
class and `grid_from_xy_bounds`, before `xy_to_indices`):

```python
def abs_xy_bounds_from_coord(patch_dir: Path) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) in absolute Lambert meters for one patch dir.

    Reads coord.npy + coord_translation.npy (mmap'd, XY-only scan). Shared by
    rasterize_network.py and rasterize_forest.py.
    """
    patch_dir = Path(patch_dir)
    coord_path = patch_dir / "coord.npy"
    transl_path = patch_dir / "coord_translation.npy"
    if not transl_path.is_file():
        raise FileNotFoundError(f"Missing coord_translation.npy under {patch_dir}")

    coord = np.load(coord_path, mmap_mode="r")
    transl = np.load(transl_path)
    if coord.ndim != 2 or coord.shape[1] < 2:
        raise ValueError(f"Unexpected coord shape {coord.shape} in {coord_path}")
    if transl.shape[0] < 2:
        raise ValueError(f"Unexpected coord_translation shape {transl.shape}")

    x = np.asarray(coord[:, 0], dtype=np.float64) + float(transl[0])
    y = np.asarray(coord[:, 1], dtype=np.float64) + float(transl[1])
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        raise ValueError(f"No finite XY in {patch_dir}")
    x, y = x[finite], y[finite]
    return float(x.min()), float(y.min()), float(x.max()), float(y.max())


def load_known_missing_tiles(path: Path | None) -> set[tuple[str, str]]:
    """Load ``(split, patch_id)`` pairs expected to lack coord.npy.

    Accepts either ``missing_coord_tiles.details.csv`` (DictReader, reason=
    missing_coord_file) or a plain ``split,patch_id,...`` text file. Shared by
    rasterize_network.py and rasterize_forest.py.
    """
    if path is None:
        return set()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing tiles file not found: {path}")

    out: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        if "reason" in sample.splitlines()[0] if sample else False:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("reason") and row.get("reason") != "missing_coord_file":
                    continue
                split = (row.get("split") or "").strip().lower()
                patch_id = (row.get("patch_id") or "").strip()
                if split and patch_id:
                    out.add((split, patch_id))
        else:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = [p.strip() for p in stripped.split(",", 2)]
                if len(parts) < 2:
                    continue
                split, patch_id = parts[0].lower(), parts[1]
                if split and patch_id:
                    out.add((split, patch_id))
    return out


def default_missing_coord_details_csv() -> Path:
    """Repo ``data/flair3d_plus/missing_coord_tiles.details.csv`` (same as Flair3DDataset)."""
    # network_xy_raster_utils.py -> .../preprocessing/flair3d_plus -> repo root = parents[4]
    return (
        Path(__file__).resolve().parents[4]
        / "data"
        / "flair3d_plus"
        / "missing_coord_tiles.details.csv"
    )
```

- [ ] **Step 4: Update `rasterize_network.py` to use the shared functions**

In `pointcept/datasets/preprocessing/flair3d_plus/rasterize_network.py`:

1. Replace the import block (currently the `try`/`except ImportError` pair importing from
   `network_label_utils` and `network_xy_raster_utils`) with:

```python
try:
    from network_label_utils import (  # type: ignore
        NETWORK_TYPES,
        load_roi_exported_networks,
        parse_bool_flag,
    )
    from network_xy_raster_utils import (  # type: ignore
        abs_xy_bounds_from_coord,
        default_missing_coord_details_csv,
        densify_segments_to_absolute_cells,
        grid_from_xy_bounds,
        load_known_missing_tiles,
        mask_from_absolute_cells,
    )
except ImportError:  # pragma: no cover
    from pointcept.datasets.preprocessing.flair3d_plus.network_label_utils import (
        NETWORK_TYPES,
        load_roi_exported_networks,
        parse_bool_flag,
    )
    from pointcept.datasets.preprocessing.flair3d_plus.network_xy_raster_utils import (
        abs_xy_bounds_from_coord,
        default_missing_coord_details_csv,
        densify_segments_to_absolute_cells,
        grid_from_xy_bounds,
        load_known_missing_tiles,
        mask_from_absolute_cells,
    )
```

2. Delete the three now-redundant private function definitions entirely: `_default_missing_
   coord_details_csv`, `_load_known_missing_tiles`, and `_abs_xy_bounds_from_coord` (their
   bodies are now identical copies living in `network_xy_raster_utils.py`).

3. Update the three call sites to drop the leading underscore:
   - In `process_patch`: `bounds = _abs_xy_bounds_from_coord(patch_dir)` becomes
     `bounds = abs_xy_bounds_from_coord(patch_dir)`.
   - In `run`: `missing_tiles_file = _default_missing_coord_details_csv()` becomes
     `missing_tiles_file = default_missing_coord_details_csv()`, and
     `known_missing = _load_known_missing_tiles(...)` becomes
     `known_missing = load_known_missing_tiles(...)`.

No other line in `rasterize_network.py` changes — this step is a pure move, not a rewrite.

- [ ] **Step 5: Run test to verify it passes, and sanity-check `rasterize_network.py` still
  imports**

Run: `PYTHONPATH=./ pytest tests/test_network_xy_raster_shared_utils.py -v`
Expected: PASS

Run: `PYTHONPATH=./ python -c "import pointcept.datasets.preprocessing.flair3d_plus.rasterize_network"`
Expected: no error (catches a missed call-site rename or leftover reference to a deleted
private function).

- [ ] **Step 6: Commit**

```bash
git add pointcept/datasets/preprocessing/flair3d_plus/network_xy_raster_utils.py pointcept/datasets/preprocessing/flair3d_plus/rasterize_network.py tests/test_network_xy_raster_shared_utils.py
git commit -m "Extract shared preprocessing helpers out of rasterize_network.py

abs_xy_bounds_from_coord / load_known_missing_tiles /
default_missing_coord_details_csv move from private, single-use
functions in rasterize_network.py to public functions in the already-
shared network_xy_raster_utils.py, gaining test coverage for the
first time. Pure relocation -- rasterize_network.py's behavior is
unchanged. Enables rasterize_forest.py (next) to reuse them instead
of duplicating them."
```

### Part B: the new `rasterize_forest.py` script

- [ ] **Step 7: Write the failing test**

Create `tests/test_rasterize_forest.py`:

```python
"""
Tests for rasterize_forest.process_patch: reads a small synthetic GeoTIFF window,
resamples it to the target grid, and writes forest_2d.npy + meta.json.

Requires rasterio (already a dependency for Flair3D+ preprocessing). Skips cleanly
if rasterio is not importable in the current environment.

Run with: PYTHONPATH=./ pytest tests/test_rasterize_forest.py
"""

import json
import os
import tempfile
import unittest

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from pointcept.datasets.preprocessing.flair3d_plus.rasterize_forest import (
    process_patch,
)


@unittest.skipUnless(HAS_RASTERIO, "rasterio not installed")
class TestProcessPatch(unittest.TestCase):
    def _write_synthetic_tiff(self, path, xmin, ymax, pixel_m, array):
        # array is already in "north-up" (row 0 = north) orientation, as a real
        # GeoTIFF read would return it.
        transform = from_origin(xmin, ymax, pixel_m, pixel_m)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=array.dtype,
            crs="EPSG:2154",
            transform=transform,
        ) as dst:
            dst.write(array, 1)

    def test_row_orientation_is_flipped_to_south_up(self):
        # Native tiff at 0.5m (== target pixel_m, so no resampling ambiguity):
        # top-left (north-west) quadrant is forest (1), rest is non-forest (0).
        # In a north-up array, "top-left" is array[0, 0].
        native = np.zeros((4, 4), dtype=np.uint8)
        native[0, 0] = 1

        with tempfile.TemporaryDirectory() as tmp:
            tiff_path = os.path.join(tmp, "FOREST.tif")
            # xmin=0, ymax=2.0 (4 rows * 0.5m), so the grid spans x in [0,2), y in [0,2).
            self._write_synthetic_tiff(tiff_path, xmin=0.0, ymax=2.0, pixel_m=0.5, array=native)

            patch_dir = os.path.join(tmp, "patch")
            os.makedirs(patch_dir)
            coord = np.array([[0.1, 0.1, 0.0], [1.9, 1.9, 0.0]], dtype=np.float32)
            np.save(os.path.join(patch_dir, "coord.npy"), coord)
            np.save(
                os.path.join(patch_dir, "coord_translation.npy"),
                np.array([0.0, 0.0, 0.0], dtype=np.float64),
            )

            process_patch(
                patch_dir, tiff_path, pixel_m=0.5, ignore_index=2,
            )

            forest = np.load(os.path.join(patch_dir, "forest_2d.npy"))
            self.assertEqual(forest.shape, (1, 4, 4))
            # North-west corner (native[0,0]=1) is at max-y, min-x -- in the
            # south-up output grid that is the LAST row, FIRST column.
            self.assertEqual(int(forest[0, -1, 0]), 1)
            # Everywhere else should be 0.
            self.assertEqual(int(forest.sum()), 1)

            with open(os.path.join(patch_dir, "meta.json")) as f:
                meta = json.load(f)
            self.assertEqual(meta["forest_2d"]["pixel_m"], 0.5)
            self.assertEqual(meta["forest_2d"]["width"], 4)
            self.assertEqual(meta["forest_2d"]["height"], 4)
            self.assertEqual(meta["forest_2d"]["channel_order"], ["FOREST"])

    def test_resamples_non_integer_ratio_with_majority_vote(self):
        # Native 0.2m tiff, target 0.5m grid (2.5x downsample, non-integer ratio).
        # Fully-forest native tiff -> every output cell should be forest.
        native = np.ones((10, 10), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            tiff_path = os.path.join(tmp, "FOREST.tif")
            self._write_synthetic_tiff(tiff_path, xmin=0.0, ymax=2.0, pixel_m=0.2, array=native)

            patch_dir = os.path.join(tmp, "patch")
            os.makedirs(patch_dir)
            coord = np.array([[0.1, 0.1, 0.0], [1.9, 1.9, 0.0]], dtype=np.float32)
            np.save(os.path.join(patch_dir, "coord.npy"), coord)
            np.save(
                os.path.join(patch_dir, "coord_translation.npy"),
                np.array([0.0, 0.0, 0.0], dtype=np.float64),
            )

            process_patch(patch_dir, tiff_path, pixel_m=0.5, ignore_index=2)

            forest = np.load(os.path.join(patch_dir, "forest_2d.npy"))
            self.assertTrue((forest == 1).all())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 8: Run test to verify it fails**

Run: `PYTHONPATH=./ pytest tests/test_rasterize_forest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named
'pointcept.datasets.preprocessing.flair3d_plus.rasterize_forest'`

- [ ] **Step 9: Create `rasterize_forest.py`**

Create `pointcept/datasets/preprocessing/flair3d_plus/rasterize_forest.py`:

```python
"""Standalone backfill: write forest_2d.npy (1, H, W) masks per Flair3D+ tile.

Driven by the split manifest CSV (same contract as ``preprocess_flair3d_v2``):
each ``LIDARHD=True`` row must already exist under ``data_root`` with
``coord.npy``. Missing patches are hard errors (manifest is the source of
truth; disk is only checked). Known-missing tiles listed in
``missing_coord_tiles.details.csv`` are skipped.

Unlike ``rasterize_network.py`` (which rasterizes a vector graph), FOREST is
already a raster: this script reads the window of the source FOREST GeoTIFF
covering each tile's own point-cloud bounding box, resamples it (majority
vote) directly to the target ``pixel_m`` grid, and writes it out in the same
south-up ``(1, H, W)`` layout used by ``network.npy`` / ``NetworkRasterToPoint
Labels``. FOREST coverage is complete for every (dept_year, roi) couple, so
(unlike network) there is no "expected but absent" case -- every manifest
patch gets a ``forest_2d.npy``.

Example::

python pointcept/datasets/preprocessing/flair3d_plus/rasterize_forest.py \
    --data_root data/flair3d_plus \
    --source_dataset_root data/flair3d_plus/raw \
    --split_manifest_csv data/flair3d_plus/raw/scene_split_manifest_D067.csv \
    --pixel_m 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    from network_xy_raster_utils import (  # type: ignore
        abs_xy_bounds_from_coord,
        default_missing_coord_details_csv,
        grid_from_xy_bounds,
        load_known_missing_tiles,
    )
    from preprocess_flair3d_v2 import build_modality_patch_path  # type: ignore
except ImportError:  # pragma: no cover
    from pointcept.datasets.preprocessing.flair3d_plus.network_xy_raster_utils import (
        abs_xy_bounds_from_coord,
        default_missing_coord_details_csv,
        grid_from_xy_bounds,
        load_known_missing_tiles,
    )
    from pointcept.datasets.preprocessing.flair3d_plus.preprocess_flair3d_v2 import (
        build_modality_patch_path,
    )


REQUIRED_MANIFEST_COLUMNS = frozenset({"split", "dept_year", "roi", "patch_id", "LIDARHD"})


def _parse_bool_flag(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


@dataclass(frozen=True)
class ManifestPatch:
    """One LIDARHD patch listed in the split manifest."""

    split: str
    dept_year: str
    roi: str
    patch_id: str

    def patch_dir(self, data_root: Path) -> Path:
        return (
            data_root / self.split / f"{self.dept_year}_LIDARHD" / self.roi / self.patch_id
        )

    def lidar_patch_stem(self) -> str:
        return f"{self.dept_year}_LIDARHD_{self.roi}_{self.patch_id}"


def load_manifest_patches(
    split_manifest_csv: Path,
    *,
    splits: Optional[Sequence[str]] = None,
    known_missing: Optional[set] = None,
) -> Tuple[List[ManifestPatch], int]:
    """Load LIDARHD=True rows from the manifest (optionally filtered by split)."""
    if not split_manifest_csv.is_file():
        raise FileNotFoundError(f"split_manifest_csv not found: {split_manifest_csv}")

    splits_set = {s.strip().lower() for s in splits} if splits else None
    skip = known_missing or set()
    patches: List[ManifestPatch] = []
    n_skipped = 0

    with split_manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {split_manifest_csv}")
        missing_cols = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"split_manifest_csv missing columns {missing_cols}.")
        for row in reader:
            split = (row.get("split") or "").strip().lower()
            dept_year = (row.get("dept_year") or "").strip()
            roi = (row.get("roi") or "").strip()
            patch_id = (row.get("patch_id") or "").strip()
            if not split or not dept_year or not roi or not patch_id:
                continue
            if splits_set is not None and split not in splits_set:
                continue
            if not _parse_bool_flag(row.get("LIDARHD")):
                continue
            if (split, patch_id) in skip:
                n_skipped += 1
                continue
            patches.append(ManifestPatch(split, dept_year, roi, patch_id))
    return patches, n_skipped


def _read_meta(patch_dir: Path) -> dict:
    meta_path = patch_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(patch_dir: Path, meta: dict) -> None:
    with open(patch_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _load_cached_bounds(meta: dict) -> Optional[Tuple[float, float, float, float]]:
    fr = meta.get("forest_2d")
    if not isinstance(fr, dict):
        return None
    bounds = fr.get("abs_xy_bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    try:
        xmin, ymin, xmax, ymax = (float(v) for v in bounds)
    except (TypeError, ValueError):
        return None
    if not np.isfinite([xmin, ymin, xmax, ymax]).all() or xmax < xmin or ymax < ymin:
        return None
    return xmin, ymin, xmax, ymax


def process_patch(
    patch_dir,
    forest_tiff_path,
    *,
    pixel_m: float = 0.5,
    ignore_index: int = 2,
    force_reload_bounds: bool = False,
) -> dict:
    """Write forest_2d.npy and update meta.json. Returns a stats dict."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    patch_dir = Path(patch_dir)
    meta = _read_meta(patch_dir)
    bounds = None if force_reload_bounds else _load_cached_bounds(meta)
    if bounds is None:
        bounds = abs_xy_bounds_from_coord(patch_dir)
    xmin, ymin, xmax, ymax = bounds
    grid = grid_from_xy_bounds(xmin, ymin, xmax, ymax, pixel_m=pixel_m)

    with rasterio.open(str(forest_tiff_path)) as src:
        window = from_bounds(
            grid.origin_x,
            grid.origin_y,
            grid.origin_x + grid.width * grid.pixel_m,
            grid.origin_y + grid.height * grid.pixel_m,
            transform=src.transform,
        )
        raw = src.read(
            1,
            window=window,
            out_shape=(grid.height, grid.width),
            resampling=Resampling.mode,
            boundless=True,
            fill_value=ignore_index,
        )

    # rasterio reads north-up (row 0 = north/top); the Flair3D+ grid
    # convention used by network.npy / NetworkRasterToPointLabels is south-up
    # (row 0 = south, row index increases with northing) -- see
    # network_xy_raster_utils.mask_from_absolute_cells. Flip to match.
    forest = np.flipud(raw).astype(np.uint8, copy=False)
    forest = forest[np.newaxis, :, :]  # (1, H, W)

    np.save(patch_dir / "forest_2d.npy", forest)

    meta["forest_2d"] = {
        "source": "FOREST_geotiff",
        "origin_x": float(grid.origin_x),
        "origin_y": float(grid.origin_y),
        "width": int(grid.width),
        "height": int(grid.height),
        "pixel_m": float(grid.pixel_m),
        "crs": "EPSG:2154",
        "channel_order": ["FOREST"],
        "abs_xy_bounds": [xmin, ymin, xmax, ymax],
        "positive_pixel_count": int((forest == 1).sum()),
        "void_pixel_count": int((forest == ignore_index).sum()),
    }
    _write_meta(patch_dir, meta)
    return {
        "patch": str(patch_dir),
        "shape": list(forest.shape),
        "positive_pixel_count": int((forest == 1).sum()),
        "void_pixel_count": int((forest == ignore_index).sum()),
    }


def run(
    data_root: Path,
    source_dataset_root: Path,
    split_manifest_csv: Path,
    *,
    splits: Optional[List[str]] = None,
    pixel_m: float = 0.5,
    ignore_index: int = 2,
    force_reload_bounds: bool = False,
    missing_tiles_file: Optional[Path] = None,
) -> None:
    if missing_tiles_file is None:
        missing_tiles_file = default_missing_coord_details_csv()
    known_missing = load_known_missing_tiles(
        missing_tiles_file if missing_tiles_file.is_file() else None
    )
    patches, n_skipped = load_manifest_patches(
        split_manifest_csv, splits=splits, known_missing=known_missing
    )
    print(
        f"Manifest: {len(patches) + n_skipped} LIDARHD rows "
        f"({n_skipped} known-missing skipped) -> {len(patches)} to process"
    )

    n_ok = 0
    n_missing_tiff = 0
    for patch in tqdm(patches, desc="patches", unit="patch"):
        patch_dir = patch.patch_dir(data_root)
        if not (patch_dir / "coord.npy").is_file():
            raise FileNotFoundError(f"Manifest patch missing coord.npy: {patch_dir}")
        forest_tiff_path = build_modality_patch_path(
            dataset_root=str(source_dataset_root),
            modality="FOREST",
            dept_year=patch.dept_year,
            roi=patch.roi,
            lidar_patch_stem=patch.lidar_patch_stem(),
        )
        if not Path(forest_tiff_path).is_file():
            n_missing_tiff += 1
            print(f"WARNING: FOREST tiff not found, skipping: {forest_tiff_path}")
            continue
        process_patch(
            patch_dir,
            forest_tiff_path,
            pixel_m=pixel_m,
            ignore_index=ignore_index,
            force_reload_bounds=force_reload_bounds,
        )
        n_ok += 1

    print(f"Done. forest_2d.npy written for {n_ok} patches ({n_missing_tiff} missing tiffs).")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rasterize the FOREST GeoTIFF into forest_2d.npy on Flair3D+ tiles."
    )
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument(
        "--source_dataset_root",
        type=str,
        required=True,
        help="Root directory containing auxiliary modality GeoTIFFs (FOREST, etc.), "
        "same layout as preprocess_flair3d_v2's --dataset_root.",
    )
    p.add_argument("--split_manifest_csv", type=str, required=True)
    p.add_argument("--splits", type=str, nargs="*", default=None)
    p.add_argument("--pixel_m", type=float, default=0.5)
    p.add_argument("--ignore_index", type=int, default=2)
    p.add_argument("--force_reload_bounds", action="store_true")
    p.add_argument("--missing_tiles_file", type=str, default=None)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    missing = Path(args.missing_tiles_file).resolve() if args.missing_tiles_file else None
    run(
        Path(args.data_root).resolve(),
        Path(args.source_dataset_root).resolve(),
        Path(args.split_manifest_csv).resolve(),
        splits=args.splits,
        pixel_m=float(args.pixel_m),
        ignore_index=int(args.ignore_index),
        force_reload_bounds=bool(args.force_reload_bounds),
        missing_tiles_file=missing,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Run test to verify it passes**

Run: `PYTHONPATH=./ pytest tests/test_rasterize_forest.py -v`
Expected: PASS. If the `boundless=True` + `out_shape` + `resampling` combination raises a
`rasterio`/GDAL error in this environment's installed rasterio version, that will surface here
— if so, replace the `with rasterio.open(...) as src: ... src.read(...)` block in
`process_patch` with a `rasterio.vrt.WarpedVRT` sized to the target grid
(`WarpedVRT(src, width=grid.width, height=grid.height, transform=<grid's affine>,
resampling=Resampling.mode)`, then `.read(1)` from the VRT) as a fallback — but attempt the
simpler direct-read form first, since it is expected to work on any reasonably current
rasterio (the repo already pins a modern torch/cuda stack).

- [ ] **Step 11: Commit**

```bash
git add pointcept/datasets/preprocessing/flair3d_plus/rasterize_forest.py tests/test_rasterize_forest.py
git commit -m "Add rasterize_forest.py: precompute forest_2d.npy from the FOREST GeoTIFF

Standalone, additive script -- reads each tile's own point-cloud
bounding box (via the newly-shared abs_xy_bounds_from_coord),
resamples the source 0.2m FOREST tiff to a 0.5m grid via
majority-vote decimated read, and flips it to the south-up row
convention used everywhere else in this pipeline."
```

---

## Task 8: Debug multi-task config exercising `forest_2d`

**Files:**
- Create: `configs/experiment/w107/debug/multi-litept-v1m0-flair3d_forest2d_debug.py`

**Interfaces:**
- Consumes: everything from Tasks 1-6 (registry, transform, dataset loading, model pooling)
  plus `forest_2d.npy` data produced by Task 7's script for whichever tiles this config's
  `csv_manifest`/`data_root` point at.

- [ ] **Step 1: Create the debug config**

Base this on `configs/experiment/w107/7/toward_bm/multi-litept-v1m0-flair3d_2.py` (same
backbone/task composition the user is actively running), swapping the 3D `forest` task for
`forest_2d`, adding the `ExtractAbsXY`/`NetworkRasterToPointLabels` pipeline steps required for
a pixel_semantic task, and applying the repo's standard debug-speed overrides (matching
`configs/experiment/w101/7/debug/multi-spunet-v1m0-flair3d_1.py`'s `train_max_sample`/
`val_max_sample`/`total_iters`/`iter_per_epoch` pattern).

Create `configs/experiment/w107/debug/multi-litept-v1m0-flair3d_forest2d_debug.py`:

```python
"""
LitePT-Small on Flair3D+ multitask debug run: same task composition as
w107/7/toward_bm/multi-litept-v1m0-flair3d_2.py (segment v20 + forest + elevation
+ 4 nathab tile_distribution axes), except forest is swapped for its 2D
grid-pooled variant, forest_2d (mean-pooled 0.5m Lambert grid + linear head,
see docs/superpowers/specs/2026-08-09-forest-2d-task-design.md).

Debug speed overrides only (train_max_sample/val_max_sample/total_iters/
iter_per_epoch) -- everything else matches the reference config so a
successful run here validates the real multi-task wiring, not a simplified one.

Prerequisite: forest_2d.npy must already exist under each tile's scene dir --
run pointcept/datasets/preprocessing/flair3d_plus/rasterize_forest.py first.
"""

# -----------------------------------------------------------------------------
# Default
# -----------------------------------------------------------------------------
_base_ = ["../../../_base_/default_runtime.py"]

# -----------------------------------------------------------------------------
# Run-level settings
# -----------------------------------------------------------------------------

# Logging parameters
grp_exp = 1
num_exp = 1

log_task_gradient_norms = False
grad_norm_lite = True
grad_norm_lite_interval = 100
grad_norm_lite_ema_alpha = 0.1
grad_norm_lite_eps = 1e-3

# Hardware parameters
num_gpu = 1
num_worker = 8 * num_gpu
enable_amp = True

# Data parameters
batch_size = 20 * num_gpu  # total batch size across all gpus
batch_size_val = 8 * num_gpu
batch_size_test = 8 * num_gpu
test_voxel_budget = 2_000_000
val_voxel_budget = 2_000_000

grid_size = 0.1
point_max = 102400
mix_prob = 0.8

patch_size = 1024

# Debug-speed overrides.
train_max_sample = 20
val_max_sample = 100
test_max_sample = val_max_sample

# Optimization parameters
lr = 1e-3
total_iters = 15
iter_per_epoch = 5

# Features
learned_masked_feat = True
feat_keys = ["coord", "color", "strength"]
coord_feat_scale = 0.01

# Backbone pooling stride (encoder stages)
stride = (2, 2, 2, 2)

# Wandb parameters
wandb_run_name = (
    f"Flair3D+ LitePT-S multi debug forest_2d {grp_exp}.{num_exp} "
    f"stride={stride} batch_size={batch_size} lr={lr}"
)
wandb_project = "flair3d_multi"

# -----------------------------------------------------------------------------
# Multitask configuration : targets configuration
# -----------------------------------------------------------------------------
from pointcept.datasets.flair3d_config_utils import (
    FLAIR3D_TILE_DISTRIBUTION_TASKS,
    init_task_configs,
    init_task_criteria,
    FLAIR3D_COLLECT_PREFIX_LITEPT,
    init_multitask_collect_keys,
)

main_task = "segment"
nathab_keys = tuple(FLAIR3D_TILE_DISTRIBUTION_TASKS.keys())
target_keys = (main_task, "forest_2d", "elevation") + nathab_keys
# natural_habitat is loader-only (remap source), not a supervised task.
dataset_target_keys = ("natural_habitat",) + target_keys

grad_norm_lite_task_groups = {task_name: "nathab" for task_name in nathab_keys}

nathab_axis_remaps = dict(
    nathab_habitat_type=("natural_habitat", "by_habitat_type_ecological"),
    nathab_moisture_regime=("natural_habitat", "by_moisture_regime"),
    nathab_soil_chemistry=("natural_habitat", "by_soil_chemistry"),
    nathab_bioclimatic_zone=("natural_habitat", "by_climatic_domain"),
)
nathab_axis_storage_definitions = dict(natural_habitat="default")
nathab_axis_remap = dict(
    type="Flair3DLabelRemap",
    remaps=nathab_axis_remaps,
    storage_definitions=nathab_axis_storage_definitions,
)

target_scales = {}

label_definitions = dict(
    segment="v20",
)

task_configs = init_task_configs(target_keys, definitions=label_definitions)
task_criteria = init_task_criteria(task_configs)
task_criteria["elevation"] = [
    dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
]
task_weights = {task_name: 1.0 for task_name in task_configs.keys()}
task_weights["elevation"] = 0.01

del (
    init_task_configs,
    init_task_criteria,
    FLAIR3D_TILE_DISTRIBUTION_TASKS,
)

num_classes = task_configs[main_task]["num_classes"]
ignore_index = task_configs[main_task]["ignore_index"]
names = task_configs[main_task]["names"]

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=100),
    dict(type="MultiTaskEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

test_single_fragment = True
test = dict(type="MultiTaskTester", verbose=True, write_cls_iou=True)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = dict(
    type="MultiTaskSegmentorV2",
    backbone_out_channels=72,
    backbone=dict(
        type="LitePT-v1",
        in_channels=7,  # coord (3) + color (3) + strength (1)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=stride,
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(patch_size, patch_size, patch_size, patch_size, patch_size),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(patch_size, patch_size, patch_size, patch_size),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=False,
    ),
    feature_mask_values=dict(
        enable=learned_masked_feat,
        masked_feat_keys=["color", "strength"],
    ),
    task_configs=task_configs,
    main_task=main_task,
    task_criteria=task_criteria,
    task_weights=task_weights,
)

# -----------------------------------------------------------------------------
# Optimizer / scheduler
# -----------------------------------------------------------------------------
optimizer = dict(type="AdamW", lr=lr, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[lr, lr / 10],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=lr / 10)]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset_type = "Flair3DDataset"
data_root = "data/flair3d_plus"
csv_manifest = "data/flair3d_plus/raw/scene_split_manifest.csv"
missing_tiles_manifest = "data/flair3d_plus/missing_ply_preflight.txt"
too_small_tiles_manifest = "data/flair3d_plus/too_small_tiles.csv"

train_multitask_keys, val_multitask_keys, multitask_index_valid_keys = (
    init_multitask_collect_keys(
        target_keys, collect_prefix_keys=FLAIR3D_COLLECT_PREFIX_LITEPT
    )
)

del FLAIR3D_COLLECT_PREFIX_LITEPT, init_multitask_collect_keys

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=names,
    target_scales=target_scales,
    task_configs=task_configs,
    main_task=main_task,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        csv_manifest=csv_manifest,
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
        target_keys=list(dataset_target_keys),
        primary_target_key=main_task,
        max_sample=train_max_sample,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(multitask_index_valid_keys)},
            ),
            dict(type="ExtractAbsXY"),
            nathab_axis_remap,
            dict(type="CenterShift", apply_z=True),
            dict(type="Z_MinShift"),
            dict(type="Z_RandomOffset"),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=point_max, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="RandomDropColor", drop_ratio=1.0, drop_application_ratio=0.2, keep_mask=True),
            dict(type="RandomDropColor", drop_ratio=0.1, drop_application_ratio=0.5, keep_mask=True),
            dict(type="RandomDropStrength", drop_ratio=1.0, drop_application_ratio=0.2, keep_mask=True),
            dict(type="RandomDropStrength", drop_ratio=0.1, drop_application_ratio=0.5, keep_mask=True),
            dict(type="NetworkRasterToPointLabels", target_key="forest_2d"),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="Collect",
                keys=train_multitask_keys,
                feat_keys=feat_keys,
                feat_scales=dict(coord=coord_feat_scale),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        csv_manifest=csv_manifest,
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
        target_keys=list(dataset_target_keys),
        primary_target_key=main_task,
        max_sample=val_max_sample,
        transform=[
            dict(
                type="Update",
                keys_dict={"index_valid_keys": list(multitask_index_valid_keys)},
            ),
            dict(type="ExtractAbsXY"),
            nathab_axis_remap,
            dict(type="CenterShift", apply_z=True),
            dict(type="Z_MinShift"),
            dict(
                type="Copy",
                keys_dict={t: f"origin_{t}" for t in target_keys},
            ),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="NetworkRasterToPointLabels", target_key="forest_2d"),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="Collect",
                keys=val_multitask_keys,
                feat_keys=feat_keys,
                feat_scales=dict(coord=coord_feat_scale),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        csv_manifest=csv_manifest,
        missing_tiles_manifest=missing_tiles_manifest,
        too_small_tiles_manifest=too_small_tiles_manifest,
        target_keys=list(dataset_target_keys),
        primary_target_key=main_task,
        max_sample=test_max_sample,
        transform=[
            dict(type="ExtractAbsXY"),
            nathab_axis_remap,
            dict(type="CenterShift", apply_z=True),
            dict(type="Z_MinShift"),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                test_single_fragment=test_single_fragment,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="NetworkRasterToPointLabels", target_key="forest_2d"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=(
                        "coord",
                        "grid_coord",
                        "index",
                        "forest_2d",
                        "forest_2d_cell",
                        "forest_2d_pix",
                        "forest_2d_origin_x",
                        "forest_2d_origin_y",
                        "forest_2d_pixel_m",
                        "forest_2d_height",
                        "forest_2d_width",
                    ),
                    optional_keys=("inverse",),
                    feat_keys=feat_keys,
                    feat_scales=dict(coord=coord_feat_scale),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ],
        ),
    ),
)
```

- [ ] **Step 2: Verify the config parses**

Run:
```bash
PYTHONPATH=./ python -c "
from pointcept.utils.config import Config
cfg = Config.fromfile('configs/experiment/w107/debug/multi-litept-v1m0-flair3d_forest2d_debug.py')
assert 'forest_2d' in cfg.data.task_configs
print('OK:', sorted(cfg.data.task_configs.keys()))
"
```
Expected: prints `OK: ['elevation', 'forest_2d', 'nathab_bioclimatic_zone',
'nathab_habitat_type', 'nathab_moisture_regime', 'nathab_soil_chemistry', 'segment']` with no
traceback. This confirms the config is syntactically valid and `init_task_configs` resolves
`forest_2d` through the registry entry from Task 1 — it does **not** confirm an actual training
run works end-to-end, since that additionally requires real preprocessed tiles with
`forest_2d.npy` already written by Task 7's script.

- [ ] **Step 3: Commit**

```bash
git add configs/experiment/w107/debug/multi-litept-v1m0-flair3d_forest2d_debug.py
git commit -m "Add debug multi-task config exercising forest_2d

Mirrors w107/7/toward_bm/multi-litept-v1m0-flair3d_2.py (same
backbone/task composition) with forest swapped for forest_2d and the
repo's standard debug-speed overrides. Requires forest_2d.npy to
already exist on disk (run rasterize_forest.py first)."
```

---

## Final verification (after all 8 tasks)

- [ ] Run the full new-test surface together: `PYTHONPATH=./ pytest tests/test_pixel_semantic_collect_keys.py tests/test_dilated_prf_opt_out.py tests/test_pixel_semantic_raster_to_points.py tests/test_pixel_semantic_dataset_loading.py tests/test_pixel_semantic_pooling.py tests/test_pixel_semantic_test_prf.py tests/test_network_xy_raster_shared_utils.py tests/test_rasterize_forest.py -v`
  Expected: all PASS.
- [ ] Run the pre-existing regression suite to confirm nothing else broke:
  `PYTHONPATH=./ pytest tests/ -v`
  Expected: all PASS (in particular `tests/test_tile_distribution_pooling.py`, which imports
  `MultiTaskSegmentorV2` from the file touched in Task 5).
- [ ] Confirm no existing config referencing `network`/`NetworkRasterToPointLabels` needs any
  edit: `grep -rl "NetworkRasterToPointLabels" configs/ | wc -l` should still report `14`
  (same count as before this plan — none of them needed a `target_key=` argument added, since
  the default preserves their behavior).
- [ ] Report back to the user: what remains **operational** (not code) before they can actually
  train with `forest_2d` — running `rasterize_forest.py` against their real
  `--data_root`/`--source_dataset_root`/manifest on whichever tiles the debug config's
  `csv_manifest` selects, since that data does not exist yet anywhere.
