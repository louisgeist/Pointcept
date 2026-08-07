#!/usr/bin/env python3
"""
Compute per-class label distributions on Flair3D+ train (or other splits).

Uses the same scene list as Flair3DDataset (CSV manifest, LIDARHD=True, excluded tiles).
Aggregates segment, forest, land_use, and natural_habitat (point-level) plus
natural_habitat_multilabel (scene-level multi-hot presence) in one pass per scene.

Also emits point-level histograms for the four nathab tile-distribution axes
(FLAIR3D_TILE_DISTRIBUTION_TASKS), fan-out remapped from natural_habitat.npy with the
same LUTs as Flair3DLabelRemap in training (storage natural_habitat=default required).

Label definitions match training: on-disk storage defs (e.g. natural_habitat=default from
preprocess) can be remapped on the fly to a target def (e.g. by_moisture_v3) via the same
LUT logic as Flair3DLabelRemap. Axis histograms always use storage natural_habitat and the
fixed axis definitions; they do not follow --label_definitions for natural_habitat.

natural_habitat_multilabel.npy is counted as scene presence over its 15 binary labels
(no remap): percent = scenes_with_label / scenes_with_file.

With multiple splits (e.g. train,val,test), one pass writes per-split subdirs plus
an aggregated ``all/`` total under ``--output_dir``. A single split keeps a flat
layout in ``--output_dir`` (backward compatible).

Example (launch on A100 -> way faster):
python scripts/count_flair3d_train_label_distribution.py \
--data_root data/flair3d_plus \
--csv_manifest data/flair3d_plus/raw/scene_split_manifest.csv \
--split train,val,test \
--missing_tiles_manifest data/flair3d_plus/missing_ply_preflight.txt \
--too_small_tiles_manifest data/flair3d_plus/too_small_tiles.csv \
--storage_definitions natural_habitat=default \
--num_workers 24 \
--output_dir stats/flair3d/label_distribution
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_flair3d_config_utils():
    path = os.path.join(REPO_ROOT, "pointcept", "datasets", "flair3d_config_utils.py")
    spec = importlib.util.spec_from_file_location("flair3d_config_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load flair3d_config_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_flair3d_cfg = _load_flair3d_config_utils()
FLAIR3D_SEMANTIC_TARGET_KEYS = _flair3d_cfg.FLAIR3D_SEMANTIC_TARGET_KEYS
FLAIR3D_TILE_DISTRIBUTION_TASKS = _flair3d_cfg.FLAIR3D_TILE_DISTRIBUTION_TASKS
get_missing_target_fill_value = _flair3d_cfg.get_missing_target_fill_value
get_semantic_config = _flair3d_cfg.get_semantic_config
get_multilabel_classification_config = _flair3d_cfg.get_multilabel_classification_config

SEMANTIC_TASKS = tuple(FLAIR3D_SEMANTIC_TARGET_KEYS)
NATHAB_AXIS_TASKS = tuple(FLAIR3D_TILE_DISTRIBUTION_TASKS.keys())
NATHAB_AXIS_SOURCE_TASK = "natural_habitat"

MULTILABEL_TASK = "natural_habitat_multilabel"
_MULTILABEL_CFG = get_multilabel_classification_config(MULTILABEL_TASK)
MULTILABEL_CLASS_NAMES: Tuple[str, ...] = tuple(_MULTILABEL_CFG["names"])
NUM_MULTILABEL_CLASSES = int(_MULTILABEL_CFG["num_classes"])
MULTILABEL_FILENAME = f"{MULTILABEL_TASK}.npy"

# Set in main() before workers start; read by _process_scene in child processes.
_TASK_REMAP_STATE: Dict[str, "TaskRemapState"] = {}
_NATHAB_AXIS_REMAP_STATE: Dict[str, "TaskRemapState"] = {}


def _load_label_remap_module():
    module_name = "flair3d_label_remap_count_script"
    path = os.path.join(
        REPO_ROOT,
        "pointcept",
        "datasets",
        "preprocessing",
        "flair3d_plus",
        "flair3d_label_remap.py",
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load flair3d_label_remap from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class TaskRemapState:
    registry_task: str
    target_definition: str
    storage_definition: str
    stored_to_train_lut: np.ndarray | None
    missing_fill_after_remap: int
    num_classes: int
    ignore_index: int | None


def parse_definition_mapping(spec: str, *, arg_name: str) -> Dict[str, str]:
    if not spec.strip():
        return {}
    result: Dict[str, str] = {}
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"{arg_name}: expected task=definition, got {token!r}")
        task_key, definition = token.split("=", 1)
        task_key = task_key.strip()
        definition = definition.strip()
        if task_key not in SEMANTIC_TASKS:
            known = ", ".join(SEMANTIC_TASKS)
            raise ValueError(f"{arg_name}: unknown task {task_key!r}. Expected one of: {known}")
        if not definition:
            raise ValueError(f"{arg_name}: empty definition for task {task_key!r}")
        result[task_key] = definition
    return result


def _make_remap_state(
    label_remap: Any,
    *,
    registry_task: str,
    target_name: str,
    storage_name: str,
) -> TaskRemapState:
    get_definition = label_remap.get_definition
    build_stored_to_train_lut = label_remap.build_stored_to_train_lut
    target_def = get_definition(registry_task, target_name)
    stored_to_train_lut = None
    if storage_name != target_name:
        storage_def = get_definition(registry_task, storage_name)
        stored_to_train_lut = build_stored_to_train_lut(storage_def, target_def)
    cfg = label_remap.definition_to_task_config(target_def)
    return TaskRemapState(
        registry_task=registry_task,
        target_definition=target_name,
        storage_definition=storage_name,
        stored_to_train_lut=stored_to_train_lut,
        missing_fill_after_remap=int(target_def.lut[target_def.missing_fill_raw_id]),
        num_classes=int(cfg["num_classes"]),
        ignore_index=cfg["ignore_index"],
    )


def _default_storage_definition_name(task: str, get_default_definition_name) -> str:
    """Default *on-disk storage* definition name for a task -- NOT the same as its default
    *training-target* definition (get_default_definition_name), which happen to coincide
    for segment/forest/land_use but NOT for natural_habitat: preprocessing always writes
    natural_habitat.npy in the raw/finest "default" CarHab taxonomy (43 raw classes) so
    that many different targets (by_habitat_x_domain, the 4 nathab axes, ...) can all be
    fanned out from the same on-disk file on the fly -- but
    get_default_definition_name("natural_habitat") returns "by_habitat_x_domain" (the
    default *target*), not "default" (the actual storage format). Using it as the storage
    fallback silently collapses the LUT (6 raw ids -> 1 stored id, last-write-wins) instead
    of raising, corrupting every downstream count without any error. Every real training
    config already special-cases this explicitly (storage_definitions=dict(natural_habitat=
    "default")); this mirrors that here so the flag becomes optional instead of silently
    required. See also get_missing_target_fill_value's docstring in flair3d_config_utils.py
    for the same storage-vs-target distinction, hit and fixed there previously.
    """
    if task == "natural_habitat":
        return "default"
    return get_default_definition_name(task)


def build_task_remap_state(
    target_definitions: Dict[str, str],
    storage_definitions: Dict[str, str],
) -> Dict[str, TaskRemapState]:
    """Mirror Flair3DLabelRemap: storage on-disk defs -> target training defs."""
    label_remap = _load_label_remap_module()
    get_default_definition_name = label_remap.get_default_definition_name

    state: Dict[str, TaskRemapState] = {}
    for task in SEMANTIC_TASKS:
        target_name = target_definitions.get(task, get_default_definition_name(task))
        storage_name = storage_definitions.get(
            task, _default_storage_definition_name(task, get_default_definition_name)
        )
        state[task] = _make_remap_state(
            label_remap,
            registry_task=task,
            target_name=target_name,
            storage_name=storage_name,
        )
    return state


def build_nathab_axis_remap_state(
    storage_definitions: Dict[str, str],
) -> Dict[str, TaskRemapState]:
    """Fan-out remaps for nathab tile-distribution axes (from natural_habitat storage)."""
    label_remap = _load_label_remap_module()
    get_default_definition_name = label_remap.get_default_definition_name
    storage_name = storage_definitions.get(
        NATHAB_AXIS_SOURCE_TASK,
        _default_storage_definition_name(NATHAB_AXIS_SOURCE_TASK, get_default_definition_name),
    )

    state: Dict[str, TaskRemapState] = {}
    for axis, target_name in FLAIR3D_TILE_DISTRIBUTION_TASKS.items():
        state[axis] = _make_remap_state(
            label_remap,
            registry_task=NATHAB_AXIS_SOURCE_TASK,
            target_name=target_name,
            storage_name=storage_name,
        )
    return state


def _remap_state_for(task: str) -> TaskRemapState:
    if task in _TASK_REMAP_STATE:
        return _TASK_REMAP_STATE[task]
    return _NATHAB_AXIS_REMAP_STATE[task]


@dataclass(frozen=True)
class SceneRecord:
    split: str
    patch_id: str
    scene_path: str


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def parse_splits(splits_arg: str) -> set[str]:
    return {token.strip() for token in splits_arg.split(",") if token.strip()}


def parse_manifest_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def build_scene_path(data_root: str, split: str, patch_id: str, dept_year: str, roi: str) -> str:
    return os.path.join(data_root, split, f"{dept_year}_LIDARHD", roi, patch_id)


def load_hardcoded_excluded_tiles() -> set[tuple[str, str]]:
    """Flair3DDataset.get_hardcoded_excluded_tiles() (CORRUPTED_TILES + missing coord CSV)."""
    excluded = set()
    details_csv = os.path.join(REPO_ROOT, "data", "flair3d_plus", "missing_coord_tiles.details.csv")
    if not os.path.exists(details_csv):
        return excluded
    with open(details_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("reason") != "missing_coord_file":
                continue
            split = (row.get("split") or "").strip()
            patch_id = (row.get("patch_id") or "").strip()
            if split and patch_id:
                excluded.add((split, patch_id))
    return excluded


def load_missing_tiles_manifest(path: str | None) -> set[tuple[str, str]]:
    missing_tiles: set[tuple[str, str]] = set()
    if not path:
        return missing_tiles
    if not os.path.isfile(path):
        print(f"Warning: missing tiles manifest not found: {path}")
        return missing_tiles
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [part.strip() for part in stripped.split(",", 2)]
            if len(parts) < 2:
                continue
            split, patch_id = parts[0], parts[1]
            if split and patch_id:
                missing_tiles.add((split, patch_id))
    return missing_tiles


def load_too_small_tiles_manifest(path: str | None, *, train_only: bool) -> set[tuple[str, str]]:
    too_small_tiles: set[tuple[str, str]] = set()
    if not path:
        return too_small_tiles
    if not os.path.isfile(path):
        print(f"Warning: too-small tiles manifest not found: {path}")
        return too_small_tiles
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_split = (row.get("split") or "").strip()
            patch_id = (row.get("patch_id") or "").strip()
            if not row_split or not patch_id:
                continue
            if train_only and row_split != "train":
                continue
            too_small_tiles.add((row_split, patch_id))
    return too_small_tiles


def load_scene_records(
    data_root: str,
    csv_manifest: str,
    target_splits: set[str],
    excluded_tiles: set[tuple[str, str]],
) -> list[SceneRecord]:
    scene_records: list[SceneRecord] = []
    with open(csv_manifest, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "patch_id", "LIDARHD"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise KeyError(f"Missing required columns in manifest: {sorted(missing_cols)}")

        for row in reader:
            split = str(row["split"]).strip()
            patch_id = str(row["patch_id"]).strip()
            if not split or not patch_id:
                continue
            if split not in target_splits:
                continue
            if not parse_manifest_bool(row.get("LIDARHD")):
                continue
            if (split, patch_id) in excluded_tiles:
                continue

            dept_year = (row.get("dept_year") or "").strip() or patch_id.split("_", 2)[0]
            roi = (row.get("roi") or "").strip() or patch_id.split("_", 2)[1]
            scene_path = build_scene_path(data_root, split, patch_id, dept_year, roi)
            scene_records.append(SceneRecord(split=split, patch_id=patch_id, scene_path=scene_path))
    return scene_records


def _count_array_length(task_state: TaskRemapState) -> int:
    num_classes = task_state.num_classes
    ignore_index = task_state.ignore_index
    void_bin = num_classes
    if ignore_index is not None and ignore_index >= 0:
        void_bin = max(void_bin, int(ignore_index))
    return void_bin + 2


def _init_task_counts() -> Dict[str, np.ndarray]:
    return {
        task: np.zeros(_count_array_length(_TASK_REMAP_STATE[task]), dtype=np.int64)
        for task in SEMANTIC_TASKS
    }


def _init_nathab_axis_counts() -> Dict[str, np.ndarray]:
    return {
        axis: np.zeros(
            _count_array_length(_NATHAB_AXIS_REMAP_STATE[axis]), dtype=np.int64
        )
        for axis in NATHAB_AXIS_TASKS
    }


def _apply_stored_to_train(labels: np.ndarray, task: str) -> np.ndarray:
    task_state = _remap_state_for(task)
    labels = labels.reshape(-1)
    lut = task_state.stored_to_train_lut
    if lut is None:
        return labels.astype(np.int64, copy=False)
    idx = labels.astype(np.int64, copy=False)
    remapped = np.full(idx.shape, task_state.missing_fill_after_remap, dtype=np.int32)
    valid = (idx >= 0) & (idx < lut.shape[0])
    if np.any(valid):
        remapped[valid] = lut[idx[valid]]
    return remapped


def _accumulate_labels(
    counts: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    ignore_index: int | None,
) -> None:
    labels = labels.reshape(-1).astype(np.int64)
    void_bin = num_classes
    if ignore_index is not None and ignore_index >= 0:
        void_bin = max(void_bin, int(ignore_index))

    mapped = labels.copy()
    if ignore_index is not None:
        mapped[labels == ignore_index] = void_bin

    in_range = (labels >= 0) & (labels < num_classes)
    if in_range.any():
        bc = np.bincount(labels[in_range], minlength=num_classes)
        counts[:num_classes] += bc

    if ignore_index is not None:
        counts[void_bin] += int((labels == ignore_index).sum())

    other = ~(in_range | ((ignore_index is not None) & (labels == ignore_index)))
    if other.any():
        counts[void_bin + 1] += int(other.sum())


def _load_task_labels(scene_path: str, task: str, n_points: int) -> np.ndarray | None:
    path = os.path.join(scene_path, f"{task}.npy")
    if os.path.isfile(path):
        labels = np.load(path)
        return labels.reshape(-1)
    if task == "segment":
        return None
    fill = get_missing_target_fill_value(task)
    return np.full(n_points, fill, dtype=np.int32)


def _empty_multilabel_counts() -> np.ndarray:
    return np.zeros(NUM_MULTILABEL_CLASSES, dtype=np.int64)


def _load_multilabel_vector(
    scene_path: str,
) -> tuple[np.ndarray | None, str | None]:
    """Return (binary vector, error). Missing file -> (None, None)."""
    path = os.path.join(scene_path, MULTILABEL_FILENAME)
    if not os.path.isfile(path):
        return None, None
    vector = np.load(path).reshape(-1)
    if vector.shape[0] != NUM_MULTILABEL_CLASSES:
        return (
            None,
            f"{MULTILABEL_TASK} length {vector.shape[0]} != "
            f"{NUM_MULTILABEL_CLASSES}: {scene_path}",
        )
    return (vector > 0).astype(np.int64), None


def _process_scene(
    task: tuple[SceneRecord, bool],
) -> tuple[
    str,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    np.ndarray,
    Dict[str, int],
    str | None,
]:
    scene, use_fill_for_missing = task
    split = scene.split
    partial = _init_task_counts()
    axis_partial = _init_nathab_axis_counts()
    ml_counts = _empty_multilabel_counts()
    meta = {f"{t}_from_file": 0 for t in SEMANTIC_TASKS}
    meta[f"{MULTILABEL_TASK}_from_file"] = 0
    meta[f"{MULTILABEL_TASK}_all_zero"] = 0
    meta[f"{MULTILABEL_TASK}_missing"] = 0
    # Axes share natural_habitat.npy; mirror source-file presence for summary.
    for axis in NATHAB_AXIS_TASKS:
        meta[f"{axis}_from_file"] = 0

    coord_path = os.path.join(scene.scene_path, "coord.npy")
    if not os.path.isfile(coord_path):
        return (
            split,
            partial,
            axis_partial,
            ml_counts,
            meta,
            f"missing coord.npy: {scene.scene_path}",
        )

    n_points = int(np.load(coord_path, mmap_mode="r").shape[0])
    stored_natural_habitat: np.ndarray | None = None

    for task_name in SEMANTIC_TASKS:
        file_path = os.path.join(scene.scene_path, f"{task_name}.npy")
        has_file = os.path.isfile(file_path)
        if not has_file:
            if task_name == "segment":
                return (
                    split,
                    partial,
                    axis_partial,
                    ml_counts,
                    meta,
                    f"missing {task_name}.npy: {scene.scene_path}",
                )
            if not use_fill_for_missing:
                continue

        labels = _load_task_labels(scene.scene_path, task_name, n_points)
        if labels is None:
            return (
                split,
                partial,
                axis_partial,
                ml_counts,
                meta,
                f"missing {task_name}.npy: {scene.scene_path}",
            )
        if labels.shape[0] != n_points:
            return (
                split,
                partial,
                axis_partial,
                ml_counts,
                meta,
                f"{task_name} length {labels.shape[0]} != coord {n_points}: "
                f"{scene.scene_path}",
            )

        if has_file:
            meta[f"{task_name}_from_file"] = 1

        # Keep storage ids for nathab axis fan-out (independent of NH target def).
        if task_name == NATHAB_AXIS_SOURCE_TASK:
            stored_natural_habitat = labels

        remapped = _apply_stored_to_train(labels, task_name)
        task_state = _TASK_REMAP_STATE[task_name]
        _accumulate_labels(
            partial[task_name],
            remapped,
            num_classes=task_state.num_classes,
            ignore_index=task_state.ignore_index,
        )

    if stored_natural_habitat is not None:
        for axis in NATHAB_AXIS_TASKS:
            axis_labels = _apply_stored_to_train(stored_natural_habitat, axis)
            axis_state = _NATHAB_AXIS_REMAP_STATE[axis]
            _accumulate_labels(
                axis_partial[axis],
                axis_labels,
                num_classes=axis_state.num_classes,
                ignore_index=axis_state.ignore_index,
            )
            meta[f"{axis}_from_file"] = meta[f"{NATHAB_AXIS_SOURCE_TASK}_from_file"]

    ml_vector, ml_error = _load_multilabel_vector(scene.scene_path)
    if ml_error is not None:
        return split, partial, axis_partial, ml_counts, meta, ml_error
    if ml_vector is None:
        meta[f"{MULTILABEL_TASK}_missing"] = 1
    else:
        meta[f"{MULTILABEL_TASK}_from_file"] = 1
        ml_counts += ml_vector
        if not np.any(ml_vector):
            meta[f"{MULTILABEL_TASK}_all_zero"] = 1

    return split, partial, axis_partial, ml_counts, meta, None


def _merge_counts(total: Dict[str, np.ndarray], partial: Dict[str, np.ndarray]) -> None:
    for task_name, counts in partial.items():
        total[task_name] += counts


@dataclass
class SplitAccumulator:
    """Per-split (or all-splits) running totals."""

    semantic_counts: Dict[str, np.ndarray]
    axis_counts: Dict[str, np.ndarray]
    multilabel_counts: np.ndarray
    scenes_with_file: Dict[str, int]
    axis_scenes_with_file: Dict[str, int]
    multilabel_scenes_with_file: int = 0
    multilabel_scenes_all_zero: int = 0
    multilabel_scenes_missing: int = 0
    processed: int = 0
    errors: List[str] | None = None

    @classmethod
    def create(cls) -> "SplitAccumulator":
        return cls(
            semantic_counts=_init_task_counts(),
            axis_counts=_init_nathab_axis_counts(),
            multilabel_counts=_empty_multilabel_counts(),
            scenes_with_file={task: 0 for task in SEMANTIC_TASKS},
            axis_scenes_with_file={axis: 0 for axis in NATHAB_AXIS_TASKS},
            errors=[],
        )

    def add_success(
        self,
        partial: Dict[str, np.ndarray],
        axis_partial: Dict[str, np.ndarray],
        ml_partial: np.ndarray,
        meta: Dict[str, int],
    ) -> None:
        _merge_counts(self.semantic_counts, partial)
        _merge_counts(self.axis_counts, axis_partial)
        self.multilabel_counts += ml_partial
        for task_name in SEMANTIC_TASKS:
            self.scenes_with_file[task_name] += meta[f"{task_name}_from_file"]
        for axis in NATHAB_AXIS_TASKS:
            self.axis_scenes_with_file[axis] += meta[f"{axis}_from_file"]
        self.multilabel_scenes_with_file += meta[f"{MULTILABEL_TASK}_from_file"]
        self.multilabel_scenes_all_zero += meta[f"{MULTILABEL_TASK}_all_zero"]
        self.multilabel_scenes_missing += meta[f"{MULTILABEL_TASK}_missing"]
        self.processed += 1

    def add_error(self, err: str) -> None:
        assert self.errors is not None
        self.errors.append(err)


def _format_count(n: int) -> str:
    if n < 5000:
        return f"{n:,}"
    return f"{n / 1e6:.2f} M"


@dataclass(frozen=True)
class DistributionRow:
    class_id: int | None
    class_name: str
    bucket: str
    count: int
    percent: float


@dataclass(frozen=True)
class TaskDistribution:
    task: str
    scenes_with_file: int
    scenes_total: int
    total_points: int
    rows: Tuple[DistributionRow, ...]


def _build_task_distribution(
    task: str,
    counts: np.ndarray,
    scenes_with_file: int,
    scenes_total: int,
) -> TaskDistribution:
    task_state = _remap_state_for(task)
    label_remap = _load_label_remap_module()
    cfg = label_remap.definition_to_task_config(
        label_remap.get_definition(
            task_state.registry_task, task_state.target_definition
        )
    )
    class_names = cfg["names"]
    num_classes = int(cfg["num_classes"])
    ignore_index = cfg["ignore_index"]

    void_bin = num_classes
    if ignore_index is not None and ignore_index >= 0:
        void_bin = max(void_bin, int(ignore_index))

    counted = int(counts.sum())
    denom = max(counted, 1)
    rows: List[DistributionRow] = []

    for i in range(num_classes):
        n = int(counts[i])
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        rows.append(
            DistributionRow(
                class_id=i,
                class_name=name,
                bucket="class",
                count=n,
                percent=100.0 * n / denom,
            )
        )

    void_count = int(counts[void_bin])
    if void_count > 0:
        void_name = "ignore/void"
        if ignore_index is not None and ignore_index < len(class_names):
            void_name = class_names[ignore_index]
        elif ignore_index is not None:
            void_name = f"ignore_index={ignore_index}"
        rows.append(
            DistributionRow(
                class_id=ignore_index,
                class_name=void_name,
                bucket="ignore",
                count=void_count,
                percent=100.0 * void_count / denom,
            )
        )

    other = int(counts[void_bin + 1]) if counts.size > void_bin + 1 else 0
    if other > 0:
        rows.append(
            DistributionRow(
                class_id=None,
                class_name="out-of-range",
                bucket="out_of_range",
                count=other,
                percent=100.0 * other / denom,
            )
        )

    return TaskDistribution(
        task=task,
        scenes_with_file=scenes_with_file,
        scenes_total=scenes_total,
        total_points=counted,
        rows=tuple(rows),
    )


def _build_multilabel_distribution(
    counts: np.ndarray,
    *,
    scenes_with_file: int,
    scenes_total: int,
    scenes_all_zero: int,
    scenes_missing: int,
) -> TaskDistribution:
    """Scene-level multi-hot presence histogram for natural_habitat_multilabel."""
    denom = max(scenes_with_file, 1)
    rows: List[DistributionRow] = []
    for i, name in enumerate(MULTILABEL_CLASS_NAMES):
        n = int(counts[i]) if i < counts.size else 0
        rows.append(
            DistributionRow(
                class_id=i,
                class_name=name,
                bucket="class",
                count=n,
                percent=100.0 * n / denom,
            )
        )
    if scenes_all_zero > 0:
        rows.append(
            DistributionRow(
                class_id=None,
                class_name="all_zero",
                bucket="all_zero",
                count=scenes_all_zero,
                percent=100.0 * scenes_all_zero / denom,
            )
        )
    if scenes_missing > 0:
        missing_denom = max(scenes_total, 1)
        rows.append(
            DistributionRow(
                class_id=None,
                class_name="missing_file",
                bucket="missing",
                count=scenes_missing,
                percent=100.0 * scenes_missing / missing_denom,
            )
        )
    return TaskDistribution(
        task=MULTILABEL_TASK,
        scenes_with_file=scenes_with_file,
        scenes_total=scenes_total,
        total_points=scenes_with_file,
        rows=tuple(rows),
    )


def _total_distribution_row(dist: TaskDistribution) -> DistributionRow:
    return DistributionRow(
        class_id=None,
        class_name="TOTAL",
        bucket="total",
        count=dist.total_points,
        percent=100.0 if dist.total_points > 0 else 0.0,
    )


def _iter_distribution_rows(dist: TaskDistribution):
    yield from dist.rows
    yield _total_distribution_row(dist)


def _format_distribution_row(row: DistributionRow) -> str:
    if row.bucket == "class":
        label = f"{row.class_id:2d} {row.class_name[:45]:45s}"
    else:
        label = f"-- {row.class_name[:45]:45s}"
    return f"  {label}: {_format_count(row.count):>10} ({row.percent:5.2f}%)"


def _format_distribution_text(dist: TaskDistribution) -> str:
    lines = [f"=== {dist.task} ==="]
    if dist.task in NATHAB_AXIS_TASKS:
        lines.append(
            f"Scenes with on-disk {NATHAB_AXIS_SOURCE_TASK}.npy "
            f"(fan-out source): {dist.scenes_with_file}/{dist.scenes_total}"
        )
        lines.append(
            f"Axis remap: {NATHAB_AXIS_SOURCE_TASK} -> "
            f"{_NATHAB_AXIS_REMAP_STATE[dist.task].target_definition}"
        )
    else:
        lines.append(
            f"Scenes with on-disk {dist.task}.npy: "
            f"{dist.scenes_with_file}/{dist.scenes_total}"
        )
    if dist.task == MULTILABEL_TASK:
        lines.append("Unit: scene presence (multi-hot)")
        lines.append(
            "TOTAL row = scenes with on-disk file "
            "(percents for labels use this denominator)"
        )
    lines.append("-" * 72)
    lines.extend(_format_distribution_row(row) for row in _iter_distribution_rows(dist))
    lines.append("-" * 72)
    return "\n".join(lines)


def _print_task_distribution(dist: TaskDistribution) -> None:
    print(f"\n{_format_distribution_text(dist)}")


def _save_task_distribution(dist: TaskDistribution, output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{dist.task}_label_distribution.csv")
    txt_path = os.path.join(output_dir, f"{dist.task}_label_distribution.txt")

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_id", "class_name", "bucket", "count", "percent"],
        )
        writer.writeheader()
        for row in _iter_distribution_rows(dist):
            writer.writerow(
                {
                    "class_id": "" if row.class_id is None else row.class_id,
                    "class_name": row.class_name,
                    "bucket": row.bucket,
                    "count": row.count,
                    "percent": f"{row.percent:.6f}",
                }
            )

    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(_format_distribution_text(dist))
        handle.write("\n")

    return csv_path, txt_path


def _save_run_summary(
    output_dir: str,
    *,
    meta: Dict[str, Any],
    distributions: Dict[str, TaskDistribution],
    errors: List[str],
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.json")
    payload = {
        **meta,
        "tasks": {
            task: {
                "scenes_with_file": dist.scenes_with_file,
                "scenes_total": dist.scenes_total,
                "total_points": dist.total_points,
                "unit": (
                    "scene_presence"
                    if task == MULTILABEL_TASK
                    else "points"
                ),
                "csv": f"{task}_label_distribution.csv",
                "txt": f"{task}_label_distribution.txt",
            }
            for task, dist in distributions.items()
        },
        "error_count": len(errors),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    if errors:
        errors_path = os.path.join(output_dir, "errors.txt")
        with open(errors_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(errors))
            handle.write("\n")

    return summary_path


def _emit_bucket_outputs(
    *,
    bucket: str,
    acc: SplitAccumulator,
    output_dir: str,
    shared_meta: Dict[str, Any],
    quiet: bool = False,
) -> List[str]:
    """Write CSV/txt/summary for one split bucket (or ``all``). Returns saved paths."""
    if not quiet:
        print(f"\n######## bucket={bucket} (processed={acc.processed}) ########")

    distributions: Dict[str, TaskDistribution] = {}
    saved_paths: List[str] = []
    for task_name in SEMANTIC_TASKS:
        dist = _build_task_distribution(
            task_name,
            acc.semantic_counts[task_name],
            acc.scenes_with_file[task_name],
            acc.processed,
        )
        distributions[task_name] = dist
        if not quiet:
            _print_task_distribution(dist)
        csv_path, txt_path = _save_task_distribution(dist, output_dir)
        saved_paths.extend([csv_path, txt_path])

    for axis in NATHAB_AXIS_TASKS:
        dist = _build_task_distribution(
            axis,
            acc.axis_counts[axis],
            acc.axis_scenes_with_file[axis],
            acc.processed,
        )
        distributions[axis] = dist
        if not quiet:
            _print_task_distribution(dist)
        csv_path, txt_path = _save_task_distribution(dist, output_dir)
        saved_paths.extend([csv_path, txt_path])

    multilabel_dist = _build_multilabel_distribution(
        acc.multilabel_counts,
        scenes_with_file=acc.multilabel_scenes_with_file,
        scenes_total=acc.processed,
        scenes_all_zero=acc.multilabel_scenes_all_zero,
        scenes_missing=acc.multilabel_scenes_missing,
    )
    distributions[MULTILABEL_TASK] = multilabel_dist
    if not quiet:
        _print_task_distribution(multilabel_dist)
    csv_path, txt_path = _save_task_distribution(multilabel_dist, output_dir)
    saved_paths.extend([csv_path, txt_path])

    errors = acc.errors or []
    summary_path = _save_run_summary(
        output_dir,
        meta={
            **shared_meta,
            "bucket": bucket,
            "scenes_processed": acc.processed,
            f"{MULTILABEL_TASK}_missing": acc.multilabel_scenes_missing,
            f"{MULTILABEL_TASK}_all_zero": acc.multilabel_scenes_all_zero,
        },
        distributions=distributions,
        errors=errors,
    )
    saved_paths.append(summary_path)
    if errors:
        saved_paths.append(os.path.join(output_dir, "errors.txt"))

    if not quiet:
        print(f"Saved outputs under: {output_dir}")
        for path in saved_paths:
            print(f"  - {os.path.relpath(path, output_dir)}")

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data/flair3d_plus")
    parser.add_argument(
        "--csv_manifest",
        default="data/flair3d_plus/raw/scene_split_manifest.csv",
    )
    parser.add_argument("--split", default="train,val,test", help="Comma-separated splits (default: train,val,test)")
    parser.add_argument(
        "--missing_tiles_manifest",
        default="data/flair3d_plus/missing_ply_preflight.txt",
    )
    parser.add_argument(
        "--too_small_tiles_manifest",
        default="data/flair3d_plus/too_small_tiles.csv",
    )
    parser.add_argument(
        "--no_exclude_hardcoded",
        action="store_true",
        help="Do not apply Flair3DDataset hardcoded/missing-coord exclusions.",
    )
    parser.add_argument(
        "--no_exclude_missing_manifest",
        action="store_true",
        help="Do not apply missing_tiles_manifest exclusions.",
    )
    parser.add_argument(
        "--no_exclude_too_small",
        action="store_true",
        help="Do not apply too_small_tiles_manifest exclusions.",
    )
    parser.add_argument(
        "--skip_missing_optional",
        action="store_true",
        help="Skip optional tasks (forest/land_use/natural_habitat) when .npy is absent "
        "(default: use ignore fill like Flair3DDataset).",
    )
    parser.add_argument(
        "--label_definitions",
        default="",
        help="Comma-separated task=definition for training label defs "
        "(e.g. natural_habitat=by_moisture_v3). Defaults match flair3d_label_remap.",
    )
    parser.add_argument(
        "--storage_definitions",
        default="",
        help="Comma-separated task=definition for on-disk label defs "
        "(e.g. natural_habitat=default). Also used as storage for nathab axis "
        "fan-out; use natural_habitat=default to match training. "
        "Defaults match flair3d_label_remap defaults.",
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    parser.add_argument("--max_scenes", type=int, default=0, help="Debug: limit number of scenes (0=all)")
    parser.add_argument(
        "--output_dir",
        default="stats/flair3d/label_distribution",
        help="Directory to write per-task CSV/txt and summary.json. "
        "With multiple --split values, writes <output_dir>/<split>/ and "
        "<output_dir>/all/ ; with a single split, writes flat into --output_dir.",
    )
    args = parser.parse_args()

    global _TASK_REMAP_STATE, _NATHAB_AXIS_REMAP_STATE
    target_definitions = parse_definition_mapping(
        args.label_definitions,
        arg_name="--label_definitions",
    )
    storage_definitions = parse_definition_mapping(
        args.storage_definitions,
        arg_name="--storage_definitions",
    )
    _TASK_REMAP_STATE = build_task_remap_state(target_definitions, storage_definitions)
    _NATHAB_AXIS_REMAP_STATE = build_nathab_axis_remap_state(storage_definitions)

    data_root = resolve_repo_path(args.data_root)
    csv_manifest = resolve_repo_path(args.csv_manifest)
    target_splits = parse_splits(args.split)
    split_order = [s for s in ("train", "val", "test") if s in target_splits]
    for split in sorted(target_splits):
        if split not in split_order:
            split_order.append(split)

    excluded: set[tuple[str, str]] = set()
    if not args.no_exclude_hardcoded:
        excluded |= load_hardcoded_excluded_tiles()
    if not args.no_exclude_missing_manifest:
        excluded |= load_missing_tiles_manifest(resolve_repo_path(args.missing_tiles_manifest))
    if not args.no_exclude_too_small:
        train_only = target_splits == {"train"}
        excluded |= load_too_small_tiles_manifest(
            resolve_repo_path(args.too_small_tiles_manifest),
            train_only=train_only,
        )

    if not os.path.isfile(csv_manifest):
        raise FileNotFoundError(f"CSV manifest not found: {csv_manifest}")

    scenes = load_scene_records(data_root, csv_manifest, target_splits, excluded)
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]

    print(f"data_root={data_root}")
    print(f"csv_manifest={csv_manifest}")
    print(f"splits={split_order}")
    print(f"excluded tiles: {len(excluded)}")
    print(f"scenes to scan: {len(scenes)}")
    print(f"use_fill_for_missing={not args.skip_missing_optional}")
    for task_name in SEMANTIC_TASKS:
        state = _TASK_REMAP_STATE[task_name]
        remap_note = (
            " (on-the-fly remap)"
            if state.stored_to_train_lut is not None
            else ""
        )
        print(
            f"{task_name}: storage={state.storage_definition} -> "
            f"target={state.target_definition}{remap_note}"
        )
    for axis in NATHAB_AXIS_TASKS:
        state = _NATHAB_AXIS_REMAP_STATE[axis]
        print(
            f"{axis}: storage={NATHAB_AXIS_SOURCE_TASK}/{state.storage_definition} -> "
            f"target={state.target_definition} (fan-out)"
        )
    print(
        f"{MULTILABEL_TASK}: scene multi-hot presence "
        f"({NUM_MULTILABEL_CLASSES} labels, no remap)"
    )

    per_split = {split: SplitAccumulator.create() for split in split_order}
    all_acc = SplitAccumulator.create()

    use_fill = not args.skip_missing_optional
    tasks = [(scene, use_fill) for scene in scenes]
    show_progress = not args.no_progress and len(tasks) > 0

    def _consume_results(result_iter):
        for split, partial, axis_partial, ml_partial, meta, err in result_iter:
            if split not in per_split:
                per_split[split] = SplitAccumulator.create()
            if err:
                per_split[split].add_error(err)
                all_acc.add_error(err)
                continue
            per_split[split].add_success(partial, axis_partial, ml_partial, meta)
            all_acc.add_success(partial, axis_partial, ml_partial, meta)

    if args.num_workers <= 1:
        iterator = (_process_scene(t) for t in tasks)
        if show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc="Scenes", unit="scene")
        _consume_results(iterator)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            mapped = pool.map(_process_scene, tasks, chunksize=8)
            if show_progress:
                mapped = tqdm(mapped, total=len(tasks), desc="Scenes", unit="scene")
            _consume_results(mapped)

    print(f"\nProcessed scenes: {all_acc.processed}/{len(scenes)}")
    for split in split_order:
        acc = per_split[split]
        n_err = len(acc.errors or [])
        print(f"  {split}: processed={acc.processed} errors={n_err}")
    if all_acc.errors:
        print(f"Skipped scenes with errors: {len(all_acc.errors)}")
        for line in all_acc.errors[:10]:
            print(f"  - {line}")
        if len(all_acc.errors) > 10:
            print(f"  ... and {len(all_acc.errors) - 10} more")

    output_root = resolve_repo_path(args.output_dir)
    multi_split = len(split_order) > 1

    shared_meta: Dict[str, Any] = {
        "data_root": data_root,
        "csv_manifest": csv_manifest,
        "splits": split_order,
        "excluded_tiles": len(excluded),
        "scenes_scanned": len(scenes),
        "use_fill_for_missing": not args.skip_missing_optional,
        "label_definitions": {
            task_name: _TASK_REMAP_STATE[task_name].target_definition
            for task_name in SEMANTIC_TASKS
        },
        "storage_definitions": {
            task_name: _TASK_REMAP_STATE[task_name].storage_definition
            for task_name in SEMANTIC_TASKS
        },
        "nathab_axis_definitions": {
            axis: {
                "source_task": NATHAB_AXIS_SOURCE_TASK,
                "storage_definition": _NATHAB_AXIS_REMAP_STATE[axis].storage_definition,
                "target_definition": _NATHAB_AXIS_REMAP_STATE[axis].target_definition,
            }
            for axis in NATHAB_AXIS_TASKS
        },
    }

    if multi_split:
        buckets: List[Tuple[str, SplitAccumulator, str]] = [
            (split, per_split[split], os.path.join(output_root, split))
            for split in split_order
        ]
        buckets.append(("all", all_acc, os.path.join(output_root, "all")))
    else:
        only = split_order[0]
        buckets = [(only, per_split[only], output_root)]

    for bucket, acc, out_dir in buckets:
        _emit_bucket_outputs(
            bucket=bucket,
            acc=acc,
            output_dir=out_dir,
            shared_meta=shared_meta,
        )

    if multi_split:
        index_path = os.path.join(output_root, "index.json")
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    **shared_meta,
                    "buckets": {
                        bucket: {
                            "output_dir": out_dir,
                            "scenes_processed": acc.processed,
                            "error_count": len(acc.errors or []),
                        }
                        for bucket, acc, out_dir in buckets
                    },
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        print(f"\nIndex: {index_path}")


if __name__ == "__main__":
    main()
