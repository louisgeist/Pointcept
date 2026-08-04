"""
Stratified fixed subset selection for val/test (segment + natural_habitat_multilabel).

Selection runs offline via scripts/build_stratified_subset.py; training loads a CSV sidecar.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from pointcept.utils.logger import get_root_logger

NH_MULTILABEL_CLASS_NAMES: Tuple[str, ...] = (
    "temperate",
    "mediterranean",
    "alpine",
    "humid",
    "mesic",
    "dry",
    "forest",
    "open",
    "acidic",
    "basic",
    "cultivated",
    "built",
    "road",
    "mineral",
    "aquatic",
)
NUM_NH_MULTILABEL_CLASSES = len(NH_MULTILABEL_CLASS_NAMES)

SIDECAR_FIELDNAMES = (
    "split",
    "patch_id",
    "present_segment_classes",
    "nh_active_labels",
)

_SCORE_EPS = 1e-12


@dataclass
class SceneFeatures:
    split: str
    patch_id: str
    scene_path: str
    seg_hist: np.ndarray
    n_points: int
    nh_vec: np.ndarray
    order_index: int = 0


@dataclass
class SubsetSelectionResult:
    selected: List[SceneFeatures]
    meta: Dict[str, object] = field(default_factory=dict)


def _normalize_split(split) -> str:
    if isinstance(split, str):
        return split
    if isinstance(split, Sequence) and len(split) == 1:
        return str(split[0])
    raise TypeError(
        "split must be a string or a length-1 sequence for subset sidecar filtering"
    )


def load_segment_histogram(
    scene_path: str,
    ignore_index: int = -1,
) -> Tuple[np.ndarray, int]:
    segment_path = os.path.join(scene_path, "segment.npy")
    if not os.path.isfile(segment_path):
        return np.zeros(0, dtype=np.int64), 0
    labels = np.load(segment_path).reshape(-1)
    if ignore_index is not None and ignore_index >= 0:
        valid = labels != ignore_index
        labels = labels[valid]
    else:
        valid = np.ones(labels.shape[0], dtype=bool)
    n_points = int(valid.sum())
    if n_points == 0:
        return np.zeros(0, dtype=np.int64), 0
    max_cls = int(labels.max())
    hist = np.bincount(labels.astype(np.int64), minlength=max_cls + 1)
    return hist.astype(np.int64), n_points


def load_nh_multilabel_vector(
    scene_path: str,
    num_classes: int = NUM_NH_MULTILABEL_CLASSES,
) -> np.ndarray:
    nh_path = os.path.join(scene_path, "natural_habitat_multilabel.npy")
    if not os.path.isfile(nh_path):
        return np.zeros(num_classes, dtype=np.float32)
    vector = np.load(nh_path).reshape(-1).astype(np.float32)
    if vector.shape[0] != num_classes:
        raise ValueError(
            f"natural_habitat_multilabel length {vector.shape[0]} != {num_classes} "
            f"under {scene_path}"
        )
    return (vector > 0).astype(np.float32)


def present_segment_class_ids(
    seg_hist: np.ndarray,
    n_points: int,
    min_fraction: float = 0.01,
) -> List[int]:
    if n_points <= 0 or seg_hist.size == 0:
        return []
    threshold = max(1, int(np.ceil(min_fraction * n_points)))
    return [int(i) for i in np.flatnonzero(seg_hist >= threshold)]


def nh_active_label_names(nh_vec: np.ndarray) -> List[str]:
    indices = np.flatnonzero(nh_vec > 0)
    return [NH_MULTILABEL_CLASS_NAMES[int(i)] for i in indices]


def _pad_hist(hist: np.ndarray, size: int) -> np.ndarray:
    if hist.shape[0] >= size:
        return hist.astype(np.int64, copy=False)
    out = np.zeros(size, dtype=np.int64)
    if hist.size:
        out[: hist.shape[0]] = hist
    return out


def _seg_l1_error(hist: np.ndarray, total_points: int, target_t: np.ndarray) -> float:
    if total_points <= 0:
        return float(np.sum(np.abs(target_t)))
    h = hist.astype(np.float64) / float(total_points)
    return float(np.sum(np.abs(h - target_t)))


def _delta_seg(
    hist: np.ndarray,
    total_points: int,
    seg_hist: np.ndarray,
    n_points: int,
    target_t: np.ndarray,
) -> float:
    size = target_t.shape[0]
    hist_p = _pad_hist(hist, size)
    seg_p = _pad_hist(seg_hist, size)
    before = _seg_l1_error(hist_p, total_points, target_t)
    after = _seg_l1_error(hist_p + seg_p, total_points + n_points, target_t)
    return before - after


def _nh_l1_error(
    count_nh: np.ndarray,
    n_selected: int,
    target_u: np.ndarray,
    weights: np.ndarray,
) -> float:
    if n_selected <= 0:
        presence = np.zeros_like(target_u, dtype=np.float64)
    else:
        presence = count_nh.astype(np.float64) / float(n_selected)
    return float(np.sum(weights * np.abs(presence - target_u)))


def _delta_nh(
    count_nh: np.ndarray,
    n_selected: int,
    nh_vec: np.ndarray,
    target_u: np.ndarray,
    weights: np.ndarray,
) -> float:
    before = _nh_l1_error(count_nh, n_selected, target_u, weights)
    after = _nh_l1_error(count_nh + nh_vec, n_selected + 1, target_u, weights)
    return before - after


def _aggregate_state(
    selected: Sequence[SceneFeatures],
    seg_num_classes: int,
) -> Tuple[np.ndarray, int, np.ndarray]:
    hist = np.zeros(seg_num_classes, dtype=np.int64)
    total_points = 0
    count_nh = np.zeros(NUM_NH_MULTILABEL_CLASSES, dtype=np.float64)
    for scene in selected:
        hist += _pad_hist(scene.seg_hist, seg_num_classes)
        total_points += scene.n_points
        count_nh += scene.nh_vec
    return hist, total_points, count_nh


def _compute_targets(
    features: Sequence[SceneFeatures],
    seg_num_classes: int,
    use_segment: bool,
    use_nh: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_scenes = len(features)
    seg_total = np.zeros(seg_num_classes, dtype=np.int64)
    total_points = 0
    nh_total = np.zeros(NUM_NH_MULTILABEL_CLASSES, dtype=np.float64)

    for scene in features:
        if use_segment:
            seg_total += _pad_hist(scene.seg_hist, seg_num_classes)
            total_points += scene.n_points
        if use_nh:
            nh_total += scene.nh_vec

    target_t = (
        seg_total.astype(np.float64) / max(total_points, 1)
        if use_segment
        else np.zeros(seg_num_classes, dtype=np.float64)
    )
    target_u = nh_total / max(n_scenes, 1) if use_nh else np.zeros(
        NUM_NH_MULTILABEL_CLASSES, dtype=np.float64
    )
    return target_t, target_u, seg_total, nh_total


def select_stratified_subset(
    features: Sequence[SceneFeatures],
    max_sample: int,
    *,
    warm_random: Optional[int] = None,
    shuffle_seed: int = 42,
    use_segment: bool = True,
    use_nh: bool = True,
    nh_weighted: bool = True,
    nh_beta_scale: float = 0.05,
) -> SubsetSelectionResult:
    if max_sample <= 0:
        raise ValueError("max_sample must be positive")
    if not features:
        return SubsetSelectionResult(selected=[], meta={"max_sample": max_sample})

    n_scenes = len(features)
    max_sample = min(max_sample, n_scenes)
    if warm_random is None:
        warm_random = max_sample // 2
    warm_random = max(0, min(warm_random, max_sample))

    seg_num_classes = max((f.seg_hist.shape[0] for f in features), default=0)
    target_t, target_u, _, _ = _compute_targets(
        features, seg_num_classes, use_segment, use_nh
    )

    rng = np.random.RandomState(shuffle_seed)
    order = rng.permutation(n_scenes)
    indexed_features = list(features)
    for rank, idx in enumerate(order):
        indexed_features[int(idx)].order_index = int(rank)

    u_min = 1.0 / max(n_scenes, 1)
    if use_nh and nh_weighted:
        nh_weights = 1.0 / np.maximum(target_u, u_min)
    else:
        nh_weights = np.ones(NUM_NH_MULTILABEL_CLASSES, dtype=np.float64)

    total_points_all = int(sum(f.n_points for f in features))
    beta = nh_beta_scale * (total_points_all / max(n_scenes, 1))

    selected: List[SceneFeatures] = [
        indexed_features[int(idx)] for idx in order[:warm_random]
    ]
    selected_set = {id(scene) for scene in selected}

    hist, total_points, count_nh = _aggregate_state(selected, seg_num_classes)
    l1_seg_after_warm = _seg_l1_error(hist, total_points, target_t) if use_segment else 0.0
    l1_nh_after_warm = (
        _nh_l1_error(count_nh, len(selected), target_u, nh_weights) if use_nh else 0.0
    )

    greedy_steps = max_sample - warm_random
    for _ in range(greedy_steps):
        best_score = -np.inf
        best_scene: Optional[SceneFeatures] = None

        for scene in indexed_features:
            if id(scene) in selected_set:
                continue
            score = 0.0
            if use_segment:
                score += _delta_seg(
                    hist, total_points, scene.seg_hist, scene.n_points, target_t
                )
            if use_nh:
                score += beta * _delta_nh(
                    count_nh, len(selected), scene.nh_vec, target_u, nh_weights
                )
            if best_scene is None:
                best_score = score
                best_scene = scene
            elif score > best_score + _SCORE_EPS:
                best_score = score
                best_scene = scene
            elif abs(score - best_score) <= _SCORE_EPS:
                if scene.order_index < best_scene.order_index:
                    best_score = score
                    best_scene = scene

        if best_scene is None:
            break

        selected.append(best_scene)
        selected_set.add(id(best_scene))
        hist += _pad_hist(best_scene.seg_hist, seg_num_classes)
        total_points += best_scene.n_points
        count_nh += best_scene.nh_vec

    l1_seg_final = _seg_l1_error(hist, total_points, target_t) if use_segment else 0.0
    l1_nh_final = (
        _nh_l1_error(count_nh, len(selected), target_u, nh_weights) if use_nh else 0.0
    )

    selected.sort(key=lambda s: s.patch_id)
    meta = {
        "max_sample": max_sample,
        "warm_random": warm_random,
        "shuffle_seed": shuffle_seed,
        "nh_weighted": nh_weighted,
        "nh_beta_scale": nh_beta_scale,
        "use_segment": use_segment,
        "use_nh": use_nh,
        "total_scenes": n_scenes,
        "selected_scenes": len(selected),
        "l1_seg_after_warm": l1_seg_after_warm,
        "l1_nh_after_warm": l1_nh_after_warm,
        "l1_seg_final": l1_seg_final,
        "l1_nh_final": l1_nh_final,
        "target_u": {NH_MULTILABEL_CLASS_NAMES[i]: float(target_u[i]) for i in range(len(target_u))},
        "subset_u": {
            NH_MULTILABEL_CLASS_NAMES[i]: float(count_nh[i] / max(len(selected), 1))
            for i in range(len(count_nh))
        },
    }
    return SubsetSelectionResult(selected=selected, meta=meta)


def load_sidecar_keys(manifest_path: str) -> Set[Tuple[str, str]]:
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Stratified subset sidecar not found: {manifest_path}")

    keys: Set[Tuple[str, str]] = set()
    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty sidecar CSV: {manifest_path}")
        missing = {"split", "patch_id"} - set(reader.fieldnames)
        if missing:
            raise KeyError(
                f"Sidecar CSV missing columns {sorted(missing)}: {manifest_path}"
            )
        for row in reader:
            split = (row.get("split") or "").strip()
            patch_id = (row.get("patch_id") or "").strip()
            if split and patch_id:
                keys.add((split, patch_id))
    return keys


def filter_paths_by_sidecar(
    data_list: Sequence[str],
    split: str,
    sidecar_keys: Set[Tuple[str, str]],
) -> List[str]:
    split = _normalize_split(split)
    filtered = [
        path
        for path in data_list
        if (split, os.path.basename(path)) in sidecar_keys
    ]
    filtered.sort(key=os.path.basename)
    return filtered


def write_sidecar_csv(output_path: str, selected: Sequence[SceneFeatures]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIDECAR_FIELDNAMES)
        writer.writeheader()
        for scene in selected:
            present = present_segment_class_ids(scene.seg_hist, scene.n_points)
            writer.writerow(
                {
                    "split": scene.split,
                    "patch_id": scene.patch_id,
                    "present_segment_classes": ";".join(str(c) for c in present),
                    "nh_active_labels": ";".join(nh_active_label_names(scene.nh_vec)),
                }
            )


def write_sidecar_meta(output_path: str, meta: Dict[str, object]) -> None:
    meta_path = f"{output_path}.meta.json"
    os.makedirs(os.path.dirname(os.path.abspath(meta_path)), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _distribution_output_paths(output_path: str) -> Tuple[str, str]:
    base, ext = os.path.splitext(output_path)
    if ext.lower() != ".csv":
        base = output_path
    return f"{base}.distribution_segment.csv", f"{base}.distribution_nh.csv"


def _segment_distribution_rows(
    features: Sequence[SceneFeatures],
    split: str,
    stage: str,
    seg_num_classes: int,
) -> Tuple[List[Dict[str, object]], int]:
    hist, total_points, _ = _aggregate_state(features, seg_num_classes)
    rows: List[Dict[str, object]] = []
    for class_id in range(seg_num_classes):
        point_count = int(hist[class_id])
        point_fraction = (
            float(point_count) / float(total_points) if total_points > 0 else 0.0
        )
        rows.append(
            {
                "split": split,
                "stage": stage,
                "class_id": class_id,
                "point_count": point_count,
                "point_fraction": point_fraction,
            }
        )
    return rows, total_points


def _nh_distribution_rows(
    features: Sequence[SceneFeatures],
    split: str,
    stage: str,
) -> Tuple[List[Dict[str, object]], int]:
    n_scenes = len(features)
    count_nh = np.zeros(NUM_NH_MULTILABEL_CLASSES, dtype=np.int64)
    for scene in features:
        count_nh += scene.nh_vec.astype(np.int64)
    rows: List[Dict[str, object]] = []
    for label_id, label_name in enumerate(NH_MULTILABEL_CLASS_NAMES):
        scene_count = int(count_nh[label_id])
        scene_presence_fraction = (
            float(scene_count) / float(n_scenes) if n_scenes > 0 else 0.0
        )
        rows.append(
            {
                "split": split,
                "stage": stage,
                "label_id": label_id,
                "label": label_name,
                "scene_count": scene_count,
                "scene_presence_fraction": scene_presence_fraction,
            }
        )
    return rows, n_scenes


def write_distribution_csvs(
    output_path: str,
    split: str,
    full_features: Sequence[SceneFeatures],
    selected_features: Sequence[SceneFeatures],
) -> Dict[str, object]:
    seg_num_classes = max(
        max((f.seg_hist.shape[0] for f in full_features), default=0),
        max((f.seg_hist.shape[0] for f in selected_features), default=0),
    )
    segment_csv, nh_csv = _distribution_output_paths(output_path)

    segment_rows, full_points = _segment_distribution_rows(
        full_features, split, "full", seg_num_classes
    )
    subset_segment_rows, subset_points = _segment_distribution_rows(
        selected_features, split, "subset", seg_num_classes
    )
    segment_rows.extend(subset_segment_rows)

    nh_rows, full_scenes = _nh_distribution_rows(full_features, split, "full")
    subset_nh_rows, subset_scenes = _nh_distribution_rows(
        selected_features, split, "subset"
    )
    nh_rows.extend(subset_nh_rows)

    os.makedirs(os.path.dirname(os.path.abspath(segment_csv)), exist_ok=True)
    with open(segment_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "stage",
                "class_id",
                "point_count",
                "point_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(segment_rows)

    with open(nh_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "stage",
                "label_id",
                "label",
                "scene_count",
                "scene_presence_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(nh_rows)

    return {
        "distribution_segment_csv": segment_csv,
        "distribution_nh_csv": nh_csv,
        "full_point_count": full_points,
        "subset_point_count": subset_points,
        "full_scene_count": full_scenes,
        "subset_scene_count": subset_scenes,
    }


def build_scene_features(
    split: str,
    patch_id: str,
    scene_path: str,
    keys: Sequence[str],
    ignore_index: int = -1,
) -> SceneFeatures:
    use_segment = "segment" in keys
    use_nh = "natural_habitat_multilabel" in keys

    if use_segment:
        seg_hist, n_points = load_segment_histogram(scene_path, ignore_index=ignore_index)
    else:
        seg_hist, n_points = np.zeros(0, dtype=np.int64), 0

    if use_nh:
        nh_vec = load_nh_multilabel_vector(scene_path)
    else:
        nh_vec = np.zeros(NUM_NH_MULTILABEL_CLASSES, dtype=np.float32)

    return SceneFeatures(
        split=split,
        patch_id=patch_id,
        scene_path=scene_path,
        seg_hist=seg_hist,
        n_points=n_points,
        nh_vec=nh_vec,
    )


def apply_subset_selection(
    data_list: Sequence[str],
    *,
    split,
    max_sample: Optional[int] = None,
    stratified_subset_manifest: Optional[str] = None,
) -> List[str]:
    logger = get_root_logger()
    split_name = _normalize_split(split)
    original_len = len(data_list)

    if stratified_subset_manifest:
        sidecar_keys = load_sidecar_keys(stratified_subset_manifest)
        data_list = filter_paths_by_sidecar(data_list, split_name, sidecar_keys)
        missing = len(sidecar_keys) - len(data_list)
        if missing > 0:
            logger.warning(
                "Stratified subset sidecar: %d/%d entries not found in current data_list "
                "(split=%s, manifest=%s)",
                missing,
                len(sidecar_keys),
                split_name,
                stratified_subset_manifest,
            )
        logger.info(
            "Stratified subset sidecar applied: %d -> %d scenes (split=%s, manifest=%s)",
            original_len,
            len(data_list),
            split_name,
            stratified_subset_manifest,
        )
    else:
        data_list = list(data_list)

    if max_sample is None or max_sample >= len(data_list):
        return data_list

    logger.info(
        "max_sample head slice: %d -> %d scenes (split=%s)",
        len(data_list),
        max_sample,
        split_name,
    )
    return list(data_list[:max_sample])
