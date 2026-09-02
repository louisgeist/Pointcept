"""
Per-subtile multi-label natural habitat vectors for Malibu3D.

Reads on-disk natural_habitat.npy (preprocess default, stored ids 0-43) and writes
natural_habitat_multilabel.npy: a length-15 int8 multi-hot vector per subtile.
A label is set when its point fraction >= threshold (default 1%) over all subtile points
(coord.npy count).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    from climatic_domain_tile_labels import (
        SubtileScene,
        scenes_from_manifest_rows,
        scenes_from_patch_tasks,
    )
except ImportError:
    from pointcept.datasets.preprocessing.malibu3d.climatic_domain_tile_labels import (
        SubtileScene,
        scenes_from_manifest_rows,
        scenes_from_patch_tasks,
    )

MULTILABEL_CLASS_NAMES: Tuple[str, ...] = (
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
NUM_MULTILABEL_CLASSES = len(MULTILABEL_CLASS_NAMES)
NUM_STORED_NATURAL_HABITAT_IDS = 44
MULTILABEL_FILENAME = "natural_habitat_multilabel.npy"
DEFAULT_THRESHOLD = 0.01

def build_stored_id_multilabel_bitmap() -> np.ndarray:
    """Return (44, 15) bool LUT: stored default id -> active multi-label columns."""
    bitmap = np.zeros(
        (NUM_STORED_NATURAL_HABITAT_IDS, NUM_MULTILABEL_CLASSES), dtype=bool
    )
    for stored_id in range(36):
        if stored_id < 12:
            bitmap[stored_id, 0] = True  # temperate
        elif stored_id < 24:
            bitmap[stored_id, 1] = True  # mediterranean
        else:
            bitmap[stored_id, 2] = True  # alpine

        bitmap[stored_id, 3 + (stored_id % 3)] = True  # humid / mesic / dry

        if (stored_id % 12) < 6:
            bitmap[stored_id, 7] = True  # open
        else:
            bitmap[stored_id, 6] = True  # forest

        if stored_id % 6 < 3:
            bitmap[stored_id, 8] = True  # acidic
        else:
            bitmap[stored_id, 9] = True  # basic

    bitmap[36, 13] = True  # mineral
    bitmap[36, 8] = True  # acidic
    bitmap[37, 13] = True  # mineral
    bitmap[37, 9] = True  # basic
    bitmap[38, 14] = True  # aquatic
    bitmap[38, 8] = True  # acidic
    bitmap[39, 14] = True  # aquatic
    bitmap[39, 9] = True  # basic
    bitmap[40, 10] = True  # cultivated
    bitmap[41, 11] = True  # built
    bitmap[42, 12] = True  # road
    return bitmap


def count_multilabel_hits(
    stored: np.ndarray,
    bitmap: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Count per-label point hits for one subtile's stored natural_habitat ids."""
    if bitmap is None:
        bitmap = build_stored_id_multilabel_bitmap()

    idx = np.asarray(stored, dtype=np.int64).reshape(-1)
    counts = np.zeros(NUM_MULTILABEL_CLASSES, dtype=np.int64)
    if idx.size == 0:
        return counts

    valid = (idx >= 0) & (idx < NUM_STORED_NATURAL_HABITAT_IDS)
    if not np.any(valid):
        return counts

    hits = bitmap[idx[valid]]
    counts += hits.sum(axis=0)
    return counts


def compute_multilabel_vector(
    stored: np.ndarray,
    n_total: int,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    bitmap: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return length-15 int8 multi-hot vector for one subtile."""
    if n_total <= 0:
        return np.zeros(NUM_MULTILABEL_CLASSES, dtype=np.int8)

    counts = count_multilabel_hits(stored, bitmap=bitmap)
    fractions = counts.astype(np.float64) / float(n_total)
    return (fractions >= threshold).astype(np.int8)


@dataclass
class MultilabelAssignmentStats:
    n_subtiles_seen: int = 0
    n_subtiles_written: int = 0
    n_missing_coord: int = 0
    n_missing_nh: int = 0
    n_all_zero: int = 0
    label_counts: Dict[str, int] = field(default_factory=dict)


def _subtile_point_total(scene_path: str) -> Tuple[int, str]:
    coord_path = os.path.join(scene_path, "coord.npy")
    if os.path.isfile(coord_path):
        return int(np.load(coord_path, mmap_mode="r").shape[0]), ""
    return 0, "missing_coord"


def _merge_multilabel_stats(
    parts: Iterable[MultilabelAssignmentStats],
) -> MultilabelAssignmentStats:
    merged = MultilabelAssignmentStats()
    for part in parts:
        merged.n_subtiles_seen += part.n_subtiles_seen
        merged.n_subtiles_written += part.n_subtiles_written
        merged.n_missing_coord += part.n_missing_coord
        merged.n_missing_nh += part.n_missing_nh
        merged.n_all_zero += part.n_all_zero
        for name, count in part.label_counts.items():
            merged.label_counts[name] = merged.label_counts.get(name, 0) + count
    return merged


def _assign_multilabel_one_scene(
    scene_path: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    only_existing_coord: bool = True,
    bitmap: Optional[np.ndarray] = None,
) -> MultilabelAssignmentStats:
    """Worker entry point: write natural_habitat_multilabel.npy for one subtile."""
    stats = MultilabelAssignmentStats(n_subtiles_seen=1)
    n_total, coord_error = _subtile_point_total(scene_path)
    if only_existing_coord and coord_error == "missing_coord":
        stats.n_missing_coord = 1
        return stats

    nh_path = os.path.join(scene_path, "natural_habitat.npy")
    if not os.path.isfile(nh_path):
        stats.n_missing_nh = 1
        return stats

    stored = np.load(nh_path).reshape(-1)
    if n_total <= 0:
        n_total = int(stored.size)

    vector = compute_multilabel_vector(
        stored,
        n_total,
        threshold=threshold,
        bitmap=bitmap,
    )
    out_path = os.path.join(scene_path, MULTILABEL_FILENAME)
    np.save(out_path, vector)
    stats.n_subtiles_written = 1
    if not np.any(vector):
        stats.n_all_zero = 1
    for name, active in zip(MULTILABEL_CLASS_NAMES, vector):
        if active:
            stats.label_counts[name] = 1
    return stats


def assign_natural_habitat_multilabel_labels(
    scenes: Sequence[SubtileScene],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    bitmap: Optional[np.ndarray] = None,
    only_existing_coord: bool = True,
    num_workers: int = 1,
) -> MultilabelAssignmentStats:
    """Write natural_habitat_multilabel.npy for each subtile independently."""
    scene_paths = [scene.scene_path for scene in scenes]
    if not scene_paths:
        return MultilabelAssignmentStats()

    if num_workers <= 1:
        results = []
        for scene_path in tqdm(
            scene_paths,
            desc="multilabel",
            unit="subtile",
        ):
            results.append(
                _assign_multilabel_one_scene(
                    scene_path,
                    threshold=threshold,
                    only_existing_coord=only_existing_coord,
                    bitmap=bitmap,
                )
            )
        return _merge_multilabel_stats(results)

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                _assign_multilabel_one_scene,
                scene_path,
                threshold=threshold,
                only_existing_coord=only_existing_coord,
            ): scene_path
            for scene_path in scene_paths
        }
        results = []
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="multilabel",
            unit="subtile",
        ):
            results.append(future.result())
    return _merge_multilabel_stats(results)


def run_natural_habitat_multilabel_label_pass(
    output_root: str,
    tasks: Sequence[object],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    num_workers: int = 1,
    logger=None,
) -> MultilabelAssignmentStats:
    scenes = scenes_from_patch_tasks(output_root, tasks)
    stats = assign_natural_habitat_multilabel_labels(
        scenes,
        threshold=threshold,
        num_workers=num_workers,
    )
    if logger is not None:
        logger.info(
            "Natural habitat multilabel: seen=%d written=%d missing_coord=%d "
            "missing_nh=%d all_zero=%d threshold=%.4f",
            stats.n_subtiles_seen,
            stats.n_subtiles_written,
            stats.n_missing_coord,
            stats.n_missing_nh,
            stats.n_all_zero,
            threshold,
        )
    return stats


__all__ = [
    "DEFAULT_THRESHOLD",
    "MULTILABEL_CLASS_NAMES",
    "MULTILABEL_FILENAME",
    "NUM_MULTILABEL_CLASSES",
    "MultilabelAssignmentStats",
    "assign_natural_habitat_multilabel_labels",
    "build_stored_id_multilabel_bitmap",
    "compute_multilabel_vector",
    "count_multilabel_hits",
    "run_natural_habitat_multilabel_label_pass",
    "scenes_from_manifest_rows",
]
