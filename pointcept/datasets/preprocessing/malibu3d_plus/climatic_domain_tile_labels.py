"""
Tile-level climatic-domain labels for Malibu3D+ (Temperate / Mediterranean / Alpine).

Aggregates point counts from on-disk natural_habitat.npy (preprocess default, ids 0-43),
remapped via by_climatic_domain, at 1 km² tile granularity (dept_year_roi).
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

VOID_TRAIN_ID = 3
INVALID_LABEL = -1
CLIMATIC_DOMAIN_FILENAME = "climatic_domain.npy"


def patch_id_to_tile_1km(patch_id: str) -> str:
    parts = patch_id.strip().split("_", 2)
    if len(parts) < 2:
        raise ValueError(
            f"Invalid patch_id format (expected dept_year_roi_i-j): {patch_id}"
        )
    return f"{parts[0]}_{parts[1]}"


def build_scene_path(
    output_root: str,
    split: str,
    patch_id: str,
    dept_year: str,
    roi: str,
) -> str:
    return os.path.join(output_root, split, f"{dept_year}_LIDARHD", roi, patch_id)


def get_stored_to_climatic_domain_lut() -> np.ndarray:
    from pointcept.datasets.preprocessing.malibu3d_plus.malibu3d_label_remap import (
        build_stored_to_train_lut,
        get_definition,
    )

    storage = get_definition("natural_habitat", "default")
    target = get_definition("natural_habitat", "by_climatic_domain")
    return build_stored_to_train_lut(storage, target)


def remap_stored_labels(stored: np.ndarray, lut: np.ndarray) -> np.ndarray:
    idx = stored.astype(np.int64, copy=False)
    remapped = np.full(idx.shape, VOID_TRAIN_ID, dtype=np.int32)
    valid = (idx >= 0) & (idx < lut.shape[0])
    if np.any(valid):
        remapped[valid] = lut[idx[valid]]
    return remapped


@dataclass
class DomainCounts:
    n_temperate: int = 0
    n_mediterranean: int = 0
    n_alpine: int = 0
    n_void: int = 0

    @property
    def n_points(self) -> int:
        return self.n_temperate + self.n_mediterranean + self.n_alpine + self.n_void

    def add_mapped(self, mapped: np.ndarray) -> None:
        self.n_temperate += int(np.count_nonzero(mapped == 0))
        self.n_mediterranean += int(np.count_nonzero(mapped == 1))
        self.n_alpine += int(np.count_nonzero(mapped == 2))
        self.n_void += int(np.count_nonzero(mapped == VOID_TRAIN_ID))

    def __iadd__(self, other: "DomainCounts") -> "DomainCounts":
        self.n_temperate += other.n_temperate
        self.n_mediterranean += other.n_mediterranean
        self.n_alpine += other.n_alpine
        self.n_void += other.n_void
        return self


def count_climatic_domains(stored: np.ndarray, lut: np.ndarray) -> DomainCounts:
    mapped = remap_stored_labels(np.asarray(stored).reshape(-1), lut)
    counts = DomainCounts()
    counts.add_mapped(mapped)
    return counts


def classify_aggregated_counts(
    n_temperate: int,
    n_mediterranean: int,
    n_alpine: int,
) -> int:
    """Return train id 0/1/2 when exactly one habitat domain is present, else -1."""
    habitat = (
        int(n_temperate > 0),
        int(n_mediterranean > 0),
        int(n_alpine > 0),
    )
    if sum(habitat) != 1:
        return INVALID_LABEL
    if n_temperate > 0:
        return 0
    if n_mediterranean > 0:
        return 1
    return 2


@dataclass
class SubtileScene:
    split: str
    patch_id: str
    scene_path: str
    dept_year: str
    roi: str


@dataclass
class AssignmentStats:
    n_tiles_1km: int = 0
    n_pure: int = 0
    n_mixed: int = 0
    n_all_void: int = 0
    n_missing_nh: int = 0
    n_subtiles_written: int = 0
    labels_by_1km: Dict[Tuple[str, str], int] = field(default_factory=dict)


def _count_subtile(scene_path: str, lut: np.ndarray) -> Tuple[DomainCounts, str]:
    nh_path = os.path.join(scene_path, "natural_habitat.npy")
    if not os.path.isfile(nh_path):
        return DomainCounts(), "missing_nh"
    stored = np.load(nh_path).reshape(-1)
    if stored.size == 0:
        return DomainCounts(), ""
    return count_climatic_domains(stored, lut), ""


def assign_climatic_domain_labels(
    scenes: Sequence[SubtileScene],
    *,
    lut: Optional[np.ndarray] = None,
    only_existing_coord: bool = True,
) -> AssignmentStats:
    """Aggregate 1 km² domains and write climatic_domain.npy to each subtile scene dir."""
    if lut is None:
        lut = get_stored_to_climatic_domain_lut()

    grouped_counts: Dict[Tuple[str, str], DomainCounts] = defaultdict(DomainCounts)
    grouped_scenes: Dict[Tuple[str, str], List[SubtileScene]] = defaultdict(list)
    grouped_errors: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for scene in scenes:
        coord_path = os.path.join(scene.scene_path, "coord.npy")
        if only_existing_coord and not os.path.isfile(coord_path):
            continue
        tile_key = (scene.split, patch_id_to_tile_1km(scene.patch_id))
        grouped_scenes[tile_key].append(scene)
        counts, error = _count_subtile(scene.scene_path, lut)
        grouped_counts[tile_key] += counts
        if error:
            grouped_errors[tile_key].append(error)

    stats = AssignmentStats()
    for tile_key, scenes_in_tile in grouped_scenes.items():
        stats.n_tiles_1km += 1
        counts = grouped_counts[tile_key]
        errors = grouped_errors[tile_key]

        if counts.n_points == 0:
            if errors and all(e == "missing_nh" for e in errors):
                label = INVALID_LABEL
                stats.n_missing_nh += 1
            else:
                label = INVALID_LABEL
                stats.n_all_void += 1
        else:
            label = classify_aggregated_counts(
                counts.n_temperate,
                counts.n_mediterranean,
                counts.n_alpine,
            )
            if label == INVALID_LABEL:
                if (
                    counts.n_temperate == 0
                    and counts.n_mediterranean == 0
                    and counts.n_alpine == 0
                ):
                    stats.n_all_void += 1
                else:
                    stats.n_mixed += 1
            else:
                stats.n_pure += 1

        stats.labels_by_1km[tile_key] = label
        for scene in scenes_in_tile:
            out_path = os.path.join(scene.scene_path, CLIMATIC_DOMAIN_FILENAME)
            np.save(out_path, np.int32(label))
            stats.n_subtiles_written += 1

    return stats


def scenes_from_patch_tasks(
    output_root: str,
    tasks: Sequence[object],
) -> List[SubtileScene]:
    """Build SubtileScene list from preprocess_malibu3d_v2.PatchTask objects."""
    scenes: List[SubtileScene] = []
    for task in tasks:
        scenes.append(
            SubtileScene(
                split=str(task.split),
                patch_id=str(task.patch_id),
                scene_path=os.path.join(
                    output_root,
                    task.split,
                    f"{task.dept_year}_LIDARHD",
                    task.roi,
                    task.patch_id,
                ),
                dept_year=str(task.dept_year),
                roi=str(task.roi),
            )
        )
    return scenes


def run_climatic_domain_label_pass(
    output_root: str,
    tasks: Sequence[object],
    *,
    logger=None,
) -> AssignmentStats:
    scenes = scenes_from_patch_tasks(output_root, tasks)
    stats = assign_climatic_domain_labels(scenes)
    if logger is not None:
        logger.info(
            "Climatic domain labels: 1km tiles=%d pure=%d mixed=%d all_void=%d "
            "missing_nh=%d subtiles_written=%d",
            stats.n_tiles_1km,
            stats.n_pure,
            stats.n_mixed,
            stats.n_all_void,
            stats.n_missing_nh,
            stats.n_subtiles_written,
        )
    return stats


def scenes_from_manifest_rows(
    output_root: str,
    rows: Iterable[Mapping[str, str]],
    *,
    target_splits: Optional[Iterable[str]] = None,
    lidarhd_only: bool = True,
) -> List[SubtileScene]:
    split_filter = set(target_splits) if target_splits is not None else None
    scenes: List[SubtileScene] = []
    for row in rows:
        split = str(row.get("split", "")).strip()
        patch_id = str(row.get("patch_id", "")).strip()
        if not split or not patch_id:
            continue
        if split_filter is not None and split not in split_filter:
            continue
        if lidarhd_only:
            token = str(row.get("LIDARHD", "")).strip().lower()
            if token not in ("true", "1", "yes"):
                continue
        dept_year = (row.get("dept_year") or "").strip() or patch_id.split("_", 2)[0]
        roi = (row.get("roi") or "").strip() or patch_id.split("_", 2)[1]
        scene_path = build_scene_path(output_root, split, patch_id, dept_year, roi)
        scenes.append(
            SubtileScene(
                split=split,
                patch_id=patch_id,
                scene_path=scene_path,
                dept_year=dept_year,
                roi=roi,
            )
        )
    return scenes
