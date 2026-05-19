"""
PureForest semantic class definitions (13 classes).

See: https://arxiv.org/abs/2404.12064
"""

from __future__ import annotations

import os
import re
from typing import Optional

NUM_CLASSES = 13

SPLITS = ("train", "val", "test")

SPLIT_TO_LAZ_PREFIX = {
    "train": "TRAIN",
    "val": "VAL",
    "test": "TEST",
}

CLASS_NAMES = [
    "deciduous_oak",
    "evergreen_oak",
    "beech",
    "chestnut",
    "black_locust",
    "maritime_pine",
    "scotch_pine",
    "black_pine",
    "aleppo_pine",
    "fir",
    "spruce",
    "larch",
    "douglas",
]

CLASS_ID_PATTERN = re.compile(r"-C(\d+)-", re.IGNORECASE)


def parse_class_id_from_patch_stem(patch_stem: str) -> int:
    """Extract zero-based class id from a PureForest patch filename stem.

    Example: ``TRAIN-Quercus_pubescens-C0-422_8_324`` → ``0``.
    """
    match = CLASS_ID_PATTERN.search(patch_stem)
    if match is None:
        raise ValueError(
            f"Cannot parse class id from patch stem {patch_stem!r} "
            f"(expected pattern '-C{{id}}-')."
        )
    class_id = int(match.group(1))
    if class_id < 0 or class_id >= NUM_CLASSES:
        raise ValueError(
            f"Class id {class_id} out of range [0, {NUM_CLASSES - 1}] for {patch_stem!r}."
        )
    return class_id


def class_name_for_id(class_id: int) -> str:
    if class_id < 0 or class_id >= NUM_CLASSES:
        raise ValueError(f"class_id must be in [0, {NUM_CLASSES - 1}], got {class_id}.")
    return CLASS_NAMES[class_id]


def parse_split_from_patch_stem(patch_stem: str) -> Optional[str]:
    """Return train/val/test from prefix TRAIN/VAL/TEST if present."""
    prefix = patch_stem.split("-", 1)[0].lower()
    if prefix in SPLITS:
        return prefix
    return None


def laz_prefix_for_split(split: str) -> str:
    split = split.strip().lower()
    if split not in SPLIT_TO_LAZ_PREFIX:
        raise ValueError(
            f"Unknown split {split!r}; expected one of {tuple(SPLIT_TO_LAZ_PREFIX)}."
        )
    return SPLIT_TO_LAZ_PREFIX[split]


def build_laz_filename(split: str, patch_id: str) -> str:
    """LAZ basename on disk, e.g. ``TRAIN-Pinus_halepensis-C8-3_1_244.laz``."""
    return f"{laz_prefix_for_split(split)}-{patch_id}.laz"


def build_laz_path(dataset_root: str, split: str, patch_id: str) -> str:
    """Full path to expected LAZ under ``dataset_root/lidar/{split}/``."""
    return os.path.join(
        dataset_root,
        "lidar",
        split.strip().lower(),
        build_laz_filename(split, patch_id),
    )
