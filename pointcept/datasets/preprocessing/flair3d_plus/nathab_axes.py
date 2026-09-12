"""Ecological-axis layout for on-disk ``natural_habitat.npy``.

Pointcept preprocess bakes CarHab ids ``(N,)`` into ``uint8 (N, 4)`` using the
same column order / LUTs as the MALiBU3D Hugging Face release
(``Flair3D-build/scripts/hf_release/common.py``).

Column ``i`` matches ``NATHAB_AXIS_KEYS[i]`` and is produced by remapping
storage definition ``default`` (CarHab 0..43) through
``NATHAB_AXIS_DEFINITIONS[i]``.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

NATHAB_AXIS_KEYS: Tuple[str, ...] = (
    "nathab_habitat_type",
    "nathab_moisture_regime",
    "nathab_soil_chemistry",
    "nathab_bioclimatic_zone",
)
NATHAB_AXIS_DEFINITIONS: Tuple[str, ...] = (
    "by_habitat_type_ecological",
    "by_moisture_regime",
    "by_soil_chemistry",
    "by_climatic_domain",
)
NATHAB_AXIS_IGNORE_INDEX: Tuple[int, ...] = (4, 3, 2, 3)

# Stored CarHab id (definition=default, 0..43) → axis train id.
# Must match build_stored_to_train_lut(default, axis) in flair3d_label_remap.
NATHAB_AXIS_LUTS: Tuple[Tuple[int, ...], ...] = (
    # nathab_habitat_type / by_habitat_type_ecological (Void=4)
    (
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        2, 2, 3, 3, 4, 4, 4, 4,
    ),
    # nathab_moisture_regime / by_moisture_regime (Void=3)
    (
        0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
        0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
        0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
        3, 3, 3, 3, 3, 3, 3, 3,
    ),
    # nathab_soil_chemistry / by_soil_chemistry (Void=2)
    (
        0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
        0, 1, 0, 1, 2, 2, 2, 2,
    ),
    # nathab_bioclimatic_zone / by_climatic_domain (Void=3)
    (
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3,
    ),
)

NATHAB_CARHAB_MAX_ID = 43
NATHAB_ONDISK_DEFINITION = "ecological_axes"
NATHAB_NUM_AXES = len(NATHAB_AXIS_KEYS)

# meta.json extras under key "natural_habitat_layout" (alongside label_definitions).
NATHAB_LAYOUT_META: Dict[str, object] = {
    "layout": "axes",
    "dtype": "uint8",
    "shape_tail": NATHAB_NUM_AXES,
    "columns": list(NATHAB_AXIS_KEYS),
    "definitions": list(NATHAB_AXIS_DEFINITIONS),
    "ignore_index": list(NATHAB_AXIS_IGNORE_INDEX),
}


def is_nathab_axes_array(array: np.ndarray) -> bool:
    """True if ``array`` looks like baked ecological axes ``(N, 4)``."""
    values = np.asarray(array)
    return values.ndim == 2 and values.shape[1] == NATHAB_NUM_AXES


def is_nathab_carhab_array(array: np.ndarray) -> bool:
    """True if ``array`` looks like legacy CarHab ids ``(N,)``."""
    values = np.asarray(array)
    return values.ndim == 1


def validate_nathab_axes(array: np.ndarray) -> np.ndarray:
    """Validate and downcast baked axes to ``uint8 (N, 4)``."""
    values = np.asarray(array)
    if not is_nathab_axes_array(values):
        raise ValueError(
            f"natural_habitat axes: expected shape (N, {NATHAB_NUM_AXES}), "
            f"got {values.shape}"
        )
    if values.size == 0:
        return values.astype(np.uint8, copy=False).reshape(0, NATHAB_NUM_AXES)
    finite_min = int(np.min(values))
    if finite_min < 0:
        raise ValueError(f"natural_habitat axes: negative axis id {finite_min}")
    maxima = np.max(values, axis=0)
    for column, (axis_key, ignore, peak) in enumerate(
        zip(NATHAB_AXIS_KEYS, NATHAB_AXIS_IGNORE_INDEX, maxima.tolist())
    ):
        if int(peak) > int(ignore):
            raise ValueError(
                f"natural_habitat column {column} ({axis_key}): "
                f"max {int(peak)} exceeds ignore_index {ignore}"
            )
    return values.astype(np.uint8, copy=False)


def carhab_to_nathab_axes(array: np.ndarray) -> np.ndarray:
    """Map stored CarHab ids ``(N,)`` to ecological axes ``(N, 4)`` uint8.

    Accepts an already-baked ``(N, 4)`` array and only validates / downcasts it.
    """
    values = np.asarray(array)
    if is_nathab_axes_array(values):
        return validate_nathab_axes(values)
    if values.ndim != 1:
        raise ValueError(
            "natural_habitat: expected CarHab ids shape (N,) or baked axes "
            f"(N, {NATHAB_NUM_AXES}), got {values.shape}"
        )
    if values.size == 0:
        return np.zeros((0, NATHAB_NUM_AXES), dtype=np.uint8)
    finite_min = int(np.min(values))
    finite_max = int(np.max(values))
    if finite_min < 0 or finite_max > NATHAB_CARHAB_MAX_ID:
        raise ValueError(
            "natural_habitat CarHab ids in "
            f"[{finite_min}, {finite_max}] exceed [0, {NATHAB_CARHAB_MAX_ID}]"
        )
    index = values.astype(np.intp, copy=False)
    stacked = np.stack(
        [np.asarray(lut, dtype=np.uint8)[index] for lut in NATHAB_AXIS_LUTS],
        axis=1,
    )
    return stacked


def unpack_nathab_axes(
    axes: np.ndarray,
    *,
    dtype: np.dtype = np.int32,
) -> Dict[str, np.ndarray]:
    """Split ``(N, 4)`` axes into per-task ``nathab_*`` vectors."""
    validated = validate_nathab_axes(axes)
    return {
        key: validated[:, column].astype(dtype, copy=False)
        for column, key in enumerate(NATHAB_AXIS_KEYS)
    }


def nathab_axis_ignore_fills(n: int) -> Dict[str, np.ndarray]:
    """Per-axis void fill arrays of length ``n`` (missing-tile path)."""
    return {
        key: np.full(n, int(ignore), dtype=np.int32)
        for key, ignore in zip(NATHAB_AXIS_KEYS, NATHAB_AXIS_IGNORE_INDEX)
    }


def axis_key_to_definition(axis_key: str) -> str:
    try:
        index = NATHAB_AXIS_KEYS.index(axis_key)
    except ValueError as exc:
        raise KeyError(f"unknown nathab axis key: {axis_key!r}") from exc
    return NATHAB_AXIS_DEFINITIONS[index]


def axis_definition_names() -> Sequence[str]:
    return NATHAB_AXIS_DEFINITIONS
