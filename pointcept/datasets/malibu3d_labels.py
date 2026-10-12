"""
MALiBU3D label layout, as in ``labels.json`` of the Hugging Face release.

- ``segment.npy``: uint8 ``(N,)`` land cover, 15 classes, Void = 15.
- ``natural_habitat.npy``: uint8 ``(N, 4)``, one column per ecological axis in
  ``NATHAB_AXES`` order; the last class of each axis is Void.
"""

from typing import Dict, Tuple

import numpy as np

SEGMENT_NAMES: Tuple[str, ...] = (
    "Building",
    "Greenhouse",
    "Impervious surface",
    "Other soil",
    "Herbaceous",
    "Vineyard",
    "Brushwood",
    "Infrastructure",
    "Swimming pool",
    "Water",
    "Deciduous",
    "Coniferous",
    "Bridge",
    "Agricultural soil",
    "Soil under vegetation",
    "Void",
)

# Column order of natural_habitat.npy -> class names (Void last).
NATHAB_AXES: Dict[str, Tuple[str, ...]] = {
    "nathab_habitat_type": ("Open", "Forest", "Mineral", "Aquatic", "Void"),
    "nathab_moisture_regime": ("Humid", "Mesic", "Dry", "Void"),
    "nathab_soil_chemistry": ("Acidic", "Alkaline", "Void"),
    "nathab_bioclimatic_zone": ("Temperate", "Mediterranean", "Alpine", "Void"),
}
NATHAB_AXIS_KEYS: Tuple[str, ...] = tuple(NATHAB_AXES)


def semantic_task_config(names: Tuple[str, ...]) -> Dict[str, object]:
    """Task config of a label set whose last class is Void (the ignore index)."""
    return {
        "num_classes": len(names) - 1,
        "ignore_index": len(names) - 1,
        "names": list(names),
        "task_type": "semantic",
    }


def unpack_nathab_axes(axes: np.ndarray) -> Dict[str, np.ndarray]:
    """Split ``natural_habitat.npy`` ``(N, 4)`` into one int32 ``(N,)`` vector per axis."""
    axes = np.asarray(axes)
    if axes.ndim != 2 or axes.shape[1] != len(NATHAB_AXES):
        raise ValueError(
            f"natural_habitat.npy: expected shape (N, {len(NATHAB_AXES)}), got {axes.shape}"
        )
    out = {}
    for column, (key, names) in enumerate(NATHAB_AXES.items()):
        labels = axes[:, column].astype(np.int32)
        if labels.size and (labels.min() < 0 or labels.max() > len(names) - 1):
            raise ValueError(
                f"natural_habitat.npy column {column} ({key}): ids outside [0, {len(names) - 1}]"
            )
        out[key] = labels
    return out
