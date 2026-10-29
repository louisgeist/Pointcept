"""
The MALiBU3D tasks: label layout, per-task configuration, losses and pipeline keys.
Used by the configs and by MALiBU3DDataset.

MALiBU3D has five tasks, all trained jointly by the benchmark configs
(``MALIBU3D_TARGET_KEYS``):

- ``segment``: point-wise land cover, 15 classes (``SEGMENT_NAMES``).
- ``forest_2d``: binary 1 m forest raster (``pixel_semantic``).
- ``elevation``: point-wise height above the DTM, in meters (``regression``).
- ``nathab_*``: the four natural-habitat ecological axes (``NATHAB_AXES``,
  ``tile_distribution``).
- ``network``: binary 1 m road raster (``pixel_semantic``), scored with APLS at test time.

Label layout, as in ``labels.json`` of the Hugging Face release:

- ``segment.npy``: uint8 ``(N,)`` land cover, Void = 15.
- ``natural_habitat.npy``: uint8 ``(N, 4)``, one column per ecological axis in
  ``NATHAB_AXES`` order; the last class of each axis is Void.

For every target key this module provides the task config read by the multitask model,
evaluator and tester (``init_task_configs``), the losses (``init_task_criteria``) and the
Collect / index_valid keys of the data pipeline (``init_multitask_collect_keys``).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple

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


# Binary 2D pixel semantic tasks (1 m Lambert-93 masks), not point-wise. ``network`` is the
# ROADS mask (the only network graph shipped with the dataset); MALiBU3DDataset keeps that
# channel when an older ``network.npy`` also has RAILROADS / TRANSMISSION_LINES.
MALIBU3D_PIXEL_SEMANTIC_TASKS: Dict[str, Dict[str, Any]] = {
    "network": {
        "task_type": "pixel_semantic",
        "num_networks": 1,
        "num_classes": 2,
        "ignore_index": 2,
        "channel_names": ["ROADS"],
        "names": ["Background", "Foreground", "Void"],
        # Tolerance of the buffered ("relaxed") precision/recall/F1, in pixels (1 px = 1 m),
        # reported by MultiTaskEvaluator / MultiTaskTester on val and test.
        "buffer_radius_px": 3,
        "enable_dilated_prf": True,
    },
    "forest_2d": {
        "task_type": "pixel_semantic",
        "num_networks": 1,
        "num_classes": 2,
        "ignore_index": 2,
        "channel_names": ["FOREST"],
        "names": ["Not Forest", "Forest", "Void"],
        # Area-coverage class: mean-pool point features per cell (network uses the default
        # max-pool, which suits catching a single strong "line here" signal).
        "pooling": "mean",
        # Buffered metrics are meant for thin curvilinear masks (roads), not for an
        # area-coverage mask like forest.
        "enable_dilated_prf": False,
    },
}

# Natural-habitat ecological axes: point-wise categorical labels per axis, trained by
# default as tile-level class distributions. ``natural_habitat.npy`` is ``uint8 (N, 4)``;
# MALiBU3DDataset unpacks its columns into these ``nathab_*`` keys.
MALIBU3D_TILE_DISTRIBUTION_TASKS: Tuple[str, ...] = NATHAB_AXIS_KEYS

# The five tasks of the benchmark, in the order of the multitask configs.
MALIBU3D_TARGET_KEYS: Tuple[str, ...] = (
    "segment",
    "forest_2d",
    "elevation",
    *NATHAB_AXIS_KEYS,
    "network",
)

# index_valid_keys for GridSample & co. (superset of all MALiBU3D targets).
# Only **per-point** arrays belong here. Scene-level arrays (``coord_translation`` (3,),
# a ``(C, H, W)`` raster before NetworkRasterToPointLabels) must NOT be listed: GridSample
# would index them with point ids (IndexError). ``{key}`` / ``{key}_cell`` / ``{key}_pix``
# of pixel_semantic tasks are appended by NetworkRasterToPointLabels once they are
# point-wise. ``abs_xy`` is the float64 per-point XY that transform consumes.
MALIBU3D_MULTITASK_INDEX_VALID_KEYS: Tuple[str, ...] = (
    "coord",
    "color",
    "normal",
    "color_mask",
    "normal_mask",
    "superpoint",
    "strength",
    "strength_mask",
    "segment",
    "instance",
    "elevation",
    "abs_xy",
    "nathab_habitat_type",
    "nathab_moisture_regime",
    "nathab_soil_chemistry",
    "nathab_bioclimatic_zone",
)

# Backbone-specific Collect keys placed before the targets (``collect_prefix_keys``).
MALIBU3D_COLLECT_PREFIX_NONE: Tuple[str, ...] = ()
MALIBU3D_COLLECT_PREFIX_GRID: Tuple[str, ...] = ("grid_coord",)
MALIBU3D_COLLECT_PREFIX_LITEPT: Tuple[str, ...] = ("grid_coord", "grid_size")


def get_task_config(target_key: str) -> Dict[str, Any]:
    """Return the task config of one MALiBU3D target (a fresh copy).

    ``nathab_*`` axes have ``task_type="tile_distribution"``: they are trained with a
    weighted KL loss on the tile-level class distribution, unless
    ``enable_nathab_pointwise_supervision`` switches them to CE + Lovasz on the point
    labels (inference and the KL / TV metrics stay tile-level either way).
    """
    if target_key == "segment":
        return semantic_task_config(SEGMENT_NAMES)
    if target_key == "elevation":
        # The model reads the target (meters) from ``input_dict["elevation"]``; NaN
        # (no DTM) is masked out of the loss and metrics.
        return {"task_type": "regression"}
    if target_key in MALIBU3D_PIXEL_SEMANTIC_TASKS:
        return deepcopy(MALIBU3D_PIXEL_SEMANTIC_TASKS[target_key])
    if target_key in MALIBU3D_TILE_DISTRIBUTION_TASKS:
        cfg = semantic_task_config(NATHAB_AXES[target_key])
        cfg["task_type"] = "tile_distribution"
        cfg["pointwise_supervision"] = False
        return cfg
    raise KeyError(
        f"Unknown target_key '{target_key}'. Expected one of: {MALIBU3D_TARGET_KEYS}"
    )


def init_task_configs(
    target_keys: Tuple[str, ...] = MALIBU3D_TARGET_KEYS,
) -> Dict[str, Any]:
    """Return ``{task_name: task_config}`` for the given target_keys."""
    return {key: get_task_config(key) for key in target_keys}


def enable_nathab_pointwise_supervision(
    task_configs: Dict[str, Any],
) -> Dict[str, Any]:
    """Switch the nathab axes to point-wise CE + Lovasz training.

    Mutates and returns ``task_configs``. Call after ``init_task_configs`` and before
    ``init_task_criteria``. Inference and KL / TV metrics stay tile-level;
    MultiTaskEvaluator / MultiTaskTester also report mIoU when this flag is set.
    """
    for name in MALIBU3D_TILE_DISTRIBUTION_TASKS:
        if name in task_configs:
            task_configs[name]["pointwise_supervision"] = True
    return task_configs


def init_multitask_collect_keys(
    target_keys: Tuple[str, ...],
    *,
    collect_prefix_keys: Tuple[str, ...] = (),
) -> Tuple[Tuple[str, ...], ...]:
    """Return ``(train_keys, val_keys, test_keys, index_valid_keys)`` for MALiBU3D configs.

    - train: coord + collect_prefix_keys + pixel-task grid keys + target_keys
    - val: coord + collect_prefix_keys + pixel-task grid keys + ``(key, origin_key)``
      per point-wise target (``key`` alone for pixel tasks) + inverse
    - test (``post_transform`` of the fragment pipeline): coord + grid_coord (when in
      collect_prefix_keys) + index + the pixel targets and their grid keys; point-wise
      targets are read by the tester from the scene, not from the fragment.

    The grid keys of a pixel_semantic target ``key`` are ``{key}_cell`` / ``{key}_pix``
    (int64 cell indices from ``NetworkRasterToPointLabels(target_key=key)``),
    ``{key}_origin_x/y``, ``{key}_pixel_m`` and ``{key}_height/width`` (dense test).
    """
    for key in target_keys:
        get_task_config(key)  # validate
    base = ("coord",) + collect_prefix_keys

    def grid_keys(key):
        return tuple(
            f"{key}_{suffix}"
            for suffix in (
                "cell",
                "pix",
                "origin_x",
                "origin_y",
                "pixel_m",
                "height",
                "width",
            )
        )

    pixel_keys = [key for key in target_keys if key in MALIBU3D_PIXEL_SEMANTIC_TASKS]
    extra = sum((grid_keys(key) for key in pixel_keys), ())
    train_keys = base + extra + target_keys
    val_target_keys: Tuple[str, ...] = ()
    for key in target_keys:
        if key in MALIBU3D_PIXEL_SEMANTIC_TASKS:
            val_target_keys += (key,)
        else:
            val_target_keys += (key, f"origin_{key}")
    val_keys = base + extra + val_target_keys + ("inverse",)
    test_keys = ("coord",) + tuple(k for k in collect_prefix_keys if k == "grid_coord")
    test_keys += ("index",)
    # Same order as the registry (network, then forest_2d).
    for key in MALIBU3D_PIXEL_SEMANTIC_TASKS:
        if key in pixel_keys:
            test_keys += (key,) + grid_keys(key)
    return train_keys, val_keys, test_keys, MALIBU3D_MULTITASK_INDEX_VALID_KEYS


def get_missing_target_fill_value(target_key: str) -> Any:
    """Return the fallback value used when a target file is missing.

    - Semantic targets and natural-habitat axes fall back to their ignore_index (Void).
    - Elevation regression falls back to NaN so masked losses ignore it.
    - Pixel masks fall back to an all-zero ``(1, 1, 1)`` raster (no positive pixel).
    """
    cfg = get_task_config(target_key)
    if cfg["task_type"] == "regression":
        return float("nan")
    if cfg["task_type"] == "pixel_semantic":
        return np.zeros((int(cfg["num_networks"]), 1, 1), dtype=np.uint8)
    return cfg["ignore_index"]


def _ce_lovasz(ignore_index: int) -> list:
    return [
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=ignore_index,
        ),
    ]


def init_task_criteria(task_configs: Dict[str, Any]) -> Dict[str, Any]:
    """Return the losses ``{task_name: [loss cfg, ...]}`` for ``task_configs``.

    - segment, forest_2d: CE + Lovasz.
    - network: CE alone, with a 5x weight on road pixels.
    - elevation: SmoothL1 with beta=1.0 (L1/L2 transition at 1 m).
    - nathab_*: weighted KL, or CE + Lovasz when ``pointwise_supervision`` is set.
    """
    task_criteria = {}
    for task_name, task_config in task_configs.items():
        task_type = task_config["task_type"]
        if task_name == "network":
            task_criteria[task_name] = [
                dict(
                    type="CrossEntropyLoss",
                    loss_weight=1.0,
                    ignore_index=task_config["ignore_index"],
                    weight=[1.0, 5.0],  # Background, Foreground
                ),
            ]
        elif task_type == "regression":
            task_criteria[task_name] = [
                dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ]
        elif task_type == "tile_distribution" and not task_config.get(
            "pointwise_supervision"
        ):
            task_criteria[task_name] = [dict(type="WeightedKLDivLoss", loss_weight=1.0)]
        else:
            task_criteria[task_name] = _ce_lovasz(task_config["ignore_index"])
    return task_criteria
