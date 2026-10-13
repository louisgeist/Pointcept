"""
Per-task configuration of the MALiBU3D targets, used by configs and by MALiBU3DDataset.

For every target key it provides the task config read by the multitask model, evaluator
and tester (task type, number of classes, ignore index, class names, ...), the default
losses (``init_task_criteria``) and the Collect / index_valid keys of the data pipeline
(``init_multitask_collect_keys``). Targets:

- ``segment``: point-wise land cover (classes in ``malibu3d_labels``).
- ``forest_2d``, ``network``: binary 1 m rasters (``pixel_semantic``).
- ``nathab_*``: the four natural-habitat ecological axes (``tile_distribution``).
- ``elevation``: point-wise height above the DTM (``regression``).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pointcept.datasets.malibu3d_labels import (
    NATHAB_AXES,
    NATHAB_AXIS_KEYS,
    SEGMENT_NAMES,
    semantic_task_config,
)

# Point-wise semantic targets (classes in ``malibu3d_labels``).
MALIBU3D_SEMANTIC_TASKS: Tuple[str, ...] = ("segment",)

# Binary 2D pixel semantic tasks (1 m Lambert-93 masks), not point-wise.
# ``network`` trains on ROADS + RAILROADS. rasterize_network.py also writes
# TRANSMISSION_LINES as channel 2; MALiBU3DDataset slices it away.
NETWORK_CHANNEL_NAMES: Tuple[str, ...] = (
    "ROADS",
    "RAILROADS",
)
MALIBU3D_PIXEL_SEMANTIC_TASKS: Dict[str, Dict[str, Any]] = {
    "network": {
        "task_type": "pixel_semantic",
        "num_networks": 2,
        "num_classes": 2,
        "ignore_index": 2,
        "channel_names": list(NETWORK_CHANNEL_NAMES),
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
        # Buffered metrics are meant for thin curvilinear masks (road/rail), not for an
        # area-coverage mask like forest.
        "enable_dilated_prf": False,
    },
}

# Natural-habitat ecological axes: point-wise categorical labels per axis, trained by
# default as tile-level class distributions. ``natural_habitat.npy`` is ``uint8 (N, 4)``;
# MALiBU3DDataset unpacks its columns into these ``nathab_*`` keys.
MALIBU3D_TILE_DISTRIBUTION_TASKS: Tuple[str, ...] = NATHAB_AXIS_KEYS

MALIBU3D_SEMANTIC_TARGET_KEYS: Tuple[str, ...] = MALIBU3D_SEMANTIC_TASKS
MALIBU3D_PIXEL_SEMANTIC_TARGET_KEYS: Tuple[str, ...] = tuple(
    MALIBU3D_PIXEL_SEMANTIC_TASKS.keys()
)
MALIBU3D_TILE_DISTRIBUTION_TARGET_KEYS: Tuple[str, ...] = (
    MALIBU3D_TILE_DISTRIBUTION_TASKS
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

# Point-wise elevation regression. The target (meters) is multiplied by
# ``target_scale`` before the loss; NaN targets (no DTM) are masked out.
ELEVATION_TARGET_SCALE = 0.01
ELEVATION_SMOOTH_L1_BETA = 1e-2

MALIBU3D_ELEVATION: Dict[str, Any] = {
    "wandb_target_display_name": "elevation",
    "dtype": "float32",
    "unit": "meters",
    "use_nan_mask": True,
    "target_scale": ELEVATION_TARGET_SCALE,
}


def get_semantic_config(target_key: str) -> Dict[str, Any]:
    """Return the semantic task config of ``target_key`` (``task_type="semantic"``)."""
    if target_key not in MALIBU3D_SEMANTIC_TASKS:
        keys = ", ".join(sorted(MALIBU3D_SEMANTIC_TASKS))
        raise KeyError(
            f"Unknown semantic target_key '{target_key}'. Expected one of: {keys}"
        )
    return semantic_task_config(SEGMENT_NAMES)


def get_multitask_regression_task_config_elevation() -> Dict[str, Any]:
    """Task config for point-wise elevation regression (``task_type="regression"``).

    The model reads the target from ``input_dict["elevation"]``, which MALiBU3DDataset
    provides when ``"elevation"`` is in its target_keys.
    """
    out = deepcopy(MALIBU3D_ELEVATION)
    out["task_type"] = "regression"
    return out


def get_pixel_semantic_config(target_key: str) -> Dict[str, Any]:
    if target_key not in MALIBU3D_PIXEL_SEMANTIC_TASKS:
        keys = ", ".join(sorted(MALIBU3D_PIXEL_SEMANTIC_TASKS.keys()))
        raise KeyError(
            f"Unknown pixel_semantic target_key '{target_key}'. Expected one of: {keys}"
        )
    return deepcopy(MALIBU3D_PIXEL_SEMANTIC_TASKS[target_key])


def get_tile_distribution_config(target_key: str) -> Dict[str, Any]:
    """Return the config of one natural-habitat axis (``nathab_*``).

    The config has ``task_type="tile_distribution"``: by default the model is trained
    with a weighted KL loss on the tile-level class distribution. Use
    ``enable_nathab_pointwise_supervision`` to train with CE + Lovasz on the point
    labels instead; inference and the KL / TV metrics stay tile-level either way.
    """
    if target_key not in MALIBU3D_TILE_DISTRIBUTION_TASKS:
        keys = ", ".join(sorted(MALIBU3D_TILE_DISTRIBUTION_TASKS))
        raise KeyError(
            f"Unknown tile_distribution target_key '{target_key}'. Expected one of: {keys}"
        )
    cfg = semantic_task_config(NATHAB_AXES[target_key])
    cfg["task_type"] = "tile_distribution"
    cfg["pointwise_supervision"] = False
    return cfg


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


def _validate_target_keys(target_keys: Tuple[str, ...]) -> None:
    known = (
        set(MALIBU3D_SEMANTIC_TASKS)
        | set(MALIBU3D_PIXEL_SEMANTIC_TASKS.keys())
        | set(MALIBU3D_TILE_DISTRIBUTION_TASKS)
        | {"elevation"}
    )
    for key in target_keys:
        if key not in known:
            keys = ", ".join(sorted(known))
            raise KeyError(f"Unknown target_key '{key}'. Expected one of: {keys}")


def _is_pixel_semantic_target(target_key: str) -> bool:
    return target_key in MALIBU3D_PIXEL_SEMANTIC_TASKS


def init_multitask_collect_keys(
    target_keys: Tuple[str, ...],
    *,
    collect_prefix_keys: Tuple[str, ...] = (),
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Return ``(train_keys, val_keys, index_valid_keys)`` for MALiBU3D configs.

    - train: coord + collect_prefix_keys + pixel-task grid keys + target_keys
    - val: coord + collect_prefix_keys + pixel-task grid keys + ``(key, origin_key)``
      per point-wise target (``key`` alone for pixel tasks) + inverse

    The grid keys of a pixel_semantic target ``key`` are ``{key}_cell`` / ``{key}_pix``
    (int64 cell indices from ``NetworkRasterToPointLabels(target_key=key)``),
    ``{key}_origin_x/y``, ``{key}_pixel_m`` and ``{key}_height/width`` (dense test).
    Each pixel task has its own key set, so ``network`` and ``forest_2d`` can coexist.
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
        if _is_pixel_semantic_target(key):
            val_target_keys += (key,)
        else:
            val_target_keys += (key, f"origin_{key}")
    val_keys = base + extra + val_target_keys + ("inverse",)
    return train_keys, val_keys, MALIBU3D_MULTITASK_INDEX_VALID_KEYS


def init_task_configs(target_keys: Tuple[str, ...]) -> Dict[str, Any]:
    """Return ``{task_name: task_config}`` for the given target_keys."""
    out = {}
    for task_name in target_keys:
        if task_name in MALIBU3D_SEMANTIC_TASKS:
            out[task_name] = get_semantic_config(task_name)
        elif task_name == "elevation":
            out[task_name] = get_multitask_regression_task_config_elevation()
        elif task_name in MALIBU3D_PIXEL_SEMANTIC_TASKS:
            out[task_name] = get_pixel_semantic_config(task_name)
        elif task_name in MALIBU3D_TILE_DISTRIBUTION_TASKS:
            out[task_name] = get_tile_distribution_config(task_name)
        else:
            raise KeyError(
                f"Unknown task_name '{task_name}'. Expected one of: "
                f"{sorted((*MALIBU3D_SEMANTIC_TASKS, *MALIBU3D_PIXEL_SEMANTIC_TASKS.keys(), *MALIBU3D_TILE_DISTRIBUTION_TASKS, 'elevation'))}"
            )
    return out


def get_missing_target_fill_value(
    target_key: str,
    pixel_semantic_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Return the fallback value used when a target file is missing.

    - Semantic targets fall back to their ignore_index (Void).
    - Natural-habitat axes fall back to each axis' void id.
    - Elevation regression falls back to NaN so masked losses ignore it.
    - Pixel masks fall back to an all-zero ``(r, 1, 1)`` raster (no positive pixel).

    ``pixel_semantic_config`` overrides the registry lookup for pixel_semantic
    targets: pass the caller's ``task_configs[target_key]`` when it changes
    ``num_networks`` (e.g. RAILROADS dropped from ``network``) so the fallback keeps
    the same channel count.
    """
    if target_key in MALIBU3D_SEMANTIC_TASKS:
        return get_semantic_config(target_key)["ignore_index"]
    if target_key in MALIBU3D_TILE_DISTRIBUTION_TASKS:
        return get_tile_distribution_config(target_key)["ignore_index"]
    if target_key in MALIBU3D_PIXEL_SEMANTIC_TASKS:
        cfg = pixel_semantic_config or get_pixel_semantic_config(target_key)
        r = int(cfg["num_networks"])
        return np.zeros((r, 1, 1), dtype=np.uint8)
    if target_key == "elevation":
        return float("nan")
    keys = ", ".join(
        sorted(
            (
                *MALIBU3D_SEMANTIC_TASKS,
                *MALIBU3D_PIXEL_SEMANTIC_TASKS.keys(),
                *MALIBU3D_TILE_DISTRIBUTION_TASKS,
                "elevation",
            )
        )
    )
    raise KeyError(f"Unknown target_key '{target_key}'. Expected one of: {keys}")


def init_task_criteria(task_configs: Dict[str, Any]) -> Dict[str, Any]:
    """Return the default losses ``{task_name: [loss cfg, ...]}`` for ``task_configs``.

    semantic / pixel_semantic: CE + Lovasz; regression: SmoothL1; tile_distribution:
    weighted KL, or CE + Lovasz when ``pointwise_supervision`` is set.
    """
    task_criteria = {}
    for task_name, task_config in task_configs.items():
        task_type = task_config.get("task_type", "semantic")
        if task_type in ("semantic", "pixel_semantic"):
            task_criteria[task_name] = [
                dict(
                    type="CrossEntropyLoss",
                    loss_weight=1.0,
                    ignore_index=task_config["ignore_index"],
                ),
                dict(
                    type="LovaszLoss",
                    mode="multiclass",
                    loss_weight=1.0,
                    ignore_index=task_config["ignore_index"],
                ),
            ]
        elif task_type == "regression":
            task_criteria[task_name] = [
                dict(
                    type="SmoothL1Loss",
                    beta=ELEVATION_SMOOTH_L1_BETA,
                    loss_weight=1.0,
                ),
            ]
        elif task_type == "tile_distribution":
            if task_config.get("pointwise_supervision"):
                task_criteria[task_name] = [
                    dict(
                        type="CrossEntropyLoss",
                        loss_weight=1.0,
                        ignore_index=task_config["ignore_index"],
                    ),
                    dict(
                        type="LovaszLoss",
                        mode="multiclass",
                        loss_weight=1.0,
                        ignore_index=task_config["ignore_index"],
                    ),
                ]
            else:
                task_criteria[task_name] = [
                    dict(
                        type="WeightedKLDivLoss",
                        loss_weight=1.0,
                    ),
                ]
        else:
            raise KeyError(
                f"Unsupported task_type {task_type!r} for task '{task_name}'."
            )

    return task_criteria
