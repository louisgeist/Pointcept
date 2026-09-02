"""
Malibu3D multi-target label configs (semantic class names / counts and elevation regression).

Edit MALIBU3D_SEMANTIC_TASKS if your on-disk label ids differ from these defaults.
For each semantic task, names[i] is the display name for processed label id i.
num_classes is the number of classes the model trains on (logit dimension); ignore_index labels
are excluded from the loss. num_raw_classes (when present) is the number of distinct ids in
source rasters before preprocessing remap (max raw id + 1).


This file is used to configure, for each task/target:
 - Number of classes
 - Ignore index
 - Names
 - Loss
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pointcept.datasets.preprocessing.malibu3d.natural_habitat_multilabel_tile_labels import (
    MULTILABEL_CLASS_NAMES,
    NUM_MULTILABEL_CLASSES,
)

# Semantic targets: one entry per target_key used by Malibu3DDataset / configs.
MALIBU3D_SEMANTIC_TASKS: Dict[str, Dict[str, Any]] = {
    "segment": {
        "num_classes": 15,
        "ignore_index": 15,
        # V14
        "names": [
            'Building',
            'Greenhouse',
            'Impervious surface',
            'Other soil',
            'Herbaceous',
            'Vineyard',
            'Other vegetation',
            'Other infrastructures',
            'Swimming pool',
            'Water',
            'Deciduous',
            'Coniferous',
            'Bridge',
            'Agricultural soil',
            'Soil under vegetation',
            'Void',
        ],
    },
    "forest": {
        "num_classes": 2,
        "ignore_index": 2,
        "names": ["Not Forest", "Forest", "Void"],
    },
    "land_use": {
        "num_classes": 20,
        "ignore_index": -1,
        "names": [
            "Agriculture",
            "Sylviculture",
            "Activites extraction",
            "Peche et aquaculture",
            "Autres productions primaires",
            "Production secondaire",
            "Production secondaire tertiaire residentiel",
            "Production tertiaire",
            "Reseaux routiers",
            "Reseaux ferres",
            "Reseaux aeriens",
            "Reseaux fluvial maritime",
            "Autres reseaux transport",
            "Services logistiques stockage",
            "Reseaux utilite publique",
            "Usage residentiel",
            "Zones en transition",
            "Zones abandonnees",
            "Sans usage",
            "Usage inconnu",
        ],
    },
    "natural_habitat": {
        # CarHab raster uses 42=N/A and 43=Autre (routes). Preprocessing remaps to 43=void, 42=routes.
        # Missing raster samples use fill_value=42 (raw), then remap to ignore_index 43.
        "num_classes": 43,
        "num_raw_classes": 44,
        "ignore_index": 43,
        "names": [
            "Habitat ouvert sur substrat acide et humide du domaine tempéré",
            "Habitat ouvert sur substrat acide et mésique du domaine tempéré",
            "Habitat ouvert sur substrat acide et sec du domaine tempéré",
            "Habitat ouvert sur substrat basique et humide du domaine tempéré",
            "Habitat ouvert sur substrat basique et mésique du domaine tempéré",
            "Habitat ouvert sur substrat basique et sec du domaine tempéré",
            "Habitat forestier sur substrat acide et humide du domaine tempéré",
            "Habitat forestier sur substrat acide et mésique du domaine tempéré",
            "Habitat forestier sur substrat acide et sec du domaine tempéré",
            "Habitat forestier sur substrat basique et humide du domaine tempéré",
            "Habitat forestier sur substrat basique et mésique du domaine tempéré",
            "Habitat forestier sur substrat basique et sec du domaine tempéré",
            "Habitat ouvert sur substrat acide et humide du domaine méditerranéen",
            "Habitat ouvert sur substrat acide et mésique du domaine méditerranéen",
            "Habitat ouvert sur substrat acide et sec du domaine méditerranéen",
            "Habitat ouvert sur substrat basique et humide du domaine méditerranéen",
            "Habitat ouvert sur substrat basique et mésique du domaine méditerranéen",
            "Habitat ouvert sur substrat basique et sec du domaine méditerranéen",
            "Habitat forestier sur substrat acide et humide du domaine méditerranéen",
            "Habitat forestier sur substrat acide et mésique du domaine méditerranéen",
            "Habitat forestier sur substrat acide et sec du domaine méditerranéen",
            "Habitat forestier sur substrat basique et humide du domaine méditerranéen",
            "Habitat forestier sur substrat basique et mésique du domaine méditerranéen",
            "Habitat forestier sur substrat basique et sec du domaine méditerranéen",
            "Habitat ouvert sur substrat acide et humide du domaine alpin",
            "Habitat ouvert sur substrat acide et mésique du domaine alpin",
            "Habitat ouvert sur substrat acide et sec du domaine alpin",
            "Habitat ouvert sur substrat basique et humide du domaine alpin",
            "Habitat ouvert sur substrat basique et mésique du domaine alpin",
            "Habitat ouvert sur substrat basique et sec du domaine alpin",
            "Habitat forestier sur substrat acide et humide du domaine alpin",
            "Habitat forestier sur substrat acide et mésique du domaine alpin",
            "Habitat forestier sur substrat acide et sec du domaine alpin",
            "Habitat forestier sur substrat basique et humide du domaine alpin",
            "Habitat forestier sur substrat basique et mésique du domaine alpin",
            "Habitat forestier sur substrat basique et sec du domaine alpin",
            "Habitat minéral sur substrat acide",
            "Habitat minéral sur substrat basique",
            "Habitat aquatique sur substrat acide",
            "Habitat aquatique sur substrat basique",
            "Habitat cultivé",
            "Zone bâtie et autre habitat artificiel",
            "Routes & voies verrées",
            "Void",
        ],
    },
}

MALIBU3D_CLASSIFICATION_TASKS: Dict[str, Dict[str, Any]] = {
    "climatic_domain": {
        "task_type": "classification",
        "num_classes": 3,
        "ignore_index": -1,
        "names": ["Temperate", "Mediterranean", "Alpine"],
        "pooling": "mean",
    },
}

MALIBU3D_MULTILABEL_CLASSIFICATION_TASKS: Dict[str, Dict[str, Any]] = {
    "natural_habitat_multilabel": {
        "task_type": "multilabel_classification",
        "num_classes": NUM_MULTILABEL_CLASSES,
        "ignore_index": -1,
        "names": list(MULTILABEL_CLASS_NAMES),
        "pooling": "mean",
        "threshold": 0.5,
    },
}

# Binary 2D pixel semantic tasks (1 m Lambert masks). Not point-wise.
# Training / heads use ROADS + RAILROADS only. On-disk ``network.npy`` may still
# be (3, H, W) with TRANSMISSION_LINES as channel 2; Malibu3DDataset slices to these.
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
        # Buffer ("relaxed") precision/recall/F1 tolerance, in pixels (1 px = 1 m on
        # this task's Lambert grid). See MultiTaskEvaluator / MultiTaskTester
        # pixel_semantic handling (val and test).
        "buffer_radius_px": 3, # for dilated recall/precision/f1
        "enable_dilated_prf": True,
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

# Nathab ecological-axis distribution tasks: point-wise per-axis categorical labels
# derived on the fly (via Malibu3DLabelRemap fan-out) from the raw ``natural_habitat``
# CarHab asset. Values are the default ``malibu3d_label_remap`` definition name (under
# task_key="natural_habitat") each axis resolves to.
MALIBU3D_TILE_DISTRIBUTION_TASKS: Dict[str, str] = {
    "nathab_habitat_type": "by_habitat_type_ecological",
    "nathab_moisture_regime": "by_moisture_regime",
    "nathab_soil_chemistry": "by_soil_chemistry",
    "nathab_bioclimatic_zone": "by_climatic_domain",
}

MALIBU3D_SEMANTIC_TARGET_KEYS: Tuple[str, ...] = tuple(MALIBU3D_SEMANTIC_TASKS.keys())
MALIBU3D_CLASSIFICATION_TARGET_KEYS: Tuple[str, ...] = tuple(MALIBU3D_CLASSIFICATION_TASKS.keys())
MALIBU3D_MULTILABEL_CLASSIFICATION_TARGET_KEYS: Tuple[str, ...] = tuple(
    MALIBU3D_MULTILABEL_CLASSIFICATION_TASKS.keys()
)
MALIBU3D_PIXEL_SEMANTIC_TARGET_KEYS: Tuple[str, ...] = tuple(
    MALIBU3D_PIXEL_SEMANTIC_TASKS.keys()
)
MALIBU3D_TILE_DISTRIBUTION_TARGET_KEYS: Tuple[str, ...] = tuple(
    MALIBU3D_TILE_DISTRIBUTION_TASKS.keys()
)

# Keys passed to GridSample / index_valid_keys (superset of all Malibu3D targets).
# Only **per-point** arrays belong here. Scene-level tensors (e.g. ``coord_translation``
# shape (3,), or ``network`` raster (C,H,W) before NetworkRasterToPointLabels) must
# NOT be listed — GridSample would index them with point ids → IndexError.
# ``network`` / ``network_cell`` / ``network_pix`` are appended by
# NetworkRasterToPointLabels after GridSample.
# ``abs_xy`` is a pre-aug float64 per-point buffer consumed by that transform.
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
    "forest",
    "land_use",
    "natural_habitat",
    "elevation",
    "abs_xy",
    "nathab_habitat_type",
    "nathab_moisture_regime",
    "nathab_soil_chemistry",
    "nathab_bioclimatic_zone",
)

# Optional Collect prefix keys before target_keys (backbone-specific).
MALIBU3D_COLLECT_PREFIX_NONE: Tuple[str, ...] = ()
MALIBU3D_COLLECT_PREFIX_GRID: Tuple[str, ...] = ("grid_coord",)
MALIBU3D_COLLECT_PREFIX_LITEPT: Tuple[str, ...] = ("grid_coord", "grid_size")

# Point-wise elevation regression (not class indices).
ELEVATION_TARGET_SCALE = 0.01
ELEVATION_SMOOTH_L1_BETA = 1e-2

MALIBU3D_ELEVATION: Dict[str, Any] = {
    "wandb_target_display_name": "elevation",
    "dtype": "float32",
    "unit": "meters",
    "use_nan_mask": True,
    "target_scale": ELEVATION_TARGET_SCALE,
}


def get_semantic_num_raw_classes(
    target_key: str,
    definition: Optional[str] = None,
) -> int:
    """Return the number of distinct label ids in source data (max raw id + 1).

    Falls back to num_classes when num_raw_classes is not defined (no raster remap).
    When ``definition`` is set, reads from ``malibu3d_label_remap`` for that task.
    """
    cfg = get_semantic_config(target_key, definition=definition)
    return int(cfg.get("num_raw_classes", cfg["num_classes"]))


def get_semantic_config(
    target_key: str,
    definition: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a deep copy of the semantic config for the given target_key.

    When ``definition`` is omitted, uses ``DEFAULT_LABEL_DEFINITION_NAMES`` from
    ``malibu3d_label_remap`` (land_use=default, natural_habitat=by_habitat_x_domain, …).

    Adds task_type set to "semantic" for use with MultiTaskSegmentorV2.
    """
    if target_key not in MALIBU3D_SEMANTIC_TASKS:
        keys = ", ".join(sorted(MALIBU3D_SEMANTIC_TASKS.keys()))
        raise KeyError(f"Unknown semantic target_key '{target_key}'. Expected one of: {keys}")

    from pointcept.datasets.preprocessing.malibu3d.malibu3d_label_remap import (
        definition_to_task_config,
        get_default_definition_name,
        get_definition,
    )

    defn_name = definition if definition is not None else get_default_definition_name(
        target_key
    )
    return deepcopy(definition_to_task_config(get_definition(target_key, defn_name)))


def get_classification_config(target_key: str) -> Dict[str, Any]:
    if target_key not in MALIBU3D_CLASSIFICATION_TASKS:
        keys = ", ".join(sorted(MALIBU3D_CLASSIFICATION_TASKS.keys()))
        raise KeyError(
            f"Unknown classification target_key '{target_key}'. Expected one of: {keys}"
        )
    return deepcopy(MALIBU3D_CLASSIFICATION_TASKS[target_key])


def get_multilabel_classification_config(target_key: str) -> Dict[str, Any]:
    if target_key not in MALIBU3D_MULTILABEL_CLASSIFICATION_TASKS:
        keys = ", ".join(sorted(MALIBU3D_MULTILABEL_CLASSIFICATION_TASKS.keys()))
        raise KeyError(
            f"Unknown multilabel classification target_key '{target_key}'. "
            f"Expected one of: {keys}"
        )
    return deepcopy(MALIBU3D_MULTILABEL_CLASSIFICATION_TASKS[target_key])


def get_elevation_config() -> Dict[str, Any]:
    return deepcopy(MALIBU3D_ELEVATION)


def get_regression_target_scales(target_keys: Tuple[str, ...]) -> Dict[str, float]:
    """Return per-target scale factors for regression keys in target_keys."""
    scales: Dict[str, float] = {}
    if "elevation" in target_keys:
        scales["elevation"] = ELEVATION_TARGET_SCALE
    return scales


def get_multitask_regression_task_config_elevation() -> Dict[str, Any]:
    """Task config dict for point-wise elevation regression in MultiTaskSegmentorV2.

    Expects an "elevation" tensor in input_dict when Malibu3D loads elevation (elevation
    listed in target_keys), i.e. task targets are keyed by task_name.
    """
    out = deepcopy(MALIBU3D_ELEVATION)
    out["task_type"] = "regression"
    return out

def get_network_config() -> Dict[str, Any]:
    """Task config for 1 m Lambert binary network pixel segmentation."""
    return deepcopy(MALIBU3D_PIXEL_SEMANTIC_TASKS["network"])


def get_pixel_semantic_config(target_key: str) -> Dict[str, Any]:
    if target_key not in MALIBU3D_PIXEL_SEMANTIC_TASKS:
        keys = ", ".join(sorted(MALIBU3D_PIXEL_SEMANTIC_TASKS.keys()))
        raise KeyError(
            f"Unknown pixel_semantic target_key '{target_key}'. Expected one of: {keys}"
        )
    return deepcopy(MALIBU3D_PIXEL_SEMANTIC_TASKS[target_key])


def get_tile_distribution_config(
    target_key: str,
    definition: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a deep copy of the tile-distribution axis config for the given target_key.

    Each nathab axis target_key (see ``MALIBU3D_TILE_DISTRIBUTION_TASKS``) is derived
    from the ``natural_habitat`` LabelDefinition registry (raw CarHab ids), not from
    an eponymous registry entry of its own — unlike semantic tasks, where target_key
    IS the ``malibu3d_label_remap`` task_key.

    Adds task_type set to "tile_distribution" for use with MultiTaskSegmentorV2.
    """
    if target_key not in MALIBU3D_TILE_DISTRIBUTION_TASKS:
        keys = ", ".join(sorted(MALIBU3D_TILE_DISTRIBUTION_TASKS.keys()))
        raise KeyError(
            f"Unknown tile_distribution target_key '{target_key}'. Expected one of: {keys}"
        )

    from pointcept.datasets.preprocessing.malibu3d.malibu3d_label_remap import (
        definition_to_task_config,
        get_definition,
    )

    defn_name = (
        definition if definition is not None else MALIBU3D_TILE_DISTRIBUTION_TASKS[target_key]
    )
    cfg = deepcopy(definition_to_task_config(get_definition("natural_habitat", defn_name)))
    cfg["task_type"] = "tile_distribution"
    return cfg


def _validate_target_keys(target_keys: Tuple[str, ...]) -> None:
    known = (
        set(MALIBU3D_SEMANTIC_TASKS.keys())
        | set(MALIBU3D_CLASSIFICATION_TASKS.keys())
        | set(MALIBU3D_MULTILABEL_CLASSIFICATION_TASKS.keys())
        | set(MALIBU3D_PIXEL_SEMANTIC_TASKS.keys())
        | set(MALIBU3D_TILE_DISTRIBUTION_TASKS.keys())
        | {"elevation"}
    )
    for key in target_keys:
        if key not in known:
            keys = ", ".join(sorted(known))
            raise KeyError(f"Unknown target_key '{key}'. Expected one of: {keys}")


def _is_classification_target(target_key: str) -> bool:
    return target_key in MALIBU3D_CLASSIFICATION_TASKS


def _is_multilabel_classification_target(target_key: str) -> bool:
    return target_key in MALIBU3D_MULTILABEL_CLASSIFICATION_TASKS


def _is_pixel_semantic_target(target_key: str) -> bool:
    return target_key in MALIBU3D_PIXEL_SEMANTIC_TASKS


def _is_scene_level_target(target_key: str) -> bool:
    return (
        _is_classification_target(target_key)
        or _is_multilabel_classification_target(target_key)
        or _is_pixel_semantic_target(target_key)
    )


def init_multitask_collect_keys(
    target_keys: Tuple[str, ...],
    *,
    collect_prefix_keys: Tuple[str, ...] = (),
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Build train/val Collect keys and index_valid_keys for Malibu3D configs.

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
    return train_keys, val_keys, MALIBU3D_MULTITASK_INDEX_VALID_KEYS


def init_task_configs(
    target_keys: Tuple[str, ...],
    definitions: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Initialize the task config dictionary for the given target_keys.

    ``definitions`` maps task names to label definition names in ``malibu3d_label_remap``
    (must match preprocessing v2 ``--{task}_definition`` flags).
    """
    definitions = definitions or {}
    out = {}
    for task_name in target_keys:
        if task_name in MALIBU3D_SEMANTIC_TASKS:
            out[task_name] = get_semantic_config(
                task_name,
                definition=definitions.get(task_name),
            )
        elif task_name == "elevation":
            out[task_name] = get_multitask_regression_task_config_elevation()
        elif task_name in MALIBU3D_CLASSIFICATION_TASKS:
            out[task_name] = get_classification_config(task_name)
        elif task_name in MALIBU3D_MULTILABEL_CLASSIFICATION_TASKS:
            out[task_name] = get_multilabel_classification_config(task_name)
        elif task_name in MALIBU3D_PIXEL_SEMANTIC_TASKS:
            out[task_name] = get_pixel_semantic_config(task_name)
        elif task_name in MALIBU3D_TILE_DISTRIBUTION_TASKS:
            out[task_name] = get_tile_distribution_config(
                task_name,
                definition=definitions.get(task_name),
            )
        else:
            raise KeyError(
                f"Unknown task_name '{task_name}'. Expected one of: "
                f"{sorted((*MALIBU3D_SEMANTIC_TASKS.keys(), *MALIBU3D_CLASSIFICATION_TASKS.keys(), *MALIBU3D_MULTILABEL_CLASSIFICATION_TASKS.keys(), *MALIBU3D_PIXEL_SEMANTIC_TASKS.keys(), *MALIBU3D_TILE_DISTRIBUTION_TASKS.keys(), 'elevation'))}"
            )
    return out

def get_missing_target_fill_value(
    target_key: str,
    pixel_semantic_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Return the fallback value used when a target file is missing.

    - Semantic targets fallback to the on-disk *storage* definition's own ignore_index
      (not the training-taxonomy one). This matters when a target is remapped on the
      fly (e.g. via Malibu3DLabelRemap): a negative sentinel (e.g. land_use's -1) is
      treated as an invalid/out-of-range raw id by the remap, which then correctly
      substitutes *its own* target-specific void id regardless of which training
      definition is in play; a non-negative storage ignore_index (e.g.
      natural_habitat's "default" definition, 43) is a valid raw/stored id that any
      natural_habitat-derived LUT (by_habitat_x_domain, the 4 nathab axes, ...) was
      built to resolve to void. Using the *training*-taxonomy ignore_index instead
      (the previous behavior) broke this for natural_habitat specifically, since its
      training-default definition's ignore_index (11, for by_habitat_x_domain) is a
      small non-negative number that collides with a real raw CarHab id.
    - Multilabel targets fallback to ignore_index on all labels (missing NH raster).
    - Elevation regression falls back to NaN so masked losses ignore it.
    - Network pixel masks fallback to an empty (r, 1, 1) zero raster (absence).

    ``pixel_semantic_config`` optionally overrides the registry lookup for
    pixel_semantic targets (e.g. a per-config ``num_networks``/``channel_names``
    override, such as dropping RAILROADS from the ``network`` task) — pass the
    caller's ``task_configs[target_key]`` when it was constructed with such an
    override so the missing-tile fallback shape stays consistent with it.
    """
    if target_key in MALIBU3D_SEMANTIC_TASKS:
        from pointcept.datasets.preprocessing.malibu3d.malibu3d_label_remap import (
            get_definition,
        )

        storage_def = get_definition(target_key, "default")
        return int(storage_def.ignore_index)
    if target_key in MALIBU3D_CLASSIFICATION_TASKS:
        return int(get_classification_config(target_key)["ignore_index"])
    if target_key in MALIBU3D_MULTILABEL_CLASSIFICATION_TASKS:
        cfg = get_multilabel_classification_config(target_key)
        ignore_index = int(cfg.get("ignore_index", -1))
        return np.full(
            (1, int(cfg["num_classes"])),
            ignore_index,
            dtype=np.float32,
        )
    if target_key in MALIBU3D_PIXEL_SEMANTIC_TASKS:
        cfg = pixel_semantic_config or get_pixel_semantic_config(target_key)
        r = int(cfg["num_networks"])
        return np.zeros((r, 1, 1), dtype=np.uint8)
    if target_key == "elevation":
        return float("nan")
    keys = ", ".join(
        sorted(
            (
                *MALIBU3D_SEMANTIC_TASKS.keys(),
                *MALIBU3D_CLASSIFICATION_TASKS.keys(),
                *MALIBU3D_MULTILABEL_CLASSIFICATION_TASKS.keys(),
                *MALIBU3D_PIXEL_SEMANTIC_TASKS.keys(),
                "elevation",
            )
        )
    )
    raise KeyError(f"Unknown target_key '{target_key}'. Expected one of: {keys}")

def init_task_criteria(task_configs: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize the task criteria dictionary for the given target_keys.
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
        elif task_type == "classification":
            task_criteria[task_name] = [
                dict(
                    type="CrossEntropyLoss",
                    loss_weight=1.0,
                    ignore_index=task_config["ignore_index"],
                ),
            ]
        elif task_type == "multilabel_classification":
            task_criteria[task_name] = [
                dict(
                    type="BCEWithLogitsLoss",
                    loss_weight=1.0,
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
            task_criteria[task_name] = [
                dict(
                    type="WeightedKLDivLoss",
                    loss_weight=1.0,
                ),
            ]
        else:
            raise KeyError(f"Unsupported task_type {task_type!r} for task '{task_name}'.")
    
    return task_criteria