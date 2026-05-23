"""
Flair3D+ multi-target label configs (semantic class names / counts and elevation regression).

Edit FLAIR3D_SEMANTIC_TASKS if your on-disk label ids differ from these defaults.
For each semantic task, names[i] is the display name for integer class id i (0 .. num_classes - 1).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple

# Semantic targets: one entry per target_key used by Flair3DDataset / configs.
FLAIR3D_SEMANTIC_TASKS: Dict[str, Dict[str, Any]] = {
    "segment": {
        "num_classes": 15,
        "ignore_index": 15,
        # V12
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
        "names": ["Not Forest", "Forest"],
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
        "num_classes": 44,
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

FLAIR3D_SEMANTIC_TARGET_KEYS: Tuple[str, ...] = tuple(FLAIR3D_SEMANTIC_TASKS.keys())

# Point-wise elevation regression (not class indices).
FLAIR3D_ELEVATION: Dict[str, Any] = {
    "wandb_target_display_name": "elevation",
    "dtype": "float32",
    "unit": "meters",
    "use_nan_mask": True,
    "nmae_offset": 0.5,
}


def get_semantic_config(target_key: str) -> Dict[str, Any]:
    """Return a deep copy of the semantic config for the given target_key.

    Adds task_type set to "semantic" for use with MultiTaskSegmentorV2.
    """
    if target_key not in FLAIR3D_SEMANTIC_TASKS:
        keys = ", ".join(sorted(FLAIR3D_SEMANTIC_TASKS.keys()))
        raise KeyError(f"Unknown semantic target_key '{target_key}'. Expected one of: {keys}")
    out = deepcopy(FLAIR3D_SEMANTIC_TASKS[target_key])
    out["task_type"] = "semantic"
    return out


def get_elevation_config() -> Dict[str, Any]:
    return deepcopy(FLAIR3D_ELEVATION)


def get_multitask_regression_task_config_elevation() -> Dict[str, Any]:
    """Task config dict for point-wise elevation regression in MultiTaskSegmentorV2.

    Expects an "elevation" tensor in input_dict when Flair3D+ loads elevation (elevation
    listed in target_keys), i.e. task targets are keyed by task_name.
    """
    out = deepcopy(FLAIR3D_ELEVATION)
    out["task_type"] = "regression"
    return out

def init_task_configs(target_keys: Tuple[str, ...]) -> Dict[str, Any]:
    """Initialize the task config dictionary for the given target_keys.
    """
    out = {}
    for task_name in target_keys:
        if task_name in FLAIR3D_SEMANTIC_TASKS:
            out[task_name] = get_semantic_config(task_name)
        elif task_name == "elevation":
            out[task_name] = get_multitask_regression_task_config_elevation()
        else:
            raise KeyError(f"Unknown task_name '{task_name}'. Expected one of: {FLAIR3D_SEMANTIC_TASKS.keys()}")
    return out

def get_missing_target_fill_value(target_key: str) -> Any:
    """Return the fallback value used when a target file is missing.

    - Semantic targets fallback to their ignore_index.
    - Elevation regression falls back to NaN so masked losses ignore it.
    """
    if target_key in FLAIR3D_SEMANTIC_TASKS:
        return int(FLAIR3D_SEMANTIC_TASKS[target_key]["ignore_index"])
    if target_key == "elevation":
        return float("nan")
    keys = ", ".join(sorted((*FLAIR3D_SEMANTIC_TASKS.keys(), "elevation")))
    raise KeyError(f"Unknown target_key '{target_key}'. Expected one of: {keys}")

def init_task_criteria(task_configs: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize the task criteria dictionary for the given target_keys.
    """
    task_criteria = {}
    for task_name, task_config in task_configs.items():
        if task_name in FLAIR3D_SEMANTIC_TASKS:
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
        elif task_name == "elevation":
            task_criteria["elevation"] = dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
        else:
            raise KeyError(f"Unknown task_name '{task_name}'. Expected one of: {FLAIR3D_SEMANTIC_TASKS.keys()}")
    
    return task_criteria