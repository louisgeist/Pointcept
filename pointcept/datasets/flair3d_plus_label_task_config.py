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
        "num_classes": 14,
        "ignore_index": 14,
        #Inter_finerall10
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
        # CarHab-style legend; ids match natural_habitat_classes.txt / raster. N/A (42) is fill for
        # missing raster samples in preprocessing (see preprocess_flair3d.py fill_value=42).
        "num_classes": 44,
        "ignore_index": 42,
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
            "N/A",
            "Autre",
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
