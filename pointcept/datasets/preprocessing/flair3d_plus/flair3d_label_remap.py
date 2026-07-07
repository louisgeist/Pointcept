"""
Flair3D+ unified label definitions and LUT remapping. (LUT: look-up table)

Single registry for all semantic tasks (segment, forest, land_use, natural_habitat).
Used by preprocess_flair3d_v2 and flair3d_config_utils (training metadata).

Source-agnostic: remapping applies the same whether labels come from a PLY field
or a sampled GeoTIFF raster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

SEMANTIC_TASK_KEYS: Tuple[str, ...] = (
    "segment",
    "forest",
    "land_use",
    "natural_habitat",
)

# ---------------------------------------------------------------------------
# Class names (default / fine-grained taxonomies)
# ---------------------------------------------------------------------------

_SEGMENT_NAMES: Tuple[str, ...] = (
    "Building",
    "Greenhouse",
    "Impervious surface",
    "Other soil",
    "Herbaceous",
    "Vineyard",
    "Other vegetation",
    "Other infrastructures",
    "Swimming pool",
    "Water",
    "Deciduous",
    "Coniferous",
    "Bridge",
    "Agricultural soil",
    "Soil under vegetation",
    "Void",
)

# Flair3D-build label=v17 (finer13): building split by land use, soil under vegetation.
_SEGMENT_V17_NAMES: Tuple[str, ...] = (
    "Residential building",
    "Greenhouse",
    "Impervious surface",
    "Other soil",
    "Herbaceous",
    "Vineyard",
    "Brushwood",
    "Other infrastructures",
    "Swimming pool",
    "Water",
    "Deciduous",
    "Coniferous",
    "Bridge",
    "Agricultural soil",
    "Soil under vegetation",
    "Industrial building",
    "Tertiary building",
    "Building with unknown usage",
    "Void",
)

# Flair3D-build label=v18 (finer14): unknown building usage merged into void.
_SEGMENT_V18_NAMES: Tuple[str, ...] = (
    "Residential building",
    "Greenhouse",
    "Impervious surface",
    "Other soil",
    "Herbaceous",
    "Vineyard",
    "Brushwood",
    "Other infrastructures",
    "Swimming pool",
    "Water",
    "Deciduous",
    "Coniferous",
    "Bridge",
    "Agricultural soil",
    "Soil under vegetation",
    "Industrial building",
    "Tertiary building",
    "Void",
)

_FOREST_NAMES: Tuple[str, ...] = ("Not Forest", "Forest", "Void")

_LAND_USE_NAMES: Tuple[str, ...] = (
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
)

_NATURAL_HABITAT_NAMES: Tuple[str, ...] = (
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
)

_LAND_USE_COARSE_NAMES: Tuple[str, ...] = (
    "Production primaire",
    "Production secondaire et tertiaire",
    "Reseaux de transport",
    "Services et utilite publique",
    "Usage residentiel",
    "Zones en transition ou abandonnees",
    "Sans usage",
    "Usage inconnu",
)

# Raw ids 0-19 match land_use_classes.txt keys 1-20 (0-indexed).
# filtered: missing / mixed classes -> void (train id 10).
_LAND_USE_FILTERED_NAMES: Tuple[str, ...] = (
    "Agriculture",
    "Sylviculture",
    "Extraction activities",
    "Secondary production",
    "Tertiary production",
    "Road networks",
    "Rail networks",
    "Air networks",
    "Residential use",
    "No land use",
    "Void",
)

_LAND_USE_FILTERED_LUT = np.array(
    [
        0,   # 0  Agriculture
        1,   # 1  Sylviculture
        2,   # 2  Extraction activities
        10,  # 3  Fishing and aquaculture -> void
        10,  # 4  Other primary production -> void
        3,   # 5  Secondary production
        10,  # 6  Secondary, tertiary and residential (mixed) -> void
        4,   # 7  Tertiary production
        5,   # 8  Road networks
        6,   # 9  Rail networks
        7,   # 10 Air networks
        10,  # 11 Inland and maritime transport networks -> void
        10,  # 12 Other transport networks -> void
        10,  # 13 Logistics and storage services -> void
        10,  # 14 Utility networks -> void
        8,   # 15 Residential use
        10,  # 16 Transition zones -> void
        10,  # 17 Abandoned zones -> void
        9,   # 18 No land use
        10,  # 19 Unknown use -> void
    ],
    dtype=np.int32,
)

_NATURAL_HABITAT_BY_DOMAIN_NAMES: Tuple[str, ...] = (
    "Domaine tempéré",
    "Domaine méditerranéen",
    "Domaine alpin",
    "Habitat minéral",
    "Habitat aquatique",
    "Habitat cultivé",
    "Zone bâtie et autre habitat artificiel",
    "Routes & voies verrées",
    "Void",
)

_NATURAL_HABITAT_BY_HABITAT_TYPE_NAMES: Tuple[str, ...] = (
    "Habitat ouvert",
    "Habitat forestier",
    "Habitat minéral",
    "Habitat aquatique",
    "Habitat cultivé",
    "Zone bâtie et autre habitat artificiel",
    "Routes & voies verrées",
    "Void",
)

_NATURAL_HABITAT_BY_HABITAT_X_DOMAIN_NAMES: Tuple[str, ...] = (
    "Open habitat in temperate domain",
    "Forest habitat in temperate domain",
    "Open habitat in mediterranean domain",
    "Forest habitat in mediterranean domain",
    "Open habitat in alpine domain",
    "Forest habitat in alpine domain",
    "Mineral habitat",
    "Aquatic habitat",
    "Cultivated habitat",
    "Built and other artificial habitat",
    "Roads & paved tracks",
    "Void",
)

_NATURAL_HABITAT_BY_MOISTURE_NAMES: Tuple[str, ...] = (
    "Humide",
    "Mesique",
    "Sec",
    "Mineral",
    "Aquatic",
    "Cultivated",
    "Built",
    "Road",
    "Void",
)

_NATURAL_HABITAT_BY_MOISTURE_V2_NAMES: Tuple[str, ...] = (
    "Humide",
    "Mesique",
    "Sec",
    "Mineral",
    "Aquatic",
    "Anthropogenic habitat",
    "Void",
)

_NATURAL_HABITAT_BY_MOISTURE_V3_NAMES: Tuple[str, ...] = (
    "Humide",
    "Mesique",
    "Sec",
    "Mineral",
    "Aquatic",
    "Void",
)

_NATURAL_HABITAT_BY_CLIMATIC_DOMAIN_NAMES: Tuple[str, ...] = (
    "Temperate",
    "Mediterranean",
    "Alpine",
    "Void",
)

# CarHab raw id -> train id. Raw 42=N/A -> void (11); raw 43=routes -> 10.
_NATURAL_HABITAT_BY_HABITAT_X_DOMAIN_LUT = np.array(
    [
        0, 0, 0, 0, 0, 0,  # 0-5   open / temperate
        1, 1, 1, 1, 1, 1,  # 6-11  forest / temperate
        2, 2, 2, 2, 2, 2,  # 12-17 open / mediterranean
        3, 3, 3, 3, 3, 3,  # 18-23 forest / mediterranean
        4, 4, 4, 4, 4, 4,  # 24-29 open / alpine
        5, 5, 5, 5, 5, 5,  # 30-35 forest / alpine
        6, 6, 7, 7, 8, 9,  # 36-41 mineral, aquatic, cultivated, built
        11, 10,  # 42=N/A -> void, 43=routes
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class LabelDefinition:
  """Named label remap: raw source ids -> consecutive train ids via LUT."""

  name: str
  task_key: str
  num_raw_classes: int
  lut: np.ndarray
  names: Tuple[str, ...]
  ignore_index: int
  missing_fill_raw_id: int
  source_field: str = ""

  def __post_init__(self) -> None:
    if self.lut.shape != (self.num_raw_classes,):
      raise ValueError(
        f"LUT shape {self.lut.shape} != ({self.num_raw_classes},) "
        f"for {self.task_key}/{self.name}"
      )
    if len(self.names) == 0:
      raise ValueError(f"names must be non-empty for {self.task_key}/{self.name}")


@dataclass(frozen=True)
class PreprocessLabelDefinitions:
  """Bundle of label definitions passed to preprocessing workers."""

  segment: LabelDefinition
  forest: LabelDefinition
  land_use: LabelDefinition
  natural_habitat: LabelDefinition

  def to_meta_dict(self) -> Dict[str, str]:
    return {
      "segment": self.segment.name,
      "forest": self.forest.name,
      "land_use": self.land_use.name,
      "natural_habitat": self.natural_habitat.name,
    }


def build_lut_from_groups(
    num_raw: int,
    groups: Mapping[int, Sequence[int]],
    *,
    default_train_id: int,
) -> np.ndarray:
  """Build a LUT mapping raw ids to train ids via explicit groups."""
  lut = np.full(num_raw, default_train_id, dtype=np.int32)
  for train_id, raw_ids in groups.items():
    for raw_id in raw_ids:
      if raw_id < 0 or raw_id >= num_raw:
        raise ValueError(
          f"raw_id {raw_id} out of range [0, {num_raw}) for train_id {train_id}"
        )
      lut[raw_id] = int(train_id)
  return lut


def _build_segment_lut(num_raw: int = 256, void_train_id: int = 15) -> np.ndarray:
  """Identity for 0..15; overflow and invalid raw ids map to void."""
  lut = np.full(num_raw, void_train_id, dtype=np.int32)
  lut[: void_train_id + 1] = np.arange(void_train_id + 1, dtype=np.int32)
  return lut


def _build_segment_v17_lut(num_raw: int = 256, void_train_id: int = 18) -> np.ndarray:
  """Identity for 0..18 (Flair3D-build finer13); overflow maps to void."""
  lut = np.full(num_raw, void_train_id, dtype=np.int32)
  lut[: void_train_id + 1] = np.arange(void_train_id + 1, dtype=np.int32)
  return lut


def _build_segment_v18_lut(num_raw: int = 256, void_train_id: int = 17) -> np.ndarray:
  """Identity for 0..17 (Flair3D-build finer14); overflow maps to void."""
  lut = np.full(num_raw, void_train_id, dtype=np.int32)
  lut[: void_train_id + 1] = np.arange(void_train_id + 1, dtype=np.int32)
  return lut


def _build_natural_habitat_default_lut() -> np.ndarray:
  """CarHab raw 42=N/A -> train void (43); raw 43=Autre/routes -> train 42."""
  lut = np.arange(44, dtype=np.int32)
  lut[42] = 43
  lut[43] = 42
  return lut


def _build_natural_habitat_by_moisture_lut() -> np.ndarray:
  """CarHab raw ids 0-35 -> moisture (id % 3); special classes kept separate."""
  lut = np.full(44, 8, dtype=np.int32)
  for raw_id in range(36):
    lut[raw_id] = raw_id % 3
  lut[36] = 3
  lut[37] = 3
  lut[38] = 4
  lut[39] = 4
  lut[40] = 5
  lut[41] = 6
  lut[42] = 8
  lut[43] = 7
  return lut


def _build_natural_habitat_by_moisture_v2_lut() -> np.ndarray:
  """Like by_moisture but Cultivated, Built and Road -> Anthropogenic habitat."""
  lut = np.full(44, 6, dtype=np.int32)
  for raw_id in range(36):
    lut[raw_id] = raw_id % 3
  lut[36] = 3
  lut[37] = 3
  lut[38] = 4
  lut[39] = 4
  lut[40] = 5
  lut[41] = 5
  lut[42] = 6
  lut[43] = 5
  return lut


def _build_natural_habitat_by_moisture_v3_lut() -> np.ndarray:
  """Like by_moisture but Cultivated, Built and Road -> void."""
  lut = np.full(44, 5, dtype=np.int32)
  for raw_id in range(36):
    lut[raw_id] = raw_id % 3
  lut[36] = 3
  lut[37] = 3
  lut[38] = 4
  lut[39] = 4
  lut[40] = 5
  lut[41] = 5
  lut[42] = 5
  lut[43] = 5
  return lut


def _build_natural_habitat_by_climatic_domain_lut() -> np.ndarray:
  """CarHab raw ids 0-35 -> climatic domain; 36-43 -> void."""
  lut = np.full(44, 3, dtype=np.int32)
  lut[0:12] = 0
  lut[12:24] = 1
  lut[24:36] = 2
  return lut


def _make_definition(
    task_key: str,
    name: str,
    *,
    num_raw_classes: int,
    lut: np.ndarray,
    names: Tuple[str, ...],
    ignore_index: int,
    missing_fill_raw_id: int,
    source_field: str = "",
) -> LabelDefinition:
  return LabelDefinition(
    name=name,
    task_key=task_key,
    num_raw_classes=num_raw_classes,
    lut=lut.astype(np.int32, copy=False),
    names=names,
    ignore_index=ignore_index,
    missing_fill_raw_id=missing_fill_raw_id,
    source_field=source_field,
  )


def _register_segment_definitions() -> Dict[str, LabelDefinition]:
  segment_lut = _build_segment_lut()
  segment_void = 15
  base = _make_definition(
    "segment",
    "default",
    num_raw_classes=segment_lut.shape[0],
    lut=segment_lut,
    names=_SEGMENT_NAMES,
    ignore_index=segment_void,
    missing_fill_raw_id=segment_void,
    source_field="semantic",
  )
  # Upstream PLY may already use inter_finerall10 train ids; same LUT/metadata.
  inter_finerall10 = _make_definition(
    "segment",
    "inter_finerall10",
    num_raw_classes=segment_lut.shape[0],
    lut=segment_lut.copy(),
    names=_SEGMENT_NAMES,
    ignore_index=segment_void,
    missing_fill_raw_id=segment_void,
    source_field="semantic",
  )
  segment_v17_lut = _build_segment_v17_lut()
  segment_v17_void = 18
  v17 = _make_definition(
    "segment",
    "v17",
    num_raw_classes=segment_v17_lut.shape[0],
    lut=segment_v17_lut,
    names=_SEGMENT_V17_NAMES,
    ignore_index=segment_v17_void,
    missing_fill_raw_id=segment_v17_void,
    source_field="semantic",
  )
  segment_v18_lut = _build_segment_v18_lut()
  segment_v18_void = 17
  v18 = _make_definition(
    "segment",
    "v18",
    num_raw_classes=segment_v18_lut.shape[0],
    lut=segment_v18_lut,
    names=_SEGMENT_V18_NAMES,
    ignore_index=segment_v18_void,
    missing_fill_raw_id=segment_v18_void,
    source_field="semantic",
  )
  return {"default": base, "inter_finerall10": inter_finerall10, "v17": v17, "v18": v18}


def _register_forest_definitions() -> Dict[str, LabelDefinition]:
  lut = np.arange(3, dtype=np.int32)
  return {
    "default": _make_definition(
      "forest",
      "default",
      num_raw_classes=3,
      lut=lut,
      names=_FOREST_NAMES,
      ignore_index=2,
      missing_fill_raw_id=2,
      source_field="FOREST",
    ),
  }


def _register_land_use_definitions() -> Dict[str, LabelDefinition]:
  identity = np.arange(20, dtype=np.int32)
  coarse_lut = build_lut_from_groups(
    20,
    {
      0: [0, 1, 2, 3, 4],
      1: [5, 6, 7],
      2: [8, 9, 10, 11, 12],
      3: [13, 14],
      4: [15],
      5: [16, 17],
      6: [18],
      7: [19],
    },
    default_train_id=7,
  )
  return {
    "default": _make_definition(
      "land_use",
      "default",
      num_raw_classes=20,
      lut=identity,
      names=_LAND_USE_NAMES,
      ignore_index=-1,
      missing_fill_raw_id=19,
      source_field="LAND_USE",
    ),
    "coarse": _make_definition(
      "land_use",
      "coarse",
      num_raw_classes=20,
      lut=coarse_lut,
      names=_LAND_USE_COARSE_NAMES,
      ignore_index=-1,
      missing_fill_raw_id=19,
      source_field="LAND_USE",
    ),
    "filtered": _make_definition(
      "land_use",
      "filtered",
      num_raw_classes=20,
      lut=_LAND_USE_FILTERED_LUT,
      names=_LAND_USE_FILTERED_NAMES,
      ignore_index=10,
      missing_fill_raw_id=19,
      source_field="LAND_USE",
    ),
  }


def _register_natural_habitat_definitions() -> Dict[str, LabelDefinition]:
  default_lut = _build_natural_habitat_default_lut()
  by_domain_lut = build_lut_from_groups(
    44,
    {
      0: list(range(0, 12)),
      1: list(range(12, 24)),
      2: list(range(24, 36)),
      3: [36, 37],
      4: [38, 39],
      5: [40],
      6: [41],
      7: [43],  # raw Autre / routes
      8: [42],  # raw N/A -> void
    },
    default_train_id=8,
  )
  open_ids = (
    list(range(0, 6))
    + list(range(12, 18))
    + list(range(24, 30))
  )
  forest_ids = (
    list(range(6, 12))
    + list(range(18, 24))
    + list(range(30, 36))
  )
  by_habitat_type_lut = build_lut_from_groups(
    44,
    {
      0: open_ids,
      1: forest_ids,
      2: [36, 37],
      3: [38, 39],
      4: [40],
      5: [41],
      6: [43],
      7: [42],
    },
    default_train_id=7,
  )
  return {
    "default": _make_definition(
      "natural_habitat",
      "default",
      num_raw_classes=44,
      lut=default_lut,
      names=_NATURAL_HABITAT_NAMES,
      ignore_index=43,
      missing_fill_raw_id=42,
      source_field="NATURAL_HABITAT",
    ),
    "by_domain": _make_definition(
      "natural_habitat",
      "by_domain",
      num_raw_classes=44,
      lut=by_domain_lut,
      names=_NATURAL_HABITAT_BY_DOMAIN_NAMES,
      ignore_index=8,
      missing_fill_raw_id=42,
      source_field="NATURAL_HABITAT",
    ),
    "by_habitat_type": _make_definition(
      "natural_habitat",
      "by_habitat_type",
      num_raw_classes=44,
      lut=by_habitat_type_lut,
      names=_NATURAL_HABITAT_BY_HABITAT_TYPE_NAMES,
      ignore_index=7,
      missing_fill_raw_id=42,
      source_field="NATURAL_HABITAT",
    ),
    "by_habitat_x_domain": _make_definition(
      "natural_habitat",
      "by_habitat_x_domain",
      num_raw_classes=44,
      lut=_NATURAL_HABITAT_BY_HABITAT_X_DOMAIN_LUT,
      names=_NATURAL_HABITAT_BY_HABITAT_X_DOMAIN_NAMES,
      ignore_index=11,
      missing_fill_raw_id=42,
      source_field="NATURAL_HABITAT",
    ),
    "by_moisture": _make_definition(
      "natural_habitat",
      "by_moisture",
      num_raw_classes=44,
      lut=_build_natural_habitat_by_moisture_lut(),
      names=_NATURAL_HABITAT_BY_MOISTURE_NAMES,
      ignore_index=8,
      missing_fill_raw_id=42,
      source_field="NATURAL_HABITAT",
    ),
    "by_moisture_v2": _make_definition(
      "natural_habitat",
      "by_moisture_v2",
      num_raw_classes=44,
      lut=_build_natural_habitat_by_moisture_v2_lut(),
      names=_NATURAL_HABITAT_BY_MOISTURE_V2_NAMES,
      ignore_index=6,
      missing_fill_raw_id=42,
      source_field="NATURAL_HABITAT",
    ),
    "by_moisture_v3": _make_definition(
      "natural_habitat",
      "by_moisture_v3",
      num_raw_classes=44,
      lut=_build_natural_habitat_by_moisture_v3_lut(),
      names=_NATURAL_HABITAT_BY_MOISTURE_V3_NAMES,
      ignore_index=5,
      missing_fill_raw_id=42,
      source_field="NATURAL_HABITAT",
    ),
    "by_climatic_domain": _make_definition(
      "natural_habitat",
      "by_climatic_domain",
      num_raw_classes=44,
      lut=_build_natural_habitat_by_climatic_domain_lut(),
      names=_NATURAL_HABITAT_BY_CLIMATIC_DOMAIN_NAMES,
      ignore_index=3,
      missing_fill_raw_id=42,
      source_field="NATURAL_HABITAT",
    ),
  }


LABEL_DEFINITIONS: Dict[str, Dict[str, LabelDefinition]] = {
  "segment": _register_segment_definitions(),
  "forest": _register_forest_definitions(),
  "land_use": _register_land_use_definitions(),
  "natural_habitat": _register_natural_habitat_definitions(),
}

# Default definition per task (preprocess v2 CLI + training when not overridden).
DEFAULT_LABEL_DEFINITION_NAMES: Dict[str, str] = {
  "segment": "v18",
  "forest": "default",
  "land_use": "default",
  "natural_habitat": "by_habitat_x_domain",
}


def get_default_definition_name(task_key: str) -> str:
  if task_key not in DEFAULT_LABEL_DEFINITION_NAMES:
    keys = ", ".join(sorted(DEFAULT_LABEL_DEFINITION_NAMES.keys()))
    raise KeyError(f"Unknown task_key '{task_key}'. Expected one of: {keys}")
  return DEFAULT_LABEL_DEFINITION_NAMES[task_key]


def supported_definitions(task_key: str) -> Tuple[str, ...]:
  if task_key not in LABEL_DEFINITIONS:
    keys = ", ".join(sorted(LABEL_DEFINITIONS.keys()))
    raise KeyError(f"Unknown task_key '{task_key}'. Expected one of: {keys}")
  return tuple(sorted(LABEL_DEFINITIONS[task_key].keys()))


def get_definition(task_key: str, name: str) -> LabelDefinition:
  if task_key not in LABEL_DEFINITIONS:
    keys = ", ".join(sorted(LABEL_DEFINITIONS.keys()))
    raise KeyError(f"Unknown task_key '{task_key}'. Expected one of: {keys}")
  task_defs = LABEL_DEFINITIONS[task_key]
  if name not in task_defs:
    supported = ", ".join(sorted(task_defs.keys()))
    raise KeyError(
      f"Unknown definition '{name}' for task '{task_key}'. Supported: {supported}"
    )
  return task_defs[name]


def build_preprocess_label_definitions(
    segment: str = DEFAULT_LABEL_DEFINITION_NAMES["segment"],
    forest: str = DEFAULT_LABEL_DEFINITION_NAMES["forest"],
    land_use: str = DEFAULT_LABEL_DEFINITION_NAMES["land_use"],
    natural_habitat: str = DEFAULT_LABEL_DEFINITION_NAMES["natural_habitat"],
) -> PreprocessLabelDefinitions:
  return PreprocessLabelDefinitions(
    segment=get_definition("segment", segment),
    forest=get_definition("forest", forest),
    land_use=get_definition("land_use", land_use),
    natural_habitat=get_definition("natural_habitat", natural_habitat),
  )


def apply_remap(values: np.ndarray, definition: LabelDefinition) -> np.ndarray:
  """Remap raw label ids to train ids via LUT; out-of-range -> missing_fill mapping."""
  idx = values.astype(np.int64, copy=False)
  lut = definition.lut
  fallback = int(lut[definition.missing_fill_raw_id])
  result = np.full(idx.shape, fallback, dtype=np.int32)
  valid = (idx >= 0) & (idx < lut.shape[0])
  if np.any(valid):
    result[valid] = lut[idx[valid]]
  return result


def build_stored_to_train_lut(
    storage: LabelDefinition,
    target: LabelDefinition,
) -> np.ndarray:
  """Map on-disk label ids (after storage definition) to target train ids.
  
  Only for the gap :
  - Road : saved as 42, raw as 43
  - Void : saved as 43, raw as 42
  """
  if storage.num_raw_classes != target.num_raw_classes:
    raise ValueError(
      f"storage and target must share num_raw_classes "
      f"(got {storage.num_raw_classes} vs {target.num_raw_classes})"
    )
  num_raw = storage.num_raw_classes
  fallback = int(target.lut[target.missing_fill_raw_id])
  stored_lut = np.full(num_raw, fallback, dtype=np.int32)
  for raw_id in range(num_raw):
    stored_id = int(storage.lut[raw_id])
    stored_lut[stored_id] = int(target.lut[raw_id])
  return stored_lut


def definition_to_task_config(definition: LabelDefinition) -> Dict[str, object]:
  """Convert a LabelDefinition to a Flair3D semantic task config dict."""
  if definition.ignore_index >= 0:
    num_classes = definition.ignore_index
  else:
    num_classes = len(definition.names)
  return {
    "num_classes": num_classes,
    "ignore_index": definition.ignore_index,
    "names": list(definition.names),
    "num_raw_classes": definition.num_raw_classes,
    "definition": definition.name,
    "task_type": "semantic",
  }


def default_task_config(task_key: str) -> Dict[str, object]:
  """Task config for the default definition of a semantic task."""
  return definition_to_task_config(get_definition(task_key, "default"))
