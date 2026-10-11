"""
MALiBU3D unified label definitions and LUT remapping. (LUT: look-up table)

Single registry for the label-based tasks (segment, natural_habitat and its ecological axes).
Used by preprocess_malibu3d and malibu3d_config_utils (training metadata).

Source-agnostic: remapping applies the same whether labels come from a PLY field
or a sampled GeoTIFF raster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

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

# Land-cover taxonomy of the released dataset (15 classes + void).
_SEGMENT_V20_NAMES: Tuple[str, ...] = (
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

_NATURAL_HABITAT_BY_CLIMATIC_DOMAIN_NAMES: Tuple[str, ...] = (
    "Temperate",
    "Mediterranean",
    "Alpine",
    "Void",
)

_NATURAL_HABITAT_BY_HABITAT_TYPE_ECOLOGICAL_NAMES: Tuple[str, ...] = (
    "Open",
    "Forest",
    "Mineral",
    "Aquatic",
    "Void",
)

_NATURAL_HABITAT_BY_SOIL_CHEMISTRY_NAMES: Tuple[str, ...] = (
    "Acidic",
    "Alkaline",
    "Void",
)

_NATURAL_HABITAT_BY_MOISTURE_REGIME_NAMES: Tuple[str, ...] = (
    "Humide",
    "Mesique",
    "Sec",
    "Void",
)

# Open / forest CarHab raw-id groups (domain x habitat type x substrate x moisture space),
# used by `by_habitat_type_ecological`.
_NATURAL_HABITAT_OPEN_IDS: Tuple[int, ...] = tuple(
    list(range(0, 6)) + list(range(12, 18)) + list(range(24, 30))
)
_NATURAL_HABITAT_FOREST_IDS: Tuple[int, ...] = tuple(
    list(range(6, 12)) + list(range(18, 24)) + list(range(30, 36))
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
  natural_habitat: LabelDefinition

  def to_meta_dict(self) -> Dict[str, str]:
    return {
      "segment": self.segment.name,
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


def _build_segment_v20_lut(num_raw: int = 256, void_train_id: int = 15) -> np.ndarray:
  """Identity for 0..15 (the 15 land-cover classes + void); overflow maps to void."""
  lut = np.full(num_raw, void_train_id, dtype=np.int32)
  lut[: void_train_id + 1] = np.arange(void_train_id + 1, dtype=np.int32)
  return lut


def _build_natural_habitat_default_lut() -> np.ndarray:
  """CarHab raw 42=N/A -> train void (43); raw 43=Autre/routes -> train 42."""
  lut = np.arange(44, dtype=np.int32)
  lut[42] = 43
  lut[43] = 42
  return lut


def _build_natural_habitat_by_climatic_domain_lut() -> np.ndarray:
  """CarHab raw ids 0-35 -> climatic domain; 36-43 -> void."""
  lut = np.full(44, 3, dtype=np.int32)
  lut[0:12] = 0
  lut[12:24] = 1
  lut[24:36] = 2
  return lut


def _build_natural_habitat_by_soil_chemistry_lut() -> np.ndarray:
  """CarHab raw ids 0-35 -> substrate parity (acid/basic); 36-39 mineral/aquatic
  acid/basic; 40-43 (cultivated, built, N/A, routes) -> void."""
  lut = np.full(44, 2, dtype=np.int32)
  for raw_id in range(36):
    lut[raw_id] = 0 if raw_id % 6 < 3 else 1
  lut[36] = 0  # mineral, acid
  lut[37] = 1  # mineral, basic
  lut[38] = 0  # aquatic, acid
  lut[39] = 1  # aquatic, basic
  return lut


def _build_natural_habitat_by_moisture_regime_lut() -> np.ndarray:
  """CarHab raw ids 0-35 -> moisture (id % 3); everything else -> void."""
  lut = np.full(44, 3, dtype=np.int32)
  for raw_id in range(36):
    lut[raw_id] = raw_id % 3
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
  segment_v20_lut = _build_segment_v20_lut()
  segment_v20_void = 15
  v20 = _make_definition(
    "segment",
    "v20",
    num_raw_classes=segment_v20_lut.shape[0],
    lut=segment_v20_lut,
    names=_SEGMENT_V20_NAMES,
    ignore_index=segment_v20_void,
    missing_fill_raw_id=segment_v20_void,
    source_field="semantic",
  )
  return {
    "default": base,
    "v20": v20,
  }


def _register_natural_habitat_definitions() -> Dict[str, LabelDefinition]:
  default_lut = _build_natural_habitat_default_lut()
  open_ids = list(_NATURAL_HABITAT_OPEN_IDS)
  forest_ids = list(_NATURAL_HABITAT_FOREST_IDS)
  by_habitat_type_ecological_lut = build_lut_from_groups(
    44,
    {
      0: open_ids,
      1: forest_ids,
      2: [36, 37],
      3: [38, 39],
    },
    default_train_id=4,
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
    "by_habitat_type_ecological": _make_definition(
      "natural_habitat",
      "by_habitat_type_ecological",
      num_raw_classes=44,
      lut=by_habitat_type_ecological_lut,
      names=_NATURAL_HABITAT_BY_HABITAT_TYPE_ECOLOGICAL_NAMES,
      ignore_index=4,
      missing_fill_raw_id=42,
      source_field="NATURAL_HABITAT",
    ),
    "by_soil_chemistry": _make_definition(
      "natural_habitat",
      "by_soil_chemistry",
      num_raw_classes=44,
      lut=_build_natural_habitat_by_soil_chemistry_lut(),
      names=_NATURAL_HABITAT_BY_SOIL_CHEMISTRY_NAMES,
      ignore_index=2,
      missing_fill_raw_id=42,
      source_field="NATURAL_HABITAT",
    ),
    "by_moisture_regime": _make_definition(
      "natural_habitat",
      "by_moisture_regime",
      num_raw_classes=44,
      lut=_build_natural_habitat_by_moisture_regime_lut(),
      names=_NATURAL_HABITAT_BY_MOISTURE_REGIME_NAMES,
      ignore_index=3,
      missing_fill_raw_id=42,
      source_field="NATURAL_HABITAT",
    ),
  }


LABEL_DEFINITIONS: Dict[str, Dict[str, LabelDefinition]] = {
  "segment": _register_segment_definitions(),
  "natural_habitat": _register_natural_habitat_definitions(),
}

# Default definition per task (preprocessing CLI, and training when not overridden).
DEFAULT_LABEL_DEFINITION_NAMES: Dict[str, str] = {
  "segment": "v20",
  "natural_habitat": "default",
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
    natural_habitat: str = DEFAULT_LABEL_DEFINITION_NAMES["natural_habitat"],
) -> PreprocessLabelDefinitions:
  return PreprocessLabelDefinitions(
    segment=get_definition("segment", segment),
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
  """Convert a LabelDefinition to a MALiBU3D semantic task config dict."""
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
