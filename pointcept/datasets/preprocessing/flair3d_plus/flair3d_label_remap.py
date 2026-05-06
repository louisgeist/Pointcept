"""
Flair3D label remapping utilities.

This module centralizes:
- direct (single-source) remaps from one raw label field,
- fusion remaps that combine COSIA + LIDARHD agreement logic.
"""

from typing import Dict, Optional

import numpy as np


COSIA_2_FLAIR3D = np.array(
    [0, 0, 3, 1, 1, 1, 3, 3, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32
)
LIDARHD_2_FLAIR3D = np.array([1, 1, 1, 2, 0, 3, 0, 0, 3, 3, 3], dtype=np.int32)
LIDARHD_2_COARSE_B = np.array([1, 1, 2, 2, 0, 3, 0, 0, 3, 3, 3], dtype=np.int32)

COSIA_FINER_ALL = np.array(
    [0, 1, 8, 2, 3, 3, 8, 8, 4, 3, 3, 5, 6, 6, 7, 8, 8, 8, 8], dtype=np.int32
)
COSIA_FINER_BUILDING = np.array(
    [0, 3, 4, 1, 1, 1, 4, 4, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4], dtype=np.int32
)
COSIA_FINER_SOIL = np.array(
    [0, 0, 5, 3, 1, 1, 5, 5, 4, 1, 1, 2, 2, 2, 2, 5, 5, 5, 5], dtype=np.int32
)
COSIA_FINER_VEGETATION = np.array(
    [0, 0, 5, 1, 1, 1, 5, 5, 1, 1, 1, 3, 2, 2, 4, 5, 5, 5, 5], dtype=np.int32
)
COSIA_FINER_VEGETATION_BETA = np.array(
    [0, 0, 4, 1, 1, 1, 4, 4, 1, 1, 1, 3, 2, 2, 2, 4, 4, 4, 4], dtype=np.int32
)
LIDARHD_FINER = np.array([0, 0, 1, 2, 3, 6, 4, 5, 6, 6, 6], dtype=np.int32)
COSIA_FINER_ALL2 = np.array(
    [0, 1, 7, 2, 3, 3, 7, 7, 4, 3, 3, 5, 6, 6, 6, 7, 7, 7, 7], dtype=np.int32
)
COSIA_FINER_ALL3 = np.array(
    [0, 1, 8, 2, 3, 3, 8, 8, 4, 3, 3, 5, 6, 6, 8, 8, 8, 8, 8], dtype=np.int32
)
COSIA_FINER_ALL5 = np.array(
    [0, 1, 9, 2, 3, 3, 9, 9, 4, 3, 3, 5, 6, 6, 7, 9, 9, 9, 9], dtype=np.int32
)
# Finer all6: building=0, greenhouse=1, impervious_surface=2, other_soil=3, herbaceous=4,
# vineyard=5, tree=6, sursol_perenne=7, void=8. Brushwood merged into tree (no separate class).
COSIA_FINER_ALL6 = np.array(
    [0, 1, 8, 2, 3, 3, 8, 8, 4, 3, 3, 5, 6, 6, 6, 8, 8, 8, 8], dtype=np.int32
)
# Finer all7: building=0, greenhouse=1, impervious_surface=2, other_soil=3, herbaceous=4,
# vineyard=5, tree=6, sursol_perenne=7 (forced from LIDARHD), agricultural_soil=8, void=9.
COSIA_FINER_ALL7 = np.array(
    [0, 1, 9, 2, 3, 3, 9, 9, 4, 8, 8, 5, 6, 6, 6, 9, 9, 9, 9], dtype=np.int32
)
# Finer all8 (based on all6): building=0, greenhouse=1, impervious_surface=2, other_soil=3,
# herbaceous=4, vineyard=5, tree=6, sursol_perenne=7 (LIDARHD), swimming_pool=8, water=9,
# deciduous=10, coniferous=11, bridge=12 (LIDARHD), void=13.
COSIA_FINER_ALL8 = np.array(
    [0, 1, 8, 2, 3, 3, 9, 13, 4, 3, 3, 5, 10, 11, 6, 13, 6, 6, 13], dtype=np.int32
)
# Finer all9: same COSIA remap as all8; void=14; forest_ground=13 reserved (no COSIA sink in table).
COSIA_FINER_ALL9 = np.array(
    [0, 1, 8, 2, 3, 3, 9, 14, 4, 3, 3, 5, 10, 11, 6, 14, 6, 6, 14], dtype=np.int32
)


lidarhd_class_dictionary = {
    1: ("#d3d3d3", "Non classé"),
    2: ("#a0522d", "Sol"),
    3: ("#b3b94d", "Végétation basse"),
    4: ("#4e9a4e", "Végétation moyenne"),
    5: ("#1f4e1f", "Végétation haute"),
    6: ("#ff0000", "Bâtiment"),
    9: ("#1e90ff", "Eau"),
    17: ("#ffff00", "Pont"),
    64: ("#ff8c00", "Sursol pérenne"),
    65: ("#8b00ff", "Artefact"),
    66: ("#000000", "Points virtuels (modélisation)")
}

LIDARHD_NUM_CLASSES = 10
TRAINID = 0
LIDARHD_ID2TRAINID = np.ones(max(lidarhd_class_dictionary.keys())+1, dtype=np.int32)*LIDARHD_NUM_CLASSES
for k,v in lidarhd_class_dictionary.items():
    if v[1] == 'Non classé':
        # LIDARHD_ID2TRAINID[k] = LIDARHD_NUM_CLASSES (void class)
        pass
    else:
        LIDARHD_ID2TRAINID[k] = TRAINID
        TRAINID += 1

SIMPLE_LABEL_REMAPS = {
    "coarse_cosia": ("cosia_class", COSIA_2_FLAIR3D),
    "coarse_lidarhd": ("lidarhd_class", LIDARHD_2_FLAIR3D),
    "coarse_lidarhd_b": ("lidarhd_class", LIDARHD_2_COARSE_B),
    "finer_cosia_all": ("cosia_class", COSIA_FINER_ALL),
    "finer_cosia_building": ("cosia_class", COSIA_FINER_BUILDING),
    "finer_cosia_soil": ("cosia_class", COSIA_FINER_SOIL),
    "finer_cosia_vegetation": ("cosia_class", COSIA_FINER_VEGETATION),
    "finer_cosia_vegetation_beta": ("cosia_class", COSIA_FINER_VEGETATION_BETA),
    "inter_finerall2": ("cosia_class", COSIA_FINER_ALL2),
    "inter_finerall3": ("cosia_class", COSIA_FINER_ALL3),
    "inter_finerall5": ("cosia_class", COSIA_FINER_ALL5),
    "inter_finerall6": ("cosia_class", COSIA_FINER_ALL6),
    "inter_finerall7": ("cosia_class", COSIA_FINER_ALL7),
    "inter_finerall8": ("cosia_class", COSIA_FINER_ALL8),
    "inter_finerall9": ("cosia_class", COSIA_FINER_ALL9),
    "finer_lidarhd": ("lidarhd_class", LIDARHD_FINER),
}

FUSION_LABEL_REMAPS = {
    "coarse_intersection",
    "rule1",
    "inter_finerall",
    "inter_finerbuilding",
    "inter_finersoil",
    "inter_finervegetation",
    "inter_finervegetation_beta",
    "inter_finerall2",
    "inter_finerall3",
    "inter_finerall4",
    "inter_finerall5",
    "inter_finerall6",
    "inter_finerall7",
    "inter_finerall8",
    "inter_finerall9",
}

SUPPORTED_LABEL_REMAPS = FUSION_LABEL_REMAPS #sorted(set(SIMPLE_LABEL_REMAPS.keys()) | FUSION_LABEL_REMAPS)


def map_labels(mapping: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Remap labels via lookup."""
    return mapping[labels.astype(np.intp)]


def _require_field(attributes: Dict[str, np.ndarray], field: str) -> np.ndarray:
    if field not in attributes:
        raise KeyError(f"Required field '{field}' not found in PLY attributes")
    return attributes[field].astype(np.int32)


def _finer_mapping_from_mode(mode: str) -> np.ndarray:
    if mode == "inter_finerall":
        return COSIA_FINER_ALL
    if mode == "inter_finerbuilding":
        return COSIA_FINER_BUILDING
    if mode == "inter_finersoil":
        return COSIA_FINER_SOIL
    if mode == "inter_finervegetation":
        return COSIA_FINER_VEGETATION
    if mode == "inter_finervegetation_beta":
        return COSIA_FINER_VEGETATION_BETA
    if mode == "inter_finerall2":
        return COSIA_FINER_ALL2
    if mode in ("inter_finerall3", "inter_finerall4"):
        return COSIA_FINER_ALL3
    if mode == "inter_finerall5":
        return COSIA_FINER_ALL5
    if mode == "inter_finerall6":
        return COSIA_FINER_ALL6
    if mode == "inter_finerall7":
        return COSIA_FINER_ALL7
    if mode == "inter_finerall8":
        return COSIA_FINER_ALL8
    if mode == "inter_finerall9":
        return COSIA_FINER_ALL9
    raise ValueError(f"Mode '{mode}' does not define a finer mapping")


def _segment_from_fusion(attributes: Dict[str, np.ndarray], mode: str) -> np.ndarray:
    cosia = _require_field(attributes, "cosia_class")
    lidarhd = _require_field(attributes, "lidarhd_class")
    
    
    # LidarHD to consecutive labels (reproducing `Flair3DToConsecutiveLabels` in SPT)
    # Defensive fix from your previous transform: clamp unexpected lidar ids.
    lidarhd = lidarhd.copy()
    lidarhd[lidarhd > 66] = 1
    lidarhd = map_labels(LIDARHD_ID2TRAINID, lidarhd)

    coarse_cosia = map_labels(COSIA_2_FLAIR3D, cosia)
    coarse_lidarhd = map_labels(LIDARHD_2_FLAIR3D, lidarhd)

    coarse_void = 3
    agreement = coarse_cosia == coarse_lidarhd

    if mode == "coarse_intersection":
        seg = np.full(cosia.shape, coarse_void, dtype=np.int32)
        seg[agreement] = coarse_cosia[agreement]
        return seg

    if mode == "rule1":
        seg = coarse_cosia.copy()
        mask = (coarse_cosia == 1) & (coarse_lidarhd != 1) & (coarse_lidarhd != coarse_void)
        seg[mask] = coarse_lidarhd[mask]
        return seg

    # Finer-intersection family.
    finer_map = _finer_mapping_from_mode(mode)
    finer_void = int(finer_map.max())

    if mode in (
        "inter_finerall5",
        "inter_finerall6",
        "inter_finerall7",
        "inter_finerall8",
        "inter_finerall9",
    ):
        # Recompute agreement with coarse_B variant (your original behavior).
        coarse_lidarhd_b = map_labels(LIDARHD_2_COARSE_B, lidarhd)
        agreement = coarse_cosia == coarse_lidarhd_b

    if mode in (
        "inter_finerall4",
        "inter_finerall5",
        "inter_finerall6",
        "inter_finerall7",
        "inter_finerall8",
        "inter_finerall9",
    ):
        # Treat lidarhd void as "agreement" for these modes.
        agreement = agreement | (lidarhd == 10)

    seg = np.full(cosia.shape, finer_void, dtype=np.int32) # initialize with void label
    cosia_finer = map_labels(finer_map, cosia)
    seg[agreement] = cosia_finer[agreement]

    if mode in ("inter_finerall3", "inter_finerall4"):
        # Vineyard override from COSIA.
        seg[cosia == 11] = 5
        # Sursol perenne override from LIDARHD.
        seg[lidarhd == 7] = 7
    elif mode == "inter_finerall5":
        # Sursol perenne override from LIDARHD (different class id).
        seg[lidarhd == 7] = 8
    elif mode == "inter_finerall6":
        # Sursol perenne override from LIDARHD (class 7).
        seg[lidarhd == 7] = 7
    elif mode == "inter_finerall7":
        # Other infrastructure override from LIDARHD (class 7).
        seg[lidarhd == 7] = 7
    elif mode in ("inter_finerall8", "inter_finerall9"):
        # Sursol perenne (train id 7) and bridge / Pont (train id 6) from LIDARHD.
        seg[lidarhd == 7] = 7 # Override Sursol perenne
        seg[lidarhd == 6] = 12 # Override Briddge

    return seg


def build_segment(
    attributes: Dict[str, np.ndarray],
    label_definition: str,
) -> np.ndarray:
    """
    Build the final segment vector from raw PLY attributes.
    """
    # Fusion takes precedence: it builds the real labels from COSIA+LIDARHD agreement.
    # Simple remaps are fallback for single-source consecutive values only.
    if label_definition in FUSION_LABEL_REMAPS:
        return _segment_from_fusion(attributes, label_definition)
    supported = ", ".join(SUPPORTED_LABEL_REMAPS)
    raise ValueError(
        f"Unknown label_definition '{label_definition}'. Supported: {supported}"
    )
