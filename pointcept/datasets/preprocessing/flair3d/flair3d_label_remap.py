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

SIMPLE_LABEL_REMAPS = {
    "coarse_cosia": ("cosia_class", COSIA_2_FLAIR3D),
    "coarse_lidarhd": ("lidarhd_class", LIDARHD_2_FLAIR3D),
    "coarse_lidarhd_b": ("lidarhd_class", LIDARHD_2_COARSE_B),
    "finer_cosia_all": ("cosia_class", COSIA_FINER_ALL),
    "finer_cosia_building": ("cosia_class", COSIA_FINER_BUILDING),
    "finer_cosia_soil": ("cosia_class", COSIA_FINER_SOIL),
    "finer_cosia_vegetation": ("cosia_class", COSIA_FINER_VEGETATION),
    "finer_cosia_vegetation_beta": ("cosia_class", COSIA_FINER_VEGETATION_BETA),
    "finer_cosia_all2": ("cosia_class", COSIA_FINER_ALL2),
    "finer_cosia_all3": ("cosia_class", COSIA_FINER_ALL3),
    "finer_cosia_all5": ("cosia_class", COSIA_FINER_ALL5),
    "finer_cosia_all6": ("cosia_class", COSIA_FINER_ALL6),
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
}

SUPPORTED_LABEL_REMAPS = sorted(set(SIMPLE_LABEL_REMAPS.keys()) | FUSION_LABEL_REMAPS)


def _safe_take(mapping: np.ndarray, labels: np.ndarray, ignore_index: int) -> np.ndarray:
    out = np.full(labels.shape, ignore_index, dtype=np.int32)
    valid = (labels >= 0) & (labels < mapping.shape[0])
    out[valid] = mapping[labels[valid]]
    return out


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
    raise ValueError(f"Mode '{mode}' does not define a finer mapping")


def _segment_from_fusion(attributes: Dict[str, np.ndarray], mode: str, ignore_index: int) -> np.ndarray:
    cosia = _require_field(attributes, "cosia_class")
    lidarhd = _require_field(attributes, "lidarhd_class")

    # Defensive fix from your previous transform: clamp unexpected lidar ids.
    lidarhd = lidarhd.copy()
    lidarhd[lidarhd > 66] = 1

    coarse_cosia = _safe_take(COSIA_2_FLAIR3D, cosia, ignore_index)
    coarse_lidarhd = _safe_take(LIDARHD_2_FLAIR3D, lidarhd, ignore_index)

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

    if mode in ("inter_finerall5", "inter_finerall6"):
        # Recompute agreement with coarse_B variant (your original behavior).
        coarse_lidarhd_b = _safe_take(LIDARHD_2_COARSE_B, lidarhd, ignore_index)
        agreement = coarse_cosia == coarse_lidarhd_b

    if mode in ("inter_finerall4", "inter_finerall5", "inter_finerall6"):
        # Treat lidarhd void as "agreement" for these modes.
        agreement = agreement | (lidarhd == 10)

    seg = np.full(cosia.shape, finer_void, dtype=np.int32)
    cosia_finer = _safe_take(finer_map, cosia, finer_void)
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

    return seg


def build_segment(
    attributes: Dict[str, np.ndarray],
    ignore_index: int,
    label_definition: str,
) -> np.ndarray:
    """
    Build the final segment vector from raw PLY attributes.
    """
    if label_definition in SIMPLE_LABEL_REMAPS:
        source_field, source_map = SIMPLE_LABEL_REMAPS[label_definition]
        raw = _require_field(attributes, source_field)
        return _safe_take(source_map, raw, ignore_index)
    if label_definition in FUSION_LABEL_REMAPS:
        return _segment_from_fusion(attributes, label_definition, ignore_index)
    supported = ", ".join(SUPPORTED_LABEL_REMAPS)
    raise ValueError(
        f"Unknown label_definition '{label_definition}'. Supported: {supported}"
    )
