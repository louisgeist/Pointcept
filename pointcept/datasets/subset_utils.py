"""
Scene-list restriction applied by DefaultDataset after ``get_data_list``.

In order: keep the tiles listed in a CSV sidecar (``split`` + ``tile_id`` or ``patch_id``
columns), e.g. the fixed stratified validation subset used by the MALiBU3D configs; then
the scenes matching ``include_names``; then the first ``max_sample`` scenes.
"""

from __future__ import annotations

import csv
import os
from typing import List, Optional, Sequence, Set, Tuple

from pointcept.utils.logger import get_root_logger

def _normalize_split(split) -> str:
    if isinstance(split, str):
        return split
    if isinstance(split, Sequence) and len(split) == 1:
        return str(split[0])
    raise TypeError(
        "split must be a string or a length-1 sequence for subset sidecar filtering"
    )


def _split_label(split) -> str:
    if isinstance(split, str):
        return split
    if isinstance(split, Sequence):
        return ",".join(str(s) for s in split)
    return str(split)


def _normalize_include_names(include_names) -> Optional[List[str]]:
    if include_names is None:
        return None
    if isinstance(include_names, str):
        names = [include_names]
    else:
        names = list(include_names)
    out = [str(n).strip() for n in names if str(n).strip()]
    return out


def scene_matches_include_name(path: str, name: str) -> bool:
    """Loose match of a user-supplied name against a scene path.

    True if the name (or the name with its ``_LIDARHD`` token removed, e.g.
    ``D075-2021_LIDARHD_AA-S2-2``) is a substring of the path, or if a
    ``<prefix>_<rest>`` name has a scene folder that starts with ``<prefix>`` and ends
    with ``<rest>`` (e.g. ``D075_UF-S1-2``). Also used for other datasets' scene names
    (e.g. OpenGF ``T2``).
    """
    n = str(name).strip()
    if not n:
        return False
    path_s = os.path.normpath(str(path)).replace("\\", "/")
    base = os.path.basename(path_s.rstrip("/"))
    variants = [n]
    if "_LIDARHD_" in n:
        variants.append(n.replace("_LIDARHD_", "_"))
    if n.endswith("_LIDARHD"):
        variants.append(n[: -len("_LIDARHD")])
    for variant in variants:
        if variant == base or variant in path_s or variant in base:
            return True
    if "_" in n and "LIDARHD" not in n:
        dept, rest = n.split("_", 1)
        if dept and rest and base.startswith(dept) and (
            base.endswith("_" + rest) or base.endswith(rest)
        ):
            return True
    return False


def matching_include_name(path: str, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if scene_matches_include_name(path, name):
            return name
    return None


def load_sidecar_keys(manifest_path: str) -> Set[Tuple[str, str]]:
    """Read ``{(split, tile_id)}`` from a sidecar CSV (id column ``tile_id`` or ``patch_id``)."""
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Stratified subset sidecar not found: {manifest_path}")

    keys: Set[Tuple[str, str]] = set()
    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty sidecar CSV: {manifest_path}")
        id_column = "tile_id" if "tile_id" in reader.fieldnames else "patch_id"
        missing = {"split", id_column} - set(reader.fieldnames)
        if missing:
            raise KeyError(
                f"Sidecar CSV missing columns {sorted(missing)}: {manifest_path}"
            )
        for row in reader:
            split = (row.get("split") or "").strip()
            patch_id = (row.get(id_column) or "").strip()
            if split and patch_id:
                keys.add((split, patch_id))
    return keys


def filter_paths_by_sidecar(
    data_list: Sequence[str],
    split: str,
    sidecar_keys: Set[Tuple[str, str]],
) -> List[str]:
    """Keep the scenes whose ``(split, folder name)`` is in the sidecar, sorted by name."""
    split = _normalize_split(split)
    filtered = [
        path
        for path in data_list
        if (split, os.path.basename(path)) in sidecar_keys
    ]
    filtered.sort(key=os.path.basename)
    return filtered


def apply_subset_selection(
    data_list: Sequence[str],
    *,
    split,
    max_sample: Optional[int] = None,
    stratified_subset_manifest: Optional[str] = None,
    include_names: Optional[Sequence[str]] = None,
) -> List[str]:
    """Apply the sidecar filter, then ``include_names``, then the ``max_sample`` head slice.

    The sidecar filter needs a single split name.
    """
    logger = get_root_logger()
    original_len = len(data_list)

    if stratified_subset_manifest:
        split_name = _normalize_split(split)
        sidecar_keys = load_sidecar_keys(stratified_subset_manifest)
        data_list = filter_paths_by_sidecar(data_list, split_name, sidecar_keys)
        missing = len(sidecar_keys) - len(data_list)
        if missing > 0:
            logger.warning(
                "Stratified subset sidecar: %d/%d entries not found in current data_list "
                "(split=%s, manifest=%s)",
                missing,
                len(sidecar_keys),
                split_name,
                stratified_subset_manifest,
            )
        logger.info(
            "Stratified subset sidecar applied: %d -> %d scenes (split=%s, manifest=%s)",
            original_len,
            len(data_list),
            split_name,
            stratified_subset_manifest,
        )
    else:
        data_list = list(data_list)
        split_name = _split_label(split)

    names = _normalize_include_names(include_names)
    if names:
        kept = []
        matched_names = set()
        for path in data_list:
            hit = matching_include_name(path, names)
            if hit is not None:
                kept.append(path)
                matched_names.add(hit)
        unmatched = [n for n in names if n not in matched_names]
        if unmatched:
            logger.warning(
                "include_names: %d/%d names matched no scene (split=%s): %s",
                len(unmatched),
                len(names),
                split_name,
                unmatched,
            )
        logger.info(
            "include_names filter: %d -> %d scenes (split=%s, %d/%d names matched)",
            len(data_list),
            len(kept),
            split_name,
            len(matched_names),
            len(names),
        )
        data_list = kept

    if max_sample is None or max_sample >= len(data_list):
        return data_list

    logger.info(
        "max_sample head slice: %d -> %d scenes (split=%s)",
        len(data_list),
        max_sample,
        split_name,
    )
    return list(data_list[:max_sample])
