"""
Flair3D Dataset (LidarHD-like preprocessed scenes).

Scenes are expected under data_root as:
  <data_root>/<split>/<dept_year>_LIDARHD/<roi>/<scene_id>/
with assets: coord.npy, color.npy, segment.npy, optionally strength.npy, normal.npy.
"""

import os
import csv
from collections.abc import Sequence
from copy import deepcopy

import numpy as np

from .defaults import DefaultDataset
from .builder import DATASETS
from .transform import record_data_pipeline
from .flair3d_plus_label_task_config import get_missing_target_fill_value
from pointcept.utils.logger import get_root_logger

FLAIR3D_SPECIFIC_ASSETS = ("forest", "land_use", "natural_habitat", "elevation")
FLAIR3D_SEMANTIC_TARGETS = ("segment", "forest", "land_use", "natural_habitat")
FLAIR3D_REGRESSION_TARGETS = ("elevation",)
FLAIR3D_ALLOWED_TARGETS = FLAIR3D_SEMANTIC_TARGETS + FLAIR3D_REGRESSION_TARGETS


def _load_missing_lidarhd_tiles():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    details_csv = os.path.join(
        repo_root, "data", "flair3d_plus", "missing_coord_tiles.details.csv"
    )
    if not os.path.exists(details_csv):
        logger = get_root_logger()
        logger.warning(
            "Flair3D missing tiles file not found: %s. Continuing with empty hardcoded missing tiles set.",
            details_csv,
        )
        return set()

    missing_tiles = set()
    with open(details_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("reason") == "missing_coord_file":
                split = row.get("split")
                patch_id = row.get("patch_id")
                if split and patch_id:
                    missing_tiles.add((split, patch_id))
    return missing_tiles


@DATASETS.register_module()
class Flair3DDataset(DefaultDataset):
    """Dataset for Flair3D / LidarHD preprocessed Pointcept scenes.
    
    
    
    :param csv_manifest: CSV manifest file path
        Lists all the scences in the dataset. It indicates wether the LIDARHD
        is available for the scene.
        
    :param missing_tiles_manifest: Missing tiles manifest file path
        Lists all the tiles that are missing from the dataset (but that were expected
        to be there). This file is usually produced by the preprocessing script
        ("missing_ply_preflight.txt").
        
    :param target_keys: Target keys. Supports semantic multitask for "segment", "forest",
        "land_use", and "natural_habitat". "elevation" can be combined with semantic keys.
        Targets are exposed in the batch under their task name.
    :param primary_target_key: Primary semantic target. Must be included in target_keys when
        provided.
    :param **kwargs: Additional arguments passed to :class:`DefaultDataset`.
    """

    VALID_ASSETS = [*DefaultDataset.VALID_ASSETS, *FLAIR3D_SPECIFIC_ASSETS]

    CORRUPTED_TILES = set()
    
    MISSING_LIDARHD_TILES = _load_missing_lidarhd_tiles()
    
    HARDCODED_EXCLUDED_TILES = CORRUPTED_TILES | MISSING_LIDARHD_TILES
    FLAIR3D_OPTIONAL_TARGETS = ("land_use", "natural_habitat", "elevation")
    #TODO@Geist : elevation should be complete, but I noticed some missing part in D049
    # e.g.: UU-S1-15

    def __init__(
        self,
        csv_manifest=None,
        missing_tiles_manifest=None,
        too_small_tiles_manifest=None,
        target_keys=("segment",),
        primary_target_key=None,
        **kwargs,
    ):
        self.csv_manifest = csv_manifest
        if isinstance(target_keys, str):
            target_keys = [target_keys]
        elif not isinstance(target_keys, Sequence):
            raise TypeError("target_keys must be a string or a sequence of strings.")
        if len(target_keys) == 0:
            raise ValueError("target_keys must contain at least one target key.")
        normalized_target_keys = []
        for tk in target_keys:
            if tk not in FLAIR3D_ALLOWED_TARGETS:
                raise ValueError(
                    f"Unsupported target key '{tk}'. Expected one of: {FLAIR3D_ALLOWED_TARGETS}."
                )
            if tk not in normalized_target_keys:
                normalized_target_keys.append(tk)
        self.target_keys = tuple(normalized_target_keys)
        normalized_optional_target_keys = []
        for tk in self.FLAIR3D_OPTIONAL_TARGETS:
            if tk in self.target_keys:
                normalized_optional_target_keys.append(tk)
        self.optional_target_keys = tuple(normalized_optional_target_keys)
        if primary_target_key is None:
            primary_target_key = self.target_keys[0]
        if primary_target_key not in self.target_keys:
            raise ValueError(
                "primary_target_key must be present in target_keys."
            )
        self.primary_target_key = primary_target_key
        if "elevation" in self.target_keys and len(self.target_keys) > 1:
            if self.primary_target_key not in FLAIR3D_SEMANTIC_TARGETS:
                raise ValueError(
                    "When target_keys mixes elevation with semantic targets, "
                    "primary_target_key must be one of "
                    f"{FLAIR3D_SEMANTIC_TARGETS} (got {self.primary_target_key!r})."
                )

        self.missing_tiles_manifest = missing_tiles_manifest
        self._missing_tiles = None

        self.too_small_tiles_manifest = too_small_tiles_manifest
        self._too_small_tiles = None
        super().__init__(**kwargs)
        get_root_logger().info(
            "Flair3DDataset target_keys=%s, optional_target_keys=%s, primary_target_key=%s",
            self.target_keys,
            self.optional_target_keys,
            self.primary_target_key,
        )

    def _get_missing_tiles(self):
        if self._missing_tiles is not None:
            return self._missing_tiles

        missing_tiles = set()
        if not self.missing_tiles_manifest:
            self._missing_tiles = missing_tiles
            return self._missing_tiles
        elif not os.path.exists(self.missing_tiles_manifest):
            logger = get_root_logger()
            logger.warning(
                f"Flair3D missing tiles file not found: {self.missing_tiles_manifest}. Continuing with empty missing tiles set.",
            )
            self._missing_tiles = missing_tiles
            return self._missing_tiles

        with open(self.missing_tiles_manifest, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = [part.strip() for part in stripped.split(",", 2)]
                if len(parts) < 2:
                    continue
                split, patch_id = parts[0], parts[1]
                if split and patch_id:
                    missing_tiles.add((split, patch_id))

        self._missing_tiles = missing_tiles
        return self._missing_tiles
    
    def _get_too_small_tiles(self, ):
        """
        Only filtering train tiles.
        """
        if self._too_small_tiles is not None:
            return self._too_small_tiles


        too_small_tiles = set()
        if not self.too_small_tiles_manifest:
            self._too_small_tiles = too_small_tiles
            return self._too_small_tiles
        
        elif not os.path.exists(self.too_small_tiles_manifest):
            logger = get_root_logger()
            logger.warning(
                f"Flair3D too-small tiles file not found: {self.too_small_tiles_manifest}. Continuing with empty too-small tiles set.",
            )
            self._too_small_tiles = too_small_tiles
            return self._too_small_tiles

        with open(self.too_small_tiles_manifest, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_split = row.get("split")
                patch_id = row.get("patch_id")
                if row_split == "train" and row_split and patch_id:
                    too_small_tiles.add((row_split, patch_id))

        self._too_small_tiles = too_small_tiles
        return self._too_small_tiles

    def get_data_list(self):
        if self.csv_manifest is None:
            return super().get_data_list()

        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise TypeError

        hardcoded_excluded = self.HARDCODED_EXCLUDED_TILES
        missing_excluded = self._get_missing_tiles()
        too_small_excluded = self._get_too_small_tiles()
        excluded_tiles = hardcoded_excluded | missing_excluded | too_small_excluded
        logger = get_root_logger()
        raw_total = (
            len(hardcoded_excluded) + len(missing_excluded) + len(too_small_excluded)
        )
        overlap_count = raw_total - len(excluded_tiles)
        logger.info(
            (
                "Excluded tiles breakdown: "
                "hardcoded=%d, missing_manifest=%d, too_small=%d, overlap=%d, total_unique=%d"
            ),
            len(hardcoded_excluded),
            len(missing_excluded),
            len(too_small_excluded),
            overlap_count,
            len(excluded_tiles),
        )
        data_list = []
        with open(self.csv_manifest, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] in split_list and row.get('LIDARHD') == 'True':
                    if (row['split'], row['patch_id']) in excluded_tiles:
                        continue
                    dept_year = row.get('dept_year') or row['patch_id'].split('_')[0]
                    roi = row.get('roi') or row['patch_id'].split('_')[1]
                    data_list.append(os.path.join(self.data_root, row['split'], f"{dept_year}_LIDARHD", roi, row['patch_id']))
        
        return data_list

    def get_data_name(self, idx):
        """Return scene id (folder name) for logging and saving."""
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    def _is_optional_target(self, target_key):
        return target_key in self.optional_target_keys

    def _missing_target_array(self, target_key, n):
        fill_value = get_missing_target_fill_value(target_key)
        if target_key in FLAIR3D_SEMANTIC_TARGETS:
            return np.full(n, int(fill_value), dtype=np.int32)
        if target_key in FLAIR3D_REGRESSION_TARGETS:
            return np.full(n, float(fill_value), dtype=np.float32)
        raise KeyError(f"Unsupported target key: {target_key}")

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        n = int(data_dict["coord"].shape[0])
        scene = self.data_list[idx % len(self.data_list)]

        if self.target_keys == ("elevation",):
            if "elevation" not in data_dict:
                if self._is_optional_target("elevation"):
                    data_dict["elevation"] = self._missing_target_array("elevation", n)
                    data_dict["segment"] = np.full(n, -1, dtype=np.int32)
                    return data_dict
                raise FileNotFoundError(
                    f"target_keys contains 'elevation' but elevation.npy missing under scene: {scene}"
                )
            elev = np.asarray(data_dict.pop("elevation"), dtype=np.float64).reshape(-1)
            if elev.shape[0] != n:
                raise ValueError(
                    f"elevation length {elev.shape[0]} does not match coord rows {n}"
                )
            data_dict["elevation"] = elev.astype(np.float32)
            data_dict["segment"] = np.full(n, -1, dtype=np.int32)
            return data_dict

        semantic_keys = [tk for tk in self.target_keys if tk != "elevation"]
        semantic_labels = {}
        for tk in semantic_keys:
            if tk == "segment":
                labels = np.asarray(data_dict["segment"]).reshape(-1)
            else:
                if tk not in data_dict:
                    if self._is_optional_target(tk):
                        labels = self._missing_target_array(tk, n)
                    else:
                        raise FileNotFoundError(
                            f"target key '{tk}' but {tk}.npy missing under scene: {scene}"
                        )
                else:
                    labels = np.asarray(data_dict[tk]).reshape(-1)
            if labels.shape[0] != n:
                raise ValueError(
                    f"{tk} length {labels.shape[0]} does not match coord rows {n}"
                )
            semantic_labels[tk] = labels.astype(np.int32)

        for tk, labels in semantic_labels.items():
            data_dict[tk] = labels

        if "elevation" in self.target_keys:
            if "elevation" not in data_dict:
                if self._is_optional_target("elevation"):
                    data_dict["elevation"] = self._missing_target_array("elevation", n)
                else:
                    raise FileNotFoundError(
                        f"target_keys contains 'elevation' but elevation.npy missing under scene: {scene}"
                    )
            else:
                elev = np.asarray(data_dict.pop("elevation"), dtype=np.float64).reshape(-1)
                if elev.shape[0] != n:
                    raise ValueError(
                        f"elevation length {elev.shape[0]} does not match coord rows {n}"
                    )
                data_dict["elevation"] = elev.astype(np.float32)

        return data_dict

    def prepare_test_data(self, idx):
        """Full-resolution multitask targets are popped into ``result_dict`` before voxelization.

        DefaultDataset only preserves ``segment`` + optional ``origin_segment`` / ``inverse``,
        which breaks multitask evaluation on the whole scene.
        """
        with record_data_pipeline("dataset.get_data"):
            data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(name=data_dict.pop("name"))
        for key in self.target_keys:
            if key in data_dict:
                result_dict[key] = data_dict.pop(key)
        origin_keys = [
            k for k in list(data_dict.keys()) if k.startswith("origin_")
        ]
        for k in origin_keys:
            result_dict[k] = data_dict.pop(k)
        if "inverse" in data_dict:
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict
