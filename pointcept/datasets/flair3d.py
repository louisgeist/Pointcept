"""
Flair3D Dataset (LidarHD-like preprocessed scenes).

Scenes are expected under data_root as:
  <data_root>/<split>/<dept_year>_LIDARHD/<roi>/<scene_id>/
with assets: coord.npy, color.npy, segment.npy, optionally strength.npy, normal.npy.
"""

import os
import csv
from collections.abc import Sequence

from .defaults import DefaultDataset
from .builder import DATASETS
from pointcept.utils.logger import get_root_logger


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
    """Dataset for Flair3D / LidarHD preprocessed Pointcept scenes."""

    CORRUPTED_TILES = {
        # LIDARHD .ply found, but corrupted ?
        ("train", "D086-2020_AU-S1-6_3-5"),
    }
    
    MISSING_LIDARHD_TILES = _load_missing_lidarhd_tiles()
    
    HARDCODED_EXCLUDED_TILES = CORRUPTED_TILES | MISSING_LIDARHD_TILES

    def __init__(
        self,
        csv_manifest=None,
        missing_tiles_manifest=None,
        min_points=1,
        min_points_train_only=True,
        **kwargs,
    ):
        self.csv_manifest = csv_manifest
        self.missing_tiles_manifest = missing_tiles_manifest
        self.min_points = int(min_points)
        self.min_points_train_only = bool(min_points_train_only)
        self._missing_tiles = None
        super().__init__(**kwargs)

    def _get_missing_tiles(self):
        if self._missing_tiles is not None:
            return self._missing_tiles

        missing_tiles = set()
        if not self.missing_tiles_manifest or not os.path.exists(self.missing_tiles_manifest):
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

    def _get_excluded_tiles(self):
        return self.HARDCODED_EXCLUDED_TILES | self._get_missing_tiles()

    def get_data_list(self):
        if self.csv_manifest is None:
            return super().get_data_list()

        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        excluded_tiles = self._get_excluded_tiles()
        logger = get_root_logger()
        logger.info(f"Excluded {len(excluded_tiles)} tiles")
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

    def _is_non_empty_coord(self, data_dict):
        coord = data_dict.get("coord", None)
        return coord is not None and coord.shape[0] > 0

    def _get_min_points_for_scene(self, idx):
        if self.min_points <= 1:
            return 1
        if not self.min_points_train_only:
            return self.min_points
        scene_split = self.get_split_name(idx)
        return self.min_points if scene_split == "train" else 1

    def _has_enough_points(self, data_dict, idx):
        coord = data_dict.get("coord", None)
        if coord is None:
            return False
        return coord.shape[0] >= self._get_min_points_for_scene(idx)

    def _log_rejected_scene(self, idx, data_dict):
        scene_name = self.get_data_name(idx)
        scene_split = self.get_split_name(idx)
        scene_path = self.data_list[idx % len(self.data_list)]
        coord = data_dict.get("coord", None)
        num_points = int(coord.shape[0]) if coord is not None else -1
        min_points = self._get_min_points_for_scene(idx)
        logger = get_root_logger()
        logger.warning(
            "Rejected scene in Flair3DDataset: split=%s name=%s path=%s num_points=%d min_points=%d",
            scene_split,
            scene_name,
            scene_path,
            num_points,
            min_points,
        )

    def get_data(self, idx):
        total = len(self.data_list)
        for offset in range(total):
            candidate_idx = idx + offset
            data_dict = super().get_data(candidate_idx)
            if self._is_non_empty_coord(data_dict) and self._has_enough_points(
                data_dict, candidate_idx
            ):
                if offset > 0:
                    logger = get_root_logger()
                    logger.warning(
                        "Recovered from empty scene(s): requested_idx=%d recovered_idx=%d",
                        idx,
                        candidate_idx,
                    )
                return data_dict
            self._log_rejected_scene(candidate_idx, data_dict)

        raise RuntimeError(
            "All scenes were rejected in Flair3DDataset after exclusions and min_points filtering; cannot build a valid batch."
        )
