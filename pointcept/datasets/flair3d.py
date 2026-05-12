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
    """Dataset for Flair3D / LidarHD preprocessed Pointcept scenes.
    
    
    
    :param csv_manifest: CSV manifest file path
        Lists all the scences in the dataset. It indicates wether the LIDARHD
        is available for the scene.
        
    :param missing_tiles_manifest: Missing tiles manifest file path
        Lists all the tiles that are missing from the dataset (but that were expected
        to be there). This file is usually produced by the preprocessing script
        (`missing_ply_preflight.txt"`).
        
    :param **kwargs: Additional arguments
    """

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
        too_small_tiles_manifest=None,
        **kwargs,
    ):
        self.csv_manifest = csv_manifest
        
        self.missing_tiles_manifest = missing_tiles_manifest
        self._missing_tiles = None
        
        self.too_small_tiles_manifest = too_small_tiles_manifest
        self._too_small_tiles = None
        super().__init__(**kwargs)

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

    def get_data(self, idx):
        return super().get_data(idx)
