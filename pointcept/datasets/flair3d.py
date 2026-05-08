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


@DATASETS.register_module()
class Flair3DDataset(DefaultDataset):
    """Dataset for Flair3D / LidarHD preprocessed Pointcept scenes."""

    HARDCODED_EXCLUDED_TILES = {
        ("train", "D032-2019_AF-S1-V16_4-1"),
        ("train", "D033-2021_UU-S1-3_2-2"),
        ("val", "D034-2021_FU-S1-39_2-15"),
        ("train", "D038-2021_NN-S1-10_5-5"),
        ("train", "D044-2020_UA-S1-23_5-9"),
        ("train", "D046-2019_UF-S1-52_4-4"),
        ("train", "D049-2020_UU-S1-20_3-6"),
        ("train", "D051-2019_NN-S1-15_5-5"),
        ("train", "D052-2019_FA-S1-22_5-4"),
        ("train", "D063-2019_FF-S2-2_5-2"),
        ("test", "D064-2021_AA-S1-26_10-2"),
        ("val", "D065-2019_NN-S1-2_4-1"),
        ("train", "D066-2021_UU-S1-43_3-1"),
        ("val", "D067-2021_UA-S1-2_5-3"),
        ("test", "D068-2021_FF-S1-21_2-8"),
        ("test", "D069-2020_UU-S1-69_8-7"),
        ("test", "D069-2020_UU-S1-76_7-4"),
        ("train", "D070-2020_UA-S1-16_3-18"),
        ("train", "D070-2020_UF-S1-3_5-18"),
        ("test", "D071-2020_FF-S1-32_5-6"),
        ("train", "D072-2019_HV-S1-9_8-12"),
        ("train", "D086-2020_AU-S1-6_3-5"),
    }

    def __init__(self, csv_manifest=None, missing_tiles_manifest=None, **kwargs):
        self.csv_manifest = csv_manifest
        self.missing_tiles_manifest = missing_tiles_manifest
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
