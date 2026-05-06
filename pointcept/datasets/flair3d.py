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

    def __init__(self, csv_manifest=None, **kwargs):
        self.csv_manifest = csv_manifest
        super().__init__(**kwargs)

    def get_data_list(self):
        if self.csv_manifest is None:
            return super().get_data_list()

        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        data_list = []
        with open(self.csv_manifest, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] in split_list and row.get('LIDARHD') == 'True':
                    dept_year = row.get('dept_year') or row['patch_id'].split('_')[0]
                    roi = row.get('roi') or row['patch_id'].split('_')[1]
                    data_list.append(os.path.join(self.data_root, row['split'], f"{dept_year}_LIDARHD", roi, row['patch_id']))
        
        return data_list

    def get_data_name(self, idx):
        """Return scene id (folder name) for logging and saving."""
        return os.path.basename(self.data_list[idx % len(self.data_list)])
