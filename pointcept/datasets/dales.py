"""
DALES aerial LiDAR semantic segmentation benchmark, as written by
preprocessing/dales/preprocess_dales.py (``<data_root>/{train,test}/<scene>/``).

Each scene holds coord.npy, segment.npy and strength.npy (raw intensity). Labels are
the 8 DALES classes (ids 0-7); raw "unknown" points are stored as 8 (use it as
ignore_index).
"""

import os

from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class DALESDataset(DefaultDataset):
    def get_data_name(self, idx):
        """Return scene id (folder name) for logging and saving."""
        return os.path.basename(self.data_list[idx % len(self.data_list)])