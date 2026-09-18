"""
OpenGF (ground-filtering) benchmark -- preprocessed ALS tiles.

Binary segmentation: 0 = Ground, 1 = Non-ground, 2 = Outlier (kept as its own
label on disk -- see preprocessing/opengf/preprocess_opengf.py for how a
config should merge/drop/ignore it).
"""

import os

from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class OpenGFDataset(DefaultDataset):
    def get_data_name(self, idx):
        """Return scene id (folder name) for logging and saving."""
        return os.path.basename(self.data_list[idx % len(self.data_list)])
