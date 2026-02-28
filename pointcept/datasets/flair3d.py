"""
Flair3D Dataset (LidarHD-like preprocessed scenes).

Scenes are expected under data_root as:
  <data_root>/<split>/<scene_id>/
with assets: coord.npy, color.npy, segment.npy, optionally strength.npy, normal.npy.
"""

import os

from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class Flair3DDataset(DefaultDataset):
    """Dataset for Flair3D / LidarHD preprocessed Pointcept scenes."""

    def get_data_name(self, idx):
        """Return scene id (folder name) for logging and saving."""
        return os.path.basename(self.data_list[idx % len(self.data_list)])
