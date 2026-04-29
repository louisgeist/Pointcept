"""
H3D (Hessigheim 3D) benchmark — preprocessed LiDAR scenes.

Semantic labels use 11 classes with IDs 0–10 (see H3D paper, Table 1).
After preprocessing, expect the following layout::

    <data_root>/
        train/<scene_name>/*.npy
        val/<scene_name>/*.npy
        test/<scene_name>/*.npy

Each scene folder should contain coord.npy, segment.npy, and color.npy (RGB).
"""

import os

from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class H3DDataset(DefaultDataset):
    """Dataset for H3D (Hessigheim 3D) preprocessed Pointcept scenes."""

    def get_data_name(self, idx):
        """Return scene id (folder name) for logging and saving."""
        return os.path.basename(self.data_list[idx % len(self.data_list)])
