"""
H3D (Hessigheim 3D) benchmark, as written by preprocessing/h3d/preprocess_h3d.py.

Semantic labels use 11 classes with IDs 0-10 (H3D paper, Table 1); configs use 11
(Void) as ignore_index. Layout::

    <data_root>/
        train/<scene_name>/*.npy
        val/<scene_name>/*.npy
        test/<scene_name>/*.npy

Each scene folder contains coord.npy, segment.npy and color.npy (RGB in [0, 255]).
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
