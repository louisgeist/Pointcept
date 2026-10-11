"""
OpenGF ground-filtering benchmark, as written by
preprocessing/opengf/preprocess_opengf.py (``<data_root>/{train,val,test}/<scene>/``).

Binary segmentation: 0 = Ground, 1 = Non-ground. Outlier points are stored as their
own label 2; a config merges them (``RemapSegment``), drops them (``DropSegmentClass``)
or ignores them (``ignore_index=2``), see the preprocessing script docstring.
"""

import os

from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class OpenGFDataset(DefaultDataset):
    def get_data_name(self, idx):
        """Return scene id (folder name) for logging and saving."""
        return os.path.basename(self.data_list[idx % len(self.data_list)])
