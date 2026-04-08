import os

from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class DALESDataset(DefaultDataset):
    def get_data_name(self, idx):
        """Return scene id (folder name) for logging and saving."""
        return os.path.basename(self.data_list[idx % len(self.data_list)])