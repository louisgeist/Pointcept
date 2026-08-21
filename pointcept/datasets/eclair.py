import json
import os

from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class ECLAIRDataset(DefaultDataset):
    """ECLAIR aerial LiDAR semantic segmentation dataset.

    Args:
        include_pseudo: If True (default), keep all train scenes including
            pseudo-labeled tiles (`review_category == "rejected"`). If False,
            keep only approved (ground-truth) train scenes. Val/test are
            already GT-only in the official split.
    """

    def __init__(self, include_pseudo=True, **kwargs):
        self.include_pseudo = include_pseudo
        super().__init__(**kwargs)

    def get_data_list(self):
        data_list = super().get_data_list()
        if self.include_pseudo:
            return data_list

        filtered = []
        for path in data_list:
            meta_path = os.path.join(path, "meta.json")
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(
                    f"Missing meta.json for include_pseudo=False filter: {meta_path}"
                )
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("review_category") == "approved":
                filtered.append(path)
        return filtered
