import json
import os

import numpy as np

from .defaults import DefaultDataset
from .builder import DATASETS
from pointcept.utils.logger import get_root_logger


@DATASETS.register_module()
class ECLAIRDataset(DefaultDataset):
    """ECLAIR aerial LiDAR semantic segmentation dataset.

    Args:
        include_pseudo: If True (default), keep all train scenes including
            pseudo-labeled tiles (`review_category == "rejected"`). If False,
            keep only approved (ground-truth) train scenes. Val/test are
            already GT-only in the official split.
        min_points: Optional dict mapping split name to a minimum point-count
            threshold. Tiles below it are excluded, checked against
            `coord.npy`'s shape (read via `mmap_mode="r"`, so this doesn't
            load full tiles into memory — cheap even for the whole split).
            Unlike `Flair3DDataset.min_points`, no manifest column is needed
            since ECLAIR has no national-scale csv_manifest. Mirrors
            Flair3DDataset's restriction: "test" cannot be thresholded.
    """

    def __init__(self, include_pseudo=True, min_points=None, **kwargs):
        self.include_pseudo = include_pseudo
        if min_points is not None:
            if not isinstance(min_points, dict):
                raise TypeError(
                    "min_points must be a dict mapping split name to a minimum point threshold."
                )
            if "test" in min_points:
                raise ValueError(
                    "min_points must not filter the 'test' split — the benchmark requires "
                    "the full test set. Only 'train'/'val' thresholds are allowed."
                )
        self.min_points = min_points
        super().__init__(**kwargs)

    def get_data_list(self):
        data_list = super().get_data_list()

        if not self.include_pseudo:
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
            data_list = filtered

        if self.min_points and self.split in self.min_points:
            threshold = self.min_points[self.split]
            filtered = []
            excluded = 0
            for path in data_list:
                n_points = np.load(
                    os.path.join(path, "coord.npy"), mmap_mode="r"
                ).shape[0]
                if n_points < threshold:
                    excluded += 1
                    continue
                filtered.append(path)
            get_root_logger().info(
                "min_points filter (split=%s, threshold=%d): excluded=%d, "
                "final_data_list_size=%d",
                self.split,
                threshold,
                excluded,
                len(filtered),
            )
            data_list = filtered

        return data_list
