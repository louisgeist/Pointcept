"""
PureForest tile classification dataset (preprocessed Pointcept scenes).

Each sample is a 50 m x 50 m LAZ tile with a single semantic class label (13 classes).

Author: Pointcept integration
"""

import os
from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose

from pointcept.datasets.preprocessing.pureforest.pureforest_classes import (
    CLASS_NAMES,
    NUM_CLASSES,
)


# Alignment with ModelNetDataset, as it is also a classification dataset
@DATASETS.register_module()
class PureForestDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/pureforest",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        class_names=None,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.class_names = class_names if class_names is not None else CLASS_NAMES
        if len(self.class_names) != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} class names, got {len(self.class_names)}."
            )

        if test_mode:
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in PureForest {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        split_dir = os.path.join(self.data_root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"PureForest split directory not found: {split_dir}. "
                "Run preprocess_pureforest.py first."
            )
        names = sorted(
            entry
            for entry in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, entry))
            and os.path.isfile(os.path.join(split_dir, entry, "coord.npy"))
        )
        if not names:
            raise FileNotFoundError(f"No preprocessed scenes under {split_dir}.")
        return names

    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        patch_stem = self.data_list[data_idx]
        scene_dir = os.path.join(self.data_root, self.split, patch_stem)
        coord = np.load(os.path.join(scene_dir, "coord.npy")).astype(np.float32)
        color = np.load(os.path.join(scene_dir, "color.npy")).astype(np.float32)
        category_val = int(np.load(os.path.join(scene_dir, "category.npy")))
        category = np.array([category_val], dtype=np.int64)
        return dict(
            coord=coord,
            color=color,
            category=category,
            name=patch_stem,
        )

    def get_data_name(self, idx):
        return self.data_list[idx % len(self.data_list)]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        return self.transform(data_dict)

    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx)
        category = data_dict.pop("category")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))
        for i in range(len(data_dict_list)):
            data_dict_list[i] = self.post_transform(data_dict_list[i])
        return dict(
            voting_list=data_dict_list,
            category=category,
            name=self.get_data_name(idx),
        )
