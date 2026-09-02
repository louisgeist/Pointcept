"""
Malibu3D tile climatic-domain classification dataset (preprocessed Pointcept scenes).

Each sample is a LidarHD subtile with a scene-level label in climatic_domain.npy
(0=Temperate, 1=Mediterranean, 2=Alpine, -1=invalid/mixed).
"""

from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose
from .malibu3d import Malibu3DDataset

CLIMATIC_DOMAIN_CLASS_NAMES = ("Temperate", "Mediterranean", "Alpine")
NUM_CLIMATIC_DOMAIN_CLASSES = 3
CLIMATIC_DOMAIN_IGNORE_INDEX = -1


@DATASETS.register_module()
class Malibu3DClimaticDomainDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/malibu3d",
        csv_manifest=None,
        missing_tiles_manifest=None,
        too_small_tiles_manifest=None,
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        class_names=None,
    ):
        super().__init__()
        self.split = split
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.class_names = (
            class_names if class_names is not None else CLIMATIC_DOMAIN_CLASS_NAMES
        )
        if len(self.class_names) != NUM_CLIMATIC_DOMAIN_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLIMATIC_DOMAIN_CLASSES} class names, "
                f"got {len(self.class_names)}."
            )

        self._manifest_dataset = Malibu3DDataset(
            split=split,
            data_root=data_root,
            csv_manifest=csv_manifest,
            missing_tiles_manifest=missing_tiles_manifest,
            too_small_tiles_manifest=too_small_tiles_manifest,
            target_keys=("climatic_domain",),
            primary_target_key="climatic_domain",
            transform=[],
            test_mode=False,
            loop=1,
        )
        self.data_list = self._manifest_dataset.data_list

        if test_mode:
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in Malibu3DClimaticDomain {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data(self, idx):
        raw = self._manifest_dataset.get_data(idx)
        category_val = int(np.asarray(raw["climatic_domain"]).reshape(-1)[0])
        data_dict = dict(
            coord=raw["coord"].astype(np.float32),
            color=raw["color"].astype(np.float32),
            category=np.array([category_val], dtype=np.int64),
            name=self.get_data_name(idx),
        )
        if "strength" in raw:
            data_dict["strength"] = raw["strength"].astype(np.float32)
        return data_dict

    def get_data_name(self, idx):
        return self._manifest_dataset.get_data_name(idx)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

    def prepare_train_data(self, idx):
        return self.transform(self.get_data(idx))

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
