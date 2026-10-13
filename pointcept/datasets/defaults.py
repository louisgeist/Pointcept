"""
Default Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import json
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence
from torchvision.transforms import InterpolationMode

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict

from .builder import DATASETS
from .transform import Compose, TRANSFORMS, record_data_pipeline
from .subset_utils import apply_subset_selection

INTERPOLATION_MODE = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


@DATASETS.register_module()
class DefaultDataset(Dataset):
    """Scenes stored as ``<data_root>/<split>/<scene>/*.npy`` (one array per asset).

    ``max_sample``, ``stratified_subset_manifest`` and ``include_names`` restrict the
    scene list after ``get_data_list`` (see ``subset_utils.apply_subset_selection``).
    """

    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "return_number",
        "number_of_returns",
        "segment",
        "instance",
        "pose",
    ]

    def __init__(
        self,
        split="train",
        data_root="data/dataset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
        loop=1,
        max_sample=None,
        stratified_subset_manifest=None,
        include_names=None,
    ):
        super(DefaultDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.ignore_index = ignore_index
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.max_sample = max_sample
        self.stratified_subset_manifest = stratified_subset_manifest
        self.include_names = include_names

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        self.data_list = apply_subset_selection(
            self.data_list,
            split=self.split,
            max_sample=self.max_sample,
            stratified_subset_manifest=self.stratified_subset_manifest,
            include_names=self.include_names,
        )
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} {} set.".format(
                len(self.data_list), self.loop, os.path.basename(self.data_root), split
            )
        )

    def get_data_list(self):
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        data_list = []
        for split in split_list:
            if os.path.isfile(os.path.join(self.data_root, split)):
                with open(os.path.join(self.data_root, split)) as f:
                    data_list += [
                        os.path.join(self.data_root, data) for data in json.load(f)
                    ]
            else:
                data_list += glob.glob(os.path.join(self.data_root, split, "*"))
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["split"] = split

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    def get_split_name(self, idx):
        return os.path.basename(
            os.path.dirname(self.data_list[idx % len(self.data_list)])
        )

    def prepare_train_data(self, idx):
        # load data
        with record_data_pipeline("dataset.get_data"):
            data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        with record_data_pipeline("dataset.get_data"):
            data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
