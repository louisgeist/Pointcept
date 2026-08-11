"""
Flair3D Dataset (LidarHD-like preprocessed scenes).

Scenes are expected under data_root as:
  <data_root>/<split>/<dept_year>_LIDARHD/<roi>/<scene_id>/
with assets: coord.npy, color.npy, segment.npy, optionally strength.npy, normal.npy.
"""

import os
import csv
from collections.abc import Sequence
from copy import deepcopy

import numpy as np

from .defaults import DefaultDataset
from .builder import DATASETS
from .transform import record_data_pipeline
from .flair3d_config_utils import (
    FLAIR3D_CLASSIFICATION_TARGET_KEYS,
    FLAIR3D_MULTILABEL_CLASSIFICATION_TARGET_KEYS,
    FLAIR3D_PIXEL_SEMANTIC_TARGET_KEYS,
    FLAIR3D_TILE_DISTRIBUTION_TARGET_KEYS,
    get_missing_target_fill_value,
)
from pointcept.utils.logger import get_root_logger

FLAIR3D_SPECIFIC_ASSETS = (
    "forest",
    "land_use",
    "natural_habitat",
    "elevation",
    "climatic_domain",
    "natural_habitat_multilabel",
    "coord_translation",
    "network",
    "forest_2d",
)
FLAIR3D_SEMANTIC_TARGETS = ("segment", "forest", "land_use", "natural_habitat")
FLAIR3D_CLASSIFICATION_TARGETS = FLAIR3D_CLASSIFICATION_TARGET_KEYS
FLAIR3D_MULTILABEL_CLASSIFICATION_TARGETS = FLAIR3D_MULTILABEL_CLASSIFICATION_TARGET_KEYS
FLAIR3D_PIXEL_SEMANTIC_TARGETS = FLAIR3D_PIXEL_SEMANTIC_TARGET_KEYS
FLAIR3D_TILE_DISTRIBUTION_TARGETS = FLAIR3D_TILE_DISTRIBUTION_TARGET_KEYS
FLAIR3D_REGRESSION_TARGETS = ("elevation",)
FLAIR3D_ALLOWED_TARGETS = (
    FLAIR3D_SEMANTIC_TARGETS
    + FLAIR3D_CLASSIFICATION_TARGETS
    + FLAIR3D_MULTILABEL_CLASSIFICATION_TARGETS
    + FLAIR3D_PIXEL_SEMANTIC_TARGETS
    + FLAIR3D_TILE_DISTRIBUTION_TARGETS
    + FLAIR3D_REGRESSION_TARGETS
)


@DATASETS.register_module()
class Flair3DDataset(DefaultDataset):
    """Dataset for Flair3D / LidarHD preprocessed Pointcept scenes.
    
    
    
    :param csv_manifest: CSV manifest file path
        Lists all the scences in the dataset. It indicates wether the LIDARHD
        is available for the scene.
        
    :param min_points: Optional dict mapping split name ("train"/"val") to a minimum
        point-count threshold. Tiles below the threshold are excluded, using the
        "n_points" column of csv_manifest (populate it via
        scripts/analyze_flair3d_test_point_voxel_counts.py --write_manifest). Raises if
        the "n_points" column is missing/empty for a row in a thresholded split, or if
        "test" is given a threshold.

    :param target_keys: Target keys. Supports semantic multitask for "segment", "forest",
        "land_use", and "natural_habitat". "elevation" can be combined with semantic keys.
        Targets are exposed in the batch under their task name.
    :param primary_target_key: Primary semantic target. Must be included in target_keys when
        provided.
    :param task_configs: Optional dict of task_name -> task config, mirroring the
        model/eval-side ``data.task_configs``. Only pixel_semantic entries are used, to
        override the registry's per-target ``num_networks``/``channel_names`` (e.g. a
        config that trains ``network`` on ROADS only, dropping RAILROADS). Absent or
        missing entries fall back to ``FLAIR3D_PIXEL_SEMANTIC_TASKS`` defaults.
    :param **kwargs: Additional arguments passed to :class:`DefaultDataset`.
    """

    VALID_ASSETS = [*DefaultDataset.VALID_ASSETS, *FLAIR3D_SPECIFIC_ASSETS]

    FLAIR3D_OPTIONAL_TARGETS = (
        "land_use",
        "natural_habitat",
        "elevation",
        "climatic_domain",
        "natural_habitat_multilabel",
        "network",
    )
    #TODO@Geist : elevation should be complete, but I noticed some missing part in D049
    # e.g.: UU-S1-15

    def __init__(
        self,
        csv_manifest=None,
        min_points=None,
        target_keys=("segment",),
        primary_target_key=None,
        task_configs=None,
        **kwargs,
    ):
        self.csv_manifest = csv_manifest
        # Optional per-target task_config overrides (e.g. a config-local "network"
        # entry with num_networks/channel_names trimmed to drop a channel such as
        # RAILROADS). Only pixel_semantic entries are consulted here; falls back
        # to the FLAIR3D_PIXEL_SEMANTIC_TASKS registry default when absent so
        # existing configs that don't pass this keep their exact behavior.
        self.task_configs = dict(task_configs) if task_configs else {}
        if isinstance(target_keys, str):
            target_keys = [target_keys]
        elif not isinstance(target_keys, Sequence):
            raise TypeError("target_keys must be a string or a sequence of strings.")
        if len(target_keys) == 0:
            raise ValueError("target_keys must contain at least one target key.")
        normalized_target_keys = []
        for tk in target_keys:
            if tk not in FLAIR3D_ALLOWED_TARGETS:
                raise ValueError(
                    f"Unsupported target key '{tk}'. Expected one of: {FLAIR3D_ALLOWED_TARGETS}."
                )
            if tk not in normalized_target_keys:
                normalized_target_keys.append(tk)
        self.target_keys = tuple(normalized_target_keys)
        normalized_optional_target_keys = []
        for tk in self.FLAIR3D_OPTIONAL_TARGETS:
            if tk in self.target_keys:
                normalized_optional_target_keys.append(tk)
        self.optional_target_keys = tuple(normalized_optional_target_keys)
        if primary_target_key is None:
            primary_target_key = self.target_keys[0]
        if primary_target_key not in self.target_keys:
            raise ValueError(
                "primary_target_key must be present in target_keys."
            )
        self.primary_target_key = primary_target_key
        if primary_target_key in FLAIR3D_CLASSIFICATION_TARGETS:
            if any(tk in FLAIR3D_SEMANTIC_TARGETS for tk in self.target_keys):
                raise ValueError(
                    "primary_target_key cannot be a classification target when "
                    "semantic targets are also requested."
                )
        if primary_target_key in FLAIR3D_PIXEL_SEMANTIC_TARGETS:
            if any(tk in FLAIR3D_SEMANTIC_TARGETS for tk in self.target_keys):
                # Allowed later for multitask; keep mono-task simple for now.
                pass
        if "elevation" in self.target_keys and len(self.target_keys) > 1:
            if self.primary_target_key not in FLAIR3D_SEMANTIC_TARGETS:
                raise ValueError(
                    "When target_keys mixes elevation with semantic targets, "
                    "primary_target_key must be one of "
                    f"{FLAIR3D_SEMANTIC_TARGETS} (got {self.primary_target_key!r})."
                )

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
        get_root_logger().info(
            "Flair3DDataset target_keys=%s, optional_target_keys=%s, primary_target_key=%s",
            self.target_keys,
            self.optional_target_keys,
            self.primary_target_key,
        )

    def get_data_list(self):
        if self.csv_manifest is None:
            return super().get_data_list()

        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise TypeError

        logger = get_root_logger()
        data_list = []
        min_points_excluded = 0
        with open(self.csv_manifest, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] in split_list and row.get('LIDARHD') == 'True':
                    if self.min_points and row['split'] in self.min_points:
                        n_points_raw = row.get('n_points')
                        if not n_points_raw:
                            raise ValueError(
                                f"min_points is configured for split {row['split']!r} but the "
                                f"'n_points' column is missing/empty for tile {row['patch_id']!r}. "
                                "Run scripts/analyze_flair3d_test_point_voxel_counts.py "
                                f"--write_manifest for split {row['split']!r} before enabling "
                                "min_points on it."
                            )
                        if int(n_points_raw) < self.min_points[row['split']]:
                            min_points_excluded += 1
                            continue
                    dept_year = row.get('dept_year') or row['patch_id'].split('_')[0]
                    roi = row.get('roi') or row['patch_id'].split('_')[1]
                    data_list.append(os.path.join(self.data_root, row['split'], f"{dept_year}_LIDARHD", roi, row['patch_id']))

        if self.min_points:
            logger.info(
                "min_points filter: excluded=%d, final_data_list_size=%d",
                min_points_excluded,
                len(data_list),
            )
        return data_list

    def get_data_name(self, idx):
        """Return scene id (folder name) for logging and saving."""
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    def _is_optional_target(self, target_key):
        return target_key in self.optional_target_keys

    def _missing_target_array(self, target_key, n):
        fill_value = get_missing_target_fill_value(
            target_key,
            pixel_semantic_config=getattr(self, "task_configs", {}).get(target_key),
        )
        if target_key in FLAIR3D_CLASSIFICATION_TARGETS:
            return np.array([int(fill_value)], dtype=np.int64)
        if target_key in FLAIR3D_MULTILABEL_CLASSIFICATION_TARGETS:
            return np.asarray(fill_value, dtype=np.float32).reshape(1, -1)
        if target_key in FLAIR3D_PIXEL_SEMANTIC_TARGETS:
            return np.asarray(fill_value, dtype=np.uint8)
        if target_key in FLAIR3D_SEMANTIC_TARGETS:
            return np.full(n, int(fill_value), dtype=np.int32)
        if target_key in FLAIR3D_REGRESSION_TARGETS:
            return np.full(n, float(fill_value), dtype=np.float32)
        raise KeyError(f"Unsupported target key: {target_key}")

    def _load_pixel_semantic_label(self, data_dict, scene, target_key="network"):
        """Load ``{target_key}.npy`` and grid meta for a pixel semantic task.

        Training heads use ``num_networks`` channels from
        ``get_pixel_semantic_config(target_key)``. On-disk rasters may have more
        channels than the task trains on (e.g. historical ``network.npy`` with
        TRANSMISSION_LINES as channel 2); those are sliced via
        ``meta.{target_key}.channel_order`` when present, else the first ``r``
        channels.

        Empty tiles may omit ``{target_key}.npy`` and only store
        ``meta.{target_key}`` (``empty: true`` + width/height); those are
        synthesized as zeros.

        ``self.task_configs[target_key]``, when set, overrides the registry's
        ``num_networks``/``channel_names`` (e.g. to drop RAILROADS from ``network``).
        """
        import json

        from pointcept.datasets.flair3d_config_utils import (
            NETWORK_CHANNEL_NAMES,
            get_pixel_semantic_config,
        )

        cfg = getattr(self, "task_configs", {}).get(target_key) or get_pixel_semantic_config(
            target_key
        )
        r = int(cfg["num_networks"])
        default_channel_names = (
            NETWORK_CHANNEL_NAMES if target_key == "network" else (target_key,)
        )
        channel_names = list(cfg.get("channel_names") or default_channel_names)

        origin_x = 0.0
        origin_y = 0.0
        pixel_m = 1.0
        raster_meta = {}
        meta_path = os.path.join(scene, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            maybe = meta.get(target_key) or {}
            if isinstance(maybe, dict):
                raster_meta = maybe
                origin_x = float(raster_meta.get("origin_x", 0.0))
                origin_y = float(raster_meta.get("origin_y", 0.0))
                pixel_m = float(raster_meta.get("pixel_m", 1.0))

        if target_key in data_dict:
            raster = np.asarray(data_dict[target_key])
            if raster.ndim != 3:
                raise ValueError(
                    f"{target_key}.npy expected shape (C, H, W), got {raster.shape} "
                    f"under scene: {scene}"
                )
            raster = self._select_pixel_semantic_channels(
                raster,
                r=r,
                channel_names=channel_names,
                channel_order=raster_meta.get("channel_order"),
                scene=scene,
                target_key=target_key,
            )
            raster = raster.astype(np.uint8, copy=False)
        elif raster_meta:
            # Preprocess wrote meta only (empty mask) or optional missing fill path.
            h = int(raster_meta.get("height", 1))
            w = int(raster_meta.get("width", 1))
            raster = np.zeros((r, max(h, 1), max(w, 1)), dtype=np.uint8)
        elif self._is_optional_target(target_key):
            raster = self._missing_target_array(target_key, 0)
        else:
            raise FileNotFoundError(
                f"target key '{target_key}' but {target_key}.npy missing under scene: {scene}"
            )

        # Align tiny optional fill (1,1) to meta grid when present.
        if raster.shape[1] == 1 and raster.shape[2] == 1 and raster_meta:
            h = int(raster_meta.get("height", 1))
            w = int(raster_meta.get("width", 1))
            if h > 1 or w > 1:
                raster = np.zeros((r, h, w), dtype=np.uint8)

        data_dict[target_key] = raster
        # Keep origins in float64 for precise cell binning in NetworkRasterToPointLabels.
        data_dict[f"{target_key}_origin_x"] = np.asarray([origin_x], dtype=np.float64)
        data_dict[f"{target_key}_origin_y"] = np.asarray([origin_y], dtype=np.float64)
        data_dict[f"{target_key}_pixel_m"] = np.asarray([pixel_m], dtype=np.float64)
        return data_dict

    @staticmethod
    def _select_pixel_semantic_channels(
        raster, *, r, channel_names, channel_order, scene, target_key
    ):
        """Reduce on-disk ``(C, H, W)`` to the ``r`` training channels."""
        c = int(raster.shape[0])
        if c == r:
            return raster
        if c < r:
            raise ValueError(
                f"{target_key}.npy has {c} channels but task expects {r} "
                f"({channel_names}) under scene: {scene}"
            )
        if isinstance(channel_order, (list, tuple)) and len(channel_order) == c:
            name_to_idx = {str(name): i for i, name in enumerate(channel_order)}
            missing = [name for name in channel_names if name not in name_to_idx]
            if missing:
                raise ValueError(
                    f"{target_key}.npy channel_order {list(channel_order)} missing "
                    f"{missing} under scene: {scene}"
                )
            indices = [name_to_idx[name] for name in channel_names]
            return raster[indices]
        # Historical preprocess order: ROADS, RAILROADS, TRANSMISSION_LINES, ...
        return raster[:r]

    def _load_classification_label(self, data_dict, target_key, scene):
        if target_key in data_dict:
            return np.array([int(np.asarray(data_dict[target_key]).reshape(-1)[0])], dtype=np.int64)
        if self._is_optional_target(target_key):
            return self._missing_target_array(target_key, 0)
        raise FileNotFoundError(
            f"target key '{target_key}' but {target_key}.npy missing under scene: {scene}"
        )

    def _load_multilabel_classification_label(self, data_dict, target_key, scene):
        from pointcept.datasets.flair3d_config_utils import (
            get_multilabel_classification_config,
        )

        expected = int(get_multilabel_classification_config(target_key)["num_classes"])
        if target_key in data_dict:
            vector = np.asarray(data_dict[target_key], dtype=np.float32).reshape(-1)
            if vector.shape[0] != expected:
                raise ValueError(
                    f"{target_key} length {vector.shape[0]} != expected {expected} "
                    f"under scene: {scene}"
                )
            return vector.astype(np.float32).reshape(1, -1)
        if self._is_optional_target(target_key):
            return self._missing_target_array(target_key, 0)
        raise FileNotFoundError(
            f"target key '{target_key}' but {target_key}.npy missing under scene: {scene}"
        )

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        n = int(data_dict["coord"].shape[0])
        scene = self.data_list[idx % len(self.data_list)]

        if self.target_keys == ("elevation",):
            if "elevation" not in data_dict:
                if self._is_optional_target("elevation"):
                    data_dict["elevation"] = self._missing_target_array("elevation", n)
                    data_dict["segment"] = np.full(n, -1, dtype=np.int32)
                    return data_dict
                raise FileNotFoundError(
                    f"target_keys contains 'elevation' but elevation.npy missing under scene: {scene}"
                )
            elev = np.asarray(data_dict.pop("elevation"), dtype=np.float64).reshape(-1)
            if elev.shape[0] != n:
                raise ValueError(
                    f"elevation length {elev.shape[0]} does not match coord rows {n}"
                )
            data_dict["elevation"] = elev.astype(np.float32)
            data_dict["segment"] = np.full(n, -1, dtype=np.int32)
            return data_dict

        pointwise_keys = [
            tk
            for tk in self.target_keys
            if tk not in FLAIR3D_CLASSIFICATION_TARGETS
            and tk not in FLAIR3D_MULTILABEL_CLASSIFICATION_TARGETS
            and tk not in FLAIR3D_PIXEL_SEMANTIC_TARGETS
            and tk not in FLAIR3D_TILE_DISTRIBUTION_TARGETS
            and tk != "elevation"
        ]
        classification_keys = [
            tk for tk in self.target_keys if tk in FLAIR3D_CLASSIFICATION_TARGETS
        ]
        multilabel_classification_keys = [
            tk
            for tk in self.target_keys
            if tk in FLAIR3D_MULTILABEL_CLASSIFICATION_TARGETS
        ]
        pixel_semantic_keys = [
            tk for tk in self.target_keys if tk in FLAIR3D_PIXEL_SEMANTIC_TARGETS
        ]
        semantic_labels = {}
        for tk in pointwise_keys:
            if tk == "segment":
                labels = np.asarray(data_dict["segment"]).reshape(-1)
            else:
                if tk not in data_dict:
                    if self._is_optional_target(tk):
                        labels = self._missing_target_array(tk, n)
                    else:
                        raise FileNotFoundError(
                            f"target key '{tk}' but {tk}.npy missing under scene: {scene}"
                        )
                else:
                    labels = np.asarray(data_dict[tk]).reshape(-1)
            if labels.shape[0] != n:
                raise ValueError(
                    f"{tk} length {labels.shape[0]} does not match coord rows {n}"
                )
            semantic_labels[tk] = labels.astype(np.int32)

        for tk, labels in semantic_labels.items():
            data_dict[tk] = labels

        for tk in classification_keys:
            data_dict[tk] = self._load_classification_label(data_dict, tk, scene)

        for tk in multilabel_classification_keys:
            data_dict[tk] = self._load_multilabel_classification_label(
                data_dict, tk, scene
            )

        for tk in pixel_semantic_keys:
            data_dict = self._load_pixel_semantic_label(data_dict, scene, target_key=tk)

        if "elevation" in self.target_keys:
            if "elevation" not in data_dict:
                if self._is_optional_target("elevation"):
                    data_dict["elevation"] = self._missing_target_array("elevation", n)
                else:
                    raise FileNotFoundError(
                        f"target_keys contains 'elevation' but elevation.npy missing under scene: {scene}"
                    )
            else:
                elev = np.asarray(data_dict.pop("elevation"), dtype=np.float64).reshape(-1)
                if elev.shape[0] != n:
                    raise ValueError(
                        f"elevation length {elev.shape[0]} does not match coord rows {n}"
                    )
                data_dict["elevation"] = elev.astype(np.float32)

        return data_dict

    def prepare_test_data(self, idx):
        """Full-resolution multitask targets are popped into `result_dict` before voxelization.

        DefaultDataset only preserves `segment` + optional `origin_segment` / `inverse`,
        which breaks multitask evaluation on the whole scene.
        """
        with record_data_pipeline("dataset.get_data"):
            data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(name=data_dict.pop("name"))
        for key in self.target_keys:
            if key not in data_dict:
                continue
            if key in FLAIR3D_PIXEL_SEMANTIC_TARGETS:
                # Keep raster in data_dict so fragments still carry it for the
                # pixel_semantic head; also expose GT at scene level for metrics.
                result_dict[key] = deepcopy(data_dict[key])
            else:
                result_dict[key] = data_dict.pop(key)
        origin_keys = [
            k for k in list(data_dict.keys()) if k.startswith("origin_")
        ]
        for k in origin_keys:
            result_dict[k] = data_dict.pop(k)
        if "inverse" in data_dict:
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
