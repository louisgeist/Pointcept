"""
MALiBU3D dataset: LiDAR HD tiles preprocessed into Pointcept scene folders.

Tiles live under ``<data_root>/<split>/<dept_year>_LIDARHD/<roi>/<tile_id>/`` with
coord.npy (float32, relative to the float64 coord_translation.npy), color.npy,
strength.npy, segment.npy and, depending on the tile, natural_habitat.npy ((N, 4)
ecological axes), elevation.npy and the 1 m rasters forest_2d.npy / network.npy.
"""

import os
from collections.abc import Sequence
from copy import deepcopy

import numpy as np

from .defaults import DefaultDataset
from .builder import DATASETS
from .transform import record_data_pipeline
from .malibu3d_config_utils import (
    MALIBU3D_PIXEL_SEMANTIC_TASKS,
    MALIBU3D_TILE_DISTRIBUTION_TASKS,
    get_missing_target_fill_value,
    get_task_config,
)
from pointcept.datasets.preprocessing.malibu3d.tile_catalog import iter_tiles
from .malibu3d_labels import NATHAB_AXIS_KEYS, unpack_nathab_axes
from pointcept.utils.logger import get_root_logger

MALIBU3D_SPECIFIC_ASSETS = (
    "natural_habitat",
    "elevation",
    "coord_translation",
    "network",
    "forest_2d",
)
MALIBU3D_SEMANTIC_TARGETS = ("segment",)
MALIBU3D_PIXEL_SEMANTIC_TARGETS = tuple(MALIBU3D_PIXEL_SEMANTIC_TASKS)
MALIBU3D_TILE_DISTRIBUTION_TARGETS = MALIBU3D_TILE_DISTRIBUTION_TASKS
MALIBU3D_REGRESSION_TARGETS = ("elevation",)
# Pixel size of the released forest_2d / network rasters (labels.json: forest_2d.pixel_m).
RELEASE_RASTER_PIXEL_M = 1.0

MALIBU3D_ALLOWED_TARGETS = (
    MALIBU3D_SEMANTIC_TARGETS
    + MALIBU3D_PIXEL_SEMANTIC_TARGETS
    + MALIBU3D_TILE_DISTRIBUTION_TARGETS
    + MALIBU3D_REGRESSION_TARGETS
)


@DATASETS.register_module()
class MALiBU3DDataset(DefaultDataset):
    """MALiBU3D tiles with any subset of the benchmark targets.

    :param csv_manifest: Tile list CSV: the MALiBU3D ``tiles.csv`` (or a subset of its
        rows), read via ``preprocessing/malibu3d/tile_catalog.py``.
        When None, scenes are globbed from ``data_root/<split>/`` like DefaultDataset.

    :param min_points: Optional dict mapping split name ("train"/"val") to a minimum
        point-count threshold. Tiles below the threshold are excluded, using the
        "n_points" column of csv_manifest. Raises if "n_points" is missing/empty for a row
        in a thresholded split, or if "test" is given a threshold.

    :param target_keys: Target keys: "segment", the pixel targets "forest_2d" and "network",
        the natural-habitat axes "nathab_*" and "elevation". Targets are exposed in the batch
        under their task name.
    :param primary_target_key: Primary semantic target. Must be included in target_keys when
        provided.
    :param **kwargs: Additional arguments passed to :class:`DefaultDataset`.
    """

    VALID_ASSETS = [*DefaultDataset.VALID_ASSETS, *MALIBU3D_SPECIFIC_ASSETS]

    # Targets that some tiles lack (no CarHab coverage, no DTM, roads not rasterized):
    # a missing file is replaced by the target's ignore/NaN fill instead of raising.
    MALIBU3D_OPTIONAL_TARGETS = (
        *NATHAB_AXIS_KEYS,
        "elevation",
        "network",
    )

    def __init__(
        self,
        csv_manifest=None,
        min_points=None,
        target_keys=("segment",),
        primary_target_key=None,
        **kwargs,
    ):
        self.csv_manifest = csv_manifest
        if isinstance(target_keys, str):
            target_keys = [target_keys]
        elif not isinstance(target_keys, Sequence):
            raise TypeError("target_keys must be a string or a sequence of strings.")
        if len(target_keys) == 0:
            raise ValueError("target_keys must contain at least one target key.")
        normalized_target_keys = []
        for tk in target_keys:
            if tk not in MALIBU3D_ALLOWED_TARGETS:
                raise ValueError(
                    f"Unsupported target key '{tk}'. Expected one of: {MALIBU3D_ALLOWED_TARGETS}."
                )
            if tk not in normalized_target_keys:
                normalized_target_keys.append(tk)
        self.target_keys = tuple(normalized_target_keys)
        normalized_optional_target_keys = []
        for tk in self.MALIBU3D_OPTIONAL_TARGETS:
            if tk in self.target_keys:
                normalized_optional_target_keys.append(tk)
        self.optional_target_keys = tuple(normalized_optional_target_keys)
        if primary_target_key is None:
            primary_target_key = self.target_keys[0]
        if primary_target_key not in self.target_keys:
            raise ValueError("primary_target_key must be present in target_keys.")
        self.primary_target_key = primary_target_key
        if "elevation" in self.target_keys and len(self.target_keys) > 1:
            if self.primary_target_key not in MALIBU3D_SEMANTIC_TARGETS:
                raise ValueError(
                    "When target_keys mixes elevation with semantic targets, "
                    "primary_target_key must be one of "
                    f"{MALIBU3D_SEMANTIC_TARGETS} (got {self.primary_target_key!r})."
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
        # tile_id -> (origin_x, origin_y) of its 1 m rasters, from tiles.csv. Used when the
        # tile has no meta.json, which is the case of the Hugging Face release.
        self.raster_origins = {}
        self._warned_missing_network = False

        super().__init__(**kwargs)
        get_root_logger().info(
            "MALiBU3DDataset target_keys=%s, optional_target_keys=%s, primary_target_key=%s",
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
        for tile in iter_tiles(self.csv_manifest, split_list):
            if self.min_points and tile.split in self.min_points:
                if tile.n_points is None:
                    raise ValueError(
                        f"min_points is configured for split {tile.split!r} but the "
                        f"'n_points' column is missing/empty for tile {tile.tile_id!r}. "
                        "Use the MALiBU3D tiles.csv, which has n_points for every tile."
                    )
                if tile.n_points < self.min_points[tile.split]:
                    min_points_excluded += 1
                    continue
            data_list.append(tile.tile_dir(self.data_root))
            if tile.raster_origin_x is not None and tile.raster_origin_y is not None:
                self.raster_origins[tile.tile_id] = (
                    tile.raster_origin_x,
                    tile.raster_origin_y,
                )

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
        fill_value = get_missing_target_fill_value(target_key)
        if target_key in MALIBU3D_PIXEL_SEMANTIC_TARGETS:
            return np.asarray(fill_value, dtype=np.uint8)
        if target_key in MALIBU3D_SEMANTIC_TARGETS:
            return np.full(n, int(fill_value), dtype=np.int32)
        if target_key in MALIBU3D_TILE_DISTRIBUTION_TARGETS:
            return np.full(n, int(fill_value), dtype=np.int32)
        if target_key in MALIBU3D_REGRESSION_TARGETS:
            return np.full(n, float(fill_value), dtype=np.float32)
        raise KeyError(f"Unsupported target key: {target_key}")

    def _unpack_natural_habitat_axes(self, data_dict, n, scene):
        """Unpack ``natural_habitat.npy`` ``(N, 4)`` ecological axes into ``nathab_*`` int32 vectors.

        The packed array is removed so the point-wise label reshape cannot flatten it.
        """
        if "natural_habitat" not in data_dict:
            return data_dict
        nh = np.asarray(data_dict.pop("natural_habitat"))
        if nh.shape[0] != n:
            raise ValueError(
                f"natural_habitat rows {nh.shape[0]} != coord rows {n} under scene: {scene}"
            )
        data_dict.update(unpack_nathab_axes(nh))
        return data_dict

    def _load_pixel_semantic_label(self, data_dict, scene, target_key="network"):
        """Load ``{target_key}.npy`` and grid meta for a pixel semantic task.

        The task trains on the ``channel_names`` of its task config (ROADS for
        ``network``). On-disk rasters may have more channels (``rasterize_network.py``
        writes ROADS, RAILROADS, TRANSMISSION_LINES); the trained ones are selected by
        name via ``meta.{target_key}.channel_order`` when present, else the first
        channels are kept.

        Tiles without any positive pixel have no ``{target_key}.npy``, only
        ``meta.{target_key}`` (``empty: true`` + width/height); they get an all-zero
        raster. Grid origin and pixel size come from meta.json, or from tiles.csv
        (``raster_origins``) when there is no meta.json.
        """
        import json

        cfg = get_task_config(target_key)
        r = int(cfg["num_networks"])
        channel_names = list(cfg["channel_names"])

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
        if (
            not raster_meta
            and target_key in data_dict
            and os.path.basename(scene) in getattr(self, "raster_origins", {})
        ):
            # Hugging Face layout: no meta.json; 1 m rasters georeferenced by tiles.csv.
            origin_x, origin_y = self.raster_origins[os.path.basename(scene)]
            pixel_m = RELEASE_RASTER_PIXEL_M

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
            # Empty mask: rasterize_network.py wrote the grid size in meta.json only.
            h = int(raster_meta.get("height", 1))
            w = int(raster_meta.get("width", 1))
            raster = np.zeros((r, max(h, 1), max(w, 1)), dtype=np.uint8)
        elif self._is_optional_target(target_key):
            if target_key == "network" and not getattr(
                self, "_warned_missing_network", False
            ):
                self._warned_missing_network = True
                get_root_logger().warning(
                    "network.npy missing under %s: its road labels are ignored. The released "
                    "dataset ships road graphs, not masks; rasterize them first with "
                    "pointcept/datasets/preprocessing/malibu3d/rasterize_network.py.",
                    scene,
                )
            raster = self._missing_target_array(target_key, 0)
        else:
            raise FileNotFoundError(
                f"target key '{target_key}' but {target_key}.npy missing under scene: {scene}"
            )

        # Expand the (r, 1, 1) missing-target fill to the meta grid when it is known.
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
        # No channel_order: assume rasterize_network.py's order (ROADS, RAILROADS, ...).
        return raster[:r]

    def get_data(self, idx):
        """Load a tile and attach its targets, keyed by task name.

        Point-wise labels become int32 ``(N,)`` (nathab axes are unpacked from
        natural_habitat.npy), pixel rasters keep ``(C, H, W)`` plus their grid
        origin / pixel size, elevation becomes float32 ``(N,)``. Missing optional
        targets get their fill value. For an elevation-only run, ``segment`` is set
        to -1 so the generic pipeline still finds a label array.
        """
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

        data_dict = self._unpack_natural_habitat_axes(data_dict, n, scene)

        pointwise_keys = [
            tk
            for tk in self.target_keys
            if tk not in MALIBU3D_PIXEL_SEMANTIC_TARGETS
            and tk not in MALIBU3D_TILE_DISTRIBUTION_TARGETS
            and tk != "elevation"
        ]
        pixel_semantic_keys = [
            tk for tk in self.target_keys if tk in MALIBU3D_PIXEL_SEMANTIC_TARGETS
        ]
        tile_distribution_keys = [
            tk for tk in self.target_keys if tk in MALIBU3D_TILE_DISTRIBUTION_TARGETS
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

        for tk in tile_distribution_keys:
            if tk not in data_dict:
                if self._is_optional_target(tk):
                    data_dict[tk] = self._missing_target_array(tk, n)
                else:
                    raise FileNotFoundError(
                        f"target key '{tk}' but natural_habitat.npy missing or "
                        f"incomplete under scene: {scene}"
                    )
            else:
                labels = np.asarray(data_dict[tk]).reshape(-1)
                if labels.shape[0] != n:
                    raise ValueError(
                        f"{tk} length {labels.shape[0]} does not match coord rows {n}"
                    )
                data_dict[tk] = labels.astype(np.int32)

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
                elev = np.asarray(data_dict.pop("elevation"), dtype=np.float64).reshape(
                    -1
                )
                if elev.shape[0] != n:
                    raise ValueError(
                        f"elevation length {elev.shape[0]} does not match coord rows {n}"
                    )
                data_dict["elevation"] = elev.astype(np.float32)

        return data_dict

    def prepare_test_data(self, idx):
        """Test-time sample with every target kept at full resolution.

        Same as ``DefaultDataset.prepare_test_data``, except that all ``target_keys``
        (not only ``segment``), the ``origin_*`` keys and ``inverse`` are moved to the
        result dict before voxelization, so multitask metrics cover the whole tile.
        """
        with record_data_pipeline("dataset.get_data"):
            data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(name=data_dict.pop("name"))
        for key in self.target_keys:
            if key not in data_dict:
                continue
            if key in MALIBU3D_PIXEL_SEMANTIC_TARGETS:
                # Copy, not pop: fragments still need the raster for the
                # pixel_semantic head.
                result_dict[key] = deepcopy(data_dict[key])
            else:
                result_dict[key] = data_dict.pop(key)
        origin_keys = [k for k in list(data_dict.keys()) if k.startswith("origin_")]
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
