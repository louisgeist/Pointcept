"""
MALiBU3D land-cover dataset for a vanilla Pointcept checkout.

Copy this file to ``pointcept/datasets/malibu3d.py`` and add
``from .malibu3d import MALiBU3DDataset`` to ``pointcept/datasets/__init__.py``.
It subclasses :class:`pointcept.datasets.DefaultDataset` and reads only the
land-cover assets (``coord.npy``, ``color.npy``, ``strength.npy``, ``segment.npy``).
It does not depend on the multitask code shipped with the MALiBU3D training repo.

On-disk layout (after extracting the Hugging Face release):

    {data_root}/{split}/{dept_year}_LIDARHD/{roi}/{tile_id}/
        coord.npy     float32 (N, 3), meters, relative to the tile origin
        color.npy     uint8 (N, 3) RGB
        strength.npy  float32 (N,) intensity in [0, 1]
        segment.npy   uint8 (N,) class id, Void = 15

``tiles.csv`` (next to the tiles) lists every tile. Coordinates stay in the
tile-local frame: ``GridSample`` voxels them in meters, and
:class:`MALiBU3DPrepareFeat` copies them into ``coord_feat`` scaled by 0.01
for the backbone features.

The file also registers :class:`MALiBU3DCrossEntropyLoss` and
:class:`MALiBU3DLovaszLoss`, the upstream losses with a guard for batches in
which every point is Void (about 0.4 % of the tiles are entirely Void).
"""

import csv
import os
from collections.abc import Sequence

import numpy as np

from pointcept.datasets.builder import DATASETS
from pointcept.datasets.defaults import DefaultDataset
from pointcept.datasets.transform import TRANSFORMS
from pointcept.models.losses.builder import LOSSES
from pointcept.models.losses.lovasz import LovaszLoss
from pointcept.models.losses.misc import CrossEntropyLoss
from pointcept.utils.logger import get_root_logger

# Trainable classes, in id order. Void is the next id (MALIBU3D_IGNORE_INDEX) and
# is not a class the head predicts.
MALIBU3D_CLASS_NAMES = (
    "Building",
    "Greenhouse",
    "Impervious surface",
    "Other soil",
    "Herbaceous",
    "Vineyard",
    "Brushwood",
    "Infrastructure",
    "Swimming pool",
    "Water",
    "Deciduous",
    "Coniferous",
    "Bridge",
    "Agricultural soil",
    "Soil under vegetation",
)
MALIBU3D_NUM_CLASSES = len(MALIBU3D_CLASS_NAMES)
MALIBU3D_IGNORE_INDEX = MALIBU3D_NUM_CLASSES  # Void stored as 15 in segment.npy

_REQUIRED_COLUMNS = ("split", "dept_year", "roi", "tile_id")
_NA_TOKENS = frozenset({"", "<na>", "na", "none", "n/a", "nan", "null"})
_DEFAULT_MIN_POINTS = {"train": 1000}


def _parse_optional_int(raw):
    token = (raw or "").strip()
    if token.lower() in _NA_TOKENS:
        return None
    return int(token)


def _tile_dir(data_root, split, dept_year, roi, tile_id):
    return os.path.join(data_root, split, f"{dept_year}_LIDARHD", roi, tile_id)


@TRANSFORMS.register_module()
class MALiBU3DPrepareFeat(object):
    """Build backbone features without changing the coordinates used for sampling.

    ``coord`` stays in meters so ``GridSample(grid_size=0.1)`` is a 10 cm voxel
    grid. ``coord_feat`` is ``coord * coord_scale`` (0.01 in the paper) and is
    what ``Collect`` should concatenate. ``strength`` is reshaped to ``(N, 1)``
    so vanilla ``Collect`` can ``cat`` it with ``coord_feat`` and ``color``.

    Run this after the last geometric transform and after ``NormalizeColor``,
    and before ``ToTensor``.
    """

    def __init__(self, coord_scale=0.01):
        self.coord_scale = float(coord_scale)

    def __call__(self, data_dict):
        if "coord" not in data_dict:
            raise KeyError("MALiBU3DPrepareFeat requires 'coord'.")
        if "strength" not in data_dict:
            raise KeyError(
                "MALiBU3DPrepareFeat requires 'strength' (strength.npy on each tile)."
            )
        coord = data_dict["coord"]
        data_dict["coord_feat"] = _scale_like(coord, self.coord_scale)
        data_dict["strength"] = _as_column(data_dict["strength"])
        return data_dict


def _scale_like(coord, scale):
    """Return ``coord * scale`` with ``coord``'s dtype, without writing into ``coord``."""
    if isinstance(coord, np.ndarray):
        return coord * np.array(scale, dtype=coord.dtype)
    # torch.Tensor (and anything else that multiplies in its own dtype).
    return coord * coord.new_tensor(scale)


def _as_column(strength):
    if isinstance(strength, np.ndarray):
        strength = np.asarray(strength)
        if strength.ndim == 1:
            return strength.reshape(-1, 1)
        if strength.ndim == 2 and strength.shape[1] == 1:
            return strength
        raise ValueError(
            f"strength must have shape (N,) or (N, 1), got {tuple(strength.shape)}."
        )
    if strength.ndim == 1:
        return strength.reshape(-1, 1)
    if strength.ndim == 2 and strength.shape[1] == 1:
        return strength
    raise ValueError(
        f"strength must have shape (N,) or (N, 1), got {tuple(strength.shape)}."
    )


# Upstream CrossEntropyLoss (mean) is NaN and upstream LovaszLoss crashes when every
# point of a batch is ignore_index. MALiBU3D has entirely Void tiles (4 of the 2,000
# validation tiles; validation runs one tile per batch), so the config uses these two.
def _all_ignored(target, ignore_index):
    return ignore_index is not None and not bool((target != ignore_index).any())


@LOSSES.register_module()
class MALiBU3DCrossEntropyLoss(CrossEntropyLoss):
    """Upstream ``CrossEntropyLoss`` that returns 0 instead of NaN on an all-Void batch."""

    def __init__(self, ignore_index=-1, **kwargs):
        super().__init__(ignore_index=ignore_index, **kwargs)
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        if _all_ignored(target, self.ignore_index):
            return pred.sum() * 0.0  # 0-dim, in the graph, zero gradient
        return super().forward(pred, target)


@LOSSES.register_module()
class MALiBU3DLovaszLoss(LovaszLoss):
    """Upstream ``LovaszLoss`` that returns 0 instead of crashing on an all-Void batch."""

    def forward(self, y_pred, y_true):
        if _all_ignored(y_true, self.ignore_index):
            return y_pred.sum() * 0.0
        return super().forward(y_pred, y_true)


@DATASETS.register_module()
class MALiBU3DDataset(DefaultDataset):
    """MALiBU3D tiles for 15-class land-cover segmentation.

    :param csv_manifest: Tile table. Defaults to ``{data_root}/tiles.csv``.
        A partial extract (one region, or the toy set) must point this at a CSV
        that lists only the tiles present on disk.
    :param min_points: ``{split: minimum n_points}``. Defaults to
        ``{"train": 1000}``, the filter used by the paper. Pass ``{}`` to keep
        every tile. ``"test"`` is rejected: the benchmark test split is complete.
    :param subset_manifest: Optional CSV with a ``tile_id`` column (and, when
        present, a ``split`` column). Only listed tiles of the current split are
        kept. ``None`` uses the whole split. The paper's
        ``val_subset_2000tiles.csv`` is this kind of file; it is not the default.
    :param data_root: Root of the extracted dataset. Defaults to ``data/malibu3d``.
    :param ignore_index: Label skipped by the loss. Defaults to 15 (Void). Ids are
        not remapped.
    """

    def __init__(
        self,
        csv_manifest=None,
        min_points=None,
        subset_manifest=None,
        **kwargs,
    ):
        if min_points is None:
            min_points = dict(_DEFAULT_MIN_POINTS)
        if not isinstance(min_points, dict):
            raise TypeError(
                "min_points must be a dict mapping split name to a minimum point "
                f"count, got {type(min_points).__name__}."
            )
        normalized_min_points = {}
        for split_name, threshold in min_points.items():
            key = str(split_name).strip().lower()
            if key == "test":
                raise ValueError(
                    "min_points must not filter the 'test' split — the benchmark "
                    "requires the full test set. Only 'train'/'val' thresholds are "
                    "allowed."
                )
            normalized_min_points[key] = int(threshold)
        self.min_points = normalized_min_points
        self.subset_manifest = subset_manifest

        if "data_root" not in kwargs:
            kwargs["data_root"] = "data/malibu3d"
        if csv_manifest is None:
            csv_manifest = os.path.join(kwargs["data_root"], "tiles.csv")
        self.csv_manifest = csv_manifest
        if "ignore_index" not in kwargs:
            kwargs["ignore_index"] = MALIBU3D_IGNORE_INDEX

        super().__init__(**kwargs)
        get_root_logger().info(
            "MALiBU3DDataset csv_manifest=%s min_points=%s subset_manifest=%s",
            self.csv_manifest,
            self.min_points,
            self.subset_manifest,
        )

    def get_data_list(self):
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = list(self.split)
        else:
            raise NotImplementedError(
                f"split must be a string or a sequence of strings, got {self.split!r}."
            )
        split_list = [str(split).strip().lower() for split in split_list]

        subset_ids = None
        if self.subset_manifest:
            subset_ids = _load_subset_ids(self.subset_manifest, split_list)

        logger = get_root_logger()
        data_list = []
        missing = []
        excluded = 0
        for tile in _iter_tiles(self.csv_manifest, split_list):
            if tile["split"] in self.min_points:
                if tile["n_points"] is None:
                    raise ValueError(
                        f"min_points is configured for split {tile['split']!r} but "
                        f"'n_points' is missing/empty for tile {tile['tile_id']!r}. "
                        "Use the MALiBU3D tiles.csv, which has n_points for every tile."
                    )
                if tile["n_points"] < self.min_points[tile["split"]]:
                    excluded += 1
                    continue
            if subset_ids is not None and tile["tile_id"] not in subset_ids:
                continue
            tile_dir = _tile_dir(
                self.data_root,
                tile["split"],
                tile["dept_year"],
                tile["roi"],
                tile["tile_id"],
            )
            if not os.path.isdir(tile_dir):
                missing.append(tile_dir)
                continue
            data_list.append(tile_dir)

        if missing:
            preview = ", ".join(missing[:3])
            extra = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
            raise FileNotFoundError(
                f"{len(missing)} tile directories listed in {self.csv_manifest} "
                f"were not found under {self.data_root}: {preview}{extra}. "
                "Point csv_manifest at a CSV that lists only extracted tiles "
                "(for example tiles_toy.csv)."
            )
        if self.min_points:
            logger.info(
                "min_points filter: excluded=%d, final_data_list_size=%d",
                excluded,
                len(data_list),
            )
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        for asset in ("coord.npy", "segment.npy"):
            asset_path = os.path.join(data_path, asset)
            if not os.path.isfile(asset_path):
                raise FileNotFoundError(f"Missing {asset} in {data_path}")
        data_dict = super().get_data(idx)
        segment = np.asarray(data_dict["segment"])
        if segment.size and (
            int(segment.min()) < 0 or int(segment.max()) > MALIBU3D_IGNORE_INDEX
        ):
            raise ValueError(
                f"segment ids in {data_path} are outside "
                f"[0, {MALIBU3D_IGNORE_INDEX}] "
                f"(min={int(segment.min())}, max={int(segment.max())}). "
                f"Void is {MALIBU3D_IGNORE_INDEX} and is not remapped."
            )
        return data_dict

    def get_split_name(self, idx):
        # .../{split}/{dept_year}_LIDARHD/{roi}/{tile_id}
        tile_dir = self.data_list[idx % len(self.data_list)]
        split_dir = os.path.dirname(os.path.dirname(os.path.dirname(tile_dir)))
        return os.path.basename(split_dir)


def _iter_tiles(csv_path, splits):
    csv_path = str(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Tile CSV not found: {csv_path}")
    splits_set = {split.strip().lower() for split in splits}
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV or no header row: {csv_path}")
        columns = set(reader.fieldnames)
        missing = [column for column in _REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"Tile CSV {csv_path} is missing columns {missing}")
        has_n_points = "n_points" in columns
        for row in reader:
            split = (row.get("split") or "").strip().lower()
            dept_year = (row.get("dept_year") or "").strip()
            roi = (row.get("roi") or "").strip()
            tile_id = (row.get("tile_id") or "").strip()
            if not split or not dept_year or not roi or not tile_id:
                continue
            if split not in splits_set:
                continue
            n_points = (
                _parse_optional_int(row.get("n_points")) if has_n_points else None
            )
            yield {
                "split": split,
                "dept_year": dept_year,
                "roi": roi,
                "tile_id": tile_id,
                "n_points": n_points,
            }


def _load_subset_ids(csv_path, splits):
    csv_path = str(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Subset CSV not found: {csv_path}")
    splits_set = {split.strip().lower() for split in splits}
    tile_ids = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "tile_id" not in reader.fieldnames:
            raise ValueError(
                f"Subset CSV {csv_path} must have a 'tile_id' column "
                f"(got {reader.fieldnames})."
            )
        has_split = "split" in reader.fieldnames
        for row in reader:
            tile_id = (row.get("tile_id") or "").strip()
            if not tile_id:
                continue
            if has_split:
                row_split = (row.get("split") or "").strip().lower()
                if row_split not in splits_set:
                    continue
            tile_ids.add(tile_id)
    return tile_ids
