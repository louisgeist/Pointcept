"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections import Counter
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from torch_scatter import scatter_min
from pointcept.models.utils import offset2batch
from pointcept.utils.logger import get_root_logger


def _fmt_count(n):
    """Compact integer: 1234 -> 1.2k, 2_000_000 -> 2.0M."""
    n = int(n)
    abs_n = abs(n)
    if abs_n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs_n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _fmt_hist(values):
    """Compact ascending histogram: '2x3, 3x7, 16x33'."""
    counts = Counter(int(v) for v in values)
    return ", ".join(f"{k}x{counts[k]}" for k in sorted(counts))


def _summarize_pack_batches(batches, sizes, voxel_budget):
    """Return human-readable packing stats for logging."""
    if not batches:
        return "n_batches=0 (empty)"
    samples = [len(b) for b in batches]
    voxels = [sum(sizes[i] for i in b) for b in batches]
    util = [100.0 * v / voxel_budget for v in voxels]
    return (
        f"n_batches={len(batches)} | "
        f"scenes/batch min/mean/max={min(samples)}/"
        f"{sum(samples) / len(samples):.1f}/{max(samples)} "
        f"[{_fmt_hist(samples)}] | "
        f"voxels/batch min/mean/max="
        f"{_fmt_count(min(voxels))}/"
        f"{_fmt_count(sum(voxels) / len(voxels))}/"
        f"{_fmt_count(max(voxels))} | "
        f"budget util mean/p50="
        f"{sum(util) / len(util):.0f}%/"
        f"{float(np.median(util)):.0f}%"
    )


def load_voxel_size_csv(path):
    """Load tile_id -> n_voxels from tiles.csv (``tile_id`` and ``n_voxels`` columns).

    Rows with a missing or non-positive n_voxels are skipped.
    """
    import csv

    sizes = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty voxel size CSV: {path}")
        required = {"tile_id", "n_voxels"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise KeyError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            tile_id = (row.get("tile_id") or "").strip()
            if not tile_id:
                continue
            n_voxels = row.get("n_voxels")
            if n_voxels in (None, ""):
                continue
            size = int(float(n_voxels))
            if size <= 0:
                continue
            sizes[tile_id] = size
    return sizes


def resolve_dataset_voxel_sizes(dataset):
    """Return ``(sizes, 0)``: the n_voxels of every scene of ``dataset.data_list``.

    Sizes come from the ``n_voxels`` column of ``dataset.csv_manifest``; a missing
    column or value raises ValueError.
    """
    import os

    manifest_path = getattr(dataset, "csv_manifest", None)
    if not manifest_path:
        raise ValueError(
            "Voxel-budget packing requires dataset.csv_manifest with an n_voxels "
            "column: set data.*.csv_manifest to the MALiBU3D tiles.csv."
        )
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Voxel-budget packing: csv_manifest not found: {manifest_path}"
        )

    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        import csv

        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
    if "n_voxels" not in fields:
        raise ValueError(
            f"Voxel-budget packing requires column 'n_voxels' in csv_manifest, "
            f"but it was not found in:\n  {manifest_path}\n"
            "Use the MALiBU3D tiles.csv, which has n_voxels for every tile."
        )

    csv_sizes = load_voxel_size_csv(manifest_path)
    data_list = getattr(dataset, "data_list", None)
    if data_list is None:
        raise AttributeError("Dataset has no data_list; cannot resolve voxel sizes")

    sizes = []
    missing_ids = []
    for path in data_list:
        patch_id = os.path.basename(path.rstrip("/"))
        if patch_id in csv_sizes:
            sizes.append(int(csv_sizes[patch_id]))
            continue
        missing_ids.append(patch_id)

    if missing_ids:
        examples = ", ".join(missing_ids[:5])
        more = f" (+{len(missing_ids) - 5} more)" if len(missing_ids) > 5 else ""
        raise ValueError(
            f"Voxel-budget packing: {len(missing_ids)}/{len(data_list)} scenes have "
            f"no positive n_voxels in csv_manifest:\n  {manifest_path}\n"
            f"Examples: {examples}{more}"
        )
    return sizes, 0


def pack_indices_by_voxel_budget(sizes, voxel_budget, max_batch_size):
    """First Fit Decreasing pack by size. Oversized scenes become singleton batches.

    See: https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing
    """
    if voxel_budget <= 0:
        raise ValueError(f"voxel_budget must be > 0, got {voxel_budget}")
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be > 0, got {max_batch_size}")

    # Descending size; ascending index tie-break for determinism.
    indexed = sorted(enumerate(sizes), key=lambda item: (-int(item[1]), item[0]))
    batches = []
    batch_sums = []
    for index, size in indexed:
        size = int(size)
        if size > voxel_budget:
            batches.append([index])
            batch_sums.append(size)
            continue
        placed = False
        for b, (batch, batch_sum) in enumerate(zip(batches, batch_sums)):
            if len(batch) >= max_batch_size:
                continue
            if batch_sum + size > voxel_budget:
                continue
            batch.append(index)
            batch_sums[b] = batch_sum + size
            placed = True
            break
        if not placed:
            batches.append([index])
            batch_sums.append(size)
    return batches


class VoxelBudgetBatchSampler(torch.utils.data.Sampler):
    """Batch sampler packing scenes under a voxel (or point) budget per batch.

    Used for val/test when ``val_voxel_budget`` / ``test_voxel_budget`` is set, so
    batches of very unequal tiles fit in memory. Scenes are packed with First Fit
    Decreasing (``pack_indices_by_voxel_budget``), at most ``max_batch_size`` per
    batch; the batch list is fixed and sharded across ranks by stride ``world_size``.
    """

    def __init__(
        self,
        sizes,
        voxel_budget,
        max_batch_size=8,
        rank=0,
        world_size=1,
    ):
        self.sizes = [int(s) for s in sizes]
        self.voxel_budget = int(voxel_budget)
        self.max_batch_size = int(max_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size <= 0:
            raise ValueError(f"world_size must be > 0, got {self.world_size}")
        if not (0 <= self.rank < self.world_size):
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}")

        all_batches = pack_indices_by_voxel_budget(
            self.sizes, self.voxel_budget, self.max_batch_size
        )
        self.batches = all_batches[self.rank :: self.world_size]
        # Root logger is silent on non-zero ranks → summarize all shards from rank 0.
        if self.rank == 0:
            logger = get_root_logger()
            logger.info(
                "[VoxelBudgetBatchSampler] budget=%s max_bs=%d world_size=%d",
                _fmt_count(self.voxel_budget),
                self.max_batch_size,
                self.world_size,
            )
            for r in range(self.world_size):
                batches_r = all_batches[r :: self.world_size]
                logger.info(
                    "[VoxelBudgetBatchSampler] rank %d/%d | %s",
                    r,
                    self.world_size,
                    _summarize_pack_batches(batches_r, self.sizes, self.voxel_budget),
                )

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def build_voxel_budget_batch_sampler(
    dataset,
    voxel_budget,
    max_batch_size,
    rank=0,
    world_size=1,
):
    """Resolve per-sample n_voxels from dataset.csv_manifest and build a sampler.

    Requires n_voxels in csv_manifest for every sample.

    Returns:
        sampler: VoxelBudgetBatchSampler for this rank
        sizes: list of per-index n_voxels used for packing
        missing_csv: always 0 (missing sizes raise ValueError instead)
    """
    sizes, missing_csv = resolve_dataset_voxel_sizes(dataset)
    sampler = VoxelBudgetBatchSampler(
        sizes=sizes,
        voxel_budget=int(voxel_budget),
        max_batch_size=int(max_batch_size),
        rank=int(rank),
        world_size=int(world_size),
    )
    return sampler, sizes, missing_csv


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], list):
        batch = [torch.tensor(data) for data in batch]
        return torch.cat(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        out = {}
        for key in batch[0]:
            if "offset" in key:
                out[key] = torch.cumsum(
                    collate_fn([d[key].diff(prepend=torch.tensor([0])) for d in batch]),
                    dim=0,
                )
                continue
            out[key] = collate_fn([d[key] for d in batch])
        return out
    else:
        return default_collate(batch)


def _merge_scene_level_labels(labels):
    """Merge per-scene labels when point_collate_fn mixes scenes in pairs (Mix3D).

    Mixing keeps offset boundaries 1, 3, ... and -1, so each merged scene takes the
    label of the first scene of its pair (the last scene stays alone if B is odd).
    Supports labels of shape (B,) and (B, 1).
    """
    if labels.ndim > 1:
        n = int(labels.shape[0])
        if n <= 1:
            return labels
        if n % 2 == 0:
            return labels[0::2]
        return torch.cat([labels[0:-1:2], labels[-1:]], dim=0)

    labels = labels.reshape(-1)
    n = int(labels.shape[0])
    if n <= 1:
        return labels
    if n % 2 == 0:
        return labels[0::2]
    return torch.cat([labels[0:-1:2], labels[-1:]])


# Scene-level targets (one label per point cloud), e.g. the PureForest tile class.
_SCENE_SCALAR_LABEL_KEYS = ("category",)


def point_collate_fn(batch, mix_prob=0):
    """Collate input dicts; with probability mix_prob, merge scenes in pairs (Mix3D)."""
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if random.random() < mix_prob:
        if any(key.endswith("_offset") for key in batch.keys()):
            raise NotImplementedError(
                "Mix3D (mix_prob > 0) is not supported for multi-view batches "
                "(Sonata global_/local_ views); set mix_prob = 0."
            )
        batch["offset"] = torch.cat(
            [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
        )

        for scene_label_key in _SCENE_SCALAR_LABEL_KEYS:
            if scene_label_key in batch:
                batch[scene_label_key] = _merge_scene_level_labels(
                    batch[scene_label_key]
                )

        # Recompute grid_coord after mixing, because each scene's grid_coord was
        # independently shifted before mixing and is no longer consistent with
        # the merged coord. Only done when grid_size is available (e.g. LitePT
        # configs); other configs are unaffected.
        if "grid_coord" in batch and "grid_size" in batch:
            batch_idx = offset2batch(batch["offset"])
            scaled_coord = batch["coord"] / batch["grid_size"][0]
            grid_coord = torch.floor(scaled_coord).to(torch.int64)
            min_coord, _ = scatter_min(grid_coord, batch_idx, dim=0)
            batch["grid_coord"] = grid_coord - min_coord[batch_idx]
    return batch
