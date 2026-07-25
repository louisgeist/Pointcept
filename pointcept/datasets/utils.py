"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

from torch_scatter import scatter_min
from pointcept.models.utils import offset2batch


def load_voxel_size_csv(path):
    """Load patch_id -> n_voxels from an enriched scene_split_manifest CSV.

    Rows with missing/non-positive n_voxels are skipped.
    """
    import csv

    sizes = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty voxel size CSV: {path}")
        required = {"patch_id", "n_voxels"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise KeyError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            error = (row.get("error") or "").strip()
            if error:
                continue
            patch_id = (row.get("patch_id") or "").strip()
            if not patch_id:
                continue
            n_voxels = row.get("n_voxels")
            if n_voxels in (None, ""):
                continue
            size = int(float(n_voxels))
            if size <= 0:
                continue
            sizes[patch_id] = size
    return sizes


def resolve_dataset_voxel_sizes(dataset):
    """Return per-index n_voxels for packing from dataset.csv_manifest.

    Requires csv_manifest column n_voxels for every sample.
    Raises ValueError with a clear message if the column or values are missing.
    """
    import os

    manifest_path = getattr(dataset, "csv_manifest", None)
    if not manifest_path:
        raise ValueError(
            "Voxel-budget packing requires dataset.csv_manifest with an n_voxels "
            "column. Set data.*.csv_manifest, then enrich it with:\n"
            "  python scripts/analyze_flair3d_test_point_voxel_counts.py "
            "--write_manifest <manifest.csv> ..."
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
            "Enrich the manifest after preprocess, e.g.:\n"
            "  python scripts/analyze_flair3d_test_point_voxel_counts.py \\\n"
            f"    --csv_manifest {manifest_path} \\\n"
            f"    --write_manifest {manifest_path} \\\n"
            "    --splits val,test --grid_size 0.1"
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
            f"Examples: {examples}{more}\n"
            "Re-run analyze_flair3d_test_point_voxel_counts.py --write_manifest "
            "for the missing splits/scenes."
        )
    return sizes, 0


def pack_indices_by_voxel_budget(sizes, voxel_budget, max_batch_size):
    """Greedy pack after ascending sort by size. Oversized scenes become singleton batches."""
    if voxel_budget <= 0:
        raise ValueError(f"voxel_budget must be > 0, got {voxel_budget}")
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be > 0, got {max_batch_size}")

    indexed = sorted(enumerate(sizes), key=lambda item: (item[1], item[0]))
    batches = []
    current = []
    current_sum = 0
    for index, size in indexed:
        size = int(size)
        if size > voxel_budget:
            if current:
                batches.append(current)
                current = []
                current_sum = 0
            batches.append([index])
            continue
        if current and (
            current_sum + size > voxel_budget or len(current) >= max_batch_size
        ):
            batches.append(current)
            current = []
            current_sum = 0
        current.append(index)
        current_sum += size
    if current:
        batches.append(current)
    return batches


class VoxelBudgetBatchSampler(torch.utils.data.Sampler):
    """BatchSampler packing dataset indices by a voxel (or point) budget.

    Scenes are sorted ascending by size then packed greedily. Batches are
    deterministically sharded across distributed ranks (stride by world_size).
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
        if "img_num" in batch[0].keys():
            max_img_num = max([d["img_num"] for d in batch])
        batch = {
            key: (
                (
                    collate_fn([d[key] for d in batch])
                    if "offset" not in key
                    # offset -> bincount -> concat bincount-> concat offset
                    else torch.cumsum(
                        collate_fn(
                            [d[key].diff(prepend=torch.tensor([0])) for d in batch]
                        ),
                        dim=0,
                    )
                )
                if "correspondence" not in key
                else collate_fn(
                    [
                        F.pad(
                            d[key].permute(0, 2, 1),
                            (0, max_img_num - d[key].shape[1]),
                            value=-1,
                        ).permute(0, 2, 1)
                        for d in batch
                    ]
                )
            )
            for key in batch[0]
        }
        return batch
    else:
        return default_collate(batch)


def pairwise_concatenate(tensors_list, dim=0):
    new_list = []
    n = len(tensors_list)
    for i in range(0, n - 1, 2):
        tensor1 = tensors_list[i]
        tensor2 = tensors_list[i + 1]
        concatenated_tensor = torch.cat([tensor1, tensor2], dim=dim)
        new_list.append(concatenated_tensor)
    if n % 2 != 0:
        new_list.append(tensors_list[-1])
    return new_list


def regroup_batch(batch, N, original_offsets, data_keys):
    num_segments = len(original_offsets)
    grouped_segments = {key: [[] for _ in range(N)] for key in data_keys}
    start_idx = 0
    for i in range(num_segments):
        end_idx = original_offsets[i]
        group_idx = i % N
        for key in data_keys:
            segment = batch[key][start_idx:end_idx]
            grouped_segments[key][group_idx].append(segment)
        start_idx = end_idx

    segment_lengths = original_offsets[1:] - original_offsets[:-1]
    segment_lengths = torch.cat([original_offsets[:1], segment_lengths])
    new_lengths_order = []
    split_lengths_order = []
    for i in range(N):
        segment_lengths_i = segment_lengths[i::N]
        n = segment_lengths_i.shape[0]
        num_pairs = n // 2
        paired_part = segment_lengths_i[: num_pairs * 2]
        reshaped_part = paired_part.view(num_pairs, 2, *segment_lengths_i.shape[1:])
        sums = torch.sum(reshaped_part, dim=1)
        first_parts = reshaped_part[:, 0]
        if n % 2 != 0:
            last_element = segment_lengths_i[-1]
            last_element = last_element.unsqueeze(0)
            final_result = torch.cat([sums, last_element], dim=0)
        else:
            final_result = sums
        new_lengths_order.append(final_result)
        split_lengths_order.append(first_parts)
    new_lengths_order = torch.stack(new_lengths_order, dim=1)
    new_lengths_order = new_lengths_order.flatten()
    new_offsets = torch.cumsum(new_lengths_order, dim=0)
    split_lengths_order = torch.stack(split_lengths_order, dim=1).flatten()

    new_batch = {}
    new_batch_imgs = []
    new_batch_img_num = 0
    img_num_offset = torch.cat(
        [torch.tensor([0]), torch.cumsum(batch["img_num"], dim=0)]
    )
    for key in data_keys:
        final_segments_in_order = []
        for i in range(N):
            grouped_segments[key][i] = pairwise_concatenate(
                grouped_segments[key][i], dim=0
            )
        for i in range(len(grouped_segments[key][0])):
            for j in range(N):
                final_segments_in_order.append(grouped_segments[key][j][i])
        new_batch[key] = torch.vstack(final_segments_in_order)
        if "correspondence" in key:
            current_start = 0
            N0, v, n_dim = new_batch[key].shape
            v2 = v * 2
            batch_correspondence_mix = -torch.ones(
                (N0, v2, n_dim),
                dtype=new_batch[key].dtype,
                device=new_batch[key].device,
            )
            for k, end in enumerate(new_offsets):
                len_part1 = split_lengths_order[k]
                split_point = current_start + len_part1
                if split_point > current_start:
                    mask1 = torch.any(
                        new_batch[key][current_start:split_point]
                        != torch.tensor([-1, -1]),
                        dim=2,
                    )
                    valid_index1 = torch.where(mask1)
                    if len(valid_index1[1]) == 0:
                        count1 = 0
                    else:
                        count1 = max(valid_index1[1])
                    batch_correspondence_mix[current_start:split_point, 0:count1] = (
                        new_batch[key][current_start:split_point, 0:count1]
                    )
                    if k % N == 0 and N == 2:
                        new_batch_imgs.append(
                            batch["images"][
                                img_num_offset[k // N * 2] : img_num_offset[k // N * 2]
                                + count1
                            ]
                        )
                if end > split_point:
                    mask2 = torch.any(
                        new_batch[key][split_point:end] != torch.tensor([-1, -1]), dim=2
                    )
                    valid_index2 = torch.where(mask2)
                    if len(valid_index2[1]) == 0:
                        count2 = 0
                    else:
                        count2 = max(valid_index2[1])
                    batch_correspondence_mix[
                        split_point:end, count1 : count1 + count2
                    ] = new_batch[key][split_point:end, 0:count2]
                    if k % N == 0 and N == 2:
                        new_batch_imgs.append(
                            batch["images"][
                                img_num_offset[k // N * 2 + 1] : img_num_offset[
                                    k // N * 2 + 1
                                ]
                                + count2
                            ]
                        )
                current_start = end

            if N == 2:
                new_batch_img_num = torch.tensor([i.shape[0] for i in new_batch_imgs])
                new_batch_imgs = torch.vstack(new_batch_imgs)
            else:
                new_batch_img_num = None
                new_batch_imgs = None
            new_batch[key] = batch_correspondence_mix
    return new_batch, new_offsets, new_batch_imgs, new_batch_img_num


def _merge_scene_level_labels(labels):
    """Merge scalar per-scene labels when point_collate_fn pair-mixes offset.

    Pair-mix keeps boundaries at offset[1], offset[3], ... and offset[-1], so each
    merged group uses the first scene label of the pair (last scene alone if odd).
    Supports scalar scene labels (B,) and (B, 1).
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


def _merge_scene_level_multihot_labels(labels):
    """OR-aggregate multi-hot scene labels when pair-mixing offset."""
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
    n = int(labels.shape[0])
    if n <= 1:
        return labels

    if n % 2 == 0:
        paired = labels.view(n // 2, 2, -1)
        return torch.maximum(paired[:, 0], paired[:, 1])

    paired = labels[:-1].view((n - 1) // 2, 2, -1)
    merged = torch.maximum(paired[:, 0], paired[:, 1])
    return torch.cat([merged, labels[-1:]], dim=0)


# Scene-level targets (one label per point cloud before mix).
_SCENE_SCALAR_LABEL_KEYS = ("climatic_domain", "category")
_SCENE_MULTIHOT_LABEL_KEYS = ("natural_habitat_multilabel",)
_SCENE_LABEL_KEYS = _SCENE_SCALAR_LABEL_KEYS + _SCENE_MULTIHOT_LABEL_KEYS


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if random.random() < mix_prob:
        valid_keys = [
            "coord",
            "grid_coord",
            "origin_coord",
            "color",
            "normal",
            "feat",
            "correspondence",
        ]
        if "instance" in batch.keys():
            offset = batch["offset"]
            start = 0
            num_instance = 0
            for i in range(len(offset)):
                if i % 2 == 0:
                    num_instance = max(batch["instance"][start : offset[i]])
                if i % 2 != 0:
                    mask = batch["instance"][start : offset[i]] != -1
                    batch["instance"][start : offset[i]] += num_instance * mask
                start = offset[i]
        offset_assets = [asset for asset in batch.keys() if "offset" in asset]
        for offset_asset in offset_assets:
            batch[offset_asset] = torch.cat(
                [batch[offset_asset][1:-1:2], batch[offset_asset][-1].unsqueeze(0)],
                dim=0,
            )

        for scene_label_key in _SCENE_SCALAR_LABEL_KEYS:
            if scene_label_key in batch:
                batch[scene_label_key] = _merge_scene_level_labels(
                    batch[scene_label_key]
                )

        for scene_label_key in _SCENE_MULTIHOT_LABEL_KEYS:
            if scene_label_key in batch:
                batch[scene_label_key] = _merge_scene_level_multihot_labels(
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
        offset_assets = [asset for asset in batch.keys() if "_offset" in asset]
        for offset_asset in offset_assets:
            offset_prefix = offset_asset.split("_")[0]
            valid_keys_with_prefix = [
                offset_prefix + "_" + valid_key for valid_key in valid_keys
            ]
            valid_keys_with_prefix = [
                valid_key_with_prefix
                for valid_key_with_prefix in valid_keys_with_prefix
                if valid_key_with_prefix in batch.keys()
            ]
            if "global" in offset_asset:
                N = 2
            elif "local" in offset_asset:
                N = 4
            updated_batch, new_offset, imgs, img_num = regroup_batch(
                batch, N, batch[offset_asset], valid_keys_with_prefix
            )
            batch[offset_asset] = new_offset
            batch.update(updated_batch)
            if "global" in offset_asset:
                batch["images"] = imgs
                batch["img_num"] = img_num

        if "img_num" in batch.keys():
            n = batch["img_num"].shape[0]
            num_pairs = n // 2
            len_pairs = num_pairs * 2
            pairs_tensor = batch["img_num"][:len_pairs]

            if num_pairs == 0:
                pass
            else:
                summed_pairs = pairs_tensor.view(-1, 2).sum(dim=1)
                if n % 2 != 0:
                    last_element = batch["img_num"][-1:]
                    result = torch.cat((summed_pairs, last_element))
                else:
                    result = summed_pairs
                batch["img_num"] = result
    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))
