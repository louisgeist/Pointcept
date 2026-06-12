from functools import partial
import weakref
import torch
import math
import torch.utils.data
import torch.distributed as dist
from torch.utils.data import Sampler

import pointcept.utils.comm as comm
from pointcept.datasets.utils import point_collate_fn
from pointcept.datasets import ConcatDataset
from pointcept.utils.env import set_seed


class MultiDatasetDummySampler:
    def __init__(self):
        self.dataloader = None

    def set_epoch(self, epoch):
        if comm.get_world_size() > 1:
            for dataloader in self.dataloader.dataloaders:
                dataloader.sampler.set_epoch(epoch)
        return


class MultiDatasetDataloader:
    """
    Multiple Datasets Dataloader, batch data from a same dataset and mix up ratio determined by loop of each sub dataset.
    The overall length is determined by the main dataset (first) and loop of concat dataset.
    """

    def __init__(
        self,
        concat_dataset: ConcatDataset,
        batch_size_per_gpu: int,
        num_worker_per_gpu: int,
        mix_prob=0,
        seed=None,
    ):
        self.datasets = concat_dataset.datasets
        self.ratios = [dataset.loop for dataset in self.datasets]
        # reset data loop, original loop serve as ratios
        for dataset in self.datasets:
            dataset.loop = 1
        # determine union training epoch by main dataset
        self.datasets[0].loop = concat_dataset.loop
        # build sub-dataloaders
        num_workers = num_worker_per_gpu // len(self.datasets)
        self.dataloaders = []
        for dataset_id, dataset in enumerate(self.datasets):
            if comm.get_world_size() > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                sampler = None

            init_fn = (
                partial(
                    self._worker_init_fn,
                    dataset_id=dataset_id,
                    num_workers=num_workers,
                    num_datasets=len(self.datasets),
                    rank=comm.get_rank(),
                    seed=seed,
                )
                if seed is not None
                else None
            )
            self.dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size_per_gpu,
                    shuffle=(sampler is None),
                    num_workers=num_worker_per_gpu,
                    sampler=sampler,
                    collate_fn=partial(point_collate_fn, mix_prob=mix_prob),
                    pin_memory=True,
                    worker_init_fn=init_fn,
                    drop_last=True,
                    persistent_workers=True,
                )
            )
        self.sampler = MultiDatasetDummySampler()
        self.sampler.dataloader = weakref.proxy(self)

    def __iter__(self):
        iterator = [iter(dataloader) for dataloader in self.dataloaders]
        while True:
            for i in range(len(self.ratios)):
                for _ in range(self.ratios[i]):
                    try:
                        batch = next(iterator[i])
                    except StopIteration:
                        if i == 0:
                            return
                        else:
                            iterator[i] = iter(self.dataloaders[i])
                            batch = next(iterator[i])
                    yield batch

    def __len__(self):
        main_data_loader_length = len(self.dataloaders[0])
        return (
            main_data_loader_length // self.ratios[0] * sum(self.ratios)
            + main_data_loader_length % self.ratios[0]
        )

    @staticmethod
    def _worker_init_fn(worker_id, num_workers, dataset_id, num_datasets, rank, seed):
        worker_seed = (
            num_workers * num_datasets * rank
            + num_workers * dataset_id
            + worker_id
            + seed
        )
        set_seed(worker_seed)


def _sample_iter_limited_indices(
    num_requested, dataset_size, shuffle, needs_replacement, generator
):
    """Build the flat index list consumed by one iter-limited epoch.

    Three mutually exclusive strategies:
    - ``shuffle=False``: deterministic cycling via ``arange % dataset_size``
      (indices may repeat, but this is ordered reuse, not random sampling).
    - ``shuffle=True`` and ``needs_replacement``: ``randint`` (random with
      replacement).
    - ``shuffle=True`` and not ``needs_replacement``: ``randperm`` truncated
      (random without replacement).
    """
    if not shuffle:
        return torch.arange(num_requested) % dataset_size
    if needs_replacement:
        return torch.randint(
            0, dataset_size, (num_requested,), generator=generator
        )
    return torch.randperm(dataset_size, generator=generator)[:num_requested]


def _compute_needs_replacement(shuffle, num_requested, dataset_size):
    """
    Compute whether replacement is needed for the given parameters.
    
    - If shuffle is False, no replacement is needed.
    - If shuffle is True, replacement is needed if the number of 
        requested indices is greater than the dataset size.
    """
    if not shuffle:
        return False
    return num_requested > dataset_size


class IterLimitedSampler(Sampler):
    """Random subsample per epoch with a fixed number of optimizer steps.

    Each epoch yields exactly ``iter_per_epoch * batch_size`` indices.

    When ``shuffle=True`` (the default used by the trainer):
    - if ``num_samples <= dataset_size``: random **without** replacement
      (``randperm``);
    - if ``num_samples > dataset_size``: random **with** replacement
      (``randint``).

    When ``shuffle=False``, indices follow a fixed cyclic order
    (``0, 1, …, dataset_size-1, 0, …``). Repeats can occur when
    ``num_samples > dataset_size``, but that is deterministic cycling,
    not probabilistic sampling.

    ``needs_replacement`` is set at init: ``True`` when
    ``num_samples > dataset_size`` and ``shuffle=True``.
    """

    def __init__(
        self,
        dataset,
        iter_per_epoch,
        batch_size,
        shuffle=True,
        seed=0,
    ):
        self.dataset_size = len(dataset)
        self.iter_per_epoch = iter_per_epoch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_samples = iter_per_epoch * batch_size
        self.needs_replacement = _compute_needs_replacement(
            self.shuffle, self.num_samples, self.dataset_size
        )

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = _sample_iter_limited_indices(
            self.num_samples,
            self.dataset_size,
            self.shuffle,
            self.needs_replacement,
            g,
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class IterLimitedDistributedSampler(Sampler):
    """DDP variant of :class:`IterLimitedSampler`.

    All ranks build the same global index list (``total_size`` indices) from
    a shared ``seed + epoch``, then each rank takes its slice. The replacement
    decision uses this **global** budget (``total_size = num_samples *
    num_replicas``) so that ranks receive disjoint indices when sampling
    without replacement.

    ``needs_replacement`` uses ``total_size`` instead of per-rank
    ``num_samples``; see :class:`IterLimitedSampler` for other semantics.
    """

    def __init__(
        self,
        dataset,
        iter_per_epoch,
        batch_size_per_gpu,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
    ):
        if num_replicas is None:
            if not dist.is_available():
                num_replicas = 1
                rank = 0
            else:
                try:
                    num_replicas = dist.get_world_size()
                    rank = dist.get_rank()
                except Exception:
                    num_replicas = 1
                    rank = 0

        self.dataset_size = len(dataset)
        self.iter_per_epoch = iter_per_epoch
        self.batch_size_per_gpu = batch_size_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_samples = iter_per_epoch * batch_size_per_gpu
        self.total_size = self.num_samples * self.num_replicas
        self.needs_replacement = _compute_needs_replacement(
            self.shuffle, self.total_size, self.dataset_size
        )

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = _sample_iter_limited_indices(
            self.total_size,
            self.dataset_size,
            self.shuffle,
            self.needs_replacement,
            g,
        )
        indices = indices[
            self.rank * self.num_samples : (self.rank + 1) * self.num_samples
        ]
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedImbalancedSampler(Sampler):
    def __init__(
        self,
        dataset,
        sampled_dataset_index,
        sampled_dataset_limit,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
    ):
        if num_replicas is None:
            if not dist.is_available():
                num_replicas = 1
                rank = 0
            else:
                try:
                    num_replicas = dist.get_world_size()
                    rank = dist.get_rank()
                except:
                    num_replicas = 1
                    rank = 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.sampled_dataset_index = sampled_dataset_index
        self.sampled_dataset_limit = sampled_dataset_limit
        self.shuffle = shuffle

        self.lengths = [len(d) for d in dataset.datasets]
        self.offsets = [0] + list(
            torch.cumsum(torch.tensor(self.lengths), 0)[:-1].numpy()
        )
        self.total_size_per_epoch = (
            sum(self.lengths)
            - self.lengths[self.sampled_dataset_index]
            + self.sampled_dataset_limit
        )
        self.num_samples = math.ceil(self.total_size_per_epoch / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []
        for i, length in enumerate(self.lengths):
            if i == self.sampled_dataset_index:
                sub_indices = (
                    torch.randperm(length, generator=g)[: self.sampled_dataset_limit]
                    + self.offsets[i]
                )
                indices.append(sub_indices)
            else:
                sub_indices = torch.arange(length) + self.offsets[i]
                indices.append(sub_indices)

        final_indices = torch.cat(indices)

        if self.shuffle:
            final_indices = final_indices[
                torch.randperm(len(final_indices), generator=g)
            ]

        final_indices = final_indices.tolist()
        padding_size = self.total_size - len(final_indices)
        if padding_size <= len(final_indices):
            final_indices += final_indices[:padding_size]
        else:
            final_indices += (
                final_indices * math.ceil(padding_size / len(final_indices))
            )[:padding_size]

        assert len(final_indices) == self.total_size

        offset = self.num_samples * self.rank
        subsample_indices = final_indices[offset : offset + self.num_samples]

        return iter(subsample_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
