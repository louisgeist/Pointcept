import torch
import torch.utils.data
import torch.distributed as dist
from torch.utils.data import Sampler



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


def _draw_cross_epoch_indices(num_requested, dataset_size, unseen_pool, generator):
    """Draw indices from a persistent unseen pool across trainer epochs.

    When the remaining pool is smaller than ``num_requested``, the leftover
    indices are discarded and a fresh full shuffle is created before drawing.
    """
    if unseen_pool is None or len(unseen_pool) < num_requested:
        unseen_pool = torch.randperm(dataset_size, generator=generator).tolist()
    selected = unseen_pool[:num_requested]
    unseen_pool = unseen_pool[num_requested:]
    return selected, unseen_pool


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

    When ``shuffle=True`` and ``num_samples <= dataset_size``, indices are drawn
    without replacement from a persistent unseen pool across epochs. The pool
    is fully reshuffled when the remaining unseen indices are fewer than
    ``num_samples``.

    When ``num_samples > dataset_size`` and ``shuffle=True``, random **with**
    replacement (``randint``) is used within the epoch.

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
        self._unseen_pool = None
        self.num_samples = iter_per_epoch * batch_size
        self.needs_replacement = _compute_needs_replacement(
            self.shuffle, self.num_samples, self.dataset_size
        )

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle and not self.needs_replacement:
            indices, self._unseen_pool = _draw_cross_epoch_indices(
                self.num_samples,
                self.dataset_size,
                self._unseen_pool,
                g,
            )
            return iter(indices)
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
        self._unseen_pool = None
        self.num_samples = iter_per_epoch * batch_size_per_gpu
        self.total_size = self.num_samples * self.num_replicas
        self.needs_replacement = _compute_needs_replacement(
            self.shuffle, self.total_size, self.dataset_size
        )

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle and not self.needs_replacement:
            indices, self._unseen_pool = _draw_cross_epoch_indices(
                self.total_size,
                self.dataset_size,
                self._unseen_pool,
                g,
            )
        else:
            indices = _sample_iter_limited_indices(
                self.total_size,
                self.dataset_size,
                self.shuffle,
                self.needs_replacement,
                g,
            ).tolist()
        indices = indices[
            self.rank * self.num_samples : (self.rank + 1) * self.num_samples
        ]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


