"""
Lovasz Loss
refer https://arxiv.org/abs/1705.08790

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from typing import Optional
from itertools import filterfalse
import torch
from torch.nn.modules.loss import _Loss

from .builder import LOSSES

MULTICLASS_MODE: str = "multiclass"


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_softmax_flat(probas, labels):
    """Multi-class Lovasz-Softmax loss, averaged over the classes present in labels.
    Args:
        @param probas: [P, C] Class probabilities at each prediction (between 0 and 1)
        @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
    """
    if probas.numel() == 0:
        # only void pixels: return a 0-dim zero that stays in the graph
        # (zero gradient), so it can be summed with the other losses
        return probas.sum()
    if probas.size(1) == 1:
        raise ValueError("Sigmoid output possible only with 1 class")
    losses = []
    for c in labels.unique():
        fg = (labels == c).type_as(probas)  # foreground for class c
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return mean(losses)


def _flatten_probas(probas, labels, ignore=None):
    """Flattens predictions in the batch"""
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)

    C = probas.size(1)
    probas = torch.movedim(probas, 1, -1)  # [B, C, Di, Dj, ...] -> [B, Di, Dj, ..., C]
    probas = probas.contiguous().view(-1, C)  # [P, C]

    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Nan-mean compatible with generators."""
    values = iter(values)
    if ignore_nan:
        values = filterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


@LOSSES.register_module()
class LovaszLoss(_Loss):
    def __init__(
        self,
        mode: str,
        ignore_index: Optional[int] = None,
        loss_weight: float = 1.0,
    ):
        """Multi-class Lovasz-Softmax loss for segmentation, computed over the whole batch.
        Args:
            mode: 'multiclass' (the only mode supported)
            ignore_index: Label that indicates ignored points (does not contribute to loss)
        Shape
             - **y_pred** - torch.Tensor of shape (N, C)
             - **y_true** - torch.Tensor of shape (N,)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode == MULTICLASS_MODE, f"Unsupported Lovasz mode {mode!r}."
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self, y_pred, y_true):
        y_pred = y_pred.softmax(dim=1)
        loss = _lovasz_softmax_flat(*_flatten_probas(y_pred, y_true, self.ignore_index))
        return loss * self.loss_weight
