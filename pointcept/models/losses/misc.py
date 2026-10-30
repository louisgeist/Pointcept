"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.utils.misc import kl_divergence_rows
from .builder import LOSSES


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        # nn.CrossEntropyLoss returns NaN when every target is ignore_index;
        # return a zero that stays in the graph instead (zero gradient).
        if (target != self.ignore_index).any():
            return self.loss(pred, target) * self.loss_weight
        return pred.sum() * 0.0


@LOSSES.register_module()
class WeightedKLDivLoss(nn.Module):
    """Annotation-count-weighted KL(target || pred) over per-tile distributions.

    Loss of the ``tile_distribution`` tasks of MultiTaskSegmentorV2.
    ``pred``/``target``: (M, C) probabilities (not logits) for M tiles that have
    at least one non-void point for this axis; ``weight``: (M,) non-void point
    counts per tile. Returns sum(weight * KL(target||pred)) / sum(weight).
    """

    def __init__(self, loss_weight=1.0, eps=1e-8):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, pred, target, weight=None):
        if pred.numel() == 0:
            return pred.new_zeros(())
        kl = kl_divergence_rows(target, pred, eps=self.eps)
        if weight is None:
            weight = torch.ones_like(kl)
        return (weight * kl).sum() / weight.sum().clamp(min=self.eps) * self.loss_weight


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    """Smooth L1 on finite elements only (ignores NaN/Inf in pred or target).

    Used for elevation (height above ground) regression, where NaN targets mark
    points without a valid DTM value.
    """

    def __init__(self, beta=1.0, loss_weight=1.0):
        super().__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        mask = torch.isfinite(pred) & torch.isfinite(target)
        if mask.sum() == 0:
            return pred.sum() * 0.0
        diff = F.smooth_l1_loss(
            pred[mask], target[mask], beta=self.beta, reduction="sum"
        )
        return diff / mask.sum().clamp(min=1) * self.loss_weight
