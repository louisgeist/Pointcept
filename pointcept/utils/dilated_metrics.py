"""Buffer ("relaxed") precision/recall/F1 for thin binary pixel masks.

Standard evaluation for curvilinear structures (road/rail extraction): a predicted
foreground pixel counts as correct if it lies within ``radius_px`` of any GT foreground
pixel (precision), and a GT foreground pixel counts as found if it lies within
``radius_px`` of any predicted foreground pixel (recall). Plain IoU on two masks
dilated by the same radius is not used here — it saturates towards 1 as the radius
grows and conflates false alarms with missed segments into a single number, whereas
buffer precision/recall keeps the two error modes separate.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def dilated_precision_recall_counts(
    pred_fg: np.ndarray,
    gt_fg: np.ndarray,
    valid: np.ndarray,
    radius_px: int,
    connectivity: int = 8,
) -> Tuple[float, float, float, float]:
    """Buffer precision/recall counts for one scene/channel.

    ``pred_fg`` / ``gt_fg`` / ``valid`` are same-shape boolean arrays (``valid`` marks
    pixels eligible to be scored at all, e.g. observed and not void/ignore_index).
    ``pred_fg`` and ``gt_fg`` must already be masked by ``valid`` (False elsewhere).

    Returns ``(precision_num, precision_denom, recall_num, recall_denom)``.
    """
    # Deferred import: pointcept.datasets.__init__ pulls in models/engines, which
    # import this module, so importing morph_dilate_mask at module load time would
    # create a circular import.
    from pointcept.datasets.preprocessing.flair3d_plus.network_xy_raster_utils import (
        morph_dilate_mask,
    )

    pred_fg = np.asarray(pred_fg, dtype=bool)
    gt_fg = np.asarray(gt_fg, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if pred_fg.shape != gt_fg.shape or pred_fg.shape != valid.shape:
        raise ValueError(
            "pred_fg/gt_fg/valid must share the same shape, got "
            f"{pred_fg.shape}, {gt_fg.shape}, {valid.shape}"
        )
    radius_px = int(radius_px)
    if radius_px < 0:
        raise ValueError(f"radius_px must be >= 0, got {radius_px}")

    gt_dilated = morph_dilate_mask(gt_fg, iterations=radius_px, connectivity=connectivity)
    pred_dilated = morph_dilate_mask(
        pred_fg, iterations=radius_px, connectivity=connectivity
    )

    precision_num = float(np.count_nonzero(pred_fg & gt_dilated & valid))
    precision_denom = float(np.count_nonzero(pred_fg & valid))
    recall_num = float(np.count_nonzero(gt_fg & pred_dilated & valid))
    recall_denom = float(np.count_nonzero(gt_fg & valid))
    return precision_num, precision_denom, recall_num, recall_denom


def precision_recall_f1(
    num_p: float,
    denom_p: float,
    num_r: float,
    denom_r: float,
    eps: float = 1e-10,
) -> Tuple[float, float, float]:
    """Precision/recall/F1 from separate numerator/denominator pairs.

    Precision and recall are allowed different numerators (buffer precision/recall
    count hits against different reference masks), so this takes 4 scalars rather
    than a single TP/FP/FN triple.
    """
    precision = num_p / (denom_p + eps)
    recall = num_r / (denom_r + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return float(precision), float(recall), float(f1)
