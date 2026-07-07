"""
Multi-label classification metrics (per-label TP/FP/FN, macro/micro F1, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MultilabelStats:
    tp: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    fp: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    fn: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    hamming_correct: int = 0
    hamming_total: int = 0
    subset_correct: int = 0
    subset_total: int = 0

    def ensure_num_labels(self, num_labels: int) -> None:
        if self.tp.shape[0] == num_labels:
            return
        self.tp = np.zeros(num_labels, dtype=np.int64)
        self.fp = np.zeros(num_labels, dtype=np.int64)
        self.fn = np.zeros(num_labels, dtype=np.int64)


def binarize_multilabel_predictions(
    pred: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Binarize probabilities or logits already passed through sigmoid."""
    arr = np.asarray(pred, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return (arr >= threshold).astype(np.int64)


def accumulate_multilabel_stats(
    pred: np.ndarray,
    target: np.ndarray,
    stats: MultilabelStats,
) -> None:
    """Update ``stats`` from binary pred/target arrays of shape (N, C)."""
    pred_bin = np.asarray(pred).astype(np.int64)
    target_bin = np.asarray(target).astype(np.int64)
    if pred_bin.ndim == 1:
        pred_bin = pred_bin.reshape(1, -1)
    if target_bin.ndim == 1:
        target_bin = target_bin.reshape(1, -1)
    if pred_bin.shape != target_bin.shape:
        raise ValueError(
            f"pred/target shape mismatch: {pred_bin.shape} vs {target_bin.shape}"
        )
    num_labels = pred_bin.shape[1]
    stats.ensure_num_labels(num_labels)

    pred_bool = pred_bin.astype(bool)
    target_bool = target_bin.astype(bool)
    stats.tp += np.count_nonzero(pred_bool & target_bool, axis=0)
    stats.fp += np.count_nonzero(pred_bool & ~target_bool, axis=0)
    stats.fn += np.count_nonzero(~pred_bool & target_bool, axis=0)
    stats.hamming_correct += int(np.count_nonzero(pred_bin == target_bin))
    stats.hamming_total += int(pred_bin.size)
    stats.subset_correct += int(np.all(pred_bin == target_bin, axis=1).sum())
    stats.subset_total += int(pred_bin.shape[0])


def compute_multilabel_metrics(
    stats: MultilabelStats,
    names: Sequence[str],
) -> Dict[str, object]:
    """Compute summary metrics from accumulated multi-label statistics.
    Expects binary predictions and targets that were accumulated via
    :func:`accumulate_multilabel_stats` (shape ``(N, C)`` per batch).
    Args:
        stats: Aggregated counts (TP/FP/FN per label, Hamming and subset
            tallies). Typically merged across batches or ranks before calling
            this function.
        names: Human-readable label names, length ``C``. Used as keys in
            ``per_label``.
    Returns:
        A dict with the following keys:
        - ``macro_f1`` (float): Mean of per-label F1 scores. Each label
          contributes equally regardless of frequency.
        - ``micro_f1`` (float): F1 computed from global TP/FP/FN sums across
          all labels and samples. Dominated by frequent labels.
        - ``subset_accuracy`` (float): Fraction of samples whose entire label
          vector matches exactly (all-or-nothing). Stricter than Hamming
          accuracy.
        - ``hamming_accuracy`` (float): Fraction of individual label bits
          predicted correctly: ``hamming_correct / hamming_total``. Equivalent
          to ``1 - hamming_loss`` in the standard multi-label definition.
        - ``per_label`` (dict): Per-label ``precision``, ``recall``, and ``f1``,
          keyed by ``names[i]``.
        - ``precision``, ``recall``, ``f1`` (np.ndarray): Per-label arrays of
          length ``C`` (same values as ``per_label``, kept for vectorized use).
    Notes:
        Per-label precision/recall/F1 use the usual definitions:
        ``precision = TP / (TP + FP)``, ``recall = TP / (TP + FN)``,
        ``F1 = 2 * precision * recall / (precision + recall)``.
        A small epsilon (1e-10) is added to denominators to avoid division
        by zero.
    """
    num_labels = len(names)
    if stats.tp.size == 0 and stats.hamming_total == 0:
        empty_f1 = np.zeros(num_labels, dtype=np.float64)
        per_label = {
            str(name): {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            for name in names
        }
        return {
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "subset_accuracy": 0.0,
            "hamming_accuracy": 0.0,
            "per_label": per_label,
            "precision": empty_f1,
            "recall": empty_f1,
            "f1": empty_f1,
        }

    precision = stats.tp.astype(np.float64) / (stats.tp + stats.fp + 1e-10)
    recall = stats.tp.astype(np.float64) / (stats.tp + stats.fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    micro_tp = float(stats.tp.sum())
    micro_fp = float(stats.fp.sum())
    micro_fn = float(stats.fn.sum())
    micro_precision = micro_tp / (micro_tp + micro_fp + 1e-10)
    micro_recall = micro_tp / (micro_tp + micro_fn + 1e-10)
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-10)
    )

    hamming_accuracy = (
        float(stats.hamming_correct) / float(stats.hamming_total)
        if stats.hamming_total > 0
        else 0.0
    )
    subset_accuracy = (
        float(stats.subset_correct) / float(stats.subset_total)
        if stats.subset_total > 0
        else 0.0
    )

    per_label: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(names):
        per_label[str(name)] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
        }

    return {
        "macro_f1": float(np.mean(f1)) if f1.size else 0.0,
        "micro_f1": float(micro_f1),
        "subset_accuracy": subset_accuracy,
        "hamming_accuracy": hamming_accuracy,
        "per_label": per_label,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def merge_multilabel_stats(stats_list: Sequence[MultilabelStats]) -> MultilabelStats:
    merged = MultilabelStats()
    for stats in stats_list:
        if stats.tp.size == 0:
            continue
        merged.ensure_num_labels(int(stats.tp.shape[0]))
        merged.tp += stats.tp
        merged.fp += stats.fp
        merged.fn += stats.fn
        merged.hamming_correct += stats.hamming_correct
        merged.hamming_total += stats.hamming_total
        merged.subset_correct += stats.subset_correct
        merged.subset_total += stats.subset_total
    return merged


def multilabel_stats_to_tensors(stats: MultilabelStats) -> Tuple[np.ndarray, ...]:
    """Flatten stats for distributed all_reduce: tp, fp, fn, hamming x2, subset x2."""
    return (
        stats.tp.astype(np.float64),
        stats.fp.astype(np.float64),
        stats.fn.astype(np.float64),
        np.array([stats.hamming_correct, stats.hamming_total], dtype=np.float64),
        np.array([stats.subset_correct, stats.subset_total], dtype=np.float64),
    )


def multilabel_stats_from_tensors(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    hamming: np.ndarray,
    subset: np.ndarray,
) -> MultilabelStats:
    return MultilabelStats(
        tp=tp.astype(np.int64),
        fp=fp.astype(np.int64),
        fn=fn.astype(np.int64),
        hamming_correct=int(hamming[0]),
        hamming_total=int(hamming[1]),
        subset_correct=int(subset[0]),
        subset_total=int(subset[1]),
    )
