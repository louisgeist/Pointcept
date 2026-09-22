#!/usr/bin/env python3
"""Offline probe on PureForest pooled embeddings (linear or MLP head).

Loads ``{train,val,test}.npz`` produced by
``scripts/extract_pureforest_pooled_embeddings.py``, builds four feature views
(mean / max / concat / sum), fits a classification head, selects the best
config on val, then reports test once.

Heads (``--head``):

- ``linear`` (default): multinomial logistic regression with an L2 ``C`` grid.
  Backends: omit ``--device`` for CPU sklearn; set ``--device cuda|cpu`` for
  torch LBFGS (train/val uploaded once per aggregation).
- ``mlp``: PureForest Fig.7-style head
  ``Linear(C→hidden) → LeakyReLU(0.2) → Dropout → Linear(hidden→K)``
  trained with Adam + CE. Always uses torch; ``--device`` defaults to ``cuda``
  if available else ``cpu``. Grid: ``agg × lr × weight_decay × dropout``.

Encoder-scale ablation (``--scale-slice``): pooled feats are the finest-first
multiscale concat (``enc_channels`` / ``channel_blocks``). Slice levels before
aggregation, e.g. ``[:2]`` / ``[2:]`` / ``full``. Blocks come from
``--channel-blocks`` or ``meta.json`` → extract config.

Usage::

    python scripts/probe_pureforest_sklearn.py \\
      --embeddings-dir stats/pureforest/embeddings/sonata_outdoor \\
      --output-dir stats/pureforest/sklearn_probe/sonata_outdoor \\
      -v

    python scripts/probe_pureforest_sklearn.py \\
      --embeddings-dir stats/pureforest/embeddings/litept_b_malibu3d_ms \\
      --output-dir stats/pureforest/sklearn_probe/litept_b_malibu3d_ms_torch \\
      --device cuda -v

    python scripts/probe_pureforest_sklearn.py \\
      --embeddings-dir stats/pureforest/embeddings/kpconvx_malibu3d_ms \\
      --output-dir stats/pureforest/sklearn_probe/kpconvx_malibu3d_ms_mlp \\
      --head mlp --device cuda -v

    python scripts/probe_pureforest_sklearn.py \\
      --embeddings-dir stats/pureforest/embeddings/sonata_outdoor_ms \\
      --output-dir stats/pureforest/sklearn_probe_scales/sonata_outdoor_ms/scale_p0-2 \\
      --scale-slice '[:2]' --device cuda -v

Progress is written after every grid fit to ``grid_progress.json``
(and ``best_so_far.pkl`` when the val winner changes). Re-run the same
command to resume; pass ``--fresh`` to start over. Use a distinct
``--output-dir`` when switching head/device/solver/scale-slice so progress
files do not mix.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


AGG_NAMES = ("mean", "max", "concat", "sum")
PROGRESS_NAME = "grid_progress.json"
BEST_CKPT_NAME = "best_so_far.pkl"

DEFAULT_MLP_LRS = (1e-3, 2e-3, 5e-3, 1e-2, 2e-2)
DEFAULT_MLP_WDS = (0.0, 1e-4, 1e-3, 1e-2)
DEFAULT_MLP_DROPOUTS = (0.0,)


def _float_key(value: float) -> str:
    """Stable key for a float hyperparam (avoid float repr mismatches on resume)."""
    return f"{float(value):.12g}"


def _c_key(c_value: float) -> str:
    return _float_key(c_value)


def _pair_key(agg: str, c_value: float) -> str:
    return f"{agg}|{_c_key(c_value)}"


def _mlp_pair_key(agg: str, lr: float, wd: float, dropout: float) -> str:
    return f"{agg}|{_float_key(lr)}|{_float_key(wd)}|{_float_key(dropout)}"


def _val_metrics_compact(metrics_val: dict) -> dict:
    return {k: metrics_val[k] for k in ("allAcc", "mAcc", "mIoU", "macro_f1")}


def _row_for_json(row: dict) -> dict:
    out = {
        "agg": row["agg"],
        "feat_dim": int(row["feat_dim"]),
        "val": _val_metrics_compact(row["val"]),
        "elapsed_s": row.get("elapsed_s"),
    }
    if "C" in row and row["C"] is not None:
        out["C"] = float(row["C"])
    if "lr" in row and row["lr"] is not None:
        out["lr"] = float(row["lr"])
        out["weight_decay"] = float(row["weight_decay"])
        out["dropout"] = float(row["dropout"])
        out["hidden"] = int(row["hidden"])
        out["epochs_ran"] = row.get("epochs_ran")
        out["best_epoch"] = row.get("best_epoch")
    return out


def load_progress(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.get("grid_results") or [])
    if isinstance(data, list):
        return data
    return []


def write_progress_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def save_best_ckpt(path: Path, best: dict) -> None:
    """Persist winner clf+scaler so a resumed run can evaluate test without refit."""
    blob = {
        "agg": best["agg"],
        "score": best["score"],
        "feat_dim": best["feat_dim"],
        "val": best["val"],
        "clf": best["clf"],
        "scaler": best["scaler"],
        "head": best.get("head", "linear"),
    }
    if best.get("C") is not None:
        blob["C"] = best["C"]
    for key in ("lr", "weight_decay", "dropout", "hidden"):
        if key in best and best[key] is not None:
            blob[key] = best[key]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_best_ckpt(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def load_split(embeddings_dir: Path, split: str):
    path = embeddings_dir / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing embeddings file: {path}")
    with np.load(path, allow_pickle=True) as data:
        payload = {
            "names": data["names"],
            "category": data["category"].astype(np.int64),
            "mean_feat": data["mean_feat"].astype(np.float32),
            "max_feat": data["max_feat"].astype(np.float32),
        }
        class_names = (
            data["class_names"].tolist()
            if "class_names" in data.files
            else None
        )
    return payload, class_names


def build_features(mean_feat, max_feat, agg: str):
    if agg == "mean":
        return mean_feat
    if agg == "max":
        return max_feat
    if agg == "concat":
        return np.concatenate([mean_feat, max_feat], axis=1)
    if agg == "sum":
        return mean_feat + max_feat
    raise ValueError(f"Unknown aggregation {agg!r}; expected one of {AGG_NAMES}.")


def parse_scale_slice(spec: str) -> slice | int:
    """Parse ``full`` / ``:`` / ``[:2]`` / ``[2:]`` / ``[1:3]`` / ``[0]``."""
    s = str(spec).strip()
    if s in ("", ":", "full", "[:]", "[::]"):
        return slice(None)
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(
            f"Invalid --scale-slice {spec!r}; expected e.g. 'full', '[:2]', '[2:]', '[1:3]', '[0]'."
        )
    inner = s[1:-1].strip()
    if not inner or inner == ":" or inner == "::":
        return slice(None)
    if ":" not in inner:
        return int(inner)

    parts = inner.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid --scale-slice {spec!r}.")

    def _maybe_int(token: str):
        token = token.strip()
        return None if token == "" else int(token)

    start = _maybe_int(parts[0])
    stop = _maybe_int(parts[1])
    step = _maybe_int(parts[2]) if len(parts) == 3 else None
    return slice(start, stop, step)


def is_full_scale_slice(parsed: slice | int) -> bool:
    return isinstance(parsed, slice) and parsed == slice(None)


def resolve_block_indices(n_blocks: int, parsed: slice | int) -> list[int]:
    indices = list(range(n_blocks))
    if isinstance(parsed, int):
        return [indices[parsed]]
    selected = indices[parsed]
    if isinstance(selected, int):
        return [selected]
    selected = list(selected)
    if not selected:
        raise ValueError(
            f"Scale slice {parsed!r} selects no blocks out of {n_blocks}."
        )
    return selected


def channel_ranges(blocks: tuple[int, ...]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for width in blocks:
        end = start + int(width)
        ranges.append((start, end))
        start = end
    return ranges


def apply_scale_slice(
    feat: np.ndarray, blocks: tuple[int, ...], block_indices: list[int]
) -> np.ndarray:
    """Keep finest-first encoder levels listed in ``block_indices``."""
    ranges = channel_ranges(blocks)
    parts = [feat[:, ranges[i][0] : ranges[i][1]] for i in block_indices]
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=1)


def format_scale_slice(parsed: slice | int) -> str:
    if isinstance(parsed, int):
        return f"[{parsed}]"
    if parsed == slice(None):
        return "full"
    start, stop, step = parsed.start, parsed.stop, parsed.step
    if step is None or step == 1:
        if start is None and stop is not None:
            return f"[:{stop}]"
        if start is not None and stop is None:
            return f"[{start}:]"
        if start is not None and stop is not None:
            return f"[{start}:{stop}]"
    body = ""
    body += "" if start is None else str(start)
    body += ":"
    body += "" if stop is None else str(stop)
    if step is not None:
        body += f":{step}"
    return f"[{body}]"


def load_channel_blocks_from_meta(embeddings_dir: Path) -> tuple[int, ...] | None:
    meta_path = embeddings_dir / "meta.json"
    if not meta_path.is_file():
        return None
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    config = meta.get("config")
    if not config:
        return None
    cfg_path = Path(config)
    if not cfg_path.is_file():
        cfg_path = REPO_ROOT / config
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"meta.json config not found: {config} (resolved {cfg_path})"
        )
    from pointcept.utils.config import Config

    cfg = Config.fromfile(str(cfg_path))
    blocks = None
    model_cfg = cfg.get("model")
    if model_cfg is not None:
        blocks = model_cfg.get("channel_blocks")
    if blocks is None:
        blocks = cfg.get("enc_channels")
    if blocks is None:
        return None
    return tuple(int(c) for c in blocks)


def resolve_channel_blocks(
    embeddings_dir: Path,
    cli_blocks: list[int] | None,
    feat_dim: int,
    scale_slice_spec: str,
) -> tuple[int, ...] | None:
    """Return encoder channel blocks, or None when the slice is full and unused."""
    parsed = parse_scale_slice(scale_slice_spec)
    if cli_blocks is not None:
        blocks = tuple(int(c) for c in cli_blocks)
    else:
        blocks = load_channel_blocks_from_meta(embeddings_dir)
    if is_full_scale_slice(parsed):
        if blocks is not None and sum(blocks) != feat_dim:
            raise ValueError(
                f"sum(channel_blocks)={sum(blocks)} != feat_dim={feat_dim}."
            )
        return blocks
    if blocks is None:
        raise SystemExit(
            "Non-full --scale-slice requires --channel-blocks or a meta.json "
            "config that defines model.channel_blocks / enc_channels."
        )
    if sum(blocks) != feat_dim:
        raise ValueError(
            f"sum(channel_blocks)={sum(blocks)} != feat_dim={feat_dim}."
        )
    return blocks


def classification_metrics(y_true, y_pred, num_classes, ignore_index=-1):
    from pointcept.utils.misc import (
        f1_scores_from_hist,
        intersection_and_union,
        mean_acc_from_hist,
        mean_iou_from_hist,
    )

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    intersection, union, target = intersection_and_union(
        y_pred, y_true, num_classes, ignore_index=ignore_index
    )
    m_iou = float(mean_iou_from_hist(intersection, union))
    m_acc = float(mean_acc_from_hist(intersection, target, union=union))
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    f1, macro_f1 = f1_scores_from_hist(intersection, union, target)
    iou_class = (intersection / (union + 1e-10)).astype(np.float64)
    return {
        "allAcc": acc,
        "mAcc": m_acc,
        "mIoU": m_iou,
        "macro_f1": float(macro_f1),
        "per_class_iou": iou_class.tolist(),
        "per_class_f1": f1.tolist(),
        "intersection": intersection.tolist(),
        "union": union.tolist(),
        "target": target.tolist(),
    }


def _sklearn_version_tuple():
    import sklearn

    return tuple(int(x) for x in sklearn.__version__.split(".")[:2])


def _make_logistic_regression(c_value, class_weight, max_iter, n_jobs, seed, solver="lbfgs"):
    """Build LogisticRegression with kwargs compatible across sklearn versions.

    Default ``solver='lbfgs'`` (multinomial softmax). ``newton-cholesky`` can be
    faster when ``n_samples >> n_features * n_classes``, but builds a dense
    Hessian of size ``(n_features * n_classes)^2`` — heavy for large ``concat``.

    Do **not** pass ``multi_class=`` on sklearn >= 1.5: the kwarg is deprecated
    and triggers ``FutureWarning`` even when set to ``'multinomial'``.
    """
    from sklearn.linear_model import LogisticRegression
    import inspect

    params = inspect.signature(LogisticRegression.__init__).parameters
    major_minor = _sklearn_version_tuple()
    kwargs = {
        "C": float(c_value),
        "solver": str(solver),
        "max_iter": int(max_iter),
        "random_state": int(seed),
    }
    if "class_weight" in params:
        kwargs["class_weight"] = class_weight
    if "multi_class" in params and major_minor < (1, 5):
        kwargs["multi_class"] = "multinomial"
    if "n_jobs" in params and n_jobs is not None and major_minor < (1, 8):
        kwargs["n_jobs"] = n_jobs
    return LogisticRegression(**kwargs)


class TorchLogisticClassifier:
    """Minimal sklearn-like predictor wrapping a fitted linear head (CPU numpy)."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray, n_iter: int):
        # weight: (num_classes, feat_dim), bias: (num_classes,)
        self.coef_ = np.asarray(weight, dtype=np.float64)
        self.intercept_ = np.asarray(bias, dtype=np.float64)
        self.n_iter_ = np.asarray([int(n_iter)])

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        logits = x @ self.coef_.T + self.intercept_
        return np.argmax(logits, axis=1).astype(np.int64)


def _build_mlp_sequential(in_features: int, num_classes: int, hidden: int, dropout: float):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(int(in_features), int(hidden), bias=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=float(dropout)),
        nn.Linear(int(hidden), int(num_classes), bias=True),
    )


class TorchMLPHead:
    """PureForest Fig.7-style probe head: Linear→hidden→Dropout→Linear→K."""

    def __init__(self, in_features: int, num_classes: int, hidden: int, dropout: float):
        self.hidden = int(hidden)
        self.dropout_p = float(dropout)
        self.net = _build_mlp_sequential(in_features, num_classes, hidden, dropout)

    def __call__(self, x):
        return self.net(x)

    def parameters(self):
        return self.net.parameters()

    def train(self, mode: bool = True):
        self.net.train(mode)
        return self

    def eval(self):
        self.net.eval()
        return self

    def to(self, device):
        self.net.to(device)
        return self

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state_dict):
        return self.net.load_state_dict(state_dict)


class TorchMLPClassifier:
    """sklearn-like predictor wrapping a fitted MLP head (CPU numpy / torch)."""

    def __init__(
        self,
        state_dict: dict,
        in_features: int,
        num_classes: int,
        hidden: int,
        dropout: float,
        n_iter: int,
        best_epoch: int,
    ):
        self.state_dict_ = {k: np.asarray(v) for k, v in state_dict.items()}
        self.in_features_ = int(in_features)
        self.num_classes_ = int(num_classes)
        self.hidden_ = int(hidden)
        self.dropout_ = float(dropout)
        self.n_iter_ = np.asarray([int(n_iter)])
        self.best_epoch_ = int(best_epoch)
        self._torch_module = None

    def _module(self):
        import torch

        if self._torch_module is None:
            head = TorchMLPHead(
                self.in_features_,
                self.num_classes_,
                self.hidden_,
                dropout=0.0,  # eval-time; weights already trained
            )
            sd = {k: torch.as_tensor(v) for k, v in self.state_dict_.items()}
            head.load_state_dict(sd)
            head.eval()
            self._torch_module = head
        return self._torch_module

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
        with torch.no_grad():
            logits = self._module()(x_t)
            pred = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
        return pred


def _class_weight_values(y, num_classes: int, class_weight):
    """Per-class weights for CE / LogisticRegression.

    Modes:
      - None / ``\"none\"``: no reweighting (returns None)
      - ``\"balanced\"``: ``w_k = N / (K * n_k)`` (sklearn-compatible)
      - ``\"sqrt\"``: ``w_k = N / (K * sqrt(n_k))`` (softer than balanced)

    Returns None or a length-``num_classes`` float64 array (0 for empty classes).
    """
    if class_weight is None or class_weight == "none":
        return None
    if class_weight not in ("balanced", "sqrt"):
        raise ValueError(
            f"Unsupported class_weight={class_weight!r}; "
            "expected none/balanced/sqrt"
        )
    y = np.asarray(y).reshape(-1).astype(np.int64, copy=False)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    n_samples = float(y.shape[0])
    w = np.zeros(num_classes, dtype=np.float64)
    for k, c in enumerate(counts):
        if c <= 0:
            continue
        denom = float(c) if class_weight == "balanced" else float(np.sqrt(c))
        w[k] = n_samples / (num_classes * denom)
    return w


def _class_weight_for_sklearn(y, num_classes: int, class_weight):
    """None, ``'balanced'``, or a ``{class_id: weight}`` dict for LogisticRegression."""
    if class_weight is None or class_weight == "none":
        return None
    if class_weight == "balanced":
        return "balanced"
    values = _class_weight_values(y, num_classes, class_weight)
    assert values is not None
    return {k: float(values[k]) for k in range(num_classes) if values[k] > 0}


def _class_weight_tensor_from_counts(y_train_t, num_classes: int, class_weight):
    """Build CE class weights on the same device as ``y_train_t``."""
    import torch

    if class_weight is None or class_weight == "none":
        return None
    y_cpu = y_train_t.detach().cpu().numpy()
    values = _class_weight_values(y_cpu, num_classes, class_weight)
    if values is None:
        return None
    return torch.tensor(values, dtype=torch.float32, device=y_train_t.device)


def prepare_torch_agg_bundle(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    device: str,
    num_classes: int,
    class_weight,
):
    """StandardScaler on CPU, then upload train/val once to ``device``.

    Returns ``(scaler, x_train_t, y_train_t, x_val_t, weight_ce, vram_bytes)``.
    """
    import torch
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    x_val_s = scaler.transform(x_val).astype(np.float32, copy=False)

    x_train_t = torch.tensor(x_train_s, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train.astype(np.int64, copy=False), dtype=torch.int64, device=device)
    x_val_t = torch.tensor(x_val_s, dtype=torch.float32, device=device)
    weight_ce = _class_weight_tensor_from_counts(y_train_t, num_classes, class_weight)

    vram_bytes = (
        x_train_t.numel() * x_train_t.element_size()
        + y_train_t.numel() * y_train_t.element_size()
        + x_val_t.numel() * x_val_t.element_size()
    )
    if weight_ce is not None:
        vram_bytes += weight_ce.numel() * weight_ce.element_size()
    return scaler, x_train_t, y_train_t, x_val_t, weight_ce, vram_bytes


def fit_torch_logistic_tensors(
    *,
    x_train_t,
    y_train_t,
    x_val_t,
    weight_ce,
    c_value: float,
    max_iter: int,
    seed: int,
    num_classes: int,
):
    """LBFGS multinomial logistic on **already-resident** device tensors.

    Objective (sklearn-compatible, intercept not regularized)::

        minimize  C * sum_i CE(y_i, softmax(W x_i + b)) + 0.5 ||W||^2
    """
    import torch
    import torch.nn.functional as F

    device = x_train_t.device
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    feat_dim = x_train_t.shape[1]
    model = torch.nn.Linear(feat_dim, num_classes, bias=True).to(device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)

    c_value = float(c_value)
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=int(max_iter),
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    n_closure = 0

    def closure():
        nonlocal n_closure
        opt.zero_grad(set_to_none=True)
        logits = model(x_train_t)
        ce = F.cross_entropy(logits, y_train_t, weight=weight_ce, reduction="sum")
        l2 = 0.5 * model.weight.pow(2).sum()
        loss = c_value * ce + l2
        loss.backward()
        n_closure += 1
        return loss

    opt.step(closure)

    # LBFGS stores true outer iteration count on the first parameter's state
    # (distinct from n_closure, which also counts line-search evals).
    first_param = next(model.parameters())
    n_lbfgs = int(opt.state.get(first_param, {}).get("n_iter", 0))

    with torch.no_grad():
        pred_val = model(x_val_t).argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        weight = model.weight.detach().cpu().numpy()
        bias = model.bias.detach().cpu().numpy()

    clf = TorchLogisticClassifier(weight=weight, bias=bias, n_iter=n_lbfgs)
    clf.n_closure_ = int(n_closure)
    return clf, pred_val


def fit_torch_mlp_tensors(
    *,
    x_train_t,
    y_train_t,
    x_val_t,
    y_val_np: np.ndarray,
    weight_ce,
    lr: float,
    weight_decay: float,
    dropout: float,
    hidden: int,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    num_classes: int,
    select_metric: str,
):
    """Adam + CE MLP head on already-resident device tensors; early-stop on val."""
    import torch
    import torch.nn.functional as F
    from copy import deepcopy

    device = x_train_t.device
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    feat_dim = int(x_train_t.shape[1])
    n_train = int(x_train_t.shape[0])
    model = TorchMLPHead(feat_dim, num_classes, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    best_score = -1.0
    best_epoch = 0
    best_state = None
    best_pred_val = None
    epochs_without_improve = 0
    epochs_ran = 0

    bs = int(batch_size)
    use_minibatches = bs > 0 and bs < n_train

    for epoch in range(1, int(epochs) + 1):
        epochs_ran = epoch
        model.train(True)
        if use_minibatches:
            perm = torch.randperm(n_train, device=device)
            for start in range(0, n_train, bs):
                idx = perm[start : start + bs]
                opt.zero_grad(set_to_none=True)
                logits = model(x_train_t[idx])
                loss = F.cross_entropy(logits, y_train_t[idx], weight=weight_ce)
                loss.backward()
                opt.step()
        else:
            opt.zero_grad(set_to_none=True)
            logits = model(x_train_t)
            loss = F.cross_entropy(logits, y_train_t, weight=weight_ce)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(x_val_t).argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        metrics_val = classification_metrics(
            y_val_np, pred_val, num_classes=num_classes
        )
        score = float(metrics_val[select_metric])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )
            best_pred_val = pred_val
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= int(patience):
                break

    assert best_state is not None and best_pred_val is not None
    state_np = {k: v.numpy() for k, v in best_state.items()}
    clf = TorchMLPClassifier(
        state_dict=state_np,
        in_features=feat_dim,
        num_classes=num_classes,
        hidden=hidden,
        dropout=dropout,
        n_iter=epochs_ran,
        best_epoch=best_epoch,
    )
    return clf, best_pred_val


def release_torch_agg_bundle(x_train_t, y_train_t, x_val_t, weight_ce, device: str) -> None:
    """Drop resident agg tensors before the next aggregation."""
    import torch

    del x_train_t, y_train_t, x_val_t, weight_ce
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def fit_and_eval(
    *,
    x_train,
    y_train,
    x_val,
    y_val,
    class_weight,
    max_iter,
    n_jobs,
    seed,
    solver: str = "lbfgs",
    backend: str = "sklearn",
    device: str = "cpu",
    num_classes: int | None = None,
    head: str = "linear",
    c_value: float | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    dropout: float | None = None,
    hidden: int = 32,
    epochs: int = 100,
    patience: int = 20,
    batch_size: int = 0,
    select_metric: str = "mIoU",
):
    """One-shot fit helper (winner refit). Supports linear and mlp heads."""
    from sklearn.preprocessing import StandardScaler

    if num_classes is None:
        num_classes = int(max(int(y_train.max()), int(y_val.max())) + 1)

    if head == "mlp":
        scaler, x_t, y_t, x_v, w_ce, _ = prepare_torch_agg_bundle(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            device=device,
            num_classes=num_classes,
            class_weight=class_weight,
        )
        clf, pred_val = fit_torch_mlp_tensors(
            x_train_t=x_t,
            y_train_t=y_t,
            x_val_t=x_v,
            y_val_np=np.asarray(y_val),
            weight_ce=w_ce,
            lr=float(lr),
            weight_decay=float(weight_decay),
            dropout=float(dropout),
            hidden=int(hidden),
            epochs=int(epochs),
            patience=int(patience),
            batch_size=int(batch_size),
            seed=seed,
            num_classes=num_classes,
            select_metric=select_metric,
        )
        release_torch_agg_bundle(x_t, y_t, x_v, w_ce, device)
        return clf, scaler, pred_val

    if backend == "torch":
        scaler, x_t, y_t, x_v, w_ce, _ = prepare_torch_agg_bundle(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            device=device,
            num_classes=num_classes,
            class_weight=class_weight,
        )
        clf, pred_val = fit_torch_logistic_tensors(
            x_train_t=x_t,
            y_train_t=y_t,
            x_val_t=x_v,
            weight_ce=w_ce,
            c_value=float(c_value),
            max_iter=max_iter,
            seed=seed,
            num_classes=num_classes,
        )
        release_torch_agg_bundle(x_t, y_t, x_v, w_ce, device)
        return clf, scaler, pred_val

    if backend != "sklearn":
        raise ValueError(f"Unknown backend {backend!r}")

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    clf = _make_logistic_regression(
        c_value=float(c_value),
        class_weight=_class_weight_for_sklearn(y_train, num_classes, class_weight),
        max_iter=max_iter,
        n_jobs=n_jobs,
        seed=seed,
        solver=solver,
    )
    clf.fit(x_train_s, y_train)
    pred_val = clf.predict(x_val_s)
    return clf, scaler, pred_val


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--embeddings-dir",
        required=True,
        help="Directory with train.npz / val.npz [/ test.npz].",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write metrics.json (default: embeddings-dir/sklearn_probe).",
    )
    parser.add_argument(
        "--head",
        default="linear",
        choices=["linear", "mlp"],
        help="Probe head: linear logistic (default) or PureForest-style MLP→32→K.",
    )
    parser.add_argument(
        "--aggs",
        nargs="+",
        default=list(AGG_NAMES),
        choices=list(AGG_NAMES),
        help="Feature aggregations to evaluate.",
    )
    parser.add_argument(
        "--Cs",
        nargs="+",
        type=float,
        default=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3],
        help="Inverse L2 strength grid for --head linear. Larger C = less regularization.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=32,
        help="MLP bottleneck width (--head mlp). Default 32 (PureForest Fig.7).",
    )
    parser.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        default=list(DEFAULT_MLP_LRS),
        help="Adam learning-rate grid for --head mlp.",
    )
    parser.add_argument(
        "--wds",
        nargs="+",
        type=float,
        default=list(DEFAULT_MLP_WDS),
        help="Adam weight-decay grid for --head mlp.",
    )
    parser.add_argument(
        "--dropouts",
        nargs="+",
        type=float,
        default=list(DEFAULT_MLP_DROPOUTS),
        help="Dropout grid for --head mlp.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max Adam epochs for --head mlp.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stop patience (epochs without val improve) for --head mlp.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Adam mini-batch size for --head mlp (0 = full-batch).",
    )
    parser.add_argument(
        "--class-weight",
        default="none",
        choices=["none", "balanced", "sqrt"],
        help=(
            "Per-class CE / LogisticRegression weights: none (default); "
            "balanced ∝ 1/n_k; sqrt ∝ 1/sqrt(n_k) (softer than balanced)."
        ),
    )
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--solver",
        default="lbfgs",
        choices=["lbfgs", "newton-cholesky", "newton-cg", "sag", "saga"],
        help="sklearn LogisticRegression solver (linear head only; ignored with --device).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device (e.g. cuda / cuda:0 / cpu). Required implicitly for "
            "--head mlp (defaults to cuda if available else cpu). For --head "
            "linear, omitting --device keeps the sklearn CPU backend."
        ),
    )
    parser.add_argument(
        "--select-metric",
        default="mIoU",
        choices=["mIoU", "allAcc", "mAcc", "macro_f1"],
        help="Val metric used to pick best config (and MLP early stopping).",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Only run train/val selection (no test.npz required).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log each grid cell before fit, wall time, and iteration counts.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore/delete grid_progress.json and best_so_far.pkl; start the grid from scratch.",
    )
    parser.add_argument(
        "--channel-blocks",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Finest-first encoder channel widths that sum to feat_dim "
            "(e.g. Sonata: 48 96 192 384 512). Default: read "
            "model.channel_blocks / enc_channels from embeddings meta.json config."
        ),
    )
    parser.add_argument(
        "--scale-slice",
        default="full",
        help=(
            "Which encoder levels to keep before aggregation "
            "(finest-first). Examples: full, [:2], [2:], [1:3], [0]. "
            "Default full (no channel slicing)."
        ),
    )
    return parser.parse_args()


def _progress_key_from_row(row: dict, head: str) -> str:
    if head == "mlp":
        return _mlp_pair_key(
            row["agg"],
            float(row["lr"]),
            float(row["weight_decay"]),
            float(row["dropout"]),
        )
    return _pair_key(row["agg"], float(row["C"]))


def _best_from_blob(best_blob: dict) -> dict:
    best = {
        "score": float(best_blob["score"]),
        "agg": best_blob["agg"],
        "clf": best_blob["clf"],
        "scaler": best_blob["scaler"],
        "val": best_blob["val"],
        "feat_dim": int(best_blob["feat_dim"]),
        "head": best_blob.get("head", "linear"),
        "C": best_blob.get("C"),
        "lr": best_blob.get("lr"),
        "weight_decay": best_blob.get("weight_decay"),
        "dropout": best_blob.get("dropout"),
        "hidden": best_blob.get("hidden"),
    }
    return best


def _best_label(best: dict) -> str:
    if best.get("head") == "mlp" or best.get("lr") is not None:
        return (
            f"agg={best['agg']} lr={best['lr']:g} wd={best['weight_decay']:g} "
            f"do={best['dropout']:g} hidden={best.get('hidden')}"
        )
    return f"agg={best['agg']} C={best['C']}"


def main():
    args = parse_args()
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        import sklearn
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is required. Install with: pip install scikit-learn"
        ) from exc

    head = str(args.head)
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir or (embeddings_dir / "sklearn_probe"))
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_NAME
    best_ckpt_path = output_dir / BEST_CKPT_NAME

    if args.fresh:
        for p in (progress_path, best_ckpt_path):
            if p.is_file():
                p.unlink()
                print(f"[probe] --fresh: removed {p}", flush=True)

    train, class_names_train = load_split(embeddings_dir, "train")
    val, class_names_val = load_split(embeddings_dir, "val")

    raw_feat_dim = int(train["mean_feat"].shape[1])
    if int(val["mean_feat"].shape[1]) != raw_feat_dim:
        raise SystemExit(
            f"train/val feat_dim mismatch: {raw_feat_dim} vs {val['mean_feat'].shape[1]}"
        )
    scale_slice_parsed = parse_scale_slice(args.scale_slice)
    channel_blocks = resolve_channel_blocks(
        embeddings_dir,
        args.channel_blocks,
        raw_feat_dim,
        args.scale_slice,
    )
    scale_slice_label = format_scale_slice(scale_slice_parsed)
    block_indices: list[int] | None = None
    if not is_full_scale_slice(scale_slice_parsed):
        assert channel_blocks is not None
        block_indices = resolve_block_indices(len(channel_blocks), scale_slice_parsed)
        train["mean_feat"] = apply_scale_slice(
            train["mean_feat"], channel_blocks, block_indices
        )
        train["max_feat"] = apply_scale_slice(
            train["max_feat"], channel_blocks, block_indices
        )
        val["mean_feat"] = apply_scale_slice(
            val["mean_feat"], channel_blocks, block_indices
        )
        val["max_feat"] = apply_scale_slice(
            val["max_feat"], channel_blocks, block_indices
        )
        print(
            f"[probe] scale_slice={scale_slice_label}  "
            f"channel_blocks={list(channel_blocks)}  "
            f"levels={block_indices}  "
            f"feat_dim {raw_feat_dim} → {train['mean_feat'].shape[1]}",
            flush=True,
        )
    elif channel_blocks is not None:
        print(
            f"[probe] scale_slice=full  channel_blocks={list(channel_blocks)}  "
            f"feat_dim={raw_feat_dim}",
            flush=True,
        )

    class_names = class_names_train or class_names_val
    if class_names is None:
        from pointcept.datasets.preprocessing.pureforest.pureforest_classes import (
            CLASS_NAMES,
        )

        class_names = list(CLASS_NAMES)
    num_classes = len(class_names)

    class_weight = None if args.class_weight == "none" else args.class_weight
    results: list[dict] = []
    done_keys: set[str] = set()
    best = None

    # Backend / device resolution.
    if head == "mlp":
        import torch

        backend = "torch"
        if args.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = str(args.device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[probe] CUDA unavailable -- falling back to --device cpu", flush=True)
            device = "cpu"
    else:
        backend = "torch" if args.device else "sklearn"
        device = "cpu"
        if backend == "torch":
            import torch

            device = str(args.device)
            if device.startswith("cuda") and not torch.cuda.is_available():
                print(
                    "[probe] CUDA unavailable -- falling back to --device cpu",
                    flush=True,
                )
                device = "cpu"

    prior = load_progress(progress_path)
    if prior:
        for row in prior:
            results.append(row)
            done_keys.add(_progress_key_from_row(row, head))
        cell_name = "(agg, lr, wd, do)" if head == "mlp" else "(agg, C)"
        print(
            f"[probe] resume: loaded {len(done_keys)} finished {cell_name} "
            f"from {progress_path}",
            flush=True,
        )
        best_blob = load_best_ckpt(best_ckpt_path)
        if best_blob is not None:
            best = _best_from_blob(best_blob)
            print(
                f"[probe] resume: best so far {_best_label(best)} "
                f"score={best['score']:.4f}",
                flush=True,
            )
        elif results:
            for row in results:
                score = float(row["val"][args.select_metric])
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "agg": row["agg"],
                        "clf": None,
                        "scaler": None,
                        "val": row["val"],
                        "feat_dim": int(row["feat_dim"]),
                        "head": head,
                        "C": row.get("C"),
                        "lr": row.get("lr"),
                        "weight_decay": row.get("weight_decay"),
                        "dropout": row.get("dropout"),
                        "hidden": row.get("hidden", args.hidden if head == "mlp" else None),
                    }

    if head == "mlp":
        n_grid = (
            len(args.aggs) * len(args.lrs) * len(args.wds) * len(args.dropouts)
        )
        print(
            f"[probe] head=mlp hidden={args.hidden}  "
            f"train={train['category'].shape[0]}  val={val['category'].shape[0]}  "
            f"feat_dim={train['mean_feat'].shape[1]}  n_classes={num_classes}  "
            f"aggs={args.aggs}  lrs={args.lrs}  wds={args.wds}  "
            f"dropouts={args.dropouts}"
        )
        print(
            f"[probe] backend=torch device={device} Adam+CE  "
            f"epochs={args.epochs} patience={args.patience} "
            f"batch_size={args.batch_size or 'full'}  grid={n_grid} fits  "
            f"select={args.select_metric}  progress→{progress_path}",
            flush=True,
        )
    else:
        n_grid = len(args.aggs) * len(args.Cs)
        print(
            f"[probe] head=linear  "
            f"train={train['category'].shape[0]}  val={val['category'].shape[0]}  "
            f"feat_dim={train['mean_feat'].shape[1]}  n_classes={num_classes}  "
            f"aggs={args.aggs}  Cs={args.Cs}"
        )
        if backend == "sklearn":
            print(
                f"[probe] backend=sklearn solver={args.solver} multiclass=multinomial "
                f"(sklearn {sklearn.__version__})  grid={n_grid} fits  "
                f"select={args.select_metric}  progress→{progress_path}",
                flush=True,
            )
            if args.solver == "newton-cholesky" and "concat" in args.aggs:
                d = int(train["mean_feat"].shape[1])
                print(
                    f"[probe] warning: newton-cholesky + concat uses Hessian size "
                    f"~({2 * d}*{num_classes})^2 floats — expect high RAM",
                    flush=True,
                )
        else:
            print(
                f"[probe] backend=torch device={device} multiclass=multinomial "
                f"(resident train/val tensors per agg)  grid={n_grid} fits  "
                f"select={args.select_metric}  progress→{progress_path}",
                flush=True,
            )

    fit_kwargs = dict(
        class_weight=class_weight,
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
        seed=args.seed,
        backend=backend,
        device=device,
        num_classes=num_classes,
        solver=args.solver,
        head=head,
        hidden=args.hidden,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        select_metric=args.select_metric,
    )

    def _record_fit(
        *,
        grid_i: int,
        agg: str,
        feat_dim: int,
        clf,
        scaler,
        pred_val,
        elapsed: float,
        c_value: float | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
        dropout: float | None = None,
        hidden: int | None = None,
    ) -> None:
        nonlocal best
        metrics_val = classification_metrics(
            val["category"], pred_val, num_classes=num_classes
        )
        row = {
            "agg": agg,
            "feat_dim": int(feat_dim),
            "val": metrics_val,
            "elapsed_s": round(elapsed, 3),
            "C": float(c_value) if c_value is not None else None,
            "lr": float(lr) if lr is not None else None,
            "weight_decay": float(weight_decay) if weight_decay is not None else None,
            "dropout": float(dropout) if dropout is not None else None,
            "hidden": int(hidden) if hidden is not None else None,
        }
        if head == "mlp":
            row["epochs_ran"] = int(np.max(clf.n_iter_))
            row["best_epoch"] = int(getattr(clf, "best_epoch_", row["epochs_ran"]))
            done_keys.add(_mlp_pair_key(agg, lr, weight_decay, dropout))
            cell = (
                f"agg={agg:6s} lr={lr:<8g} wd={weight_decay:<8g} do={dropout:<4g}"
            )
        else:
            done_keys.add(_pair_key(agg, c_value))
            cell = f"agg={agg:6s} C={c_value:<8g}"

        results.append(row)
        score = metrics_val[args.select_metric]
        n_iter = getattr(clf, "n_iter_", None)
        n_iter_str = (
            f"  n_iter={int(np.max(n_iter))}" if n_iter is not None else ""
        )
        n_closure = getattr(clf, "n_closure_", None)
        if args.verbose and n_closure is not None and n_iter is not None:
            n_iter_str += f"  n_closure={int(n_closure)}"
        if head == "mlp" and args.verbose:
            n_iter_str += f"  best_epoch={row['best_epoch']}"
        print(
            f"[probe] ({grid_i}/{n_grid}) {cell} "
            f"val/{args.select_metric}={score:.4f}  "
            f"allAcc={metrics_val['allAcc']:.4f}  "
            f"mIoU={metrics_val['mIoU']:.4f}  "
            f"macro_f1={metrics_val['macro_f1']:.4f}  "
            f"({elapsed:.1f}s{n_iter_str})",
            flush=True,
        )
        if (
            head == "linear"
            and args.verbose
            and n_iter is not None
            and int(np.max(n_iter)) >= args.max_iter
        ):
            print(
                f"[probe] warning: LBFGS hit max_iter={args.max_iter} "
                f"(agg={agg} C={c_value:g}); consider raising --max-iter",
                flush=True,
            )
        if best is None or score > best["score"]:
            best = {
                "score": float(score),
                "agg": agg,
                "clf": clf,
                "scaler": scaler,
                "val": metrics_val,
                "feat_dim": int(feat_dim),
                "head": head,
                "C": float(c_value) if c_value is not None else None,
                "lr": float(lr) if lr is not None else None,
                "weight_decay": float(weight_decay) if weight_decay is not None else None,
                "dropout": float(dropout) if dropout is not None else None,
                "hidden": int(hidden) if hidden is not None else None,
            }
            save_best_ckpt(best_ckpt_path, best)
            if args.verbose:
                print(
                    f"[probe]   ^ new best {args.select_metric}={score:.4f} "
                    f"(saved {best_ckpt_path.name})",
                    flush=True,
                )
        best_payload = {
            "agg": best["agg"],
            "score": best["score"],
            "feat_dim": best["feat_dim"],
            "val": _val_metrics_compact(best["val"]),
            "head": best.get("head", head),
        }
        if best.get("C") is not None:
            best_payload["C"] = best["C"]
        if best.get("lr") is not None:
            best_payload["lr"] = best["lr"]
            best_payload["weight_decay"] = best["weight_decay"]
            best_payload["dropout"] = best["dropout"]
            best_payload["hidden"] = best["hidden"]
        write_progress_atomic(
            progress_path,
            {
                "embeddings_dir": str(embeddings_dir),
                "head": head,
                "backend": backend,
                "solver": args.solver if backend == "sklearn" and head == "linear" else None,
                "device": device if backend == "torch" else None,
                "select_metric": args.select_metric,
                "class_weight": class_weight,
                "hidden": args.hidden if head == "mlp" else None,
                "updated_at_fit": grid_i,
                "n_grid": n_grid,
                "best": best_payload,
                "grid_results": [_row_for_json(r) for r in results],
            },
        )

    grid_i = 0

    if head == "mlp":
        mlp_cells = [
            (lr, wd, do)
            for lr in args.lrs
            for wd in args.wds
            for do in args.dropouts
        ]
        for agg in args.aggs:
            x_train = build_features(train["mean_feat"], train["max_feat"], agg)
            x_val = build_features(val["mean_feat"], val["max_feat"], agg)
            feat_dim = int(x_train.shape[1])
            pending = [
                cell
                for cell in mlp_cells
                if _mlp_pair_key(agg, *cell) not in done_keys
            ]
            bundle = None
            scaler = None
            if pending:
                scaler, x_t, y_t, x_v, w_ce, vram_bytes = prepare_torch_agg_bundle(
                    x_train=x_train,
                    y_train=train["category"],
                    x_val=x_val,
                    device=device,
                    num_classes=num_classes,
                    class_weight=class_weight,
                )
                bundle = (x_t, y_t, x_v, w_ce)
                if args.verbose:
                    print(
                        f"[probe] agg={agg} uploaded to {device}: "
                        f"X_train={tuple(x_t.shape)} X_val={tuple(x_v.shape)} "
                        f"resident≈{vram_bytes / (1024 ** 2):.1f} MiB "
                        f"({len(pending)} mlp cells)",
                        flush=True,
                    )
            for lr, wd, do in mlp_cells:
                grid_i += 1
                key = _mlp_pair_key(agg, lr, wd, do)
                if key in done_keys:
                    print(
                        f"[probe] ({grid_i}/{n_grid}) skip agg={agg:6s} "
                        f"lr={lr:<8g} wd={wd:<8g} do={do:<4g} (already in progress)",
                        flush=True,
                    )
                    continue
                assert bundle is not None and scaler is not None
                x_t, y_t, x_v, w_ce = bundle
                if args.verbose:
                    print(
                        f"[probe] ({grid_i}/{n_grid}) fitting agg={agg} "
                        f"lr={lr:g} wd={wd:g} do={do:g} hidden={args.hidden} "
                        f"on {device} ...",
                        flush=True,
                    )
                t0 = time.perf_counter()
                clf, pred_val = fit_torch_mlp_tensors(
                    x_train_t=x_t,
                    y_train_t=y_t,
                    x_val_t=x_v,
                    y_val_np=val["category"],
                    weight_ce=w_ce,
                    lr=lr,
                    weight_decay=wd,
                    dropout=do,
                    hidden=args.hidden,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    num_classes=num_classes,
                    select_metric=args.select_metric,
                )
                elapsed = time.perf_counter() - t0
                _record_fit(
                    grid_i=grid_i,
                    agg=agg,
                    feat_dim=feat_dim,
                    clf=clf,
                    scaler=scaler,
                    pred_val=pred_val,
                    elapsed=elapsed,
                    lr=lr,
                    weight_decay=wd,
                    dropout=do,
                    hidden=args.hidden,
                )
            if bundle is not None:
                x_t, y_t, x_v, w_ce = bundle
                release_torch_agg_bundle(x_t, y_t, x_v, w_ce, device)
    else:
        for agg in args.aggs:
            x_train = build_features(train["mean_feat"], train["max_feat"], agg)
            x_val = build_features(val["mean_feat"], val["max_feat"], agg)
            feat_dim = int(x_train.shape[1])

            if backend == "torch":
                pending_cs = [
                    c for c in args.Cs if _pair_key(agg, c) not in done_keys
                ]
                bundle = None
                scaler = None
                if pending_cs:
                    scaler, x_t, y_t, x_v, w_ce, vram_bytes = prepare_torch_agg_bundle(
                        x_train=x_train,
                        y_train=train["category"],
                        x_val=x_val,
                        device=device,
                        num_classes=num_classes,
                        class_weight=class_weight,
                    )
                    bundle = (x_t, y_t, x_v, w_ce)
                    if args.verbose:
                        print(
                            f"[probe] agg={agg} uploaded to {device}: "
                            f"X_train={tuple(x_t.shape)} X_val={tuple(x_v.shape)} "
                            f"resident≈{vram_bytes / (1024 ** 2):.1f} MiB "
                            f"({len(pending_cs)} C values)",
                            flush=True,
                        )
                for c_value in args.Cs:
                    grid_i += 1
                    key = _pair_key(agg, c_value)
                    if key in done_keys:
                        print(
                            f"[probe] ({grid_i}/{n_grid}) skip agg={agg:6s} "
                            f"C={c_value:<8g} (already in progress)",
                            flush=True,
                        )
                        continue
                    assert bundle is not None and scaler is not None
                    x_t, y_t, x_v, w_ce = bundle
                    if args.verbose:
                        print(
                            f"[probe] ({grid_i}/{n_grid}) fitting agg={agg} "
                            f"C={c_value:g} on {device} ...",
                            flush=True,
                        )
                    t0 = time.perf_counter()
                    clf, pred_val = fit_torch_logistic_tensors(
                        x_train_t=x_t,
                        y_train_t=y_t,
                        x_val_t=x_v,
                        weight_ce=w_ce,
                        c_value=c_value,
                        max_iter=args.max_iter,
                        seed=args.seed,
                        num_classes=num_classes,
                    )
                    elapsed = time.perf_counter() - t0
                    _record_fit(
                        grid_i=grid_i,
                        agg=agg,
                        feat_dim=feat_dim,
                        clf=clf,
                        scaler=scaler,
                        pred_val=pred_val,
                        elapsed=elapsed,
                        c_value=c_value,
                    )
                if bundle is not None:
                    x_t, y_t, x_v, w_ce = bundle
                    release_torch_agg_bundle(x_t, y_t, x_v, w_ce, device)
                continue

            # sklearn: scale+fit per (agg, C) as before
            for c_value in args.Cs:
                grid_i += 1
                key = _pair_key(agg, c_value)
                if key in done_keys:
                    print(
                        f"[probe] ({grid_i}/{n_grid}) skip agg={agg:6s} C={c_value:<8g} "
                        f"(already in progress)",
                        flush=True,
                    )
                    continue
                if args.verbose:
                    print(
                        f"[probe] ({grid_i}/{n_grid}) fitting agg={agg} C={c_value:g} "
                        f"X_train={x_train.shape} ...",
                        flush=True,
                    )
                t0 = time.perf_counter()
                clf, scaler, pred_val = fit_and_eval(
                    x_train=x_train,
                    y_train=train["category"],
                    x_val=x_val,
                    y_val=val["category"],
                    c_value=c_value,
                    **fit_kwargs,
                )
                elapsed = time.perf_counter() - t0
                _record_fit(
                    grid_i=grid_i,
                    agg=agg,
                    feat_dim=feat_dim,
                    clf=clf,
                    scaler=scaler,
                    pred_val=pred_val,
                    elapsed=elapsed,
                    c_value=c_value,
                )

    assert best is not None
    # If we resumed without a pickled winner, refit best once for test.
    if best.get("clf") is None or best.get("scaler") is None:
        print(
            f"[probe] refitting winner {_best_label(best)} for test...",
            flush=True,
        )
        x_train = build_features(train["mean_feat"], train["max_feat"], best["agg"])
        x_val = build_features(val["mean_feat"], val["max_feat"], best["agg"])
        clf, scaler, pred_val = fit_and_eval(
            x_train=x_train,
            y_train=train["category"],
            x_val=x_val,
            y_val=val["category"],
            c_value=best.get("C"),
            lr=best.get("lr"),
            weight_decay=best.get("weight_decay"),
            dropout=best.get("dropout"),
            **fit_kwargs,
        )
        best["clf"] = clf
        best["scaler"] = scaler
        best["val"] = classification_metrics(
            val["category"], pred_val, num_classes=num_classes
        )
        best["score"] = float(best["val"][args.select_metric])
        save_best_ckpt(best_ckpt_path, best)
    elif "per_class_f1" not in best["val"]:
        x_val = build_features(val["mean_feat"], val["max_feat"], best["agg"])
        pred_val = best["clf"].predict(best["scaler"].transform(x_val))
        best["val"] = classification_metrics(
            val["category"], pred_val, num_classes=num_classes
        )

    print(f"[probe] best on val: {_best_label(best)} {args.select_metric}={best['score']:.4f}")

    _split_metric_keys = (
        "allAcc",
        "mAcc",
        "mIoU",
        "macro_f1",
        "per_class_iou",
        "per_class_f1",
    )

    def _metrics_subset(metrics: dict) -> dict:
        return {k: metrics[k] for k in _split_metric_keys if k in metrics}

    def _print_split_metrics(split: str, metrics: dict) -> None:
        print(
            f"[probe] {split} (winner only): "
            f"allAcc={metrics['allAcc']:.4f}  "
            f"mIoU={metrics['mIoU']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if "per_class_iou" in metrics:
            print(f"[probe] {split} per-class IoU:", flush=True)
            for name, iou in zip(class_names, metrics["per_class_iou"]):
                print(f"  {name:40s}  {iou:.4f}", flush=True)

    # Train metrics for the winner (cheap predict; no refit).
    x_train_best = build_features(train["mean_feat"], train["max_feat"], best["agg"])
    pred_train = best["clf"].predict(best["scaler"].transform(x_train_best))
    metrics_train = classification_metrics(
        train["category"], pred_train, num_classes=num_classes
    )
    _print_split_metrics("train", metrics_train)

    best_report = {
        "agg": best["agg"],
        "feat_dim": best["feat_dim"],
        "head": best.get("head", head),
        "train": _metrics_subset(metrics_train),
        "val": _metrics_subset(best["val"]),
    }
    if best.get("C") is not None:
        best_report["C"] = best["C"]
    if best.get("lr") is not None:
        best_report["lr"] = best["lr"]
        best_report["weight_decay"] = best["weight_decay"]
        best_report["dropout"] = best["dropout"]
        best_report["hidden"] = best["hidden"]

    report = {
        "embeddings_dir": str(embeddings_dir),
        "head": head,
        "backend": backend,
        "solver": args.solver if backend == "sklearn" and head == "linear" else None,
        "device": device if backend == "torch" else None,
        "class_names": class_names,
        "num_classes": num_classes,
        "select_metric": args.select_metric,
        "class_weight": class_weight,
        "aggs": list(args.aggs),
        "scale_slice": scale_slice_label,
        "scale_slice_raw": str(args.scale_slice),
        "channel_blocks": list(channel_blocks) if channel_blocks is not None else None,
        "block_indices": block_indices,
        "raw_feat_dim": raw_feat_dim,
        "grid_results": [_row_for_json(r) for r in results],
        "best": best_report,
    }
    if head == "linear":
        report["Cs"] = [float(c) for c in args.Cs]
    else:
        report["hidden"] = int(args.hidden)
        report["lrs"] = [float(x) for x in args.lrs]
        report["wds"] = [float(x) for x in args.wds]
        report["dropouts"] = [float(x) for x in args.dropouts]
        report["epochs"] = int(args.epochs)
        report["patience"] = int(args.patience)
        report["batch_size"] = int(args.batch_size)

    if not args.skip_test:
        test_path = embeddings_dir / "test.npz"
        if not test_path.is_file():
            print(f"[probe] no test.npz at {test_path} -- skipping test.")
        else:
            test, _ = load_split(embeddings_dir, "test")
            if block_indices is not None:
                assert channel_blocks is not None
                if int(test["mean_feat"].shape[1]) != raw_feat_dim:
                    raise SystemExit(
                        f"test feat_dim {test['mean_feat'].shape[1]} != "
                        f"raw_feat_dim {raw_feat_dim}"
                    )
                test["mean_feat"] = apply_scale_slice(
                    test["mean_feat"], channel_blocks, block_indices
                )
                test["max_feat"] = apply_scale_slice(
                    test["max_feat"], channel_blocks, block_indices
                )
            x_test = build_features(test["mean_feat"], test["max_feat"], best["agg"])
            x_test_s = best["scaler"].transform(x_test)
            pred_test = best["clf"].predict(x_test_s)
            metrics_test = classification_metrics(
                test["category"], pred_test, num_classes=num_classes
            )
            report["best"]["test"] = _metrics_subset(metrics_test)
            _print_split_metrics("test", metrics_test)

            pred_path = output_dir / "best_test_predictions.npz"
            save_kwargs = dict(
                names=test["names"],
                category=test["category"],
                pred=pred_test.astype(np.int64),
                agg=best["agg"],
                head=np.asarray(best.get("head", head)),
            )
            if best.get("C") is not None:
                save_kwargs["C"] = best["C"]
            if best.get("lr") is not None:
                save_kwargs["lr"] = best["lr"]
                save_kwargs["weight_decay"] = best["weight_decay"]
                save_kwargs["dropout"] = best["dropout"]
                save_kwargs["hidden"] = best["hidden"]
            np.savez_compressed(pred_path, **save_kwargs)
            print(f"[probe] wrote {pred_path}")

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[probe] wrote {metrics_path}")


if __name__ == "__main__":
    main()
