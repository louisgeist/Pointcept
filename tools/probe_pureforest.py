"""
PureForest linear probe of a frozen backbone (PureForest column of Tab. 3 of the MALiBU3D paper).

1. Extraction: one forward pass of the frozen backbone over every train / val / test tile; the
   multi-scale features are mean- and max-pooled per tile and saved to ``{split}.npz``.
2. Probe: for every pooling in ``probe.aggs`` (mean, max, their concatenation, their sum) and every
   ``C`` in ``probe.Cs``, a multinomial logistic regression (standardized features, L2 penalty
   ``0.5 ||W||^2 + C * sum of cross-entropies``, full-batch L-BFGS) is fitted on train. The pair
   with the best validation mIoU is evaluated on test.

Configs: ``configs/pureforest/probe-*.py``. Usage::

    export PYTHONPATH=./
    python tools/probe_pureforest.py --config-file configs/pureforest/probe-litept-b-v1m0-pureforest.py

Outputs go to ``--work-dir`` (default ``exp/pureforest/<config name>``): ``{split}.npz``,
``metrics.json`` and ``test_predictions.npz``. Existing ``{split}.npz`` files are reused, so the
probe can be refitted without a new extraction (``--force-extract`` redoes it).

Eval-time serialization of the transformer backbones shuffles the point orders, so the pooled
features vary slightly from one run to the next; ``seed`` in the config makes a run reproducible.
"""

import argparse
import copy
import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pointcept.datasets import collate_fn
from pointcept.datasets.builder import build_dataset
from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import pack_indices_by_voxel_budget
from pointcept.models.builder import build_model
from pointcept.utils.config import Config, DictAction
from pointcept.utils.misc import (
    f1_scores_from_hist,
    intersection_and_union,
    mean_acc_from_hist,
    mean_iou_from_hist,
)

SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-file", required=True, metavar="FILE")
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="override config entries"
    )
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--device", default=None, help="default: cuda if available")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="tiles per extraction batch (max tiles per batch with --point-budget)",
    )
    parser.add_argument(
        "--point-budget",
        type=int,
        default=None,
        help="pack extraction batches up to this many raw points (PureForest tiles vary ~10x "
        "in size, so this bounds the memory of a batch better than --batch-size)",
    )
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument(
        "--amp", action="store_true", help="fp16 autocast during extraction"
    )
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- extraction


def install_sdpa_flash_shim(module):
    """Route ``module.flash_attn.flash_attn_varlen_qkvpacked_func`` through SDPA when
    flash-attn is not installed (same attention, unfused kernel). No-op otherwise."""
    if getattr(module, "flash_attn", None) is not None:
        return False

    class _SDPAFlashShim:
        @staticmethod
        def flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None
        ):
            n_pad, _, h, ch = qkv.shape
            q_all, k_all, v_all = qkv.unbind(dim=1)
            out = torch.empty(n_pad, h, ch, device=qkv.device, dtype=qkv.dtype)
            starts = cu_seqlens[:-1].tolist()
            ends = cu_seqlens[1:].tolist()
            for s, e in zip(starts, ends):
                if e <= s:
                    continue
                q = q_all[s:e].permute(1, 0, 2).unsqueeze(0)
                k = k_all[s:e].permute(1, 0, 2).unsqueeze(0)
                v = v_all[s:e].permute(1, 0, 2).unsqueeze(0)
                o = F.scaled_dot_product_attention(q, k, v, scale=softmax_scale)
                out[s:e] = o.squeeze(0).permute(1, 0, 2)
            return out

    module.flash_attn = _SDPAFlashShim()
    return True


def force_native_conv_algo(model):
    """Use spconv's native gather-scatter convolution in every sparse conv layer. Some
    spconv/cumm builds find no auto-tuned kernel once several conv shapes are used in one
    process; the native algorithm avoids the tuner (slower, fine for a single pass)."""
    try:
        import spconv.pytorch as spconv
        from spconv.pytorch.core import ConvAlgo
    except ImportError:
        return 0
    n = 0
    for m in model.modules():
        if isinstance(m, spconv.conv.SparseConvolution):
            m.algo = ConvAlgo.Native
            n += 1
    return n


def load_weight(model, path, key_map=None):
    """Load the backbone (and learned mask values) of a checkpoint; every model key must be
    found. Other checkpoint keys (task heads, Sonata teacher, ...) are ignored."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    weight = OrderedDict()
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        for old, new in (key_map or {}).items():
            if key.startswith(old):
                key = new + key[len(old) :]
        weight[key] = value
    missing, unexpected = model.load_state_dict(weight, strict=False)
    if missing:
        raise KeyError(f"{len(missing)} model keys missing from {path}: {missing[:10]}")
    print(f"[probe] loaded {path} ({len(unexpected)} checkpoint keys unused)")


def build_extractor(cfg, device):
    model_cfg = copy.deepcopy(cfg.model)
    backbone_type = model_cfg["backbone"]["type"]
    if backbone_type == "LitePT-v1":
        import pointcept.models.litept.litept_v1 as module

        install_sdpa_flash_shim(module)
    elif backbone_type == "PT-v3-malibu":
        import pointcept.models.point_transformer_v3.point_transformer_v3_malibu as module

        install_sdpa_flash_shim(module)
    model = build_model(model_cfg)
    force_native_conv_algo(model)
    load_weight(model, cfg.weight, cfg.get("weight_key_map"))
    return model.to(device).eval()


def split_transform(transform):
    """Split the pipeline after GridSample, whose test mode returns a list of fragments."""
    transform = list(transform)
    i = next(i for i, t in enumerate(transform) if t["type"] == "GridSample")
    assert transform[i]["mode"] == "test" and transform[i]["test_single_fragment"]
    return Compose(transform[: i + 1]), Compose(transform[i + 1 :])


class TileDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, pre_transform, post_transform):
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.post_transform = post_transform

    def __len__(self):
        return len(self.dataset.data_list)

    def __getitem__(self, idx):
        data = self.dataset.get_data(idx)
        category = int(data["category"][0])
        fragments = self.pre_transform(data)
        assert len(fragments) == 1
        sample = self.post_transform(fragments[0])
        return sample, self.dataset.get_data_name(idx), category


def _collate(batch):
    samples, names, categories = zip(*batch)
    return collate_fn(list(samples)), list(names), list(categories)


def raw_point_counts(dataset):
    """Points per tile from the size of coord.npy (float32 (N, 3), 128-byte header): an upper
    bound of the voxelized count, used only to pack batches."""
    sizes = []
    for name in dataset.data_list:
        path = os.path.join(dataset.data_root, dataset.split, name, "coord.npy")
        sizes.append(max(1, (os.path.getsize(path) - 128) // 12))
    return sizes


@torch.no_grad()
def extract_split(model, dataset, transform, device, args):
    pre_transform, post_transform = split_transform(transform)
    loader_kwargs = dict(
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1
    if args.point_budget is not None:
        loader_kwargs["batch_sampler"] = pack_indices_by_voxel_budget(
            raw_point_counts(dataset),
            voxel_budget=args.point_budget,
            max_batch_size=args.batch_size,
        )
    else:
        loader_kwargs["batch_size"] = args.batch_size
        loader_kwargs["shuffle"] = False
    loader = torch.utils.data.DataLoader(
        TileDataset(dataset, pre_transform, post_transform), **loader_kwargs
    )

    names, categories, mean_feats, max_feats = [], [], [], []
    for input_dict, batch_names, batch_categories in tqdm(
        loader, desc=f"extract {dataset.split}"
    ):
        input_dict = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in input_dict.items()
        }
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            mean_feat, max_feat = model(input_dict)
        mean_feats.append(mean_feat.float().cpu().numpy().astype(np.float16))
        max_feats.append(max_feat.float().cpu().numpy().astype(np.float16))
        names.extend(batch_names)
        categories.extend(batch_categories)
    return dict(
        names=np.asarray(names),
        category=np.asarray(categories, dtype=np.int64),
        mean_feat=np.concatenate(mean_feats),
        max_feat=np.concatenate(max_feats),
    )


# ----------------------------------------------------------------------------- probe


def pooled_features(split, agg):
    mean_feat = split["mean_feat"].astype(np.float32)
    max_feat = split["max_feat"].astype(np.float32)
    if agg == "mean":
        return mean_feat
    if agg == "max":
        return max_feat
    if agg == "concat":
        return np.concatenate([mean_feat, max_feat], axis=1)
    if agg == "sum":
        return mean_feat + max_feat
    raise ValueError(f"Unknown aggregation {agg!r}")


def fit_logistic(x_train, y_train, num_classes, c_value, max_iter, seed):
    """Multinomial logistic regression, sklearn objective (intercept not penalized):
    minimize C * sum_i CE(y_i, W x_i + b) + 0.5 ||W||^2, full-batch L-BFGS."""
    torch.manual_seed(seed)
    linear = torch.nn.Linear(x_train.shape[1], num_classes).to(x_train.device)
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.zeros_(linear.bias)
    optimizer = torch.optim.LBFGS(
        linear.parameters(),
        lr=1.0,
        max_iter=max_iter,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = (
            c_value * F.cross_entropy(linear(x_train), y_train, reduction="sum")
            + 0.5 * linear.weight.pow(2).sum()
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return linear


def classification_metrics(target, pred, num_classes):
    intersection, union, target_hist = intersection_and_union(
        pred, target, num_classes, ignore_index=-1
    )
    f1, macro_f1 = f1_scores_from_hist(intersection, union, target_hist)
    return dict(
        allAcc=float((pred == target).mean()),
        mAcc=float(mean_acc_from_hist(intersection, target_hist, union=union)),
        mIoU=float(mean_iou_from_hist(intersection, union)),
        macro_f1=float(macro_f1),
        per_class_iou=(intersection / (union + 1e-10)).tolist(),
        per_class_f1=f1.tolist(),
    )


def run_probe(cfg, splits, device):
    from sklearn.preprocessing import StandardScaler

    num_classes = len(cfg.data.names)
    probe = cfg.probe
    grid, best = [], None
    for agg in probe.aggs:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(pooled_features(splits["train"], agg))
        x_val = scaler.transform(pooled_features(splits["val"], agg))
        x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
        x_val = torch.tensor(x_val, dtype=torch.float32, device=device)
        y_train = torch.tensor(splits["train"]["category"], device=device)
        for c_value in probe.Cs:
            linear = fit_logistic(
                x_train, y_train, num_classes, float(c_value), probe.max_iter, cfg.seed
            )
            with torch.no_grad():
                pred_val = linear(x_val).argmax(dim=1).cpu().numpy()
            val = classification_metrics(
                splits["val"]["category"], pred_val, num_classes
            )
            grid.append(dict(agg=agg, C=float(c_value), val_mIoU=val["mIoU"]))
            print(f"[probe] agg={agg:<6} C={c_value:<8g} val mIoU={val['mIoU']:.4f}")
            if best is None or val["mIoU"] > best["val"]["mIoU"]:
                best = dict(
                    agg=agg,
                    C=float(c_value),
                    val=val,
                    scaler=scaler,
                    weight=linear.weight.detach().cpu().double().numpy(),
                    bias=linear.bias.detach().cpu().double().numpy(),
                )

    x_test = best["scaler"].transform(pooled_features(splits["test"], best["agg"]))
    pred_test = np.argmax(x_test @ best["weight"].T + best["bias"], axis=1)
    test = classification_metrics(splits["test"]["category"], pred_test, num_classes)
    print(
        f"[probe] best agg={best['agg']} C={best['C']:g}: val mIoU={best['val']['mIoU']:.4f}"
        f"  test mIoU={test['mIoU']:.4f}"
    )
    metrics = dict(
        config=cfg.filename,
        weight=cfg.weight,
        class_names=list(cfg.data.names),
        grid=grid,
        best=dict(agg=best["agg"], C=best["C"], val=best["val"], test=test),
    )
    return metrics, pred_test


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    work_dir = args.work_dir or os.path.join(
        "exp", "pureforest", os.path.splitext(os.path.basename(args.config_file))[0]
    )
    os.makedirs(work_dir, exist_ok=True)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    splits, model = {}, None
    for split in SPLITS:
        path = os.path.join(work_dir, f"{split}.npz")
        if args.force_extract or not os.path.isfile(path):
            if model is None:
                model = build_extractor(cfg, device)
            seed_everything(cfg.seed)
            dataset = build_dataset(dict(cfg.data[split], transform=None))
            np.savez_compressed(
                path, **extract_split(model, dataset, cfg.transform, device, args)
            )
            print(f"[probe] wrote {path}")
        with np.load(path) as data:
            splits[split] = {k: data[k] for k in data.files}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metrics, pred_test = run_probe(cfg, splits, device)
    with open(os.path.join(work_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(
        os.path.join(work_dir, "test_predictions.npz"),
        names=splits["test"]["names"],
        category=splits["test"]["category"],
        pred=pred_test,
    )
    print(f"[probe] wrote {work_dir}/metrics.json")


if __name__ == "__main__":
    main()
