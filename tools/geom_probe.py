"""
Geometric linear probe of frozen point features (hybrid-SSL diagnostics).

Fits a closed-form ridge regression from the frozen per-point features of a probe config's
backbone to the point normals (from the ShapeNetPart files, even when they are not an input) and
to the height z, on training shapes, and reports on test shapes:
- normals: mean cosine and mean |cosine| (orientation-free) between predicted and true normals;
- height: R^2 and mean absolute error.
High geometric scores together with a low part-segmentation probe point to features that encode
local geometry (the shortcut the masked points leave visible) rather than parts.

Usage (from the repository root):
    PYTHONPATH=. python tools/geom_probe.py \
        --config-file configs/shapenet_part/probe-lin-litept-s-v2m0-shapenet-part.py \
        --weight <sonata checkpoint> --save-path <dir>
Without --weight the backbone keeps its random initialisation (baseline).
"""

import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import torch

from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.config import Config


def ridge_fit(x, y, lam):
    """Ridge regression with a bias, on standardised features (float64, on the
    device of x)."""
    x, y = x.double(), y.double()
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    xs = torch.cat([(x - mean) / std, torch.ones_like(x[:, :1])], 1)
    eye = torch.eye(xs.shape[1], dtype=xs.dtype, device=xs.device)
    eye[-1, -1] = 0  # no penalty on the bias
    w = torch.linalg.solve(xs.T @ xs + lam * xs.shape[0] * eye, xs.T @ y)
    return (
        lambda z: torch.cat([(z.double() - mean) / std, torch.ones_like(z[:, :1])], 1)
        @ w
    )


def geom_metrics(pred, normal, height):
    """pred: [N, 4] (normal xyz, height); normal: [N, 3]; height: [N]."""
    cos = torch.nn.functional.cosine_similarity(pred[:, :3], normal, dim=1)
    residual = ((pred[:, 3] - height) ** 2).sum()
    total = ((height - height.mean()) ** 2).sum()
    return dict(
        normal_cos=float(cos.mean()),
        normal_abs_cos=float(cos.abs().mean()),
        height_r2=float(1 - residual / total),
        height_mae=float((pred[:, 3] - height).abs().mean()),
    )


def load_backbone_weight(model, weight, keyword):
    checkpoint = torch.load(weight, map_location="cuda", weights_only=False)
    state = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        key = key if key.startswith("module.") else "module." + key
        key = key.replace(keyword, "module.backbone", 1)
        state[key[len("module.") :]] = value
    info = model.load_state_dict(state, strict=False)
    return info.missing_keys


@torch.no_grad()
def extract(model, dataset, num_shapes, points_per_shape, batch_size, rng):
    index = rng.choice(len(dataset), size=min(num_shapes, len(dataset)), replace=False)
    feats, normals, heights = [], [], []
    for start in range(0, len(index), batch_size):
        batch = collate_fn([dataset[i] for i in index[start : start + batch_size]])
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            point = model(batch, return_point=True)["point"]
        feat = point.feat.float()
        assert feat.shape[0] == batch["coord"].shape[0]
        begin = 0
        for end in batch["offset"].tolist():
            pick = begin + torch.randperm(end - begin, device="cuda")[:points_per_shape]
            # kept on the CPU: ~4 GB of features for 2000 shapes, next to a training run
            feats.append(feat[pick].cpu())
            normals.append(batch["normal"][pick].double().cpu())
            heights.append(batch["coord"][pick, 2].double().cpu())
            begin = end
    return torch.cat(feats), torch.cat(normals), torch.cat(heights)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weight", default=None)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--num-train-shapes", type=int, default=2000)
    parser.add_argument("--num-test-shapes", type=int, default=1000)
    parser.add_argument("--points-per-shape", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = build_model(cfg.model).cuda().eval()
    missing = []
    if args.weight:
        keyword = f"module.{cfg.get('probe_branch', 'student')}.backbone"
        missing = load_backbone_weight(model, args.weight, keyword)
        assert all(k.startswith("seg_head") for k in missing), missing

    grid_size = cfg.grid_size
    transform = [
        dict(type="NormalizeCoord"),
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect", keys=("coord", "grid_coord", "normal"), feat_keys=("coord",)
        ),
    ]
    data_root = cfg.data.train.data_root
    datasets = {
        split: build_dataset(
            dict(
                type="ShapeNetPartDataset",
                split=splits,
                data_root=data_root,
                transform=transform,
            )
        )
        for split, splits in (("train", ["train", "val"]), ("test", "test"))
    }
    x, n, h = extract(
        model,
        datasets["train"],
        args.num_train_shapes,
        args.points_per_shape,
        32,
        rng,
    )
    predict = ridge_fit(x, torch.cat([n, h[:, None]], 1), args.ridge)
    metrics = dict(train=geom_metrics(predict(x), n, h))
    x, n, h = extract(
        model,
        datasets["test"],
        args.num_test_shapes,
        args.points_per_shape,
        32,
        rng,
    )
    metrics["test"] = geom_metrics(predict(x), n, h)
    metrics.update(
        weight=args.weight,
        config=args.config_file,
        feature_dim=int(x.shape[1]),
        missing_keys=missing,
    )
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "geom_probe.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
