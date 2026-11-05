"""
Evaluate a trained ShapeNetPart segmentor (supervised model or linear probe) with the metrics of
pointcept/utils/partseg_metrics.py: mIoU / mAcc / allAcc over the 50 parts at the original points
(main), and the per-shape instance / category mIoU (secondary).

Runs the validation pipeline of the run's own config (data.val) on the test split, with the
weights of a checkpoint saved by that run (e.g. model/model_last.pth), and writes the metrics to
--save-path/partseg_metrics.json.

Usage (from the repository root):
    PYTHONPATH=. python tools/eval_partseg.py --config-file <run>/config.py \
        --weight <run>/model/model_last.pth --save-path <dir>
"""

import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import torch

from pointcept.datasets import build_dataset, collate_fn
from pointcept.engines.hooks.evaluator import update_partseg_metrics
from pointcept.models import build_model
from pointcept.utils.config import Config
from pointcept.utils.partseg_metrics import PartSegMetrics


@torch.no_grad()
def evaluate(model, dataset, num_classes, batch_size=32, num_workers=8):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    meter = PartSegMetrics(dataset.categories, dataset.category2part, num_classes)
    for input_dict in loader:
        input_dict = {
            k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
            for k, v in input_dict.items()
        }
        update_partseg_metrics(meter, model(input_dict)["seg_logits"], input_dict)
    return meter.compute()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)
    torch.manual_seed(args.seed)
    np.random.seed(
        args.seed
    )  # the val GridSample(mode="train") draws one point per voxel
    model = build_model(cfg.model).cuda().eval()
    checkpoint = torch.load(args.weight, map_location="cuda", weights_only=False)
    state = OrderedDict(
        (k[len("module.") :] if k.startswith("module.") else k, v)
        for k, v in checkpoint["state_dict"].items()
    )
    model.load_state_dict(state, strict=True)
    metrics = evaluate(model, build_dataset(cfg.data.val), cfg.data.num_classes)
    out = {k: v for k, v in metrics.items() if isinstance(v, float)}
    out.update(
        iou_class=metrics["iou_class"].tolist(),
        weight=args.weight,
        config=args.config_file,
        epoch=checkpoint.get("epoch"),
    )
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "partseg_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "iou_class"}, indent=2))


if __name__ == "__main__":
    main()
