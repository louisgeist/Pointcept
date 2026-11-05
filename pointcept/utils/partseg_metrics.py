"""
ShapeNetPart metrics, shared by ShapeNetPartSegEvaluator, ShapeNetPartSegTester and
tools/eval_partseg.py.

Main protocol (hybrid-SSL): semantic segmentation over the 50 parts. Argmax over the 50 classes
without category masking, a 50 x 50 confusion accumulated over every original point of the split
(predictions mapped back through the voxelisation inverse), mIoU = mean of the 50 IoUs, plus mAcc
and allAcc.

Secondary (logged, not the selection metric): the per-shape ShapeNetPart protocol, instance mIoU
(mean over shapes) and category mIoU (mean over the 16 categories of the per-shape mIoU), with the
plain argmax and with the argmax restricted to the parts of the shape's category (``*_catmask``).
"""

import numpy as np
import torch
import torch.distributed as dist

from pointcept.utils.misc import masked_part_argmax, shape_part_miou


class PartSegMetrics:
    def __init__(self, categories, category2part, num_classes=50):
        self.categories = list(categories)
        self.category2part = category2part
        self.num_classes = num_classes
        self.intersection = np.zeros(num_classes, dtype=np.float64)
        self.union = np.zeros(num_classes, dtype=np.float64)
        self.target = np.zeros(num_classes, dtype=np.float64)
        # row 0: plain argmax, row 1: argmax restricted to the category's parts
        self.shape_iou_sum = np.zeros((2, len(self.categories)), dtype=np.float64)
        self.shape_count = np.zeros(len(self.categories), dtype=np.float64)

    def update(self, logits, target, cls_token):
        """One shape: logits (or probabilities) [N, num_classes] and labels [N] of its
        original points, cls_token its category index."""
        target = target.long()
        pred = logits.argmax(dim=1)
        confusion = torch.bincount(
            target * self.num_classes + pred, minlength=self.num_classes**2
        ).view(self.num_classes, self.num_classes)
        confusion = confusion.double().cpu().numpy()  # rows: target, columns: pred
        intersection = np.diag(confusion)
        self.intersection += intersection
        self.union += confusion.sum(0) + confusion.sum(1) - intersection
        self.target += confusion.sum(1)
        parts = self.category2part[self.categories[cls_token]]
        target_np = target.cpu().numpy()
        for row, p in enumerate((pred, masked_part_argmax(logits, parts))):
            self.shape_iou_sum[row, cls_token] += shape_part_miou(
                p.cpu().numpy(), target_np, parts
            )
        self.shape_count[cls_token] += 1

    def sync(self):
        """Sum the accumulators over the distributed processes."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        if dist.get_world_size() == 1:
            return
        flat = np.concatenate(
            [
                self.intersection,
                self.union,
                self.target,
                self.shape_iou_sum.ravel(),
                self.shape_count,
            ]
        )
        flat = torch.tensor(flat, device="cuda", dtype=torch.float64)
        dist.all_reduce(flat)
        flat = flat.cpu().numpy()
        c, k = self.num_classes, len(self.categories)
        self.intersection, self.union, self.target = (
            flat[:c],
            flat[c : 2 * c],
            flat[2 * c : 3 * c],
        )
        self.shape_iou_sum = flat[3 * c : 3 * c + 2 * k].reshape(2, k)
        self.shape_count = flat[3 * c + 2 * k :]

    def compute(self):
        iou_class = self.intersection / (self.union + 1e-10)
        acc_class = self.intersection / (self.target + 1e-10)
        present = self.target > 0
        seen = self.shape_count > 0
        ins = self.shape_iou_sum.sum(1) / (self.shape_count.sum() + 1e-10)
        per_cat = self.shape_iou_sum / (self.shape_count + 1e-10)
        cat = per_cat[:, seen].mean(1) if seen.any() else np.zeros(2)
        return dict(
            mIoU=float(iou_class.mean()),
            mAcc=float(acc_class[present].mean()) if present.any() else 0.0,
            allAcc=float(self.intersection.sum() / (self.target.sum() + 1e-10)),
            ins_mIoU=float(ins[0]),
            cat_mIoU=float(cat[0]),
            ins_mIoU_catmask=float(ins[1]),
            cat_mIoU_catmask=float(cat[1]),
            iou_class=iou_class,
            acc_class=acc_class,
            shape_iou_per_category=per_cat[0],
            shape_count=self.shape_count.copy(),
        )
