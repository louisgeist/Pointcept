"""Run one released MALiBU3D backbone on one preprocessed tile and print what it predicts.

Weights come from the Hugging Face repo (``hf auth login`` first while it is private) or from a local file:

    export PYTHONPATH=$PWD
    python scripts/hf_release/forward_example.py --model litept-b --tile <tile_name> --split val
    python scripts/hf_release/forward_example.py --model ptv3 --weight ckpt_cleaned/ptv3/model.pth --tile <tile_name>
    python scripts/hf_release/forward_example.py --model sonata-pretrained --tile <tile_name>   # features only

Multitask models (sonata-ft, litept-b, ptv3, spunet, kpconvx) print, per task, the predicted classes /
regression statistics, and the accuracy / mIoU against the tile's labels for ``segment``. ``sonata-pretrained``
has no task head: it prints the shape and statistics of the backbone features. ``--save`` dumps the
predictions to a compressed ``.npz``. A single GPU (cuda:0) is required (spconv).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from pointcept.datasets import build_dataset
from pointcept.datasets.utils import collate_fn
from pointcept.models import build_model
from pointcept.utils.config import Config

REPO = Path(__file__).resolve().parents[2]
HF_REPO = "LouisGeist/MALiBU3D-backbones"
CONFIGS = {  # released folder -> default config of this repository
    "sonata-ft": "configs/flair3d_default/multi-sonata-ft-v1m0-flair3d.py",
    "litept-b": "configs/flair3d_default/multi-litept-b-v1m0-flair3d.py",
    "ptv3": "configs/flair3d_default/multi-ptv3-v1m0-flair3d.py",
    "spunet": "configs/flair3d_default/multi-spunet-v1m0-flair3d.py",
    "kpconvx": "configs/flair3d_default/multi-kpconvx-v1m0-flair3d.py",
    "sonata-pretrained": "configs/flair3d_default/probe/sonata-v1m2-flair3d-lin-grid.py",
}


def load_model(name, weight, cfg):
    model = build_model(cfg.model).cuda()
    sd = torch.load(weight, map_location="cpu", weights_only=True)["state_dict"]
    if name == "sonata-pretrained":
        # backbone only: the probe heads of the GridProbe model stay randomly initialised, and
        # embedding.mask_token belongs to the masked-SSL objective, so it is ignored here
        res = model.load_state_dict(sd, strict=False)
        assert all(k.startswith("heads.") for k in res.missing_keys), res.missing_keys
        assert set(res.unexpected_keys) <= {"backbone.embedding.mask_token"}, res.unexpected_keys
    else:
        model.load_state_dict(sd, strict=True)
    return model.eval()


def build_input(cfg, tile, split, csv_manifest, max_points):
    """Run the config's val pipeline on one tile; optionally keep the `max_points` points nearest its centre."""
    dcfg = dict(cfg.data.val)
    dcfg.pop("stratified_subset_manifest", None)
    dcfg.pop("max_sample", None)
    dcfg.update(split=split, min_points=None)
    if csv_manifest:
        dcfg["csv_manifest"] = csv_manifest
    ds = build_dataset(dcfg)
    names = [os.path.basename(p) for p in ds.data_list]
    if tile not in names:
        sys.exit(f"tile {tile!r} not found in split {split!r} of {dcfg['csv_manifest']} "
                 f"({len(names)} tiles, e.g. {names[:3]})")
    item = ds[names.index(tile)]
    n = item["coord"].shape[0]
    if max_points and n > max_points:
        d = ((item["coord"][:, :2] - item["coord"][:, :2].mean(0, keepdim=True)) ** 2).sum(1)
        keep = torch.argsort(d)[:max_points].sort().values
        for k, v in list(item.items()):
            if k.startswith("origin_") or k == "inverse":
                item.pop(k)
            elif torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == n:
                item[k] = v[keep]
        item["offset"] = torch.tensor([keep.numel()])
    item = collate_fn([item])
    return {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in item.items()}


def seg_scores(pred, gt, ignore):
    m = gt != ignore
    p, g = pred[m], gt[m]
    ious = []
    for c in np.unique(g):
        inter, union = ((p == c) & (g == c)).sum(), ((p == c) | (g == c)).sum()
        ious.append(inter / union)
    return float((p == g).mean()), float(np.mean(ious))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(CONFIGS))
    ap.add_argument("--weight", help="local model.pth; default: download <model>/model.pth from the HF repo")
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--revision", default=None, help="HF branch / tag / commit")
    ap.add_argument("--tile", required=True, help="tile directory name, as listed in the manifest")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--csv-manifest", default=None, help="override the config's tile table (tiles.csv)")
    ap.add_argument("--max-points", type=int, default=0, help="keep only the N points nearest the tile centre (0 = all)")
    ap.add_argument("--save", default=None, help="write predictions to this .npz")
    args = ap.parse_args()

    weight = args.weight
    if weight is None:
        from huggingface_hub import hf_hub_download
        weight = hf_hub_download(args.hf_repo, f"{args.model}/model.pth", revision=args.revision)
        print("downloaded", weight)

    cfg = Config.fromfile(str(REPO / CONFIGS[args.model]))
    model = load_model(args.model, weight, cfg)
    inp = build_input(cfg, args.tile, args.split, args.csv_manifest, args.max_points)
    n = int(inp["coord"].shape[0])
    print(f"{args.model}: {n} points after voxelisation, {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params")

    dump = {"coord": inp["coord"].cpu().numpy()}
    if args.model == "sonata-pretrained":
        feat, _ = model._forward_backbone(inp)
        feat = feat.float()
        print(f"features: shape={tuple(feat.shape)} mean={feat.mean():.3f} std={feat.std():.3f} "
              f"finite={bool(torch.isfinite(feat).all())}")
        dump["feat"] = feat.cpu().numpy()
    else:
        out = model(inp)
        for name, logits in out["seg_logits_by_task"].items():
            pred = logits.argmax(1).cpu().numpy()
            dump[f"{name}_pred"] = pred
            line = f"{name:26s} classes predicted: {len(np.unique(pred))}"
            if name == cfg.data.main_task and "segment" in inp:
                acc, miou = seg_scores(pred, inp["segment"].cpu().numpy(), int(cfg.data.ignore_index))
                line += f" | acc={acc:.3f} mIoU(present classes)={miou:.3f}"
            print(line)
        for name, pred in out["reg_pred_by_task"].items():
            p = pred.float().cpu().numpy().reshape(-1)
            dump[f"{name}_pred"] = p
            print(f"{name:26s} mean={p.mean():.2f} std={p.std():.2f} min={p.min():.2f} max={p.max():.2f}")
    if args.save:
        np.savez_compressed(args.save, **dump)
        print("saved", args.save)


if __name__ == "__main__":
    main()
