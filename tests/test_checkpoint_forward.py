"""Regression test: local checkpoints still load and give sensible predictions after code changes.

Run after touching models / datasets / configs / transforms (needs the real pointcept env, a GPU
and the local D067 val tiles + ckpt/malibu3d/*; skipped otherwise):

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest tests/test_checkpoint_forward.py -v

For each model (multitask LitePT-B / PTv3 / SpUNet / KPConvX from ckpt/malibu3d/*, and the
Sonata SSL outdoor backbone) it:
  1. builds the model from its config and loads the checkpoint with a STRICT report of
     missing / unexpected / shape-mismatched keys (CheckpointLoader only logs "Missing keys"
     with strict=False and silently ignores unexpected ones);
  2. runs one eval-mode forward on a centre crop (30k voxels) of two small val tiles;
  3. asserts predictions against the on-disk labels (thresholds well below the observed values,
     so only real breakage trips them).

Supervised multitask models: segment acc / mIoU, elevation MAE vs a constant prediction.
Sonata (SSL, no head): spatially-split kNN probe on the frozen features (checkerboard of 6 m
cells: fit on even cells, predict odd cells); must beat a random-init control.

CLI (dump predictions .npz + figures + summary.json, to eyeball the zone):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/test_checkpoint_forward.py \
      --models litept_b kpconvx --tile D067-2021_AU-S1-21_3-6:val --out-dir stats/ckpt_sanity

cuda:1 is broken for spconv on hecate -> always cuda:0.
"""

import argparse
import json
import os
import random
import time
from collections import OrderedDict

import numpy as np
import pytest
import torch

from pointcept.datasets import build_dataset
from pointcept.datasets.utils import collate_fn
from pointcept.models import build_model
from pointcept.utils.config import Config

MODELS = {
    "litept_b": ("configs/flair3d_default/multi-litept-b-v1m0-flair3d.py",
                 "ckpt/malibu3d/litept_b_multitask/model_best.pth", "supervised"),
    "ptv3": ("configs/flair3d_default/multi-ptv3-v1m0-flair3d.py",
             "ckpt/malibu3d/ptv3_multitask/model_best.pth", "supervised"),
    "spunet": ("configs/flair3d_default/multi-spunet-v1m0-flair3d.py",
               "ckpt/malibu3d/spunet_multitask/model_best.pth", "supervised"),
    "kpconvx": ("configs/flair3d_default/multi-kpconvx-v1m0-flair3d.py",
                "ckpt/malibu3d/kpconvx_multitask/model_best.pth", "supervised"),
    "sonata": ("configs/flair3d_default/probe/sonata-v1m2-flair3d-lin-grid.py",
               "ckpt/malibu3d/sonata_outdoor/epoch_120.pth", "ssl"),
}
# The malibu3d SpUNet ckpt was trained with stride=3 (3x3x3 down/up convs; downstream probe configs
# such as configs/experiment/w110/2/grid_seed_h3d/spunet-*.py set it) but multi-spunet-v1m0-flair3d.py
# leaves the default stride=2 -> 8 shape mismatches without this override.
BACKBONE_OVERRIDES = {"spunet": dict(stride=3)}
# Sonata SSL ckpt stores student/teacher; the probe config loads the student backbone.
SSL_KEY_REPLACE = ("module.student.backbone", "module.backbone")
MANIFESTS = {
    "val": "data/flair3d_plus/raw/scene_split_manifest_D067.csv",
    "test": "data/flair3d_plus/raw/scene_split_manifest_D075.csv",
    "train": "data/flair3d_plus/raw/scene_split_manifest_D067.csv",
}


def load_weights(model, path, ssl):
    """Strict-report load. Returns (info dict, checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    weight = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        if not k.startswith("module."):
            k = "module." + k
        if ssl:
            if k.startswith("module.teacher."):
                continue
            k = k.replace(SSL_KEY_REPLACE[0], SSL_KEY_REPLACE[1], 1)
        weight[k[len("module."):]] = v
    own = model.state_dict()
    shape_bad = [k for k, v in weight.items() if k in own and own[k].shape != v.shape]
    for k in shape_bad:
        weight.pop(k)
    res = model.load_state_dict(weight, strict=False)
    missing = list(res.missing_keys)
    unexpected = list(res.unexpected_keys)
    info = dict(
        n_ckpt_keys=len(weight) + len(shape_bad), n_model_keys=len(own),
        missing=missing, unexpected=unexpected, shape_mismatch=shape_bad,
        epoch=ckpt.get("epoch"), best_metric_value=ckpt.get("best_metric_value"),
    )
    return info


def build_input(cfg, tile, split, max_points, ds_cache):
    """Run the config's *val* pipeline on one tile and centre-crop to max_points."""
    key = (id(cfg), split)
    if key not in ds_cache:
        dcfg = dict(cfg.data.val)
        dcfg.pop("stratified_subset_manifest", None)
        dcfg.pop("max_sample", None)
        dcfg.update(split=split, csv_manifest=MANIFESTS[split], min_points=None)
        ds_cache[key] = build_dataset(dcfg)
    ds = ds_cache[key]
    names = [os.path.basename(p) for p in ds.data_list]
    if tile not in names:
        raise SystemExit(f"tile {tile} not in manifest for split {split}")
    item = ds[names.index(tile)]
    n = item["coord"].shape[0]
    if n > max_points:
        c = item["coord"][:, :2]
        d = ((c - c.mean(0, keepdim=True)) ** 2).sum(1)
        keep = torch.argsort(d)[:max_points].sort().values
        for k, v in list(item.items()):
            if k.startswith("origin_") or k == "inverse":
                item.pop(k)
            elif torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == n:
                item[k] = v[keep]
        item["offset"] = torch.tensor([keep.numel()])
    item = collate_fn([item])
    return {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in item.items()}


def seg_metrics(pred, gt, ignore, num_classes):
    m = gt != ignore
    p, g = pred[m], gt[m]
    acc = float((p == g).mean())
    vals, cnt = np.unique(g, return_counts=True)
    majority = float(cnt.max() / cnt.sum())
    ious = {}
    for c in vals:
        inter = float(((p == c) & (g == c)).sum())
        union = float(((p == c) | (g == c)).sum())
        ious[int(c)] = inter / union if union else float("nan")
    return dict(acc=acc, majority_baseline=majority, miou_present=float(np.nanmean(list(ious.values()))),
                iou_per_class=ious, n_valid=int(m.sum()))


@torch.no_grad()
def run_supervised(cfg, model, inp):
    out = model(inp)
    ign = int(cfg.data.ignore_index)
    nc = int(cfg.data.num_classes)
    res, dump = {}, {}
    seg_pred = out["seg_logits"].argmax(1).cpu().numpy()
    seg_gt = inp["segment"].cpu().numpy()
    res["segment"] = seg_metrics(seg_pred, seg_gt, ign, nc)
    res["segment"]["pred_hist"] = np.bincount(seg_pred, minlength=nc).tolist()
    res["finite"] = bool(all(torch.isfinite(v).all() for v in out["seg_logits_by_task"].values()))
    dump.update(coord=inp["coord"].cpu().numpy(), seg_gt=seg_gt, seg_pred=seg_pred)
    for name, logits in out["seg_logits_by_task"].items():
        tc = cfg.data.task_configs[name]
        if name == cfg.data.main_task or name not in inp or inp[name].shape[0] != logits.shape[0]:
            continue
        if tc.get("task_type") != "semantic":
            continue
        g = inp[name].cpu().numpy()
        p = logits.argmax(1).cpu().numpy()
        res[name] = seg_metrics(p, g, int(tc["ignore_index"]), int(tc["num_classes"]))
        res[name].pop("iou_per_class")
    for name, pred in out["reg_pred_by_task"].items():
        p = pred.float().cpu().numpy().reshape(-1)
        res[f"{name}_pred_stats"] = dict(mean=float(p.mean()), std=float(p.std()),
                                         min=float(p.min()), max=float(p.max()))
        if name in inp and inp[name].shape[0] == p.shape[0]:
            g = inp[name].float().cpu().numpy().reshape(-1)
            ok = np.isfinite(g) & (g > -1e4)
            res[name] = dict(mae=float(np.abs(p[ok] - g[ok]).mean()),
                             gt_mean=float(g[ok].mean()), gt_std=float(g[ok].std()),
                             const_pred_mae=float(np.abs(g[ok] - np.median(g[ok])).mean()))
            dump[f"{name}_gt"], dump[f"{name}_pred"] = g, p
    return res, dump


CELL = 6.0
MIN_CLASS_TEST_PTS = 200


@torch.no_grad()
def run_ssl(cfg, model, inp, k=20):
    feat, _ = model._forward_backbone(inp)
    feat = feat.float()
    res = dict(feat_dim=int(feat.shape[1]), finite=bool(torch.isfinite(feat).all()),
               feat_std=float(feat.std()), feat_mean_abs=float(feat.abs().mean()))
    coord = inp["coord"]
    gt = inp["segment"]
    ign = int(cfg.data.ignore_index)
    f = torch.nn.functional.normalize(feat - feat.mean(0, keepdim=True), dim=1)
    # checkerboard of CELL-m blocks: fit on even cells, predict odd cells (no left/right split --
    # half-tiles can be single-class). Only cell borders leak spatially, and the random-init
    # control absorbs any residual coordinate-driven signal.
    cell = torch.floor((coord[:, :2] - coord[:, :2].min(0).values) / CELL).long()
    even = (cell.sum(1) % 2) == 0
    tr = even & (gt != ign)
    te = (~even) & (gt != ign)
    sim = f[te] @ f[tr].T
    idx = sim.topk(k, dim=1).indices
    votes = torch.nn.functional.one_hot(gt[tr][idx], int(cfg.data.num_classes)).sum(1)
    pred = votes.argmax(1).cpu().numpy()
    g = gt[te].cpu().numpy()
    m = seg_metrics(pred, g, ign, int(cfg.data.num_classes))
    res["knn_split"] = m
    bal = [float((pred[g == c] == c).mean()) for c in np.unique(g)]
    m["balanced_acc"] = float(np.mean(bal))
    # Same, restricted to classes with >= MIN_CLASS_TEST_PTS test points: a class with a few dozen
    # points flips its recall from ~0.2 to ~0.7 depending on which point GridSample draws per voxel,
    # i.e. +-0.1 on the mean of ~5 classes, which makes the plain balanced_acc a coin flip.
    n_per_class = np.array([int((g == c).sum()) for c in np.unique(g)])
    m["balanced_acc_robust"] = float(np.mean(np.array(bal)[n_per_class >= MIN_CLASS_TEST_PTS]))
    # PCA-RGB of features for the figure
    centered = feat - feat.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(centered, q=3, center=False)
    pca = (centered @ V[:, :3]).cpu().numpy()
    pca = (pca - np.percentile(pca, 2, axis=0)) / (np.percentile(pca, 98, axis=0) - np.percentile(pca, 2, axis=0) + 1e-9)
    dump = dict(coord=coord.cpu().numpy(), seg_gt=gt.cpu().numpy(), pca=np.clip(pca, 0, 1),
                knn_test_mask=te.cpu().numpy(), knn_pred=pred)
    return res, dump


def figure(path, title, kind, dump, ign, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    xy = dump["coord"][:, :2]

    def cls_rgb(a):
        out = np.array([cmap(int(v) % 20)[:3] for v in a])
        out[a == ign] = (0.85, 0.85, 0.85)
        return out

    if kind == "supervised":
        panels = [("GT segment", cls_rgb(dump["seg_gt"])), ("pred segment", cls_rgb(dump["seg_pred"])),
                  ("error", np.where((dump["seg_pred"] != dump["seg_gt"])[:, None] & (dump["seg_gt"] != ign)[:, None],
                                     [[0.9, 0.1, 0.1]], [[0.8, 0.8, 0.8]]))]
        if "elevation_pred" in dump:
            panels.append(("elevation pred (m)", dump["elevation_pred"]))
    else:
        te = dump["knn_test_mask"]
        pred_full = np.full(len(xy), ign)
        pred_full[te] = dump["knn_pred"]
        panels = [("GT segment", cls_rgb(dump["seg_gt"])), ("feature PCA-RGB", dump["pca"]),
                  ("kNN pred (odd cells)", cls_rgb(pred_full))]
    fig, axs = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
    for ax, (t, c) in zip(np.atleast_1d(axs), panels):
        if c.ndim == 1:
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, s=0.4, cmap="viridis")
            plt.colorbar(sc, ax=ax, fraction=0.046)
        else:
            ax.scatter(xy[:, 0], xy[:, 1], c=c, s=0.4)
        ax.set_title(t, fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


DEFAULT_TILES = [("D067-2021_AU-S1-21_3-6", "val"), ("D067-2021_AN-S1-15_1-7", "val")]
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SEED = 0


def _seed_all(seed):
    """Seed every RNG the data pipeline may touch (torch, numpy, random)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_model(mname, tiles, max_points=30000, control=False, out_dir=None, ds_cache=None):
    """Load + forward one model on `tiles`; returns the summary dict (and dumps if out_dir)."""
    ds_cache = {} if ds_cache is None else ds_cache
    cfg_path, ckpt_path, kind = MODELS[mname]
    cfg = Config.fromfile(cfg_path)
    cfg.model.backbone.update(BACKBONE_OVERRIDES.get(mname, {}))
    names = list(cfg.data.names)
    ign = int(cfg.data.ignore_index)
    summary = {}
    for variant in ["trained"] + (["random_init"] if control else []):
        _seed_all(SEED)
        model = build_model(cfg.model).cuda()
        if variant == "trained":
            summary["load"] = load_info = load_weights(model, ckpt_path, ssl=(kind == "ssl"))
            print(f"[{mname}] ckpt epoch={load_info['epoch']} best={load_info['best_metric_value']} "
                  f"missing={len(load_info['missing'])} unexpected={len(load_info['unexpected'])} "
                  f"shape_mismatch={len(load_info['shape_mismatch'])}")
        model.eval()
        for tile, split in tiles:
            t0 = time.time()
            _seed_all(SEED)  # the val pipeline's GridSample(mode="train") draws np.random points per voxel
            inp = build_input(cfg, tile, split, max_points, ds_cache)
            n = int(inp["coord"].shape[0])
            res, dump = (run_supervised if kind == "supervised" else run_ssl)(cfg, model, inp)
            res["n_points"], res["seconds"] = n, round(time.time() - t0, 2)
            summary.setdefault(variant, {})[tile] = res
            headline = res["segment"] if kind == "supervised" else res["knn_split"]
            print(f"[{mname}/{variant}] {tile} n={n} acc={headline['acc']:.3f} "
                  f"mIoU(present)={headline['miou_present']:.3f} majority={headline['majority_baseline']:.3f}"
                  + (f" balAcc={headline['balanced_acc']:.3f}" if "balanced_acc" in headline else "")
                  + (f" elevMAE={res['elevation']['mae']:.2f}m (const {res['elevation']['const_pred_mae']:.2f}m)"
                     if "elevation" in res else ""))
            if variant == "trained" and out_dir:
                np.savez_compressed(os.path.join(out_dir, f"{mname}__{tile}.npz"), **dump)
                figure(os.path.join(out_dir, f"{mname}__{tile}.png"), f"{mname} | {tile} | {n} pts",
                       kind, dump, ign, names)
        del model
        torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--tile", action="append", default=None, help="<tile_name>:<split>")
    ap.add_argument("--max-points", type=int, default=30000)
    ap.add_argument("--out-dir", default="stats/ckpt_sanity")
    ap.add_argument("--control", action="store_true", help="also run the random-init control")
    args = ap.parse_args()
    tiles = [tuple(t.split(":")) for t in args.tile] if args.tile else DEFAULT_TILES
    os.makedirs(args.out_dir, exist_ok=True)
    ds_cache, summary = {}, {}
    for mname in args.models:
        summary[mname] = run_model(mname, tiles, args.max_points, args.control, args.out_dir, ds_cache)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote", os.path.join(args.out_dir, "summary.json"))


# ----------------------------------------------------------------------------- pytest
# Thresholds sit well below the values observed on 2026-10-09 (acc >= 0.83, mIoU(present) >= 0.55,
# Sonata balanced acc >= 0.89) so only real breakage trips them, not retraining noise.
SEG_ACC_MIN = 0.75
SEG_MIOU_MIN = 0.40
# Sonata thresholds, on balanced_acc_robust (classes with >= MIN_CLASS_TEST_PTS test points).
# Measured over 24 seeds of the data draw (2026-10-09): trained 0.903-0.933, random-init control
# 0.835-0.901, trained - random >= 0.027 (mean 0.04-0.05). The test is also seeded (SEED), so it is
# deterministic; the thresholds still leave room for the seed-to-seed spread above.
SONATA_BALACC_MIN = 0.85
SONATA_MARGIN_OVER_RANDOM = 0.01
SSL_EXPECTED_UNEXPECTED = ("student.mask_head", "student.unmask_head", "backbone.embedding.mask_token")

_RESULTS = {}


def _require_env(mname):
    pytest.importorskip("spconv")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    _, ckpt_path, _ = MODELS[mname]
    ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
    if not os.path.isfile(ckpt_path):
        pytest.skip(f"checkpoint not available locally: {ckpt_path}")
    for tile, split in DEFAULT_TILES:
        dept, roi = tile.split("_")[0], tile.split("_")[1]
        if not os.path.isdir(os.path.join(REPO_ROOT, "data/flair3d_plus", split, f"{dept}_LIDARHD", roi, tile)):
            pytest.skip(f"tile not mirrored locally: {tile}")


@pytest.fixture(scope="module")
def ds_cache():
    return {}


def _result(mname, ds_cache):
    _require_env(mname)
    if mname not in _RESULTS:
        cwd = os.getcwd()
        os.chdir(REPO_ROOT)  # configs / manifests / ckpt paths are repo-relative
        try:
            _RESULTS[mname] = run_model(mname, DEFAULT_TILES, control=(MODELS[mname][2] == "ssl"),
                                        ds_cache=ds_cache)
        finally:
            os.chdir(cwd)
    return _RESULTS[mname]


@pytest.mark.parametrize("mname", [m for m, v in MODELS.items() if v[2] == "supervised"])
def test_supervised_checkpoint_forward(mname, ds_cache):
    r = _result(mname, ds_cache)
    load = r["load"]
    assert not load["missing"], f"{mname}: checkpoint keys missing in model: {load['missing'][:5]}"
    assert not load["unexpected"], f"{mname}: unexpected checkpoint keys: {load['unexpected'][:5]}"
    assert not load["shape_mismatch"], f"{mname}: shape mismatch (config != checkpoint): {load['shape_mismatch'][:5]}"
    for tile, res in r["trained"].items():
        assert res["finite"], f"{mname}/{tile}: non-finite logits"
        seg = res["segment"]
        assert seg["acc"] >= SEG_ACC_MIN, f"{mname}/{tile}: segment acc {seg['acc']:.3f} < {SEG_ACC_MIN}"
        assert seg["miou_present"] >= SEG_MIOU_MIN, f"{mname}/{tile}: mIoU {seg['miou_present']:.3f} < {SEG_MIOU_MIN}"
    # elevation: on the tile with real relief the net must clearly beat a constant prediction
    tile1 = DEFAULT_TILES[0][0]
    elev = r["trained"][tile1]["elevation"]
    assert elev["mae"] < 0.5 * elev["const_pred_mae"], f"{mname}/{tile1}: elevation no better than constant: {elev}"


def test_sonata_backbone_checkpoint_forward(ds_cache):
    r = _result("sonata", ds_cache)
    load = r["load"]
    assert not [k for k in load["missing"] if k.startswith("backbone.")], "Sonata backbone keys missing"
    assert not load["shape_mismatch"], f"Sonata shape mismatch: {load['shape_mismatch'][:5]}"
    bad = [k for k in load["unexpected"] if not any(k.startswith(p) or k == p for p in SSL_EXPECTED_UNEXPECTED)]
    assert not bad, f"Sonata: unexpected keys beyond SSL heads / mask_token: {bad[:5]}"
    for tile, _ in DEFAULT_TILES:
        trained = r["trained"][tile]
        assert trained["finite"], f"sonata/{tile}: non-finite features"
        bal = trained["knn_split"]["balanced_acc_robust"]
        rand = r["random_init"][tile]["knn_split"]["balanced_acc_robust"]
        assert bal >= SONATA_BALACC_MIN, f"sonata/{tile}: kNN balanced acc {bal:.3f} < {SONATA_BALACC_MIN}"
        assert bal >= rand + SONATA_MARGIN_OVER_RANDOM, (
            f"sonata/{tile}: trained features ({bal:.3f}) not better than random init ({rand:.3f})"
        )


if __name__ == "__main__":
    main()
