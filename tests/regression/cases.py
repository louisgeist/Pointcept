"""Regression cases for the release cleanup (usage: tests/regression/harness.py).

Outputs are keyed by case *name*, never by path: when a config or checkpoint moves (cleanup, rename),
update its path here and keep the name, so the new capture still lines up with the reference.
"""

import glob

# ----------------------------------------------------------------------------- stage "build"
# Every config matched here is parsed, its model built (seeded init) and its datasets instantiated.
BUILD_GLOBS = [
    "configs/_base_/**/*.py",
    "configs/flair3d_default/**/*.py",
    "configs/flair3d/*.py",
    "configs/dales/*.py",
    "configs/h3d/*.py",
    "configs/eclair/**/*.py",
    "configs/opengf/*.py",
    "configs/pureforest/*.py",
    "configs/experiment/**/*.py",
]


def build_configs():
    out = set()
    for g in BUILD_GLOBS:
        out.update(p for p in glob.glob(g, recursive=True) if "__pycache__" not in p)
    return sorted(out)


# ----------------------------------------------------------------------------- stage "forward"
CKPT = {
    "litept-b": "ckpt_cleaned/litept-b/model.pth",
    "ptv3": "ckpt_cleaned/ptv3/model.pth",
    "spunet": "ckpt_cleaned/spunet/model.pth",
    "kpconvx": "ckpt_cleaned/kpconvx/model.pth",
    "sonata-ft": "ckpt_cleaned/sonata-ft/model.pth",
    "sonata-pretrained": "ckpt_cleaned/sonata-pretrained/model.pth",
}
# Jean Zay job id (in the `weight=` path of the probe configs) -> released checkpoint.
JOB_TO_CKPT = {
    "873542": "litept-b",
    "1095469": "ptv3",
    "1052217": "spunet",
    "1159986": "kpconvx",
    "862680": "sonata-pretrained",
    "litept_b_multitask": "litept-b",
    "ptv3_multitask": "ptv3",
    "spunet_multitask": "spunet",
    "kpconvx_multitask": "kpconvx",
    "sonata_outdoor": "sonata-pretrained",
}
# Probe configs whose `weight` is unset (filled in at launch time).
PROBE_CKPT_OVERRIDE = {
    "flair3d_default/probe/sonata-v1m2-flair3d-lin-grid": "sonata-pretrained",
    "flair3d_default/probe/sonata-v1m2-flair3d-lin-grid-wide": "sonata-pretrained",
    "flair3d_default/probe/sonata-v1m2-flair3d-lin-grid-toy": "sonata-pretrained",
    "flair3d_default/probe/sonata-v1m2-flair3d-lin": "sonata-pretrained",
}

# Local tiles. D075/D068/D073 lack forest_2d.npy on hecate, so everything uses D067.
FLAIR3D_MANIFEST = {
    "train": "data/flair3d_plus/raw/tiles_D067.csv",
    "val": "data/flair3d_plus/raw/tiles_D067.csv",
}
FLAIR3D_TILES = [
    ("D067-2021_AU-S1-21_3-6", "val"),
    ("D067-2021_AN-S1-15_1-7", "val"),
    ("D067-2021_FF-S1-29_1-2", "val"),
]

# Supervised multitask models: full forward, every output head dumped.
MULTITASK = {
    "multi-litept-b": ("configs/flair3d_default/multi-litept-b-v1m0-flair3d.py", "litept-b"),
    "multi-ptv3": ("configs/flair3d_default/multi-ptv3-v1m0-flair3d.py", "ptv3"),
    "multi-spunet": ("configs/flair3d_default/multi-spunet-v1m0-flair3d.py", "spunet"),
    "multi-kpconvx": ("configs/flair3d_default/multi-kpconvx-v1m0-flair3d.py", "kpconvx"),
    "multi-sonata-ft": ("configs/flair3d_default/multi-sonata-ft-v1m0-flair3d.py", "sonata-ft"),
}

PROBE_DIRS = ["configs/dales", "configs/h3d", "configs/eclair", "configs/opengf", "configs/pureforest",
              "configs/flair3d_default/probe"]


def probe_cases():
    """GridProbe configs: frozen-backbone features on one val sample. name = config basename."""
    out = {}
    for d in PROBE_DIRS:
        for p in sorted(glob.glob(f"{d}/*.py")):
            if "grid" in p or "lin_" in p:
                out[p.split("configs/", 1)[1][:-3]] = p
    return out


# ----------------------------------------------------------------------------- stage "train"
# Short end-to-end tools/train.py runs (train + val + test) on local tiles, fixed seed.
_FLAIR_SMALL = [
    "seed=0", "enable_wandb=False", "num_worker=2", "num_worker_test=2",
    "batch_size=2", "batch_size_val=1", "batch_size_test=1",
    "total_iters=40", "iter_per_epoch=20", "eval_every=1",
    f"data.train.csv_manifest={FLAIR3D_MANIFEST['train']}",
    f"data.val.csv_manifest={FLAIR3D_MANIFEST['val']}",
    "data.val.stratified_subset_manifest=", "data.val.max_sample=4",
    # D075/D068 test tiles lack forest_2d.npy locally: test on D067 val tiles instead.
    "data.test.split=val", f"data.test.csv_manifest={FLAIR3D_MANIFEST['val']}", "data.test.max_sample=2",
]
_PROBE_BASE = [
    "seed=0", "enable_wandb=False", "num_worker=2", "num_worker_test=2",
    "batch_size=4", "batch_size_val=1", "batch_size_test=1", "epoch=2", "eval_epoch=2",
]
_PROBE_SMALL = _PROBE_BASE + ["data.train.max_sample=8", "data.val.max_sample=2", "data.test.max_sample=1"]
# PureForestDataset has no max_sample: use the 26-tile-per-split toy copy instead.
_PUREFOREST_SMALL = _PROBE_BASE + [f"data.{s}.data_root=data/pureforest_toy" for s in ("train", "val", "test")]
TRAIN = {
    "train-multi-litept-b": ("configs/flair3d_default/multi-litept-b-v1m0-flair3d.py", _FLAIR_SMALL),
    "train-multi-spunet": ("configs/flair3d_default/multi-spunet-v1m0-flair3d.py", _FLAIR_SMALL),
    "train-multi-ptv3": ("configs/flair3d_default/multi-ptv3-v1m0-flair3d.py", _FLAIR_SMALL),
    "train-multi-kpconvx": ("configs/flair3d_default/multi-kpconvx-v1m0-flair3d.py", _FLAIR_SMALL),
    "train-multi-sonata-ft": ("configs/flair3d_default/multi-sonata-ft-v1m0-flair3d.py",
                              _FLAIR_SMALL + [f"weight={CKPT['sonata-pretrained']}"]),
    "train-segment-litept-b": ("configs/flair3d_default/segment/litept-b-v1m0-flair3d.py", _FLAIR_SMALL),
    "train-elevation-litept-b": ("configs/flair3d_default/elevation/litept-b-v1m0-flair3d.py", _FLAIR_SMALL),
    "train-forest-litept": ("configs/flair3d_default/forest/litept-v1m0-flair3d.py", _FLAIR_SMALL),
    "train-nathab-litept": ("configs/flair3d_default/natural_habitat/litept-v1m0-flair3d.py", _FLAIR_SMALL),
    "train-network-litept": ("configs/flair3d_default/network/litept-v1m0-flair3d.py", _FLAIR_SMALL),
    "train-pretrain-sonata": ("configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py", [
        "seed=0", "enable_wandb=False", "num_worker=2", "batch_size=2",
        "total_iters=40", "iter_per_epoch=20", f"data.train.csv_manifest={FLAIR3D_MANIFEST['train']}"]),
    "train-probe-h3d-litept-b": ("configs/h3d/litept-b-v1m0-h3d-lin_enc.py",
                                 _PROBE_SMALL + [f"weight={CKPT['litept-b']}"]),
    "train-probe-opengf-spunet": ("configs/opengf/spunet-v1m0-opengf-lin-grid-enc.py",
                                  _PROBE_SMALL + [f"weight={CKPT['spunet']}"]),
    "train-probe-pureforest-litept-b": ("configs/pureforest/cls-litept-b-v1m0-pureforest-lin-grid-enc.py",
                                        _PUREFOREST_SMALL + [f"weight={CKPT['litept-b']}"]),
    "train-probe-flair3d-sonata": ("configs/flair3d_default/probe/sonata-v1m2-flair3d-lin-grid.py",
                                   _FLAIR_SMALL + [f"weight={CKPT['sonata-pretrained']}"]),
}
