#!/usr/bin/env python3
"""Inference-speed benchmark: LitePT-B / PTv3 / KPConvX / SpUNet / Sonata on Flair3D+.

Measures batch_size=1 test-time throughput (points/sec) for the 5 models
currently used on Flair3D, on an identical set of tiles, breaking the per-tile
cost into CPU (dataset load + transform pipeline), CPU->GPU transfer, and
"true" GPU compute -- reproducing exactly what the tester does per fragment
(pointcept/engines/test.py), but instrumented and without the mIoU/APLS
bookkeeping.

Reference configs (already harmonized on data_root="data/flair3d_plus",
segment="v20", feat_keys=["coord","color","strength"], grid_size=0.1,
test_single_fragment=True, crop=None) -- see
configs/flair3d_default/multi-{litept-b,ptv3,kpconvx,spunet}-v1m0-flair3d.py
(MultiTaskSegmentorV2) and configs/flair3d_default/probe/sonata-v1m2-flair3d-lin.py
(DefaultSegmentorV2, PT-v3m2 encoder + linear head). The single-probe Sonata
config has no data.test (val-only periodic probes); its test pipeline is
borrowed from sonata-v1m2-flair3d-lin-grid.py (same tiles / voxelize, no
config file is edited). No config is copied or modified; they are loaded via
Config.fromfile exactly like tools/train.py / tools/test.py.

Weights are randomly initialized (no --weight / CheckpointLoader): none of
the architectures branch on weight *values*, only on point-cloud geometry,
so a trained checkpoint isn't needed for a pure speed comparison.

The full post-transform fragment dict (coord, grid_coord, feat, offset, plus
the network_*/forest_2d_* pixel-head raster metadata + labels on the 4
multitask configs) is passed to the model unmodified, matching real test-time
behavior: the network/forest_2d pixel-semantic heads in these multitask
configs only run their forward pass when their raster metadata is present in
input_dict (see MultiTaskSegmentorV2._compute_pixel_logits), so stripping
"targets" out would silently skip part of the real per-tile GPU cost. The
Sonata lin-probe Collect is segment-only (no pixel heads).

Same tiles across all backbones are pinned once via `include_names`
(pointcept/datasets/subset_utils.py) so a fair, apples-to-apples comparison
is guaranteed regardless of manifest/split overrides.

Examples::

  # Local dry run on Hecate (D067 has no local test split -- use val).
  export PYTHONPATH=$PWD
  python scripts/bench_inference_speed.py \\
    --csv-manifest data/flair3d_plus/raw/scene_split_manifest_D067.csv --split val \\
    --num-tiles 15 --num-warmup 5 --device cuda:0

  # Real run on A100 (Jean Zay, full national manifest, test split).
  python scripts/bench_inference_speed.py \\
    --csv-manifest data/flair3d_plus/raw/scene_split_manifest.csv --split test \\
    --num-tiles 200 --num-warmup 10 --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIGS = {
    "litept_b": "configs/flair3d_default/multi-litept-b-v1m0-flair3d.py",
    "ptv3": "configs/flair3d_default/multi-ptv3-v1m0-flair3d.py",
    "kpconvx": "configs/flair3d_default/multi-kpconvx-v1m0-flair3d.py",
    "spunet": "configs/flair3d_default/multi-spunet-v1m0-flair3d.py",
    # PT-v3m2 encoder (enc_mode=True, 1232-ch concat) + linear head. Not the
    # SSL MultiView graph, and not PT-v3-malibu (`ptv3` above, decoder + multitask).
    "sonata": "configs/flair3d_default/probe/sonata-v1m2-flair3d-lin.py",
}

# sonata-v1m2-flair3d-lin.py has train/val only; reuse the grid-probe test split
# (same Flair3D+ tiles, test_single_fragment=True, no max_sample).
TEST_PIPELINE_FALLBACK = {
    "sonata": "configs/flair3d_default/probe/sonata-v1m2-flair3d-lin-grid.py",
}

PER_TILE_FIELDS = [
    "backbone",
    "tile_idx",
    "patch_id",
    "num_points",
    "cpu_ms",
    "transfer_ms",
    "gpu_ms",
    "total_ms",
    "warmup",
    "oom",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=list(DEFAULT_CONFIGS.keys()),
        choices=list(DEFAULT_CONFIGS.keys()),
        help="Subset of backbones to benchmark (default: all 5).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        metavar="NAME=PATH",
        help="Override one or more config paths, e.g. ptv3=configs/foo.py (default: "
        "the 4 flair3d_default multi-task configs + Sonata lin-probe, unmodified).",
    )
    parser.add_argument(
        "--csv-manifest",
        default=None,
        help="Override data.test.csv_manifest for all backbones (default: keep each "
        "config's own, the national manifest -- only fully mirrored on Jean Zay).",
    )
    parser.add_argument(
        "--split", default="test", help="Override data.test.split for all backbones."
    )
    parser.add_argument(
        "--num-tiles",
        type=int,
        default=60,
        help="Total tiles benchmarked per backbone, including warmup tiles.",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Leading tiles run but excluded from aggregated stats "
        "(cudnn/kernel-selection warmup).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=True,
        help="Pin CPU tensors before the H2D transfer (default: on).",
    )
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Wrap the GPU forward in torch.autocast (default: off, fp32, matching "
        "MultiTaskTester's real eval-time behavior).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Default: stats/flair3d/inference_speed_bench/<timestamp>/",
    )
    return parser.parse_args()


def resolve_config_paths(args):
    paths = dict(DEFAULT_CONFIGS)
    if args.configs:
        for item in args.configs:
            name, sep, path = item.partition("=")
            if not sep or name not in paths:
                raise ValueError(
                    f"--configs entries must be NAME=PATH with NAME in {list(paths)}, "
                    f"got {item!r}"
                )
            paths[name] = path
    return {name: paths[name] for name in args.backbones}


def build_test_dataset(cfg_path, csv_manifest, split, include_names=None, name=None):
    from pointcept.utils.config import Config
    from pointcept.datasets.builder import build_dataset

    cfg = Config.fromfile(str(cfg_path))
    if "test" in cfg.data:
        test_cfg = cfg.data.test
    else:
        fallback = TEST_PIPELINE_FALLBACK.get(name)
        if not fallback:
            raise ValueError(
                f"config {cfg_path} has no data.test and no TEST_PIPELINE_FALLBACK "
                f"for name={name!r}"
            )
        print(
            f"[bench][{name}] no data.test in {cfg_path}; "
            f"using test pipeline from {fallback}"
        )
        test_cfg = Config.fromfile(str(fallback)).data.test
    test_cfg = copy.deepcopy(test_cfg)
    if csv_manifest is not None:
        test_cfg["csv_manifest"] = csv_manifest
    if split is not None:
        test_cfg["split"] = split
    if include_names is not None:
        test_cfg["include_names"] = list(include_names)
    dataset = build_dataset(test_cfg)
    return cfg, dataset


def pick_tile_names(cfg_path, csv_manifest, split, num_tiles, name=None):
    _, dataset = build_test_dataset(cfg_path, csv_manifest, split, name=name)
    if len(dataset.data_list) < num_tiles:
        raise ValueError(
            f"Only {len(dataset.data_list)} tiles available for split={split!r} "
            f"(config={cfg_path}, csv_manifest override={csv_manifest!r}), need "
            f"--num-tiles={num_tiles}."
        )
    return [dataset.get_data_name(i) for i in range(num_tiles)]


def benchmark_backbone(name, cfg_path, tile_names, args):
    from pointcept.datasets.utils import collate_fn
    from pointcept.models.builder import build_model

    cfg, dataset = build_test_dataset(
        cfg_path,
        args.csv_manifest,
        args.split,
        include_names=tile_names,
        name=name,
    )
    if len(dataset.data_list) != len(tile_names):
        raise RuntimeError(
            f"[{name}] pinned tile list resolved to {len(dataset.data_list)} scenes, "
            f"expected {len(tile_names)} -- some patch_ids didn't match under this "
            "backbone's data_root/csv_manifest (are all configs really unified?)."
        )

    torch.manual_seed(args.seed)
    model = build_model(cfg.model).to(args.device)
    model.eval()

    device_type = "cuda" if str(args.device).startswith("cuda") else "cpu"
    records = []
    for tile_idx in range(len(tile_names)):
        is_warmup = tile_idx < args.num_warmup
        try:
            t0 = time.perf_counter()
            data_dict = dataset[tile_idx]
            fragment = data_dict["fragment_list"][0]
            input_dict = collate_fn([fragment])
            if args.pin_memory:
                for k, v in input_dict.items():
                    if torch.is_tensor(v) and v.device.type == "cpu":
                        input_dict[k] = v.pin_memory()
            cpu_ms = (time.perf_counter() - t0) * 1000.0

            start_ev, end_ev = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start_ev.record()
            gpu_input = {
                k: (v.to(args.device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in input_dict.items()
            }
            end_ev.record()
            torch.cuda.synchronize()
            transfer_ms = start_ev.elapsed_time(end_ev)

            num_points = int(gpu_input["coord"].shape[0])

            start_ev, end_ev = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start_ev.record()
            with torch.no_grad():
                with torch.autocast(device_type=device_type, enabled=args.amp):
                    model(gpu_input)
            end_ev.record()
            torch.cuda.synchronize()
            gpu_ms = start_ev.elapsed_time(end_ev)

            total_ms = cpu_ms + transfer_ms + gpu_ms
            records.append(
                dict(
                    backbone=name,
                    tile_idx=tile_idx,
                    patch_id=tile_names[tile_idx],
                    num_points=num_points,
                    cpu_ms=cpu_ms,
                    transfer_ms=transfer_ms,
                    gpu_ms=gpu_ms,
                    total_ms=total_ms,
                    warmup=is_warmup,
                    oom=False,
                )
            )
            tag = "warmup" if is_warmup else "     "
            print(
                f"[bench][{name}] tile {tile_idx + 1:3d}/{len(tile_names)} [{tag}] "
                f"{tile_names[tile_idx]:35s} pts={num_points:8,d} "
                f"cpu={cpu_ms:7.2f}ms xfer={transfer_ms:6.3f}ms gpu={gpu_ms:7.2f}ms "
                f"total={total_ms:7.2f}ms"
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[bench][{name}] OOM on tile {tile_idx} ({tile_names[tile_idx]}) -- skipped")
            records.append(
                dict(
                    backbone=name,
                    tile_idx=tile_idx,
                    patch_id=tile_names[tile_idx],
                    num_points=None,
                    cpu_ms=None,
                    transfer_ms=None,
                    gpu_ms=None,
                    total_ms=None,
                    warmup=is_warmup,
                    oom=True,
                )
            )

    del model
    torch.cuda.empty_cache()
    return records


def summarize(records):
    measured = [r for r in records if not r["warmup"] and not r["oom"]]
    n_failed = sum(1 for r in records if r["oom"])
    if not measured:
        return dict(n_measured=0, n_failed=n_failed)

    def col(key):
        return np.array([r[key] for r in measured], dtype=np.float64)

    cpu, transfer, gpu, total, pts = (
        col("cpu_ms"),
        col("transfer_ms"),
        col("gpu_ms"),
        col("total_ms"),
        col("num_points"),
    )
    return dict(
        n_measured=len(measured),
        n_failed=n_failed,
        cpu_ms_mean=float(cpu.mean()),
        cpu_ms_median=float(np.median(cpu)),
        cpu_ms_std=float(cpu.std()),
        transfer_ms_mean=float(transfer.mean()),
        transfer_ms_median=float(np.median(transfer)),
        transfer_ms_std=float(transfer.std()),
        gpu_ms_mean=float(gpu.mean()),
        gpu_ms_median=float(np.median(gpu)),
        gpu_ms_std=float(gpu.std()),
        total_ms_mean=float(total.mean()),
        total_ms_median=float(np.median(total)),
        total_ms_std=float(total.std()),
        num_points_mean=float(pts.mean()),
        num_points_min=float(pts.min()),
        num_points_max=float(pts.max()),
        pts_per_sec_gpu=float(pts.sum() / (gpu.sum() / 1000.0)),
        pts_per_sec_e2e=float(pts.sum() / (total.sum() / 1000.0)),
    )


def _mean_std(s, key, decimals):
    return f"{s[f'{key}_mean']:.{decimals}f}±{s[f'{key}_std']:.{decimals}f}"


def print_summary_table(summaries):
    header = (
        f"{'backbone':10s} {'n_ok':>5s} {'n_fail':>6s} {'cpu_ms (mean±std)':>16s} "
        f"{'xfer_ms (mean±std)':>16s} {'gpu_ms (mean±std)':>16s} "
        f"{'total_ms (mean±std)':>18s} {'pts/s(GPU)':>12s} {'pts/s(e2e)':>12s}"
    )
    print(header)
    print("-" * len(header))
    for name, s in summaries.items():
        if s["n_measured"] == 0:
            print(f"{name:10s} {0:5d} {s['n_failed']:6d}  -- all measured tiles failed --")
            continue
        print(
            f"{name:10s} {s['n_measured']:5d} {s['n_failed']:6d} "
            f"{_mean_std(s, 'cpu_ms', 2):>16s} {_mean_std(s, 'transfer_ms', 3):>16s} "
            f"{_mean_std(s, 'gpu_ms', 2):>16s} {_mean_std(s, 'total_ms', 2):>18s} "
            f"{s['pts_per_sec_gpu']:12,.0f} {s['pts_per_sec_e2e']:12,.0f}"
        )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    cfg_paths = resolve_config_paths(args)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else REPO_ROOT
        / "stats"
        / "flair3d"
        / "inference_speed_bench"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    first_name = args.backbones[0]
    print(
        f"[bench] resolving {args.num_tiles} pinned tiles from '{first_name}' "
        f"({cfg_paths[first_name]}, split={args.split!r}) ..."
    )
    tile_names = pick_tile_names(
        cfg_paths[first_name], args.csv_manifest, args.split, args.num_tiles, name=first_name
    )
    print(f"[bench] pinned {len(tile_names)} tiles (first 3: {tile_names[:3]})")

    all_records = []
    summaries = {}
    for name in args.backbones:
        print(f"\n[bench] === {name} ({cfg_paths[name]}) ===")
        records = benchmark_backbone(name, cfg_paths[name], tile_names, args)
        all_records.extend(records)
        summaries[name] = summarize(records)

    print("\n=== Summary (warmup tiles excluded: first "
          f"{args.num_warmup}/{args.num_tiles}) ===")
    print_summary_table(summaries)

    per_tile_csv = out_dir / "per_tile.csv"
    with open(per_tile_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_TILE_FIELDS)
        writer.writeheader()
        writer.writerows(all_records)

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w") as f:
        json.dump(
            dict(
                args={k: str(v) for k, v in vars(args).items()},
                configs={k: str(v) for k, v in cfg_paths.items()},
                tile_names=tile_names,
                summaries=summaries,
            ),
            f,
            indent=2,
        )

    print(f"\n[bench] wrote {per_tile_csv}")
    print(f"[bench] wrote {summary_json}")


if __name__ == "__main__":
    main()
