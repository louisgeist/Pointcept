#!/usr/bin/env python3
"""Inference-speed benchmark: LitePT-B / PTv3 / KPConvX / SpUNet / Sonata on Flair3D+.

Measures batch_size=1 test-time throughput (points/sec) for the 5 models
currently used on Flair3D, on an identical seeded tile sample. Two passes per
backbone:

- sequential: exclusive CPU (dataset load + transform) / H2D / GPU via
  ``torch.cuda.Event`` -- diagnosis of where time goes in isolation.
- pipeline: DataLoader workers + prefetch (as TesterBase, but batch_size=1,
  no voxel-budget packing) -- stall waiting on the loader, then H2D / GPU.
  ``pts/s(pipeline)`` is the figure to cite for "how fast do we infer".

Reproduces the tester fragment forward (pointcept/engines/test.py) without
mIoU/APLS bookkeeping.

A CPU-only page-cache warmup on the sampled tiles runs before the first
backbone so LitePT is not the only one paying cold Lustre I/O. GPU warmup
(``--num-warmup`` leading tiles, cudnn/kernel selection) is independent and
applies to both passes.

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

Same tiles across all backbones: names are sampled once from the first
config's ``data.test`` (``--tile-sample random`` shuffles with ``--seed``,
``first`` keeps CSV order) and looked up by name in every other dataset.

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
    "mode",
    "tile_idx",
    "patch_id",
    "num_points",
    "cpu_ms",
    "stall_ms",
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
        "(cudnn/kernel-selection warmup). Applied to both sequential and pipeline.",
    )
    parser.add_argument(
        "--tile-sample",
        choices=("random", "first"),
        default="random",
        help="How to pick --num-tiles from data.test: seeded shuffle (default) or "
        "CSV / data_list order (the old first-N behaviour).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for --tile-sample random and torch.manual_seed (model init).",
    )
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=True,
        help="Pin CPU tensors before the H2D transfer (default: on).",
    )
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument(
        "--cache-warmup",
        dest="cache_warmup",
        action="store_true",
        default=True,
        help="CPU-only pass over the sampled tiles before the first backbone, so "
        "all backbones see a hot page cache (default: on).",
    )
    parser.add_argument("--no-cache-warmup", dest="cache_warmup", action="store_false")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers for the pipeline pass (default: "
        "cfg.num_worker_per_gpu, or num_worker // num_gpu).",
    )
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


def build_test_dataset(cfg_path, csv_manifest, split, name=None):
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
    dataset = build_dataset(test_cfg)
    return cfg, dataset


def resolve_num_workers(cfg, args):
    if args.num_workers is not None:
        return int(args.num_workers)
    per_gpu = getattr(cfg, "num_worker_per_gpu", None)
    if per_gpu is not None:
        return int(per_gpu)
    num_worker = int(getattr(cfg, "num_worker", 0) or 0)
    num_gpu = max(int(getattr(cfg, "num_gpu", 1) or 1), 1)
    return num_worker // num_gpu


def select_tile_names(dataset, num_tiles, seed, mode, *, cfg_path, csv_manifest, split):
    n = len(dataset.data_list)
    if n < num_tiles:
        raise ValueError(
            f"Only {n} tiles available for split={split!r} "
            f"(config={cfg_path}, csv_manifest override={csv_manifest!r}), need "
            f"--num-tiles={num_tiles}."
        )
    names = [dataset.get_data_name(i) for i in range(n)]
    if mode == "first":
        return names[:num_tiles]
    if mode == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        return [names[int(i)] for i in perm[:num_tiles]]
    raise ValueError(f"Unknown --tile-sample {mode!r}")


def name_to_index(dataset):
    mapping = {}
    for i in range(len(dataset.data_list)):
        name = dataset.get_data_name(i)
        if name in mapping:
            raise RuntimeError(f"duplicate data.test name {name!r} at indices "
                               f"{mapping[name]} and {i}")
        mapping[name] = i
    return mapping


def resolve_indices(dataset, tile_names, backbone_name):
    mapping = name_to_index(dataset)
    missing = [n for n in tile_names if n not in mapping]
    if missing:
        raise RuntimeError(
            f"[{backbone_name}] {len(missing)} sampled tile(s) missing from data.test "
            f"(first missing: {missing[0]!r})"
        )
    return [mapping[n] for n in tile_names]


def dept_years(tile_names):
    return sorted({n.split("_")[0] for n in tile_names})


def warmup_page_cache(dataset, indices, tile_names):
    n = len(indices)
    print(f"[bench] cache warmup: loading {n} tiles (CPU only, discarded) ...")
    for i, idx in enumerate(indices):
        dataset[idx]
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"[bench] cache warmup {i + 1}/{n} {tile_names[i]}")


def _empty_record(name, mode, tile_idx, patch_id, is_warmup, oom=True):
    return dict(
        backbone=name,
        mode=mode,
        tile_idx=tile_idx,
        patch_id=patch_id,
        num_points=None,
        cpu_ms=None,
        stall_ms=None,
        transfer_ms=None,
        gpu_ms=None,
        total_ms=None,
        warmup=is_warmup,
        oom=oom,
    )


def _pin_cpu_tensors(input_dict):
    for k, v in input_dict.items():
        if torch.is_tensor(v) and v.device.type == "cpu":
            input_dict[k] = v.pin_memory()
    return input_dict


def _timed_h2d_and_forward(model, input_dict, args, device_type):
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
    return transfer_ms, gpu_ms, num_points


def _bench_collate(batch):
    from pointcept.datasets.utils import collate_fn

    fragment = batch[0]["fragment_list"][0]
    return collate_fn([fragment])


def benchmark_sequential(name, dataset, indices, tile_names, model, args):
    from pointcept.datasets.utils import collate_fn

    device_type = "cuda" if str(args.device).startswith("cuda") else "cpu"
    records = []
    for tile_idx, data_idx in enumerate(indices):
        is_warmup = tile_idx < args.num_warmup
        patch_id = tile_names[tile_idx]
        try:
            t0 = time.perf_counter()
            data_dict = dataset[data_idx]
            fragment = data_dict["fragment_list"][0]
            input_dict = collate_fn([fragment])
            if args.pin_memory:
                input_dict = _pin_cpu_tensors(input_dict)
            cpu_ms = (time.perf_counter() - t0) * 1000.0

            transfer_ms, gpu_ms, num_points = _timed_h2d_and_forward(
                model, input_dict, args, device_type
            )
            total_ms = cpu_ms + transfer_ms + gpu_ms
            records.append(
                dict(
                    backbone=name,
                    mode="sequential",
                    tile_idx=tile_idx,
                    patch_id=patch_id,
                    num_points=num_points,
                    cpu_ms=cpu_ms,
                    stall_ms=None,
                    transfer_ms=transfer_ms,
                    gpu_ms=gpu_ms,
                    total_ms=total_ms,
                    warmup=is_warmup,
                    oom=False,
                )
            )
            tag = "warmup" if is_warmup else "     "
            print(
                f"[bench][{name}][sequential] tile {tile_idx + 1:3d}/{len(tile_names)} "
                f"[{tag}] {patch_id:35s} pts={num_points:8,d} "
                f"cpu={cpu_ms:7.2f}ms xfer={transfer_ms:6.3f}ms gpu={gpu_ms:7.2f}ms "
                f"total={total_ms:7.2f}ms"
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[bench][{name}][sequential] OOM on tile {tile_idx} ({patch_id}) -- skipped")
            records.append(_empty_record(name, "sequential", tile_idx, patch_id, is_warmup))
    return records


def benchmark_pipeline(name, dataset, indices, tile_names, model, args, num_workers):
    device_type = "cuda" if str(args.device).startswith("cuda") else "cpu"
    subset = torch.utils.data.Subset(dataset, indices)
    loader_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        collate_fn=_bench_collate,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1
        loader_kwargs["persistent_workers"] = True
    loader = torch.utils.data.DataLoader(subset, **loader_kwargs)
    print(
        f"[bench][{name}][pipeline] DataLoader num_workers={num_workers} "
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'n/a')} "
        f"pin_memory={args.pin_memory}"
    )

    records = []
    loader_iter = iter(loader)
    try:
        for tile_idx in range(len(tile_names)):
            is_warmup = tile_idx < args.num_warmup
            patch_id = tile_names[tile_idx]
            try:
                t0 = time.perf_counter()
                input_dict = next(loader_iter)
                stall_ms = (time.perf_counter() - t0) * 1000.0

                transfer_ms, gpu_ms, num_points = _timed_h2d_and_forward(
                    model, input_dict, args, device_type
                )
                total_ms = stall_ms + transfer_ms + gpu_ms
                records.append(
                    dict(
                        backbone=name,
                        mode="pipeline",
                        tile_idx=tile_idx,
                        patch_id=patch_id,
                        num_points=num_points,
                        cpu_ms=None,
                        stall_ms=stall_ms,
                        transfer_ms=transfer_ms,
                        gpu_ms=gpu_ms,
                        total_ms=total_ms,
                        warmup=is_warmup,
                        oom=False,
                    )
                )
                tag = "warmup" if is_warmup else "     "
                print(
                    f"[bench][{name}][pipeline] tile {tile_idx + 1:3d}/{len(tile_names)} "
                    f"[{tag}] {patch_id:35s} pts={num_points:8,d} "
                    f"stall={stall_ms:7.2f}ms xfer={transfer_ms:6.3f}ms gpu={gpu_ms:7.2f}ms "
                    f"total={total_ms:7.2f}ms"
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(
                    f"[bench][{name}][pipeline] OOM on tile {tile_idx} ({patch_id}) -- skipped"
                )
                records.append(
                    _empty_record(name, "pipeline", tile_idx, patch_id, is_warmup)
                )
    finally:
        del loader_iter
        del loader
    return records


def benchmark_backbone(name, cfg, dataset, indices, tile_names, args):
    from pointcept.models.builder import build_model

    torch.manual_seed(args.seed)
    model = build_model(cfg.model).to(args.device)
    model.eval()
    num_workers = resolve_num_workers(cfg, args)

    seq_records = benchmark_sequential(name, dataset, indices, tile_names, model, args)
    pipe_records = benchmark_pipeline(
        name, dataset, indices, tile_names, model, args, num_workers
    )

    del model
    torch.cuda.empty_cache()
    return seq_records, pipe_records


def summarize(records):
    measured = [r for r in records if not r["warmup"] and not r["oom"]]
    n_failed = sum(1 for r in records if r["oom"])
    mode = records[0]["mode"] if records else None
    if not measured:
        return dict(n_measured=0, n_failed=n_failed, mode=mode)

    def col(key):
        return np.array([r[key] for r in measured], dtype=np.float64)

    transfer, gpu, total, pts = (
        col("transfer_ms"),
        col("gpu_ms"),
        col("total_ms"),
        col("num_points"),
    )
    out = dict(
        n_measured=len(measured),
        n_failed=n_failed,
        mode=mode,
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
    if mode == "pipeline":
        stall = col("stall_ms")
        out.update(
            stall_ms_mean=float(stall.mean()),
            stall_ms_median=float(np.median(stall)),
            stall_ms_std=float(stall.std()),
            pts_per_sec_pipeline=out["pts_per_sec_e2e"],
            tiles_per_sec=float(len(measured) / (total.sum() / 1000.0)),
        )
    else:
        cpu = col("cpu_ms")
        out.update(
            cpu_ms_mean=float(cpu.mean()),
            cpu_ms_median=float(np.median(cpu)),
            cpu_ms_std=float(cpu.std()),
        )
    return out


def _mean_std(s, key, decimals):
    return f"{s[f'{key}_mean']:.{decimals}f}±{s[f'{key}_std']:.{decimals}f}"


def print_summary_table(summaries, mode):
    if mode == "sequential":
        header = (
            f"{'backbone':10s} {'n_ok':>5s} {'n_fail':>6s} {'cpu_ms (mean±std)':>18s} "
            f"{'xfer_ms (mean±std)':>18s} {'gpu_ms (mean±std)':>18s} "
            f"{'total_ms (mean±std)':>20s} {'pts/s(GPU)':>12s} {'pts/s(e2e)':>12s}"
        )
    else:
        header = (
            f"{'backbone':10s} {'n_ok':>5s} {'n_fail':>6s} {'stall_ms (mean±std)':>20s} "
            f"{'xfer_ms (mean±std)':>18s} {'gpu_ms (mean±std)':>18s} "
            f"{'total_ms (mean±std)':>20s} {'pts/s(GPU)':>12s} "
            f"{'pts/s(pipeline)':>16s} {'tiles/s':>8s}"
        )
    print(header)
    print("-" * len(header))
    for name, modes in summaries.items():
        s = modes[mode]
        if s["n_measured"] == 0:
            print(f"{name:10s} {0:5d} {s['n_failed']:6d}  -- all measured tiles failed --")
            continue
        if mode == "sequential":
            print(
                f"{name:10s} {s['n_measured']:5d} {s['n_failed']:6d} "
                f"{_mean_std(s, 'cpu_ms', 2):>18s} {_mean_std(s, 'transfer_ms', 3):>18s} "
                f"{_mean_std(s, 'gpu_ms', 2):>18s} {_mean_std(s, 'total_ms', 2):>20s} "
                f"{s['pts_per_sec_gpu']:12,.0f} {s['pts_per_sec_e2e']:12,.0f}"
            )
        else:
            print(
                f"{name:10s} {s['n_measured']:5d} {s['n_failed']:6d} "
                f"{_mean_std(s, 'stall_ms', 2):>20s} {_mean_std(s, 'transfer_ms', 3):>18s} "
                f"{_mean_std(s, 'gpu_ms', 2):>18s} {_mean_std(s, 'total_ms', 2):>20s} "
                f"{s['pts_per_sec_gpu']:12,.0f} {s['pts_per_sec_pipeline']:16,.0f} "
                f"{s['tiles_per_sec']:8.2f}"
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
        f"[bench] sampling {args.num_tiles} tiles of data.test "
        f"(split={args.split!r}, tile_sample={args.tile_sample!r}, seed={args.seed}) ..."
    )

    cfg0, dataset0 = build_test_dataset(
        cfg_paths[first_name], args.csv_manifest, args.split, name=first_name
    )
    tile_names = select_tile_names(
        dataset0,
        args.num_tiles,
        args.seed,
        args.tile_sample,
        cfg_path=cfg_paths[first_name],
        csv_manifest=args.csv_manifest,
        split=args.split,
    )
    indices0 = resolve_indices(dataset0, tile_names, first_name)
    depts = dept_years(tile_names)
    print(
        f"[bench] {len(tile_names)} tiles from {len(depts)} dept_year "
        f"(first 3: {tile_names[:3]}; depts e.g. {depts[:8]})"
    )

    if args.cache_warmup:
        warmup_page_cache(dataset0, indices0, tile_names)
    else:
        print("[bench] cache warmup disabled (--no-cache-warmup)")

    all_records = []
    summaries = {}
    for i, name in enumerate(args.backbones):
        print(f"\n[bench] === {name} ({cfg_paths[name]}) ===")
        if i == 0:
            cfg, dataset, indices = cfg0, dataset0, indices0
        else:
            cfg, dataset = build_test_dataset(
                cfg_paths[name], args.csv_manifest, args.split, name=name
            )
            indices = resolve_indices(dataset, tile_names, name)
        seq_records, pipe_records = benchmark_backbone(
            name, cfg, dataset, indices, tile_names, args
        )
        all_records.extend(seq_records)
        all_records.extend(pipe_records)
        summaries[name] = dict(
            sequential=summarize(seq_records),
            pipeline=summarize(pipe_records),
        )

    excluded = f"warmup tiles excluded: first {args.num_warmup}/{args.num_tiles}"
    print(f"\n=== Summary sequential ({excluded}) ===")
    print_summary_table(summaries, "sequential")
    print(f"\n=== Summary pipeline ({excluded}) ===")
    print_summary_table(summaries, "pipeline")

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
                tile_sample=args.tile_sample,
                seed=args.seed,
                tile_names=tile_names,
                dept_years=depts,
                summaries=summaries,
            ),
            f,
            indent=2,
        )

    print(f"\n[bench] wrote {per_tile_csv}")
    print(f"[bench] wrote {summary_json}")


if __name__ == "__main__":
    main()
