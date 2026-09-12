#!/usr/bin/env python3
"""VRAM breakdown for GridProbeSegmentorV2 (see pointcept/models/grid_probe.py
and GridProbeTrainer in pointcept/engines/train.py).

Answers "where does the VRAM go" for a grid-probe run: how much is the
frozen-backbone forward pass + shared multiscale feature, how much is the
per-input_norm unitsphere cache, and how much is added per active probe head
(forward + loss + backward + optimizer.step()) -- reproducing the exact
per-step code shape of GridProbeTrainer.run_step, but with
torch.cuda.reset_peak_memory_stats() checkpoints inserted between stages.

Uses synthetic input_dict batches (random coord/grid_coord/feat/segment)
shaped like the real Flair3D+ pipeline output (batch_size, point_max points
per sample) instead of going through Flair3DDataset -- this only measures
*shapes*-driven VRAM, not learned representation quality, and sidesteps the
Hecate/Jean-Zay data-mirroring gap (see CLAUDE.md) and segment-version
mismatches entirely. Backbone weights are randomly initialized (no
--weight/CheckpointLoader): irrelevant for memory, only shapes/dtypes matter.

If flash_attn is not importable (e.g. no prebuilt wheel for this torch+cuda
combo), PT-v3m2's flash-attention branch is monkeypatched to route through
torch.nn.functional.scaled_dot_product_attention instead (same non-materializing
windowed-attention memory profile on Ampere+, just slower -- see
_install_sdpa_flash_shim). This is a LOCAL-ONLY approximation: absolute
backbone numbers from a shimmed run are indicative, not bit-identical to a
real flash_attn run on the target GPU.

Examples::

  # Fixed-cost breakdown (backbone fwd + shared feat + norm caches) and a
  # per-probe step-by-step trace at N=21 probes (the user's observed ceiling
  # at batch_size=12), from the wide grid config:
  python scripts/sonata/diagnose_grid_probe_vram.py \\
    --config-file configs/flair3d_default/probe/sonata-v1m2-flair3d-lin-grid-wide.py \\
    --num-probes 21 --detail

  # Sweep probe count to fit fixed-cost / marginal-per-probe-cost:
  python scripts/sonata/diagnose_grid_probe_vram.py \\
    --config-file configs/flair3d_default/probe/sonata-v1m2-flair3d-lin-grid-wide.py \\
    --sweep 1 4 8 12 16 21 32 48 64
"""

from __future__ import annotations

import argparse
import copy
import gc
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MB = 2**20
GB = 2**30


def _install_sdpa_flash_shim():
    """Route PT-v3m2's flash_attn_varlen_qkvpacked_func through SDPA.

    Real call site passes qkv already reshaped to (N_pad, 3, H, C/H) with
    cu_seqlens marking window boundaries -- windows are only padded to a
    multiple of patch_size when a sample already exceeds patch_size (see
    SerializedAttention.get_padding_and_inverse), so windows are genuinely
    ragged and must be attended to one at a time, not reshaped as if uniform.
    """
    import pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata as ptv3mod

    if ptv3mod.flash_attn is not None:
        return False

    class _SDPAFlashShim:
        @staticmethod
        def flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None
        ):
            n_pad, _, h, ch = qkv.shape
            q_all, k_all, v_all = qkv.unbind(dim=1)
            out = torch.empty(n_pad, h, ch, device=qkv.device, dtype=qkv.dtype)
            dp = dropout_p if isinstance(dropout_p, (int, float)) else 0.0
            starts = cu_seqlens[:-1].tolist()
            ends = cu_seqlens[1:].tolist()
            for s, e in zip(starts, ends):
                if e <= s:
                    continue
                q = q_all[s:e].permute(1, 0, 2).unsqueeze(0)
                k = k_all[s:e].permute(1, 0, 2).unsqueeze(0)
                v = v_all[s:e].permute(1, 0, 2).unsqueeze(0)
                o = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, scale=softmax_scale)
                out[s:e] = o.squeeze(0).permute(1, 0, 2)
            return out

    ptv3mod.flash_attn = _SDPAFlashShim()
    return True


def _make_synthetic_batch(*, batch_size, points_per_sample, in_channels, num_classes, ignore_index, device):
    p = batch_size * points_per_sample
    # coord/0.1 (not raw independent randint per axis): i.i.d.-uniform grid_coord
    # over a huge range makes an unrealistically sparse cloud (~1 point per
    # voxel, next to no GridPooling reduction) that can trip spconv's kernel
    # autotuner into an int32-overflow assert on the CPE SubMConv3d -- real
    # LiDAR tiles cluster far more densely than uniform noise.
    coord = torch.rand(p, 3, device=device) * 100.0
    grid_coord = (coord / 0.1).long()
    feat = torch.randn(p, in_channels, device=device)
    offset = torch.tensor(
        [points_per_sample * (i + 1) for i in range(batch_size)], device=device, dtype=torch.int64
    )
    segment = torch.randint(0, num_classes + 1, (p,), device=device, dtype=torch.int64)
    segment[segment == num_classes] = ignore_index  # sprinkle some ignore_index, like real data
    return dict(coord=coord, grid_coord=grid_coord, feat=feat, offset=offset, segment=segment)


def _cuda_mem():
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def _reset():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def build_raw_model(cfg, num_probes, device):
    from pointcept.models.builder import build_model

    model_cfg = copy.deepcopy(cfg.model)
    all_probes = list(model_cfg["probes"].items())
    if num_probes is not None:
        if num_probes > len(all_probes):
            raise ValueError(f"--num-probes {num_probes} > {len(all_probes)} probes in config")
        all_probes = all_probes[:num_probes]
    model_cfg["probes"] = dict(all_probes)
    model = build_model(model_cfg).to(device)
    return model


def build_optimizers(raw_model, device):
    from pointcept.utils.config import ConfigDict
    from pointcept.utils.optimizer import build_optimizer

    optimizers = {}
    for name in raw_model.probe_names:
        opt_cfg = ConfigDict(copy.deepcopy(dict(raw_model.probe_configs[name]["optimizer"])))
        optimizers[name] = build_optimizer(opt_cfg, params=raw_model.probe_head_parameters(name))
    return optimizers


def run_one_step(
    *,
    raw_model,
    optimizers,
    input_dict,
    enable_amp,
    amp_dtype,
    detail,
):
    """Mirrors GridProbeTrainer.run_step, with memory checkpoints between stages.

    Returns (stage_records, per_probe_records) where stage_records covers
    backbone+shared-feat-prep and the aggregate optimizer-step phase, and
    per_probe_records (only if detail=True) covers each active probe's own
    forward+loss+backward increment.
    """
    if version_ge_2_4():
        auto_cast = partial(torch.amp.autocast, device_type="cuda")
    else:
        auto_cast = torch.cuda.amp.autocast

    stage_records = []
    per_probe_records = []

    for opt in optimizers.values():
        opt.zero_grad()

    _reset()
    alloc0, res0 = _cuda_mem()
    with auto_cast(enabled=enable_amp, dtype=amp_dtype):
        feat_by_norm, point, active = raw_model.prepare_batch(input_dict)
        del point  # mirrors the GridProbeTrainer.run_step fix (see train.py)
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_res = torch.cuda.max_memory_reserved()
    alloc1, res1 = _cuda_mem()
    stage_records.append(
        dict(
            stage="backbone_fwd+shared_feat+norm_cache",
            alloc_before_mb=alloc0 / MB,
            alloc_after_mb=alloc1 / MB,
            peak_alloc_mb=peak_alloc / MB,
            peak_reserved_mb=peak_res / MB,
            n_norm_kinds=len(feat_by_norm),
        )
    )

    target = input_dict.get(raw_model.target_key)
    with auto_cast(enabled=enable_amp, dtype=amp_dtype):
        for name in active:
            if detail:
                _reset()
            alloc_before, res_before = _cuda_mem()
            logits = raw_model.probe_logits(name, feat_by_norm)
            task_loss = raw_model.criteria_by_task[name](logits, target)
            task_loss.backward()
            del logits
            if detail:
                torch.cuda.synchronize()
                peak_a = torch.cuda.max_memory_allocated()
                peak_r = torch.cuda.max_memory_reserved()
                alloc_after, res_after = _cuda_mem()
                per_probe_records.append(
                    dict(
                        probe=name,
                        alloc_before_mb=alloc_before / MB,
                        alloc_after_mb=alloc_after / MB,
                        peak_alloc_mb=peak_a / MB,
                        peak_reserved_mb=peak_r / MB,
                    )
                )
    torch.cuda.synchronize()
    heads_peak_alloc = torch.cuda.max_memory_allocated()
    heads_peak_res = torch.cuda.max_memory_reserved()
    alloc2, res2 = _cuda_mem()
    stage_records.append(
        dict(
            stage="all_heads_fwd+loss+backward",
            alloc_before_mb=alloc1 / MB,
            alloc_after_mb=alloc2 / MB,
            peak_alloc_mb=heads_peak_alloc / MB,
            peak_reserved_mb=heads_peak_res / MB,
            n_norm_kinds=None,
        )
    )

    _reset()
    alloc2b, res2b = _cuda_mem()
    for name, opt in optimizers.items():
        opt.step()
    torch.cuda.synchronize()
    opt_peak_alloc = torch.cuda.max_memory_allocated()
    opt_peak_res = torch.cuda.max_memory_reserved()
    alloc3, res3 = _cuda_mem()
    stage_records.append(
        dict(
            stage="optimizer_step (all probes)",
            alloc_before_mb=alloc2b / MB,
            alloc_after_mb=alloc3 / MB,
            peak_alloc_mb=opt_peak_alloc / MB,
            peak_reserved_mb=opt_peak_res / MB,
            n_norm_kinds=None,
        )
    )

    overall_peak_alloc = max(peak_alloc, heads_peak_alloc, opt_peak_alloc)
    overall_peak_res = max(peak_res, heads_peak_res, opt_peak_res)
    return stage_records, per_probe_records, overall_peak_alloc, overall_peak_res


def version_ge_2_4():
    from packaging import version

    return version.parse(torch.__version__) >= version.parse("2.4")


def print_stage_table(stage_records, title):
    print(f"\n=== {title} ===")
    header = f"{'stage':38s} {'alloc_before':>13s} {'alloc_after':>12s} {'peak_alloc':>11s} {'peak_reserved':>14s}"
    print(header)
    for r in stage_records:
        print(
            f"{r['stage']:38s} {r['alloc_before_mb']:11.0f}MB {r['alloc_after_mb']:10.0f}MB "
            f"{r['peak_alloc_mb']:9.0f}MB {r['peak_reserved_mb']:12.0f}MB"
        )


def print_per_probe_table(per_probe_records):
    if not per_probe_records:
        return
    print(f"\n=== per-probe forward+loss+backward increments (n={len(per_probe_records)}) ===")
    header = f"{'#':>3s} {'probe':45s} {'alloc_before':>13s} {'alloc_after':>12s} {'peak_alloc':>11s} {'delta(after-before)':>20s}"
    print(header)
    for i, r in enumerate(per_probe_records):
        delta = r["alloc_after_mb"] - r["alloc_before_mb"]
        print(
            f"{i:3d} {r['probe']:45s} {r['alloc_before_mb']:11.0f}MB {r['alloc_after_mb']:10.0f}MB "
            f"{r['peak_alloc_mb']:9.0f}MB {delta:18.1f}MB"
        )
    deltas = [r["alloc_after_mb"] - r["alloc_before_mb"] for r in per_probe_records]
    peaks = [r["peak_alloc_mb"] - r["alloc_before_mb"] for r in per_probe_records]
    print(
        f"\nresident growth per probe: min={min(deltas):.1f}MB mean={sum(deltas)/len(deltas):.1f}MB "
        f"max={max(deltas):.1f}MB (first={deltas[0]:.1f}MB, last={deltas[-1]:.1f}MB)"
    )
    print(
        f"transient (peak-before) per probe: min={min(peaks):.1f}MB mean={sum(peaks)/len(peaks):.1f}MB "
        f"max={max(peaks):.1f}MB"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--num-probes", type=int, default=None, help="Slice cfg.model.probes to the first N (dict order). Default: all probes in the config.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override cfg.batch_size (per-GPU, single-GPU diagnostic).")
    parser.add_argument("--points-per-sample", type=int, default=None, help="Override cfg.point_max (worst-case: SphereCrop cap).")
    parser.add_argument("--detail", action="store_true", help="Per-probe forward/backward memory trace (stage table always printed).")
    parser.add_argument("--sweep", type=int, nargs="+", default=None, help="Probe-count values to sweep (fresh model rebuilt per value); fits fixed-cost + marginal-cost-per-probe.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")
    device = "cuda"
    torch.manual_seed(args.seed)

    shimmed = _install_sdpa_flash_shim()
    if shimmed:
        print(
            "[warn] flash_attn not importable here -- PT-v3m2 attention routed through "
            "torch SDPA instead (same non-materializing memory profile on Ampere+, but "
            "NOT the real flash_attn kernel: absolute numbers are indicative, not "
            "bit-identical to a real flash_attn run on the target GPU)."
        )

    from pointcept.utils.config import Config

    cfg = Config.fromfile(args.config_file)
    batch_size = args.batch_size or cfg.batch_size
    points_per_sample = args.points_per_sample or cfg.point_max
    total_points = batch_size * points_per_sample
    enable_amp = bool(getattr(cfg, "enable_amp", False))
    amp_dtype_name = getattr(cfg, "amp_dtype", "float16")
    amp_dtype = dict(float16=torch.float16, bfloat16=torch.bfloat16)[amp_dtype_name]
    in_channels = cfg.model["backbone"]["in_channels"]
    backbone_out_channels = cfg.model["backbone_out_channels"]
    num_classes = cfg.model["num_classes"]
    ignore_index = cfg.model["ignore_index"]
    total_probes_in_cfg = len(cfg.model["probes"])

    print(f"config: {args.config_file}")
    print(f"batch_size={batch_size} points_per_sample={points_per_sample} total_points={total_points:,}")
    print(f"enable_amp={enable_amp} amp_dtype={amp_dtype_name} backbone_out_channels={backbone_out_channels}")
    print(f"probes in config: {total_probes_in_cfg}")

    bytes_per_elem = 2 if enable_amp else 4
    feat_gb = total_points * backbone_out_channels * bytes_per_elem / GB
    print(
        f"\n[analytic] shared backbone feat [{total_points:,}, {backbone_out_channels}] "
        f"@ {bytes_per_elem}B/elem = {feat_gb:.2f} GB (paid once, independent of probe count)"
    )
    print(
        "[analytic] each DISTINCT input_norm kind other than None used by the active "
        f"probes adds another ~{feat_gb:.2f} GB (unitsphere cache) -- see _input_norm_cache."
    )

    input_dict = _make_synthetic_batch(
        batch_size=batch_size,
        points_per_sample=points_per_sample,
        in_channels=in_channels,
        num_classes=num_classes,
        ignore_index=ignore_index,
        device=device,
    )

    def run_for_n(n_probes, detail):
        torch.cuda.empty_cache()
        _reset()
        raw_model = build_raw_model(cfg, n_probes, device)
        optimizers = build_optimizers(raw_model, device)
        # warmup: first call pays cudnn/kernel-selection + CUDA-context costs
        # that would otherwise contaminate the measured numbers.
        run_one_step(
            raw_model=raw_model,
            optimizers=optimizers,
            input_dict=input_dict,
            enable_amp=enable_amp,
            amp_dtype=amp_dtype,
            detail=False,
        )
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        stage_records, per_probe_records, overall_peak_alloc, overall_peak_res = run_one_step(
            raw_model=raw_model,
            optimizers=optimizers,
            input_dict=input_dict,
            enable_amp=enable_amp,
            amp_dtype=amp_dtype,
            detail=detail,
        )
        del raw_model, optimizers
        return stage_records, per_probe_records, overall_peak_alloc, overall_peak_res

    if args.sweep:
        print(f"\n{'='*100}\nSWEEP over probe counts: {args.sweep}\n{'='*100}")
        results = []
        for n in args.sweep:
            if n > total_probes_in_cfg:
                print(f"skip N={n}: only {total_probes_in_cfg} probes in config")
                continue
            stage_records, _, overall_peak_alloc, overall_peak_res = run_for_n(n, detail=False)
            results.append((n, overall_peak_alloc / MB, overall_peak_res / MB))
            print(f"N={n:4d} probes -> overall peak_alloc={overall_peak_alloc/GB:.2f}GB peak_reserved={overall_peak_res/GB:.2f}GB")

        if len(results) >= 2:
            import numpy as np

            ns = np.array([r[0] for r in results], dtype=float)
            peaks_gb = np.array([r[1] / 1024 for r in results], dtype=float)
            slope, intercept = np.polyfit(ns, peaks_gb, 1)
            print(
                f"\n[fit] peak_alloc(N) ~= {intercept:.2f} GB + {slope*1024:.1f} MB * N_probes "
                f"(linear fit over N={list(ns.astype(int))})"
            )
            print(
                "      intercept = fixed cost (backbone fwd + shared feat + norm caches + CUDA context)\n"
                "      slope     = marginal VRAM per additional probe head"
            )
    else:
        n_probes = args.num_probes
        stage_records, per_probe_records, overall_peak_alloc, overall_peak_res = run_for_n(
            n_probes, detail=args.detail
        )
        title = f"N={n_probes or total_probes_in_cfg} probes, batch_size={batch_size}, points_per_sample={points_per_sample}"
        print_stage_table(stage_records, title)
        print_per_probe_table(per_probe_records)
        print(f"\noverall peak_alloc={overall_peak_alloc/GB:.2f}GB peak_reserved={overall_peak_res/GB:.2f}GB")


if __name__ == "__main__":
    main()
