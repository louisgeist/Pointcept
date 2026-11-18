"""Re-test a finished OpenGF seed ensemble (10 probes) on each OpenGF test region separately.

The paper's OpenGF test region T2 is float32-quantized in the published data (X 0.125 m, Y 0.5 m), T1 and T3
are not, so every OpenGF probe is also reported per region (coord-fix campaign rule, 2026-11-17).

Usage: python scripts/jz/retest_opengf_regions.py <seed_config.py> <seeds_model_dir> <out_dir> T3_ T2_ T1_
Builds the model from <seed_config.py>, loads <seeds_model_dir>/model_last.pth (backbone + heads), then each
probe's best-val head (probe_best_<name>.pth, as GridProbeSeedEnsembleTester does), and runs the tester of
the config on data.test restricted to each region. Writes <out_dir>/<region>.json; a region whose JSON
already exists is skipped (safe to rerun after a requeue).
Run automatically at the end of a `scripts/jz/job.slurm sweep` job on an OpenGF config.
DRY=1: CPU model, one test batch per region, no test pass (login-node check).
"""

import json
import os
import sys

import numpy as np
import torch

from pointcept.engines.defaults import default_setup
from pointcept.engines.test import TESTERS, TesterBase
from pointcept.utils.config import Config

cfg_path, model_dir, out_dir, *regions = sys.argv[1:]
DRY = os.environ.get("DRY") == "1"
if DRY:
    from collections import OrderedDict

    from pointcept.models import build_model

    def _cpu_model(self):
        model = build_model(self.cfg.model)
        sd = torch.load(self.cfg.weight, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        model.load_state_dict(
            OrderedDict(
                (k[7:] if k.startswith("module.") else k, v) for k, v in sd.items()
            ),
            strict=True,
        )
        return model

    TesterBase.build_model = _cpu_model
os.makedirs(out_dir, exist_ok=True)
for region in regions:
    out_json = os.path.join(out_dir, f"{region}.json")
    if os.path.exists(out_json) and not DRY:
        print(region, "already done:", out_json, flush=True)
        continue
    cfg = Config.fromfile(cfg_path)
    cfg.weight = os.path.join(model_dir, "model_last.pth")
    cfg.save_path = os.path.join(out_dir, region)
    cfg.resume = False
    cfg.data.test.include_names = region
    cfg = default_setup(cfg)
    os.makedirs(cfg.save_path, exist_ok=True)
    tester = TESTERS.build(dict(cfg=cfg, **cfg.test))
    raw = tester.model.module if hasattr(tester.model, "module") else tester.model
    for name in raw.probe_names:
        state = torch.load(
            os.path.join(model_dir, f"probe_best_{name}.pth"),
            map_location="cpu",
            weights_only=False,
        )
        raw.heads[name].load_state_dict(state)
    raw.active_probe = None
    if DRY:
        batch = next(iter(tester.test_loader))
        print(
            region,
            "dry ok:",
            len(tester.test_loader.dataset),
            "scenes, batch keys",
            sorted(batch)[:8],
            flush=True,
        )
        continue
    tester.test()
    by_probe = tester.test_metrics_by_probe
    keys = sorted(
        {k for m in by_probe.values() for k in m if isinstance(m[k], (int, float))}
    )
    summary = {
        k: [float(m[k]) for m in by_probe.values() if k in m] for k in keys
    }
    out = {
        "region": region,
        "n_scenes": len(tester.test_loader.dataset),
        "weight": cfg.weight,
        **{f"{k}_mean": float(np.mean(v)) for k, v in summary.items()},
        **{f"{k}_std": float(np.std(v)) for k, v in summary.items()},
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    print(
        region,
        {
            k: round(v * 100, 2)
            for k, v in out.items()
            if k.endswith("_mean") and "iou" in k.lower()
        },
        flush=True,
    )
    del tester
    torch.cuda.empty_cache()
