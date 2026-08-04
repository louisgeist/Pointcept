"""Shared helper: run tools/eval_network_apls.py from within the training/testing engines.

Used by both the standalone `tools/test.py` entrypoint and the `NetworkAPLSEvaluator` training
hook (`pointcept/engines/hooks/misc.py`) so the config/guard/error-handling logic lives in one
place. `tools/eval_network_apls.py` is a standalone script (not a package module, no
`__init__.py` under `tools/`), so it's imported here via a `sys.path` insert -- the same pattern
`eval_network_apls.py` itself uses to import its `pointcept/datasets/preprocessing/flair3d_plus/`
dependencies.
"""

import sys
from pathlib import Path

from pointcept.utils.comm import is_main_process


def run_network_apls_eval_if_configured(cfg, logger):
    """Opt-in via `cfg.network_apls_eval` (a dict); no-op if absent.

    Reads `{patch_id}_logits_network.npy` from `cfg.save_path/result` (already written by
    `MultiTaskTester`), stitches per-ROI and scores APLS. Never raises -- failures are logged
    and swallowed, since this is a convenience post-processing step, not part of the caller's
    pass/fail criteria.
    """
    opts = cfg.get("network_apls_eval", None)
    if not opts:
        return None
    if not is_main_process():
        return None

    task_configs = cfg.get("data", {}).get("task_configs", None)
    if task_configs and "network" not in task_configs:
        logger.warning(
            "cfg.network_apls_eval is set but 'network' is not in cfg.data.task_configs "
            "-- skipping APLS eval."
        )
        return None

    opts = dict(opts)
    try:
        network_graphs_root = opts.pop("network_graphs_root")
    except KeyError:
        logger.warning(
            "cfg.network_apls_eval requires 'network_graphs_root' -- skipping APLS eval."
        )
        return None

    tools_dir = str(Path(__file__).resolve().parents[2] / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import eval_network_apls

    save_path = Path(cfg.save_path) / "result"
    out_dir = Path(opts.pop("out_dir", save_path))
    try:
        return eval_network_apls.run(
            Path(cfg.data_root),
            save_path,
            Path(network_graphs_root),
            Path(cfg.csv_manifest),
            out_dir,
            **opts,
        )
    except Exception:
        logger.exception("Network APLS eval failed -- continuing.")
        return None
