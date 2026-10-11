"""Shared helper: run tools/eval_network_apls.py from within the training/testing engines.

Used by both the standalone `tools/test.py` entrypoint and the `NetworkAPLSEvaluator` training
hook (`pointcept/engines/hooks/misc.py`) so the config/guard/error-handling logic lives in one
place. `tools/eval_network_apls.py` is a standalone script (not a package module, no
`__init__.py` under `tools/`), so it's imported here via a `sys.path` insert -- the same pattern
`eval_network_apls.py` itself uses to import its `pointcept/datasets/preprocessing/malibu3d/`
dependencies.
"""

import sys
from pathlib import Path

from pointcept.utils.comm import is_main_process

# APLS metric parameters fixed by ``APLS_PROTOCOL`` in ``apls_metric.py``. A config must
# not set them; they are listed only to reject such configs with an explicit message.
_REMOVED_APLS_KEYS = (
    "apls_densify",
    "apls_snap_to_edge",
    "apls_symmetric",
    "apls_min_path_length_m",
    "apls_max_nodes_exact",
)


def validate_network_apls_eval_cfg(cfg):
    """Fail at config-parse time if ``cfg.network_apls_eval`` still sets fixed APLS params."""
    opts = cfg.get("network_apls_eval", None)
    if not opts:
        return
    stale = [k for k in _REMOVED_APLS_KEYS if k in opts]
    if stale:
        raise ValueError(
            f"network_apls_eval: {stale} can no longer be set. The APLS metric parameters "
            "are fixed (see APLS_PROTOCOL in pointcept/datasets/preprocessing/malibu3d/"
            "apls_metric.py); remove these keys from the config. Only the mask->graph "
            "parameters (threshold, morphology, merge, ...) remain configurable."
        )


def run_network_apls_eval_if_configured(cfg, logger):
    """Score network APLS after a test pass if ``cfg.network_apls_eval`` is set.

    ``cfg.network_apls_eval`` is a dict of keyword arguments for ``eval_network_apls.run``
    (plus optional ``network_graphs_root`` and ``out_dir``); no-op when absent, and on
    non-main ranks. Reads the ``{patch_id}_logits_network.npy`` files that
    ``MultiTaskTester`` wrote to ``cfg.save_path/result``, stitches them per ROI and scores
    APLS. Returns the metrics dict, or None. Never raises: failures are logged, so this
    post-processing step cannot fail the caller's test run.
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
    # None: GT graphs sit in each ROI directory (Hugging Face layout of the dataset).
    network_graphs_root = opts.pop("network_graphs_root", None)

    # Align APLS channels with the trained head (e.g. ROADS+RAILROADS only).
    if "network_types" not in opts and task_configs and "network" in task_configs:
        channel_names = task_configs["network"].get("channel_names")
        if channel_names:
            opts["network_types"] = list(channel_names)

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
            Path(network_graphs_root) if network_graphs_root else None,
            Path(cfg.csv_manifest),
            out_dir,
            **opts,
        )
    except Exception:
        logger.exception("Network APLS eval failed -- continuing.")
        return None
