"""
Persistent cumulative runtime across Slurm requeue segments.

Tracks active compute time per segment in ``${save_path}/runtime_state.json``.
Queue wait time between requeued segments is excluded.
"""

import json
import os
import time

RUNTIME_STATE_FILENAME = "runtime_state.json"


def runtime_state_path(save_path):
    return os.path.join(save_path, RUNTIME_STATE_FILENAME)


def resolve_save_path_for_runtime():
    for key in ("JOB_DIR", "POINTCEPT_SAVE_PATH"):
        value = os.environ.get(key)
        if value and os.path.isdir(value):
            return value
    return None


def _load_state(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_state(path, state):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    os.replace(tmp_path, path)


def _default_state():
    now = time.time()
    return {"cumulative_runtime_s": 0.0, "segment_wall_start": now}


def init_runtime_segment(save_path):
    """Load or create runtime state and start a new segment timer."""
    path = runtime_state_path(save_path)
    state = _load_state(path)
    if state is None:
        state = _default_state()
    else:
        state["segment_wall_start"] = time.time()
    _save_state(path, state)
    return state


def flush_runtime_segment(save_path):
    """Add elapsed segment time to cumulative runtime and reset segment start."""
    path = runtime_state_path(save_path)
    state = _load_state(path)
    if state is None:
        state = _default_state()
    now = time.time()
    segment_start = float(state.get("segment_wall_start", now))
    state["cumulative_runtime_s"] = float(state.get("cumulative_runtime_s", 0.0)) + (
        now - segment_start
    )
    state["segment_wall_start"] = now
    _save_state(path, state)
    return float(state["cumulative_runtime_s"])


def get_total_runtime_s(save_path):
    """Flush the current segment and return cumulative runtime in seconds."""
    return flush_runtime_segment(save_path)


def flush_runtime_segment_from_env():
    """Flush runtime state using JOB_DIR or POINTCEPT_SAVE_PATH, if available."""
    save_path = resolve_save_path_for_runtime()
    if save_path is None:
        return None
    if not os.path.isfile(runtime_state_path(save_path)):
        return None
    return flush_runtime_segment(save_path)
