"""
Grid-Search Config Authoring Helpers

Pure-stdlib helper for building a GridProbeSegmentorV2 `probes` dict as a
cartesian product of axes (e.g. every loss variant x every learning rate).
No torch / pointcept.datasets import — safe to import directly inside a
config file (unlike pointcept.datasets.malibu3d_config_utils, which configs
avoid importing at parse time because it pulls torch_cluster).
"""

import copy
import itertools


def _deep_merge(base, override):
    """Deep-copy `base`, then recursively merge `override` into it: dict
    values are merged key-by-key (recursing), everything else is replaced."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def cartesian_probes(base, *axes, name_fn=None):
    """Cartesian product over N axes of partial probe-config dicts, each
    combination deep-merged onto `base` (shared defaults).

    axes: any number of lists of dicts, e.g.
        losses = [dict(criteria=[ce_cfg]), dict(criteria=[focal_cfg])]
        lrs = [dict(optimizer=dict(type="AdamW", lr=v, weight_decay=0.02))
               for v in [1e-4, 5e-4, 2e-3]]
        probes = cartesian_probes(dict(dropout=0.0, grad_clip=3.0), losses, lrs)
        # -> 2 x 3 = 6 probes, auto-named probe_0_0 .. probe_1_2.

    Deep-merge means e.g. an axis entry that only sets optimizer.lr combines
    with a `base` that already sets optimizer.type/weight_decay, instead of
    wholesale-replacing the optimizer dict.

    name_fn(indices: tuple[int, ...]) -> str; default "probe_" + "_".join(indices).
    Composes freely with hand-written one-off probes: `probes.update(...)`.
    """
    if not axes:
        raise ValueError("cartesian_probes needs at least one axis.")
    name_fn = name_fn or (lambda idx: "probe_" + "_".join(map(str, idx)))
    probes = {}
    for indices in itertools.product(*(range(len(axis)) for axis in axes)):
        cfg = copy.deepcopy(base)
        for axis, i in zip(axes, indices):
            cfg = _deep_merge(cfg, axis[i])
        name = name_fn(indices)
        if name in probes:
            raise ValueError(f"Duplicate probe name {name!r} from indices {indices}.")
        probes[name] = cfg
    return probes
