"""
Standalone test pass for a saved GridProbe winner checkpoint.

GridProbeWinnerSelector.after_train() (pointcept/engines/hooks/grid_probe.py)
runs this same test pass at the end of a grid-probe job, but only when
skip_test=False *and* training reaches after_train. A sweep run without test,
or a job stopped before the end, leaves model/probe_best_<name>.pth on disk
untested. This script reproduces that path: build the model from the original
grid config (same probes dict, so the head architecture matches), load the
backbone from cfg.weight (same key remap as the CheckpointLoader hook), load
one probe head from a probe_best_<name>.pth file, make it the active probe,
and run the tester.

Usage:
    export PYTHONPATH=$PWD
    python tools/test_grid_probe_winner.py \
      --config-file configs/h3d/sonata-v1m2-h3d-lin-grid.py \
      --options save_path=exp/h3d/sonata_grid/test_probe_best \
                probe_name=<name> \
                probe_weight=exp/h3d/sonata_grid/model/probe_best_<name>.pth

probe_name must match a key of the config's `probes` dict (the
probe_best_<name>.pth suffix). probe_weight is that file's path; cfg.weight
(set in the config, or via --options weight=ckpt/<name>/model.pth) is the
frozen backbone.
"""

from collections import OrderedDict

import torch

import pointcept.utils.comm as comm
from pointcept.engines.defaults import (
    create_ddp_model,
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.launch import launch
from pointcept.engines.test import TESTERS
from pointcept.models import build_model


def main_worker(cfg):
    cfg = default_setup(cfg)

    probe_name = cfg.probe_name
    probe_weight = cfg.probe_weight
    if probe_name not in cfg.model.probes:
        raise ValueError(
            f"probe_name={probe_name!r} not found in this config's probes "
            f"dict (available: {sorted(cfg.model.probes.keys())})"
        )

    model = build_model(cfg.model)

    # Backbone: same keyword remap / strict=False as CheckpointLoader hook.
    checkpoint = torch.load(cfg.weight, map_location="cpu", weights_only=False)
    weight = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        if not key.startswith("module."):
            key = "module." + key
        if "module.student.backbone" in key:
            key = key.replace("module.student.backbone", "module.backbone", 1)
        if comm.get_world_size() == 1:
            key = key[7:]  # module.xxx.xxx -> xxx.xxx
        weight[key] = value
    missing, unexpected = model.load_state_dict(weight, strict=False)
    print(f"Backbone load from {cfg.weight} -- missing: {missing}, unexpected: {unexpected}")

    # Winning probe head (a few-KB standalone state_dict, saved by
    # GridProbeCheckpointSaver -- not nested under "heads.<name>.").
    head_state = torch.load(probe_weight, map_location="cpu", weights_only=False)
    model.heads[probe_name].load_state_dict(head_state)
    model.active_probe = probe_name

    model = create_ddp_model(
        model.cuda(), broadcast_buffers=False, find_unused_parameters=cfg.find_unused_parameters
    )

    test_cfg = dict(cfg=cfg, model=model, **cfg.test)
    tester = TESTERS.build(test_cfg)
    tester.test()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
