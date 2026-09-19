"""
Main Testing Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import wandb

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch
from pointcept.utils.network_apls import run_network_apls_eval_if_configured
from pointcept.utils.wandb_resume import init_or_resume_wandb


def main_worker(cfg):
    cfg = default_setup(cfg)
    test_cfg = dict(cfg=cfg, **cfg.test)
    tester = TESTERS.build(test_cfg)
    init_or_resume_wandb(cfg, logger=tester.logger)
    try:
        tester.test()
        run_network_apls_eval_if_configured(cfg, tester.logger)
    finally:
        if wandb.run is not None:
            wandb.finish()


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
