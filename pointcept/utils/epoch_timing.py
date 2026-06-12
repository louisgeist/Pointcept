"""
Wall-clock timing helpers for per-epoch train/validation logging.
"""

import time

import torch
import wandb

import pointcept.utils.comm as comm


def begin_epoch_wall_timing():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def finalize_epoch_wall_timing(
    trainer,
    start_time,
    current_epoch,
    prefix,
    wandb_dict=None,
    log_console=False,
):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    comm.synchronize()
    total_s = time.perf_counter() - start_time
    if comm.is_main_process():
        timing = {
            f"{prefix}/s_per_epoch": float(total_s),
            f"{prefix}/h_per_epoch": float(total_s / 3600.0),
        }
        if log_console:
            phase = "Validation" if prefix == "val" else "Train"
            trainer.logger.info(
                "%s wall time (epoch %d): %.2fs (%.4fh)",
                phase,
                current_epoch,
                total_s,
                total_s / 3600.0,
            )
        if wandb_dict is not None:
            wandb_dict.update(timing)
        elif (
            getattr(trainer.cfg, "enable_wandb", False)
            and wandb.run is not None
        ):
            wandb.log({"Epoch": current_epoch, **timing})
    return total_s
