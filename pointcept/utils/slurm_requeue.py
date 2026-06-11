"""
Slurm timeout requeue handler for Jean Zay and similar clusters.

Activated when POINTCEPT_SLURM_REQUEUE=1 and SLURM_JOB_ID are set.
Requires #SBATCH --signal=USR1@XX in the submission script.
"""

import logging
import os
import signal
import subprocess

from pointcept.utils.comm import is_main_process

_HANDLER_INSTALLED = False


def _finish_wandb_if_active():
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def _sigusr1_handler(signum, frame):
    logger = logging.getLogger(__name__)
    logger.warning("Slurm timeout signal received (signum=%s)", signum)

    if is_main_process():
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id is None:
            logger.error("SLURM_JOB_ID is not set; cannot requeue.")
        else:
            _finish_wandb_if_active()
            logger.warning("Requeuing Slurm job %s", job_id)
            subprocess.run(
                ["scontrol", "requeue", job_id],
                check=False,
            )
    else:
        logger.warning("Non-main process exiting after Slurm timeout signal.")

    os._exit(1)


def install_slurm_timeout_requeue_handler(logger=None):
    """Install SIGUSR1 handler for Slurm walltime pre-timeout requeue."""
    global _HANDLER_INSTALLED
    if _HANDLER_INSTALLED:
        return

    if os.environ.get("POINTCEPT_SLURM_REQUEUE") != "1":
        return
    if "SLURM_JOB_ID" not in os.environ:
        return

    if logger is not None:
        logger.warning(
            "Installing Slurm timeout requeue handler for job %s",
            os.environ["SLURM_JOB_ID"],
        )
    else:
        logging.getLogger(__name__).warning(
            "Installing Slurm timeout requeue handler for job %s",
            os.environ["SLURM_JOB_ID"],
        )

    signal.signal(signal.SIGUSR1, _sigusr1_handler)
    _HANDLER_INSTALLED = True
