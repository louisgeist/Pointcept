"""
Slurm timeout requeue for Jean Zay and similar clusters.

Activated when POINTCEPT_SLURM_REQUEUE=1 and SLURM_JOB_ID are set.

Uses two mechanisms (either is enough):
1. SIGUSR1 from `#SBATCH --signal=USR1@XX` or `B:USR1@XX` (fast path)
2. Background poll of `squeue` time left (works with jz_utils + srun nesting)
"""

import logging
import os
import signal
import subprocess
import sys
import threading

from pointcept.utils.comm import is_main_process

_HANDLER_INSTALLED = False
_REQUEUE_TRIGGERED = False
_REQUEUE_LOCK = threading.Lock()

_DEFAULT_MARGIN_SEC = 120
_DEFAULT_POLL_SEC = 30


def _log_requeue(msg):
    logging.getLogger(__name__).warning(msg)
    print(msg, file=sys.stderr, flush=True)


def parse_slurm_duration_to_seconds(value):
    """Parse Slurm duration strings (e.g. 2-03:00:00, 1:30:00, 59:30) to seconds."""
    if not value:
        return None
    text = value.strip().split()[0]
    if not text or text in {"N/A", "NOT_SET", "UNLIMITED", "INVALID"}:
        return None

    days = 0
    if "-" in text:
        days_part, text = text.split("-", 1)
        try:
            days = int(days_part)
        except ValueError:
            return None

    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            hours, minutes, seconds = 0, int(parts[0]), int(parts[1])
        else:
            return None
    except ValueError:
        return None

    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def get_slurm_time_left_seconds(job_id=None):
    """Return remaining walltime in seconds for a Slurm job, or None if unknown."""
    job_id = job_id or os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    try:
        # %L = time left before walltime (scontrol has no such field/format option)
        result = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-o", "%L"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    return parse_slurm_duration_to_seconds(first_line)


def _finish_wandb_if_active():
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def _trigger_slurm_requeue(reason):
    global _REQUEUE_TRIGGERED
    with _REQUEUE_LOCK:
        if _REQUEUE_TRIGGERED:
            return
        _REQUEUE_TRIGGERED = True

    _log_requeue(f"Slurm requeue triggered ({reason})")

    if is_main_process():
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id is None:
            _log_requeue("SLURM_JOB_ID is not set; cannot requeue.")
        else:
            try:
                from pointcept.utils.runtime_state import flush_runtime_segment_from_env

                flushed = flush_runtime_segment_from_env()
                if flushed is not None:
                    _log_requeue(
                        f"Flushed cumulative runtime before requeue: {flushed:.2f}s"
                    )
            except Exception:
                pass
            _finish_wandb_if_active()
            job_dir = os.environ.get("JOB_DIR")
            if job_dir:
                try:
                    open(os.path.join(job_dir, ".requeue_triggered"), "a").close()
                except OSError:
                    pass
            _log_requeue(f"Requeuing Slurm job {job_id}")
            subprocess.run(["scontrol", "requeue", job_id], check=False)
    else:
        _log_requeue("Non-main process exiting after Slurm requeue request.")

    os._exit(1)


def _sigusr1_handler(signum, frame):
    _trigger_slurm_requeue(f"SIGUSR1 signum={signum}")


def _poll_slurm_time_left():
    margin = int(os.environ.get("POINTCEPT_SLURM_REQUEUE_MARGIN_SEC", _DEFAULT_MARGIN_SEC))
    poll = int(os.environ.get("POINTCEPT_SLURM_REQUEUE_POLL_SEC", _DEFAULT_POLL_SEC))
    job_id = os.environ.get("SLURM_JOB_ID")
    _log_requeue(
        f"Slurm requeue poll watcher started for job {job_id} "
        f"(margin={margin}s, poll={poll}s)"
    )

    first_read = True
    while True:
        left = get_slurm_time_left_seconds(job_id)
        if left is not None and first_read:
            _log_requeue(f"Slurm requeue poll: {left}s left before walltime")
            first_read = False
        if left is not None and left <= margin:
            _trigger_slurm_requeue(f"time_left={left}s <= margin={margin}s")
        threading.Event().wait(poll)


def install_slurm_timeout_requeue_handler(logger=None):
    """Install SIGUSR1 handler and optional time-left poll watcher."""
    global _HANDLER_INSTALLED
    if _HANDLER_INSTALLED:
        return

    if os.environ.get("POINTCEPT_SLURM_REQUEUE") != "1":
        return
    if "SLURM_JOB_ID" not in os.environ:
        return

    job_id = os.environ["SLURM_JOB_ID"]
    msg = f"Installing Slurm timeout requeue handler for job {job_id}"
    if logger is not None:
        logger.warning(msg)
    _log_requeue(msg)

    signal.signal(signal.SIGUSR1, _sigusr1_handler)
    _HANDLER_INSTALLED = True

    if is_main_process():
        thread = threading.Thread(target=_poll_slurm_time_left, daemon=True)
        thread.start()
