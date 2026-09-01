"""Keep W&B's internal ``_step`` monotonic across process-level resumes.

Charts use the ``Epoch`` field as x-axis (see ``define_wandb_metrics``), so we
never pass ``step=epoch`` to ``wandb.log``: after a Slurm requeue that value
is below the run's internal step and W&B would drop the log. The remaining
hazard is a *new process* that resumes the same run id while the local
``_step`` counter restarts near zero and collides with already-uploaded
history (train logs of epoch N merge with leftover val logs of epoch N-73).

Call ``bump_wandb_step_on_resume`` once after ``wandb.init(resume=...)`` so
the next ``wandb.log`` lands past ``lastHistoryStep``.

The step is sourced locally (``wandb_last_step.txt`` next to
``wandb_run_id.txt`` in ``save_path``, refreshed each epoch by
``CheckpointSaver``) rather than via ``wandb.Api()``: that call ignores
``WANDB_MODE=offline`` and, on a compute node with no internet egress (e.g.
cluster), blocks in an unbounded retry loop on ConnectionError instead of
raising — silently burning the whole job's walltime before training even
starts. The local file works identically online and offline and needs no
network at all.
"""

import os


def local_last_step_path(save_path):
    return os.path.join(save_path, "wandb_last_step.txt")


def write_local_last_step(save_path, step):
    """Persist the current W&B step so a future resume can bump past it
    without any network call. Best-effort: never raises."""
    try:
        with open(local_last_step_path(save_path), "w") as f:
            f.write(str(int(step)))
    except OSError:
        pass


def read_local_last_step(save_path):
    """Read back the step written by ``write_local_last_step``, or None."""
    path = local_last_step_path(save_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            text = f.read().strip()
        return int(text) if text else None
    except (OSError, ValueError):
        return None


def compute_resume_step(local_step, last_history_step):
    """Return the ``_step`` to use after a W&B resume.

    If ``last_history_step`` is unknown (offline / API failure), keep the
    local counter unchanged rather than guessing.
    """
    local_step = int(local_step or 0)
    if last_history_step is None:
        return local_step
    return max(local_step, int(last_history_step) + 1)


def fetch_last_history_step(project, run_id, entity=None, api=None):
    """Query W&B for ``lastHistoryStep``. Returns None on any failure."""
    try:
        import wandb

        if api is None:
            api = wandb.Api()
        path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
        run = api.run(path)
        step = getattr(run, "lastHistoryStep", None)
        if step is None:
            return None
        return int(step)
    except Exception:
        return None


def bump_wandb_run_step(wandb_run, last_history_step, logger=None):
    """Assign ``wandb_run._step`` so the next log cannot collide with history.

    Returns the step that will be used. Warns (does not raise) when
    ``last_history_step`` is unknown.
    """
    local_step = int(getattr(wandb_run, "step", 0) or 0)
    if last_history_step is None:
        if logger is not None:
            logger.warning(
                "W&B resume: could not query lastHistoryStep; skipping _step bump"
            )
        return local_step
    new_step = compute_resume_step(local_step, last_history_step)
    if new_step != local_step:
        wandb_run._step = new_step
        if logger is not None:
            logger.info(
                "W&B resume: bumped _step from %s to %s (lastHistoryStep=%s)",
                local_step,
                new_step,
                last_history_step,
            )
    return new_step


def bump_wandb_step_on_resume(
    wandb_run, project, run_id, logger=None, entity=None, api=None, local_last_step=None
):
    """Bump ``wandb_run._step`` past the last step used before this resume.

    Prefers ``local_last_step`` (read from the on-disk sidecar file, no
    network needed) over the ``wandb.Api()`` fallback, which only makes
    sense for a run that is both online and missing that local record (e.g.
    a crash before the first checkpoint) — offline runs can't reach the API
    at all (see module docstring for why that call is otherwise unsafe).
    """
    if local_last_step is not None:
        return bump_wandb_run_step(wandb_run, local_last_step, logger=logger)

    mode = getattr(getattr(wandb_run, "settings", None), "mode", None)
    if mode == "offline":
        if logger is not None:
            logger.info(
                "W&B resume: run is offline and no local step record found; "
                "skipping lastHistoryStep fetch"
            )
        return int(getattr(wandb_run, "step", 0) or 0)
    last = fetch_last_history_step(project, run_id, entity=entity, api=api)
    return bump_wandb_run_step(wandb_run, last, logger=logger)
