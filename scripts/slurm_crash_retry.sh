#!/bin/sh
# Source this in a sbatch script before wrapping the training `srun` call in a
# retry loop, e.g.:
#
#   . scripts/slurm_crash_retry.sh
#   while true; do
#     ATTEMPT_START=$(date +%s)
#     srun ... bash -c "... sh scripts/train.sh ..."
#     TRAIN_EXIT_CODE=$?
#     ATTEMPT_DURATION=$(( $(date +%s) - ATTEMPT_START ))
#     [ "$TRAIN_EXIT_CODE" -eq 0 ] && break
#     should_retry_after_crash "$ATTEMPT_DURATION" "${JOB_DIR}" || break
#   done
#
# should_retry_after_crash: retry in place (same Slurm allocation, same JOB_DIR
# so scripts/train.sh resumes from model_last.pth if one was saved) up to
# POINTCEPT_MAX_CRASH_RETRIES (default 3) consecutive quick failures. The
# budget resets whenever the previous attempt ran at least
# POINTCEPT_CRASH_RESET_THRESHOLD_SEC (default 600s) before dying, so sporadic
# crashes (e.g. one bad sample) don't exhaust the budget of a long-running,
# otherwise-healthy job. Never retries if a graceful walltime-requeue
# (scripts/slurm_requeue_trap.sh / slurm_requeue_watchdog.sh /
# pointcept/utils/slurm_requeue.py) already fired for this JOB_DIR, since that
# allocation is being torn down by Slurm anyway.
should_retry_after_crash() {
  duration="$1"
  job_dir="$2"

  if [ -f "${job_dir}/.requeue_triggered" ]; then
    echo "[slurm_crash_retry] graceful walltime-requeue already triggered - this allocation is ending, not retrying in place" >&2
    return 1
  fi

  max_retries="${POINTCEPT_MAX_CRASH_RETRIES:-3}"
  reset_threshold="${POINTCEPT_CRASH_RESET_THRESHOLD_SEC:-600}"

  retry_count=0
  if [ -f "${job_dir}/.crash_retry_count" ]; then
    retry_count=$(cat "${job_dir}/.crash_retry_count")
  fi

  if [ "$duration" -ge "$reset_threshold" ]; then
    echo "[slurm_crash_retry] previous attempt ran ${duration}s (>= ${reset_threshold}s) before crashing - treating as a one-off, resetting retry budget" >&2
    retry_count=0
  fi

  if [ "$retry_count" -ge "$max_retries" ]; then
    echo "[slurm_crash_retry] ${retry_count} consecutive quick crashes (limit ${max_retries}) - giving up, needs manual investigation" >&2
    return 1
  fi

  new_retry_count=$((retry_count + 1))
  echo "$new_retry_count" > "${job_dir}/.crash_retry_count"
  echo "[slurm_crash_retry] crashed after ${duration}s, retrying in place within the same allocation (attempt ${new_retry_count}/${max_retries})" >&2
  return 0
}
