#!/bin/sh
# Poll Slurm walltime and requeue before timeout.
# Started by train_jz_utils.sh; uses repo scripts/ (not JOB_DIR/code snapshot).

set -eu

if [ -z "${SLURM_JOB_ID:-}" ]; then
  exit 0
fi

MARGIN="${POINTCEPT_SLURM_REQUEUE_MARGIN_SEC:-120}"
POLL="${POINTCEPT_SLURM_REQUEUE_POLL_SEC:-30}"

_parse_time_left() {
  python3 <<'PY'
import os
import subprocess
import sys

job_id = os.environ.get("SLURM_JOB_ID")
if not job_id:
    sys.exit(1)

# %L = time left before walltime (scontrol has no such field/format option)
result = subprocess.run(
    ["squeue", "-h", "-j", job_id, "-o", "%L"],
    capture_output=True,
    text=True,
    timeout=15,
    check=False,
)
if result.returncode != 0:
    sys.exit(1)

text = (result.stdout or "").strip().split()[0] if result.stdout else ""
if not text or text in {"N/A", "NOT_SET", "UNLIMITED", "INVALID"}:
    sys.exit(1)

days = 0
if "-" in text:
    days_part, text = text.split("-", 1)
    days = int(days_part)

parts = text.split(":")
if len(parts) == 3:
    hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
elif len(parts) == 2:
    hours, minutes, seconds = 0, int(parts[0]), int(parts[1])
else:
    sys.exit(1)

total = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
print(total)
PY
}

FIRST_READ=1
while true; do
  if left=$(_parse_time_left 2>/dev/null); then
    if [ "$FIRST_READ" = "1" ]; then
      echo "[slurm_requeue_watchdog] polling job ${SLURM_JOB_ID}: ${left}s left (margin=${MARGIN}s, poll=${POLL}s)" >&2
      FIRST_READ=0
    fi
    if [ "$left" -le "$MARGIN" ]; then
      echo "[slurm_requeue_watchdog] ${left}s left (margin=${MARGIN}s), requeuing ${SLURM_JOB_ID}" >&2
      scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || true
      exit 0
    fi
  fi
  sleep "$POLL"
done
