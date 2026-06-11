#!/bin/sh
# Source this in a sbatch script BEFORE srun when using jz_utils-style launchers.
#
#   #SBATCH --signal=B:USR1@120
#   . scripts/slurm_requeue_trap.sh
#   srun ...

if [ -n "${SLURM_JOB_ID:-}" ]; then
  _pointcept_slurm_requeue_on_usr1() {
    echo "[slurm_requeue_trap.sh] SIGUSR1 received, requeuing job ${SLURM_JOB_ID}" >&2
    scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || true
    exit 1
  }
  trap '_pointcept_slurm_requeue_on_usr1' USR1
  echo "Slurm requeue batch trap installed for job ${SLURM_JOB_ID}" >&2
fi
