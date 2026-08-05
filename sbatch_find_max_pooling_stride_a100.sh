#!/bin/bash
# Jean Zay (A100): sweep PT-v3 pooling model.backbone.stride candidates (VRAM
# probe) via scripts/find_max_pooling_stride.py, with max_size /
# batch_size_per_gpu FIXED.
#
# A100 variant of sbatch_find_max_pooling_stride.sh (H100) — same script/env
# vars, only the hardware-specific bits differ (account/constraint, cpu
# ratio, module/cuda versions, pointops build path).
#
# Usage:
#   sbatch sbatch_find_max_pooling_stride_a100.sh
#
# Optional env overrides (examples):
#   CONFIG=configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py \
#     BATCH_SIZE_PER_GPU=3 MAX_SIZE=65536 \
#     STRIDES="2,2,2,3 2,2,3,3 2,3,3,3 3,3,3,3" \
#     PROBE_STEPS=16 SOAK_STEPS=300 \
#     sbatch sbatch_find_max_pooling_stride_a100.sh
#
#   # Probe only, no soak (faster, less confidence)
#   SOAK_STEPS=0 sbatch sbatch_find_max_pooling_stride_a100.sh
#
# Defaults target the Flair3D+ Sonata pretrain config, batch_size_per_gpu=3
# (matching the upstream total batch_size=96 on 32 GPUs), max_size=65536, and
# the sweep order (2,2,2,3) -> (2,2,3,3) -> (2,3,3,3) -> (3,3,3,3) — base
# stride (2,2,2,2) is skipped since it is known to OOM outright.

#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err
#SBATCH -A uhn@a100
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --job-name=find-max-pooling-stride-a100
#SBATCH --time=06:00:00

# Do not use `set -u`: conda activate.d scripts may reference unset vars.
set -eo pipefail

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
CODE_DIR="${CODE_DIR:-${REPO_ROOT}}"
NUM_GPUS=1
CPUS=8

CONFIG="${CONFIG:-configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py}"

STRIDES="${STRIDES:-2,2,2,3 2,2,3,3 2,3,3,3 3,3,3,3}"
MAX_SIZE="${MAX_SIZE:-65536}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-3}"
MIX_PROB="${MIX_PROB:-0}"
PROBE_STEPS="${PROBE_STEPS:-16}"
SOAK_STEPS="${SOAK_STEPS:-300}"
FORCE_WORST_CASE="${FORCE_WORST_CASE:-true}"

NUM_WORKER="${NUM_WORKER:-8}"
EXTRA_OPTIONS="${EXTRA_OPTIONS:-}"

JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p "${JOB_DIR}"
cp "$0" "${JOB_DIR}/script.slurm"

{
  echo "Job ID: ${SLURM_JOB_ID}"
  echo "CONFIG: ${CONFIG}"
  echo "CODE_DIR: ${CODE_DIR}"
  echo "STRIDES: ${STRIDES}"
  echo "MAX_SIZE=${MAX_SIZE} BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU} MIX_PROB=${MIX_PROB}"
  echo "PROBE_STEPS=${PROBE_STEPS} SOAK_STEPS=${SOAK_STEPS} FORCE_WORST_CASE=${FORCE_WORST_CASE}"
  echo "Starting job at: $(date)"
  echo "Running on host: $(hostname)"
  nvidia-smi
} > "${JOB_DIR}/job_info.log"

module purge
module load miniforge/24.9.0

conda deactivate && while [ -n "${CONDA_DEFAULT_ENV:-}" ]; do conda deactivate; done

module purge
module load arch/a100
module load cuda/12.1.0
module load miniforge/24.9.0

conda activate pointcept_124
conda list > "${JOB_DIR}/conda_env.txt"

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

# Pointops built for A100 (train.sh prepends CODE_DIR but does not override an existing PYTHONPATH)
POINTOPS_PATH=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_a100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
export PYTHONPATH="${CODE_DIR}:${POINTOPS_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

ulimit -n 65536

if [ ! -f "${REPO_ROOT}/${CONFIG}" ] && [ ! -f "${CONFIG}" ]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 1
fi
if [ ! -f "${CODE_DIR}/scripts/find_max_pooling_stride.py" ]; then
  echo "ERROR: find_max_pooling_stride.py not found under CODE_DIR=${CODE_DIR}" >&2
  exit 1
fi

WORK_DIR="${JOB_DIR}/pooling_stride_search"
mkdir -p "${WORK_DIR}"

# shellcheck disable=SC2206
STRIDE_ARGS=( ${STRIDES} )

cmd=(
  python "${CODE_DIR}/scripts/find_max_pooling_stride.py"
  --config-file "${CONFIG}"
  --num-gpus "${NUM_GPUS}"
  --num-worker "${NUM_WORKER}"
  --work-dir "${WORK_DIR}"
  --csv "${WORK_DIR}/results.csv"
  --strides "${STRIDE_ARGS[@]}"
  --max-size "${MAX_SIZE}"
  --batch-size-per-gpu "${BATCH_SIZE_PER_GPU}"
  --mix-prob "${MIX_PROB}"
  --probe-steps "${PROBE_STEPS}"
  --soak-steps "${SOAK_STEPS}"
)

if [ "${FORCE_WORST_CASE}" = "false" ]; then
  cmd+=(--no-force-worst-case-scale)
fi

if [ -n "${EXTRA_OPTIONS}" ]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_OPTIONS} )
  cmd+=(--extra-options "${extra[@]}")
fi

START_TIME=$(date +%s)

echo "${cmd[*]}"
srun --unbuffered \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-${CPUS}}" \
  bash -c "
cd ${REPO_ROOT} || exit 1
export PYTHONPATH=\"${PYTHONPATH}\"
export WANDB_MODE=\"${WANDB_MODE}\"
export PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF}\"
$(printf '%q ' "${cmd[@]}")
"
STATUS=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
  echo "Job finished at: $(date)"
  echo "Duration: ${DURATION} seconds"
  echo "Results under: ${WORK_DIR}"
  if [ -f "${WORK_DIR}/results.csv" ]; then
    echo "--- ${WORK_DIR}/results.csv ---"
    cat "${WORK_DIR}/results.csv"
  fi
  nvidia-smi
} >> "${JOB_DIR}/job_info.log"

echo "Done. Summary in ${JOB_DIR}/job_info.log and ${WORK_DIR}/results.csv"
exit "${STATUS}"
