#!/bin/bash
# Jean Zay: binary-search the largest safe MultiViewGenerator.max_size (VRAM
# probe) via scripts/find_max_view_size.py, with batch_size_per_gpu FIXED.
#
# Usage:
#   sbatch sbatch_find_max_view_size.sh
#
# Optional env overrides (examples):
#   CONFIG=configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py \
#     BATCH_SIZE_PER_GPU=3 MIN_MAX_SIZE=8192 MAX_MAX_SIZE=49152 \
#     PROBE_STEPS=16 SOAK_STEPS=300 \
#     sbatch sbatch_find_max_view_size.sh
#
#   # Bracket around the known OOM (job 546886 crashed at max_size=65536,
#   # batch_size_per_gpu=2); batch_size_per_gpu=3 here is strictly worse
#   MIN_MAX_SIZE=32768 MAX_MAX_SIZE=65536 SOAK_STEPS=0 \
#     sbatch sbatch_find_max_view_size.sh
#
# Defaults target the Flair3D+ Sonata pretrain config, batch_size_per_gpu=3
# (matching the upstream total batch_size=96 on 32 GPUs).

#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err
#SBATCH -A uhn@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --job-name=find-max-view-size
#SBATCH --time=06:00:00

# Do not use `set -u`: conda activate.d scripts may reference unset vars.
set -eo pipefail

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
CODE_DIR="${CODE_DIR:-${REPO_ROOT}}"
NUM_GPUS=1
CPUS=24

CONFIG="${CONFIG:-configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py}"

MIN_MAX_SIZE="${MIN_MAX_SIZE:-8192}"
MAX_MAX_SIZE="${MAX_MAX_SIZE:-49152}"
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
  echo "MIN/MAX max_size: ${MIN_MAX_SIZE}/${MAX_MAX_SIZE}"
  echo "BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU} MIX_PROB=${MIX_PROB}"
  echo "PROBE_STEPS=${PROBE_STEPS} SOAK_STEPS=${SOAK_STEPS} FORCE_WORST_CASE=${FORCE_WORST_CASE}"
  echo "Starting job at: $(date)"
  echo "Running on host: $(hostname)"
  nvidia-smi
} > "${JOB_DIR}/job_info.log"

module purge
module load miniforge/24.9.0

conda deactivate && while [ -n "${CONDA_DEFAULT_ENV:-}" ]; do conda deactivate; done

module purge
module load arch/h100
module load cuda/12.4.1
module load miniforge/24.9.0

conda activate pointcept_124
conda list > "${JOB_DIR}/conda_env.txt"

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

POINTOPS_PATH=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_h100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
export PYTHONPATH="${CODE_DIR}:${POINTOPS_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

ulimit -n 65536

if [ ! -f "${REPO_ROOT}/${CONFIG}" ] && [ ! -f "${CONFIG}" ]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 1
fi
if [ ! -f "${CODE_DIR}/scripts/find_max_view_size.py" ]; then
  echo "ERROR: find_max_view_size.py not found under CODE_DIR=${CODE_DIR}" >&2
  exit 1
fi

WORK_DIR="${JOB_DIR}/max_size_search"
mkdir -p "${WORK_DIR}"

cmd=(
  python "${CODE_DIR}/scripts/find_max_view_size.py"
  --config-file "${CONFIG}"
  --num-gpus "${NUM_GPUS}"
  --num-worker "${NUM_WORKER}"
  --work-dir "${WORK_DIR}"
  --csv "${WORK_DIR}/results.csv"
  --min-max-size "${MIN_MAX_SIZE}"
  --max-max-size "${MAX_MAX_SIZE}"
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
