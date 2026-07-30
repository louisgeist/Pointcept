#!/bin/bash
# Jean Zay: binary-search max batch_size (train / val / test) via
# scripts/find_max_batch_size.py
#
# Usage:
#   sbatch sbatch_find_max_batch_size.sh
#
# Optional env overrides (examples):
#   MODE=train sbatch sbatch_find_max_batch_size.sh
#   MODES="train val" sbatch sbatch_find_max_batch_size.sh
#   CONFIG=configs/experiment/w105/2/10h/litept-v1m0-flair3d_12.py \
#     MIN_BS_TRAIN=2 MAX_BS_TRAIN=24 sbatch sbatch_find_max_batch_size.sh
#
# Sonata Flair3D+ (align MIX_PROB with the real config):
#   # Pretrain SSL — no Mix3D
#   CONFIG=configs/flair3d_default/pretrain-sonata-v1m2-flair3d.py \
#     MODE=train MIX_PROB=0 \
#     MIN_BS_TRAIN=1 MAX_BS_TRAIN=8 \
#     PROBE_STEPS=32 SOAK_STEPS_TRAIN=200 \
#     sbatch sbatch_find_max_batch_size.sh
#
#   # Linear probe — Mix3D as in *-lin config
#   CONFIG=configs/flair3d_default/segment/sonata-v1m2-flair3d-lin.py \
#     MODE=train MIX_PROB=0.8 \
#     MIN_BS_TRAIN=1 MAX_BS_TRAIN=8 \
#     PROBE_STEPS=32 SOAK_STEPS_TRAIN=200 \
#     sbatch sbatch_find_max_batch_size.sh
#
# Defaults match the recommended local commands (H100, 1 GPU).
# Probe overlays replace hooks (LinProbeSbatchHook disabled during search).

#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err
#SBATCH -A uhn@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --job-name=find-max-bs
#SBATCH --time=06:00:00

# Do not use `set -u`: conda activate.d scripts may reference unset vars.
set -eo pipefail

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
CODE_DIR="${CODE_DIR:-${REPO_ROOT}}"
NUM_GPUS=1
CPUS=24

CONFIG="${CONFIG:-configs/experiment/w105/2/10h/litept-v1m0-flair3d_13.py}"

# MODE=train|val|test runs a single mode. Else MODES (space-separated) defaults to all three.
if [ -n "${MODE:-}" ]; then
  MODES="${MODE}"
else
  MODES="${MODES:-train val test}"
fi

# Train search
MIN_BS_TRAIN="${MIN_BS_TRAIN:-2}"
MAX_BS_TRAIN="${MAX_BS_TRAIN:-32}"
PROBE_STEPS="${PROBE_STEPS:-64}"
SOAK_STEPS_TRAIN="${SOAK_STEPS_TRAIN:-500}"
MIX_PROB="${MIX_PROB:-1.0}"

# Val / test search
MIN_BS_EVAL="${MIN_BS_EVAL:-1}"
MAX_BS_EVAL="${MAX_BS_EVAL:-16}"
MAX_SAMPLE="${MAX_SAMPLE:-128}"
SOAK_STEPS_EVAL="${SOAK_STEPS_EVAL:-0}"

NUM_WORKER="${NUM_WORKER:-8}"
EXTRA_OPTIONS="${EXTRA_OPTIONS:-}"

JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p "${JOB_DIR}"
cp "$0" "${JOB_DIR}/script.slurm"

{
  echo "Job ID: ${SLURM_JOB_ID}"
  echo "CONFIG: ${CONFIG}"
  echo "CODE_DIR: ${CODE_DIR}"
  echo "MODES: ${MODES}"
  echo "MIN/MAX train: ${MIN_BS_TRAIN}/${MAX_BS_TRAIN}"
  echo "MIN/MAX eval:  ${MIN_BS_EVAL}/${MAX_BS_EVAL}"
  echo "PROBE_STEPS=${PROBE_STEPS} SOAK_STEPS_TRAIN=${SOAK_STEPS_TRAIN} MIX_PROB=${MIX_PROB}"
  echo "MAX_SAMPLE=${MAX_SAMPLE} SOAK_STEPS_EVAL=${SOAK_STEPS_EVAL}"
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
if [ ! -f "${CODE_DIR}/scripts/find_max_batch_size.py" ]; then
  echo "ERROR: find_max_batch_size.py not found under CODE_DIR=${CODE_DIR}" >&2
  exit 1
fi

START_TIME=$(date +%s)
FAIL=0

run_mode() {
  local mode="$1"
  local work_dir="${JOB_DIR}/batch_size_search/${mode}"
  mkdir -p "${work_dir}"

  local -a cmd=(
    python "${CODE_DIR}/scripts/find_max_batch_size.py"
    --config-file "${CONFIG}"
    --mode "${mode}"
    --num-gpus "${NUM_GPUS}"
    --num-worker "${NUM_WORKER}"
    --work-dir "${work_dir}"
    --csv "${work_dir}/results.csv"
  )

  if [ "${mode}" = "train" ]; then
    cmd+=(
      --min-bs "${MIN_BS_TRAIN}"
      --max-bs "${MAX_BS_TRAIN}"
      --probe-steps "${PROBE_STEPS}"
      --soak-steps "${SOAK_STEPS_TRAIN}"
      --mix-prob "${MIX_PROB}"
    )
  else
    cmd+=(
      --min-bs "${MIN_BS_EVAL}"
      --max-bs "${MAX_BS_EVAL}"
      --max-sample "${MAX_SAMPLE}"
      --soak-steps "${SOAK_STEPS_EVAL}"
    )
  fi

  if [ -n "${EXTRA_OPTIONS}" ]; then
    # shellcheck disable=SC2206
    local -a extra=( ${EXTRA_OPTIONS} )
    cmd+=(--extra-options "${extra[@]}")
  fi

  echo "========== Running mode=${mode} =========="
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
}

for mode in ${MODES}; do
  if ! run_mode "${mode}"; then
    echo "ERROR: mode=${mode} failed" >&2
    FAIL=1
  fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
  echo "Job finished at: $(date)"
  echo "Duration: ${DURATION} seconds"
  echo "Results under: ${JOB_DIR}/batch_size_search/"
  for mode in ${MODES}; do
    csv="${JOB_DIR}/batch_size_search/${mode}/results.csv"
    if [ -f "${csv}" ]; then
      echo "--- ${csv} ---"
      cat "${csv}"
    fi
  done
  nvidia-smi
} >> "${JOB_DIR}/job_info.log"

echo "Done. Summaries in ${JOB_DIR}/job_info.log and */results.csv"
exit "${FAIL}"
