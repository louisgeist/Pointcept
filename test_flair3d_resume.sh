#!/bin/bash
# Resume Flair3D+ multitask precise test from a finished Slurm training run.
#
# Usage (Jean Zay):
#   sbatch test_flair3d_resume.sh 1318327
#   sbatch test_flair3d_resume.sh /lustre/.../logs/slurm/1318327
#
# Verify paths locally (no Slurm):
#   bash test_flair3d_resume.sh --verify-only 1318327

#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err
#SBATCH -A unv@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --job-name=flair3d-test
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
NUM_GPUS=1
NUM_NODES=1
CPUS=24

VERIFY_ONLY=false
if [ "${1:-}" = "--verify-only" ]; then
    VERIFY_ONLY=true
    shift
fi

ARG="${1:-${SOURCE_JOB_ID:-}}"
if [[ "${ARG}" == */* ]]; then
    SAVE_PATH="${SAVE_PATH:-${ARG}}"
    SOURCE_JOB_ID="$(basename "${SAVE_PATH}")"
else
    SOURCE_JOB_ID="${ARG}"
    SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/logs/slurm/${SOURCE_JOB_ID}}"
fi

CHECKPOINT="${CHECKPOINT:-${SAVE_PATH}/model/model_best.pth}"
CONFIG="${CONFIG:-${SAVE_PATH}/config.py}"
CODE_DIR="${CODE_DIR:-${SAVE_PATH}/code}"

verify_paths() {
    local ok=true
    echo "========== Verify Flair3D+ test resume paths =========="
    echo "SOURCE_JOB_ID: ${SOURCE_JOB_ID}"
    echo "SAVE_PATH:     ${SAVE_PATH}"
    echo "CHECKPOINT:    ${CHECKPOINT}"
    echo "CONFIG:        ${CONFIG}"
    echo "CODE_DIR:      ${CODE_DIR}"
    echo "======================================================="

    if [ -z "${SOURCE_JOB_ID}" ]; then
        echo "ERROR: pass SOURCE_JOB_ID (train Slurm job id), e.g. sbatch $0 1318327" >&2
        ok=false
    fi
    if [ ! -d "${SAVE_PATH}" ]; then
        echo "ERROR: save_path not found: ${SAVE_PATH}" >&2
        ok=false
    fi
    if [ ! -f "${CONFIG}" ]; then
        echo "ERROR: config snapshot not found: ${CONFIG}" >&2
        ok=false
    fi
    if [ ! -f "${CHECKPOINT}" ]; then
        echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
        ok=false
    fi
    if [ ! -f "${CODE_DIR}/tools/test.py" ]; then
        echo "ERROR: ${CODE_DIR}/tools/test.py not found" >&2
        ok=false
    fi

    if [ "${ok}" = true ]; then
        echo "OK: all required paths exist."
        ls -lh "${CHECKPOINT}"
        return 0
    fi
    return 1
}

if [ "${VERIFY_ONLY}" = true ]; then
    verify_paths
    exit $?
fi

if [ -z "${SOURCE_JOB_ID}" ]; then
    echo "ERROR: pass SOURCE_JOB_ID as first argument, e.g. sbatch $0 1318327" >&2
    exit 1
fi

verify_paths

JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p "${JOB_DIR}"

cp "$0" "${JOB_DIR}/script.slurm"

{
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "Source train job ID: ${SOURCE_JOB_ID}"
    echo "SAVE_PATH (outputs): ${SAVE_PATH}"
    echo "CHECKPOINT: ${CHECKPOINT}"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    echo "Working directory: $(pwd)"
    echo "Python executable: $(which python)"
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

START_TIME=$(date +%s)

### --- Compiled code (H100 pointops) ---
POINTOPS_PATH=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_h100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
export PYTHONPATH="${CODE_DIR}:${POINTOPS_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

ulimit -n 65536

srun --unbuffered \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-${CPUS}}" \
  bash -c "
cd ${REPO_ROOT} || exit 1
export PYTHONPATH=\"${PYTHONPATH}\"
export WANDB_MODE=\"${WANDB_MODE}\"
export PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF}\"
python ${CODE_DIR}/tools/test.py \
  --config-file ${CONFIG} \
  --num-gpus ${NUM_GPUS} \
  --num-machines ${NUM_NODES} \
  --machine-rank 0 \
  --dist-url auto \
  --options save_path=${SAVE_PATH} weight=${CHECKPOINT}
"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    echo "Test log: ${SAVE_PATH}/test.log"
    echo "Results:  ${SAVE_PATH}/result/"
    nvidia-smi
} >> "${JOB_DIR}/job_info.log"
