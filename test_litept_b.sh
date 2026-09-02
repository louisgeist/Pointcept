#!/bin/bash
# Precise test of LitePT-B Flair3D+ multitask (job 873542).
#
# MultiTaskTester logs dilated (r=3 px = 3 m) precision/recall/F1 on ROADS:
#   [task=network] Channel ROADS Test result: precision/recall/f1 ... |
#     dilated(r=3px) precision/recall/f1 ...
#
# Uses the *current* repo (not the 873542 code snapshot) so dilated P/R/F1 is
# computed. SAVE_PATH defaults to the training dir: existing
# {tile}_pred_*.npy / {tile}_logits_network.npy caches skip the GPU forward.
#
# Usage (Jean Zay):
#   sbatch test_litept_b.sh
#
# Optional overrides:
#   SAVE_PATH=... CHECKPOINT=... CONFIG=... CODE_DIR=... EXTRA_OPTIONS='...' \
#     sbatch test_litept_b.sh
#
# Smoke test on a few ROIs (any split):
#   EXTRA_OPTIONS='data.test.split=[train,val,test] data.test.include_names=[D075-2021_AA-S2-2,D075-2021_UU-S1-4]' \
#     sbatch test_litept_b.sh
#
# Verify paths locally (no Slurm):
#   bash test_litept_b.sh --verify-only

#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err
#SBATCH -A uhn@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --job-name=3h_lptb-test
#SBATCH --time=03:00:00

# Do not use `set -u` here: conda activate.d scripts (e.g. gdal) reference
# unset variables like GDAL_DATA and fail under nounset.
set -eo pipefail

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
SOURCE_JOB_ID=873542
NUM_GPUS=1
NUM_NODES=1
CPUS=24

VERIFY_ONLY=false
if [ "${1:-}" = "--verify-only" ]; then
    VERIFY_ONLY=true
    shift
fi

SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/logs/slurm/${SOURCE_JOB_ID}}"
CHECKPOINT="${CHECKPOINT:-${SAVE_PATH}/model/model_best.pth}"
CONFIG="${CONFIG:-${SAVE_PATH}/config.py}"
# Current tree has dilated P/R/F1 in MultiTaskTester; the train snapshot may not.
CODE_DIR="${CODE_DIR:-${REPO_ROOT}}"
EXTRA_OPTIONS="${EXTRA_OPTIONS:-}"

verify_paths() {
    local ok=true
    echo "========== Verify LitePT-B precise test paths =========="
    echo "SOURCE_JOB_ID: ${SOURCE_JOB_ID}"
    echo "SAVE_PATH:     ${SAVE_PATH}"
    echo "CHECKPOINT:    ${CHECKPOINT}"
    echo "CONFIG:        ${CONFIG}"
    echo "CODE_DIR:      ${CODE_DIR}"
    echo "EXTRA_OPTIONS: ${EXTRA_OPTIONS}"
    echo "========================================================"

    if [ ! -d "${SAVE_PATH}" ]; then
        echo "ERROR: save_path not found: ${SAVE_PATH}" >&2
        ok=false
    fi
    if [ ! -f "${CONFIG}" ]; then
        echo "ERROR: config not found: ${CONFIG}" >&2
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
    if ! grep -q "dilated_precision_recall_counts" "${CODE_DIR}/pointcept/engines/test.py"; then
        echo "ERROR: CODE_DIR MultiTaskTester has no dilated P/R/F1 (wrong snapshot?)" >&2
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

verify_paths

JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p "${JOB_DIR}"

cp "$0" "${JOB_DIR}/script.slurm"

{
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "Source train job ID: ${SOURCE_JOB_ID} (LitePT-B)"
    echo "SAVE_PATH (outputs / caches): ${SAVE_PATH}"
    echo "CHECKPOINT: ${CHECKPOINT}"
    echo "CONFIG: ${CONFIG}"
    echo "CODE_DIR: ${CODE_DIR}"
    echo "EXTRA_OPTIONS: ${EXTRA_OPTIONS}"
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
  --options save_path=${SAVE_PATH} weight=${CHECKPOINT} ${EXTRA_OPTIONS}
"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    echo "Test log: ${SAVE_PATH}/test.log"
    echo "Results:  ${SAVE_PATH}/result/"
    echo "----- dilated P/R/F1 (ROADS) -----"
    grep -E "Channel ROADS Test result" "${SAVE_PATH}/test.log" || echo "(no ROADS test line found)"
    nvidia-smi
} >> "${JOB_DIR}/job_info.log"
