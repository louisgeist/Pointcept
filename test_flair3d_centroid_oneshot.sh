#!/bin/bash
# One-shot Jean Zay test: checkpoint 12.1 (job 2146226) with GridSample voxel_repr=centroid.
#
# Uses the *current* repo code + config (not the training job code snapshot), because
# voxel_repr was added after that run.
#
# Usage:
#   sbatch test_flair3d_centroid_oneshot.sh
#
# Optional overrides:
#   CONFIG=... CHECKPOINT=... sbatch test_flair3d_centroid_oneshot.sh
#   bash test_flair3d_centroid_oneshot.sh --verify-only

#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err
#SBATCH -A unv@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --job-name=flair3d-cent-test
#SBATCH --time=12:00:00

# Do not use `set -u` here: conda activate.d scripts (e.g. gdal) reference
# unset variables like GDAL_DATA and fail under nounset.
set -eo pipefail

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
NUM_GPUS=1
NUM_NODES=1
CPUS=24

CONFIG="${CONFIG:-${REPO_ROOT}/configs/experiment/w105/2/test_centroid/litept-v1m0-flair3d_14.py}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/logs/slurm/2146226/model/model_best.pth}"
# Current tree (has voxel_repr=centroid), not logs/slurm/2146226/code
CODE_DIR="${CODE_DIR:-${REPO_ROOT}}"

VERIFY_ONLY=false
if [ "${1:-}" = "--verify-only" ]; then
    VERIFY_ONLY=true
    shift
fi

verify_paths() {
    local ok=true
    echo "========== Verify Flair3D+ centroid test paths =========="
    echo "CONFIG:     ${CONFIG}"
    echo "CHECKPOINT: ${CHECKPOINT}"
    echo "CODE_DIR:   ${CODE_DIR}"
    echo "========================================================="

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
    if ! grep -q 'voxel_repr="centroid"' "${CONFIG}"; then
        echo "ERROR: config does not set voxel_repr=centroid: ${CONFIG}" >&2
        ok=false
    fi
    if ! grep -q 'voxel_repr' "${CODE_DIR}/pointcept/datasets/transform.py"; then
        echo "ERROR: CODE_DIR GridSample has no voxel_repr (wrong code snapshot?)" >&2
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
    echo "CONFIG: ${CONFIG}"
    echo "CHECKPOINT: ${CHECKPOINT}"
    echo "CODE_DIR: ${CODE_DIR}"
    echo "SAVE_PATH: ${JOB_DIR}"
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
  --options save_path=${JOB_DIR} weight=${CHECKPOINT}
"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    echo "Test log: ${JOB_DIR}/test.log"
    echo "Results:  ${JOB_DIR}/result/"
    nvidia-smi
} >> "${JOB_DIR}/job_info.log"
