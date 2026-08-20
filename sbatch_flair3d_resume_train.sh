#!/bin/bash
# Resume a crashed Flair3D+ multitask training job through scripts/train.sh, so the
# *trainer's* after_train hooks (PreciseEvaluator + NetworkAPLSEvaluator) rerun and
# log into the SAME wandb run (resumed via wandb_run_id.txt), instead of leaving the
# test/APLS metrics unsynced or starting a fresh run.
#
# Use this (not test_flair3d_resume.sh) when the crash happened *after* training, in
# the post-train hooks -- e.g. the DDP wandb.log bug in NetworkAPLSEvaluator fixed in
# a9a7b21 ("fix APLS compute in DDP"). test_flair3d_resume.sh calls tools/test.py
# directly, which never calls wandb.init() -- it would produce local test/APLS
# results but write nothing back into wandb at all.
#
# scripts/train.sh auto-resumes when JOB_DIR points at a dir with an existing
# model/model_last.pth (see its lines ~101-104): it reloads config.py + the
# checkpoint from there regardless of -r/-c. Since the train loop is
# `for epoch in range(start_epoch, max_epoch)`, if training had already finished,
# no epoch reruns -- it goes straight to after_train() hooks.
#
# Usage (Jean Zay):
#   sbatch sbatch_flair3d_resume_train.sh 1095469
#
# Verify paths locally (no Slurm):
#   bash sbatch_flair3d_resume_train.sh --verify-only 1095469
#
# After the job completes, sync from a login node (compute nodes have no internet):
#   wandb sync logs/slurm/1095469/wandb

#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err
#SBATCH -A unv@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --job-name=flair3d-multi-resume
#SBATCH --time=3:00:00

# Do not use `set -u`: conda activate.d scripts (e.g. gdal) reference unset vars.
set -eo pipefail

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
NUM_GPUS=4
CONFIG_PATH="experiment/w109/2/ptv3_wd/multi-ptv3-v1m0-flair3d_5"
EXP_NAME="multi-ptv3-v1m0-flair3d_5"

VERIFY_ONLY=false
if [ "${1:-}" = "--verify-only" ]; then
    VERIFY_ONLY=true
    shift
fi

SOURCE_JOB_ID="${1:-${SOURCE_JOB_ID:-}}"
TRAIN_JOB_DIR="${REPO_ROOT}/logs/slurm/${SOURCE_JOB_ID}"
CHECKPOINT="${TRAIN_JOB_DIR}/model/model_last.pth"
CONFIG_SNAPSHOT="${TRAIN_JOB_DIR}/config.py"

verify_paths() {
    local ok=true
    echo "========== Verify Flair3D+ train resume paths =========="
    echo "SOURCE_JOB_ID: ${SOURCE_JOB_ID}"
    echo "TRAIN_JOB_DIR: ${TRAIN_JOB_DIR}"
    echo "CHECKPOINT:    ${CHECKPOINT}"
    echo "CONFIG SNAP:   ${CONFIG_SNAPSHOT}"
    echo "==========================================================="

    if [ -z "${SOURCE_JOB_ID}" ]; then
        echo "ERROR: pass the crashed training job's Slurm ID, e.g. sbatch $0 1095469" >&2
        ok=false
    fi
    if [ ! -d "${TRAIN_JOB_DIR}" ]; then
        echo "ERROR: not found: ${TRAIN_JOB_DIR}" >&2
        ok=false
    fi
    if [ ! -f "${CONFIG_SNAPSHOT}" ]; then
        echo "ERROR: config snapshot not found: ${CONFIG_SNAPSHOT}" >&2
        ok=false
    fi
    if [ ! -f "${CHECKPOINT}" ]; then
        echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
        ok=false
    fi
    if [ ! -f "${TRAIN_JOB_DIR}/wandb_run_id.txt" ]; then
        echo "WARNING: no wandb_run_id.txt in ${TRAIN_JOB_DIR} -- a NEW wandb run would be created instead of resuming the crashed one." >&2
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
    echo "ERROR: pass SOURCE_JOB_ID as first argument, e.g. sbatch $0 1095469" >&2
    exit 1
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: no \$SLURM_JOB_ID -- this must be submitted via sbatch, not run directly:" >&2
    echo "         sbatch $0 ${SOURCE_JOB_ID}" >&2
    echo "       For a dry path check without Slurm, use --verify-only instead:" >&2
    echo "         bash $0 --verify-only ${SOURCE_JOB_ID}" >&2
    exit 1
fi

verify_paths

RESUME_JOB_DIR="${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}"
mkdir -p "${RESUME_JOB_DIR}"
cp "$0" "${RESUME_JOB_DIR}/script.slurm"

{
    echo "Resume job ID: ${SLURM_JOB_ID}"
    echo "Source (crashed) train job ID: ${SOURCE_JOB_ID}"
    echo "Resuming into: ${TRAIN_JOB_DIR}"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    nvidia-smi
} > "${RESUME_JOB_DIR}/job_info.log"

module purge
module load miniforge/24.9.0

conda deactivate && while [ -n "${CONDA_DEFAULT_ENV:-}" ]; do conda deactivate; done

module purge
module load arch/h100
module load cuda/12.4.1
module load miniforge/24.9.0

conda activate pointcept_124

conda list > "${RESUME_JOB_DIR}/conda_env.txt"

# Refresh the job's code snapshot with the CURRENT repo before resuming: the
# original snapshot under ${TRAIN_JOB_DIR}/code predates the DDP wandb.log fix
# (a9a7b21) and scripts/train.sh's JOB_DIR-based auto-resume reuses that snapshot
# as-is -- it never re-copies once model_last.pth already exists there.
cp -r "${REPO_ROOT}/scripts" "${REPO_ROOT}/tools" "${REPO_ROOT}/pointcept" "${TRAIN_JOB_DIR}/code/"

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

START_TIME=$(date +%s)

POINTOPS_PATH=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_h100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
export PYTHONPATH="${POINTOPS_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

ulimit -n 65536

export JOB_DIR="${TRAIN_JOB_DIR}"
sh scripts/train.sh -g "${NUM_GPUS}" -d flair3d -c "${CONFIG_PATH}" -n "${EXP_NAME}" -r true

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    echo "Results:  ${TRAIN_JOB_DIR}/result/"
    echo "Next step (from a login node): wandb sync ${TRAIN_JOB_DIR}/wandb"
    nvidia-smi
} >> "${RESUME_JOB_DIR}/job_info.log"
