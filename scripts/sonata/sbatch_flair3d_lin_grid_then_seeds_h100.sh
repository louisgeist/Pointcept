#!/bin/bash

# Chain: Sonata Flair3D lin-probe LR grid (segment, val mIoU) then 10 multi-task
# seed jobs. This job only runs phase 1; on success it generates seed configs
# into $JOB_DIR/seed_configs (does not dirty the git checkout) and sbatch's
# sbatch_flair3d_multitask_lin_seeds_h100.sh as an array.
#
# Requeue-safe: if grid_search_results.json already exists, phase 1 is skipped;
# if $JOB_DIR/seeds_submitted exists, the array is not submitted again.
#
# Usage:
#   sbatch scripts/sonata/sbatch_flair3d_lin_grid_then_seeds_h100.sh
#   sbatch scripts/sonata/sbatch_flair3d_lin_grid_then_seeds_h100.sh /path/to.pth [exp_name]
#   WEIGHT=... N_SEEDS=10 sbatch scripts/sonata/sbatch_flair3d_lin_grid_then_seeds_h100.sh
#   SKIP_GRID=1 GRID_DIR=logs/slurm/<old_grid> sbatch ...  # gen + submit only
#
# Jean-Zay compute-accounting tags (IMAGINE wrapper):
#   https://github.com/Archiel19/compute-accounting

#SBATCH -A uhn@h100
#SBATCH -C h100
#SBATCH --comment=flair3d,explore,evaluate
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err

# Phase 1 only (12 probes, 10k iters, val@100, skip_test). Seeds are a follow-up array.
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# H100 Jean-Zay: 24 CPU/GPU (gpu_p6).
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

#SBATCH --job-name=sonata_f3d_lin_gts

DEFAULT_WEIGHT=/lustre/fsn1/projects/rech/unv/usi32yh/logs/pointcept_logs/slurm/862680/model/epoch_120.pth
WEIGHT="${WEIGHT:-${1:-$DEFAULT_WEIGHT}}"
EXP_NAME="${EXP_NAME:-${2:-sonata_flair3d_lin_grid}}"
N_SEEDS="${N_SEEDS:-10}"
SKIP_GRID="${SKIP_GRID:-0}"
CONFIG=experiment/w112/6/sonata_flair3d_lin/sonata-v1m2-flair3d-lin-grid_1
SEED_SBATCH=scripts/sonata/sbatch_flair3d_multitask_lin_seeds_h100.sh

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Exp name: $EXP_NAME"
    echo "Weight: $WEIGHT"
    echo "Config: ${CONFIG}"
    echo "N seeds: ${N_SEEDS}"
    echo "Skip grid: ${SKIP_GRID}"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    nvidia-smi
} > ${JOB_DIR}/job_info.log

GRID_DIR="${GRID_DIR:-${JOB_DIR}}"
if [[ "${GRID_DIR}" != /* ]]; then
  GRID_DIR="${REPO_ROOT}/${GRID_DIR}"
fi
SEED_CONFIG_DIR="${JOB_DIR}/seed_configs"
WANDB_GROUP="${WANDB_GROUP:-sonata_f3d_lin_${SLURM_JOB_ID}}"

module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0

conda deactivate && while [ ! -z "$CONDA_DEFAULT_ENV" ]; do conda deactivate; done
conda activate pointcept_124

conda list > ${JOB_DIR}/conda_env.txt

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export POINTCEPT_SLURM_REQUEUE=1
cd ${REPO_ROOT}

POINTOPS_PATH=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_h100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
export PYTHONPATH="${REPO_ROOT}:${POINTOPS_PATH}${PYTHONPATH:+:$PYTHONPATH}"

START_TIME=$(date +%s)
TRAIN_RC=0

# ---------------- Phase 1: LR grid (segment, val mIoU) ----------------
if [ "${SKIP_GRID}" = "1" ]; then
    echo "phase 1 (grid): SKIP_GRID=1, using ${GRID_DIR}" | tee -a "${JOB_DIR}/job_info.log"
elif [ -f "${GRID_DIR}/grid_search_results.json" ]; then
    echo "phase 1 (grid): already done (${GRID_DIR}/grid_search_results.json) -- skipping" \
        | tee -a "${JOB_DIR}/job_info.log"
else
    if [ ! -f "$WEIGHT" ]; then
        echo "ERROR: checkpoint not found: $WEIGHT" | tee -a "${JOB_DIR}/job_info.log" >&2
        exit 1
    fi
    echo "phase 1 (grid): starting" | tee -a "${JOB_DIR}/job_info.log"
    export JOB_DIR="${GRID_DIR}"
    export EXTRA_OPTIONS="wandb_group=${WANDB_GROUP}"
    mkdir -p "${GRID_DIR}"
    sh scripts/train.sh \
      -g 1 \
      -d flair3d_default \
      -c "$CONFIG" \
      -n "$EXP_NAME" \
      -w "$WEIGHT" || TRAIN_RC=$?
    export JOB_DIR="${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}"
    if [ "${TRAIN_RC}" -ne 0 ]; then
        echo "ERROR: phase 1 failed rc=${TRAIN_RC}" | tee -a "${JOB_DIR}/job_info.log" >&2
        exit "${TRAIN_RC}"
    fi
fi

if [ ! -f "${GRID_DIR}/grid_search_results.json" ]; then
    echo "ERROR: missing ${GRID_DIR}/grid_search_results.json" | tee -a "${JOB_DIR}/job_info.log" >&2
    exit 3
fi

# ---------------- Generate 10 multi-task seed configs ----------------
echo "generating seed configs -> ${SEED_CONFIG_DIR}" | tee -a "${JOB_DIR}/job_info.log"
python scripts/sonata/gen_flair3d_multitask_lin_seeds.py \
  --grid-dir "${GRID_DIR}" \
  --output-dir "${SEED_CONFIG_DIR}" \
  --n-seeds "${N_SEEDS}" \
  | tee -a "${JOB_DIR}/job_info.log"

# ---------------- Phase 2: submit seed array ----------------
MARKER="${JOB_DIR}/seeds_submitted"
if [ -f "${MARKER}" ]; then
    echo "phase 2 (seeds): already submitted ($(cat "${MARKER}")) -- skipping" \
        | tee -a "${JOB_DIR}/job_info.log"
else
    echo "phase 2 (seeds): sbatch --array=1-${N_SEEDS} ${SEED_SBATCH}" \
        | tee -a "${JOB_DIR}/job_info.log"
    SUBMIT_OUT=$(
      SEED_CONFIG_DIR="${SEED_CONFIG_DIR}" \
      WEIGHT="${WEIGHT}" \
      WANDB_GROUP="${WANDB_GROUP}" \
      N_SEEDS="${N_SEEDS}" \
      sbatch --comment=flair3d,explore,evaluate \
        --array="1-${N_SEEDS}" \
        --export=ALL,SEED_CONFIG_DIR="${SEED_CONFIG_DIR}",WEIGHT="${WEIGHT}",WANDB_GROUP="${WANDB_GROUP}" \
        "${SEED_SBATCH}"
    ) || {
        echo "ERROR: sbatch of seed array failed" | tee -a "${JOB_DIR}/job_info.log" >&2
        echo "${SUBMIT_OUT}" | tee -a "${JOB_DIR}/job_info.log" >&2
        exit 4
    }
    echo "${SUBMIT_OUT}" | tee -a "${JOB_DIR}/job_info.log"
    echo "${SUBMIT_OUT}" > "${MARKER}"
fi

echo "Exp dir: ${JOB_DIR}" >> "${JOB_DIR}/job_info.log"
echo "GRID_DIR=${GRID_DIR}" >> "${JOB_DIR}/job_info.log"
echo "SEED_CONFIG_DIR=${SEED_CONFIG_DIR}" >> "${JOB_DIR}/job_info.log"
echo "WEIGHT=${WEIGHT}" >> "${JOB_DIR}/job_info.log"
echo "TRAIN_RC=${TRAIN_RC}" >> "${JOB_DIR}/job_info.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log

exit 0
