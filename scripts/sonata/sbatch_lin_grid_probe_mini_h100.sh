#!/bin/bash

# Mini Sonata grid-search linear probe sweep on Flair3D+ segment (1× H100).
# One array task per pretrain checkpoint epoch 10, 20, ..., 150.
# No test pass; winner val mIoU is appended to PRETRAIN_JOB_DIR/grid_probe_results.csv.
#
# Usage:
#   PRETRAIN_JOB_DIR=/lustre/fsn1/projects/rech/unv/usi32yh/logs/pointcept_logs/slurm/862680 \
#     sbatch scripts/sonata/sbatch_lin_grid_probe_mini_h100.sh
#   PRETRAIN_JOB_DIR=... sbatch --array=10,50,150 scripts/sonata/sbatch_lin_grid_probe_mini_h100.sh
#
# Jean-Zay compute-accounting tags (IMAGINE wrapper):
#   https://github.com/Archiel19/compute-accounting

#SBATCH -A uhn@h100
#SBATCH -C h100
#SBATCH --comment=flair3d,explore,evaluate
# %A_%a (not %j): on Jean-Zay, %j is the array parent id for every task,
# so 15 tasks would share one slurm.out / JOB_DIR.
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%A_%a/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%A_%a/slurm.err

# Mini grid: 10k iters, 11 probes, val@100 every 2 epochs, no test.
#SBATCH --array=10-150:10
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# H100 Jean-Zay: 24 CPU/GPU (gpu_p6).
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

#SBATCH --job-name=sonata_grid_mini_h100

DEFAULT_PRETRAIN_JOB_DIR=/lustre/fsn1/projects/rech/unv/usi32yh/logs/pointcept_logs/slurm/862680
PRETRAIN_JOB_DIR="${PRETRAIN_JOB_DIR:-$DEFAULT_PRETRAIN_JOB_DIR}"
PRETRAIN_EPOCH="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required (use --array)}"
PRETRAIN_ITERS=$((PRETRAIN_EPOCH * 1000))
WEIGHT="${WEIGHT:-${PRETRAIN_JOB_DIR}/model/epoch_${PRETRAIN_EPOCH}.pth}"
EXP_NAME="${EXP_NAME:-sonata_grid_mini_ep${PRETRAIN_EPOCH}}"
CONFIG=experiment/w109/1/sonata_grid_mini/sonata-v1m2-flair3d-lin-grid_1

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Array job/task: ${SLURM_ARRAY_JOB_ID:-n/a} / ${SLURM_ARRAY_TASK_ID:-n/a}"
    echo "Exp name: $EXP_NAME"
    echo "Weight: $WEIGHT"
    echo "Pretrain job dir: ${PRETRAIN_JOB_DIR}"
    echo "Pretrain epoch/iters: ${PRETRAIN_EPOCH}/${PRETRAIN_ITERS}"
    echo "Config: ${CONFIG}"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    nvidia-smi
} > ${JOB_DIR}/job_info.log

if [ ! -f "$WEIGHT" ]; then
    echo "ERROR: checkpoint not found: $WEIGHT" | tee -a "${JOB_DIR}/job_info.log" >&2
    exit 1
fi

module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0

conda deactivate && while [ ! -z "$CONDA_DEFAULT_ENV" ]; do conda deactivate; done
conda activate pointcept_124

conda list > ${JOB_DIR}/conda_env.txt

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ${REPO_ROOT}

POINTOPS_PATH=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_h100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
export PYTHONPATH="${POINTOPS_PATH}${PYTHONPATH:+:$PYTHONPATH}"

START_TIME=$(date +%s)

export JOB_DIR
TRAIN_RC=0
sh scripts/train.sh \
  -g 1 \
  -d flair3d_default \
  -c "$CONFIG" \
  -n "$EXP_NAME" \
  -w "$WEIGHT" || TRAIN_RC=$?

echo "Exp dir: ${JOB_DIR}" >> "${JOB_DIR}/job_info.log"
echo "WEIGHT=${WEIGHT}" >> "${JOB_DIR}/job_info.log"
echo "TRAIN_RC=${TRAIN_RC}" >> "${JOB_DIR}/job_info.log"

python scripts/sonata/append_grid_probe_result.py \
  --pretrain_job_dir "${PRETRAIN_JOB_DIR}" \
  --probe_job_dir "${JOB_DIR}" \
  --ckpt "${WEIGHT}" \
  --pretrain_epoch "${PRETRAIN_EPOCH}" \
  --train_exit_code "${TRAIN_RC}" \
  >> "${JOB_DIR}/job_info.log" 2>&1 || true

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log

exit ${TRAIN_RC}
