#!/bin/bash

# Sonata-v1m2 12-LR GridProbe on Flair3D+ segment (1x H100), val mIoU, no test.
# Default weight: pretrain job 862680 / epoch_120.pth (W_SONATA).
#
# Usage:
#   sbatch scripts/sonata/sbatch_flair3d_lin_grid_h100.sh
#   sbatch scripts/sonata/sbatch_flair3d_lin_grid_h100.sh /path/to/epoch_N.pth [exp_name]
#   WEIGHT=... EXP_NAME=... sbatch scripts/sonata/sbatch_flair3d_lin_grid_h100.sh
#
# After grid_search_results.json (or use the chained launcher):
#   python scripts/sonata/gen_flair3d_multitask_lin_seeds.py --grid-dir logs/slurm/<JOB>
#   sbatch scripts/sonata/sbatch_flair3d_multitask_lin_seeds_h100.sh
# One-shot grid then 10 seeds:
#   sbatch scripts/sonata/sbatch_flair3d_lin_grid_then_seeds_h100.sh
#
# Jean-Zay compute-accounting tags (IMAGINE wrapper):
#   https://github.com/Archiel19/compute-accounting

#SBATCH -A uhn@h100
#SBATCH -C h100
#SBATCH --comment=flair3d,explore,evaluate
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err

# 12 probes, 10k iters, val@100 every 2 epochs, skip_test.
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# H100 Jean-Zay: 24 CPU/GPU (gpu_p6).
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

#SBATCH --job-name=sonata_f3d_lin_grid

DEFAULT_WEIGHT=/lustre/fsn1/projects/rech/unv/usi32yh/logs/pointcept_logs/slurm/862680/model/epoch_120.pth
WEIGHT="${WEIGHT:-${1:-$DEFAULT_WEIGHT}}"
EXP_NAME="${EXP_NAME:-${2:-sonata_flair3d_lin_grid}}"
CONFIG=experiment/w112/6/sonata_flair3d_lin/sonata-v1m2-flair3d-lin-grid_1

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Exp name: $EXP_NAME"
    echo "Weight: $WEIGHT"
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

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log

exit ${TRAIN_RC}
