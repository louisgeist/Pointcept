#!/bin/bash

# Wide Sonata grid-search linear probe on Flair3D+ segment (1× A100).
# Default weight: pretrain job 862680 / epoch_9.pth
#
# Usage:
#   sbatch scripts/sonata/sbatch_lin_grid_probe.sh
#   sbatch scripts/sonata/sbatch_lin_grid_probe.sh /path/to/epoch_N.pth [exp_name]
#   WEIGHT=... EXP_NAME=... sbatch scripts/sonata/sbatch_lin_grid_probe.sh
#
# Jean-Zay compute-accounting tags (IMAGINE wrapper):
#   https://github.com/Archiel19/compute-accounting

#SBATCH -A uhn@a100
#SBATCH -C a100
#SBATCH --comment=flair3d,explore,evaluate
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err

# Wide grid: 10k iters, 336 probes, val@100 every 5 epochs + full winner test.
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# A100 Jean-Zay: 8 CPU/GPU (gpu_p5).
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=sonata_grid

DEFAULT_WEIGHT=/lustre/fsn1/projects/rech/unv/usi32yh/logs/pointcept_logs/slurm/862680/model/epoch_9.pth
WEIGHT="${WEIGHT:-${1:-$DEFAULT_WEIGHT}}"
EXP_NAME="${EXP_NAME:-${2:-sonata_grid_wide_ep9}}"

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Exp name: $EXP_NAME"
    echo "Weight: $WEIGHT"
    echo "Config: flair3d_default/probe/sonata-v1m2-flair3d-lin-grid-wide"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    nvidia-smi
} > ${JOB_DIR}/job_info.log

module purge
module load arch/a100
module load cuda/12.1.0
module load miniforge/24.9.0

conda deactivate && while [ ! -z "$CONDA_DEFAULT_ENV" ]; do conda deactivate; done
conda activate pointcept_124

conda list > ${JOB_DIR}/conda_env.txt

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ${REPO_ROOT}

POINTOPS_PATH=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_a100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
export PYTHONPATH="${POINTOPS_PATH}${PYTHONPATH:+:$PYTHONPATH}"

START_TIME=$(date +%s)

export JOB_DIR
TRAIN_RC=0
sh scripts/train.sh \
  -g 1 \
  -d flair3d_default \
  -c probe/sonata-v1m2-flair3d-lin-grid-wide \
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
