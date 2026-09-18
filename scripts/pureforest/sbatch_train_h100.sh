#!/bin/bash

# Launch a single PureForest config (from-scratch training or probe) via
# scripts/train.sh on 1x H100 Jean-Zay. Generic launcher: pass the config
# path relative to configs/ as $1, optional exp name as $2 (defaults to the
# config's basename).
#
# Usage:
#   sbatch scripts/pureforest/sbatch_train_h100.sh <config_path> [exp_name]
#   sbatch scripts/pureforest/sbatch_train_h100.sh \
#       experiment/w113/1/pf_scratch/cls-litept-b-v1m0-pureforest-30e-wd1_1
#
# Override walltime: sbatch --time=HH:MM:SS scripts/pureforest/sbatch_train_h100.sh ...

#SBATCH -A uhn@h100
#SBATCH -C h100
#SBATCH --comment=pureforest
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err

#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

#SBATCH --job-name=pf_train

CONFIG="${1:?Usage: sbatch scripts/pureforest/sbatch_train_h100.sh <config_path> [exp_name]}"
EXP_NAME="${2:-$(basename "$CONFIG")}"

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p "${JOB_DIR}"

cp "$0" "${JOB_DIR}/script.slurm"

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Config: $CONFIG"
    echo "Exp name: $EXP_NAME"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    nvidia-smi
} > "${JOB_DIR}/job_info.log"

module purge
module load arch/h100
module load cuda/12.4.1
module load miniforge/24.9.0

conda deactivate && while [ ! -z "$CONDA_DEFAULT_ENV" ]; do conda deactivate; done
conda activate pointcept_124

conda list > "${JOB_DIR}/conda_env.txt"

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

POINTOPS_PATH=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_h100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
export PYTHONPATH="${POINTOPS_PATH}:${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

export JOB_DIR
START_TIME=$(date +%s)

sh scripts/train.sh -g 1 -d pureforest -c "$CONFIG" -n "$EXP_NAME"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    nvidia-smi
} >> "${JOB_DIR}/job_info.log"
