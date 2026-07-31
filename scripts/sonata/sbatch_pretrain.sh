#!/bin/bash

# Sonata-v1m2 pretrain on Flair3D+ (default: 1 node × 8 A100).
# Usage: sbatch scripts/sonata/sbatch_pretrain.sh [exp_name]
#
# Multi-node: bump --nodes AND --ntasks together (1 Slurm task per node).
# --gres=gpu:N is per node; NUM_GPUS below must match N.
#
# Jean-Zay compute-accounting tags (IMAGINE wrapper):
#   https://github.com/Archiel19/compute-accounting
# Keep --comment set so non-interactive sbatch never prompts.

#SBATCH -A uhn@a100
#SBATCH -C a100
#SBATCH --comment=flair3d,baseline,pre-train
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err

#SBATCH --time=19:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread

#SBATCH --job-name=sonata_pretrain

EXP_NAME="${1:-sonata_pretrain_flair3dplus}"
NUM_GPUS=8  # must match --gres=gpu:N (per node)

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Exp name: $EXP_NAME"
    echo "Config: flair3d_default/pretrain-sonata-v1m2-flair3d"
    echo "Nodes: ${SLURM_NNODES} × ${NUM_GPUS} A100"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    echo "Working directory: $(pwd)"
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

# Pointops built for A100 (train.sh prepends CODE_DIR but does not override an existing PYTHONPATH)
POINTOPS_PATH=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_a100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
export PYTHONPATH="${POINTOPS_PATH}${PYTHONPATH:+:$PYTHONPATH}"

START_TIME=$(date +%s)

export JOB_DIR

# Batch script runs on node 0 only; srun starts one task per node so every node runs train.sh.
# train.sh uses SLURM_NODEID for --machine-rank and SLURM_NODELIST for the dist URL.
srun --unbuffered \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  bash -c "
cd ${REPO_ROOT} || exit 1
export JOB_DIR=\"${JOB_DIR}\"
export PYTHONPATH=\"${PYTHONPATH}\"
export WANDB_MODE=\"${WANDB_MODE}\"
export PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF}\"
sh scripts/train.sh \
  -g ${NUM_GPUS} \
  -m ${SLURM_NNODES} \
  -d flair3d_default \
  -c pretrain-sonata-v1m2-flair3d \
  -n \"${EXP_NAME}\"
"

echo "Exp dir: ${JOB_DIR}" >> "${JOB_DIR}/job_info.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log
