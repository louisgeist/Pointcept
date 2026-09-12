#!/bin/bash

# 10-seed Sonata-v1m2 multi-task lin-probe on Flair3D+ (1x H100 per seed).
# Requires configs generated first:
#   python scripts/sonata/gen_flair3d_multitask_lin_seeds.py --grid-dir logs/slurm/<GRID_JOB>
#
# Usage:
#   sbatch scripts/sonata/sbatch_flair3d_multitask_lin_seeds_h100.sh
#   WEIGHT=... sbatch scripts/sonata/sbatch_flair3d_multitask_lin_seeds_h100.sh
#   sbatch --array=1,3,7 scripts/sonata/sbatch_flair3d_multitask_lin_seeds_h100.sh
#
# Jean-Zay compute-accounting tags (IMAGINE wrapper):
#   https://github.com/Archiel19/compute-accounting

#SBATCH -A uhn@h100
#SBATCH -C h100
#SBATCH --comment=flair3d,explore,evaluate
# %A_%a (not %j): on Jean-Zay, %j is the array parent id for every task.
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%A_%a/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%A_%a/slurm.err

# 10k iters + PreciseEvaluator + NetworkAPLS on the full test split.
#SBATCH --array=1-10
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# H100 Jean-Zay: 24 CPU/GPU (gpu_p6).
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

#SBATCH --job-name=sonata_f3d_mt_seed

DEFAULT_WEIGHT=/lustre/fsn1/projects/rech/unv/usi32yh/logs/pointcept_logs/slurm/862680/model/epoch_120.pth
WEIGHT="${WEIGHT:-${1:-$DEFAULT_WEIGHT}}"
SEED_IDX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID required (use --array)}"
EXP_NAME="${EXP_NAME:-sonata_flair3d_multitask_lin_seed_${SEED_IDX}}"
CONFIG=experiment/w112/6/sonata_flair3d_lin/multi-sonata-v1m2-flair3d-lin-seed_${SEED_IDX}

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Array job/task: ${SLURM_ARRAY_JOB_ID:-n/a} / ${SLURM_ARRAY_TASK_ID:-n/a}"
    echo "Exp name: $EXP_NAME"
    echo "Weight: $WEIGHT"
    echo "Config: ${CONFIG}"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    nvidia-smi
} > ${JOB_DIR}/job_info.log

if [ ! -f "${REPO_ROOT}/configs/${CONFIG}.py" ]; then
    echo "ERROR: config not found: configs/${CONFIG}.py" | tee -a "${JOB_DIR}/job_info.log" >&2
    echo "Run: python scripts/sonata/gen_flair3d_multitask_lin_seeds.py --grid-dir <grid_job>" \
        | tee -a "${JOB_DIR}/job_info.log" >&2
    exit 1
fi

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
