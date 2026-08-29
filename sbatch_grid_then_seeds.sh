#!/bin/bash

# Grid-probe sweep -> best-config (by val) -> seed-ensemble robustness run,
# chained in one job (1x A100). Generic: works for any *-lin-grid* config
# (h3d / dales / eclair / flair3d, any frozen backbone) -- not Sonata-specific.
#
# Phase 1  runs the given GridProbe config; GridProbeWinnerSelector writes
#          <jobdir>/grid/grid_search_results.json.
# Phase 2  tools/grid_then_seeds.py reads the winner's full probe_config,
#          generates a 10-init seed-ensemble config, runs it; the result is
#          <jobdir>/seeds/seed_ensemble_results.json (test mIoU/mAcc/allAcc/
#          f1_macro mean +/- std) and <jobdir>/grid_then_seeds_summary.csv.
#
# Usage:
#   ./submit_grid_then_seeds.sh <grid_config> <weight.pth> [exp_name]
#   (sets Slurm --time from config path: H3D 4h / DALES 8h / ECLAIR 12h)
#   GRID_CONFIG=... WEIGHT=... [N_SEEDS=10] sbatch sbatch_grid_then_seeds.sh
#
# For the 336-probe wide flair3d grid (48h on its own): run that grid with
# scripts/sonata/sbatch_lin_grid_probe.sh first, then here with
#   EXTRA_ARGS="--skip-grid --grid-dir logs/slurm/<gridjob>"
#
# Jean-Zay compute-accounting tags (IMAGINE wrapper):
#   https://github.com/Archiel19/compute-accounting

#SBATCH -A uhn@a100
#SBATCH -C a100
#SBATCH --comment=flair3d,explore,evaluate
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# A100 Jean-Zay: 8 CPU/GPU (gpu_p5).
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=grid_then_seeds

GRID_CONFIG="${GRID_CONFIG:-${1:?grid config required (arg1 or GRID_CONFIG=)}}"
WEIGHT="${WEIGHT:-${2:?WEIGHT path required (arg2 or WEIGHT=)}}"
EXP_NAME="${EXP_NAME:-${3:-grid_then_seeds}}"
N_SEEDS="${N_SEEDS:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Exp name: $EXP_NAME"
    echo "Grid config: $GRID_CONFIG"
    echo "Weight: $WEIGHT"
    echo "N seeds: $N_SEEDS"
    echo "Slurm time limit: ${SLURM_TIME:-12:00:00}"
    echo "Extra args: ${EXTRA_ARGS:-<none>}"
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
# each tools/train.py phase installs an in-process timeout->requeue handler;
# on requeue the batch reruns and grid_then_seeds.py resumes the unfinished phase.
export POINTCEPT_SLURM_REQUEUE=1
cd ${REPO_ROOT}

POINTOPS_PATH=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_a100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
export PYTHONPATH="${REPO_ROOT}:${POINTOPS_PATH}${PYTHONPATH:+:$PYTHONPATH}"

START_TIME=$(date +%s)

export JOB_DIR
RC=0
python tools/grid_then_seeds.py \
  --grid-config "$GRID_CONFIG" \
  --weight "$WEIGHT" \
  --save-root "$JOB_DIR" \
  --n-seeds "$N_SEEDS" \
  --num-gpus 1 \
  $EXTRA_ARGS || RC=$?

echo "Exp dir: ${JOB_DIR}" >> "${JOB_DIR}/job_info.log"
echo "WEIGHT=${WEIGHT}" >> "${JOB_DIR}/job_info.log"
echo "RC=${RC}" >> "${JOB_DIR}/job_info.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log

exit ${RC}
