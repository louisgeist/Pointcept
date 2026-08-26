#!/bin/bash

# Inference-speed bench: LitePT-B / PTv3 / KPConvX / SpUNet on Flair3D+ (1× A100).
# Calls scripts/bench_inference_speed.py with the "real run" defaults
# (national manifest, test split, 200 tiles). Weights are randomly initialized.
#
# Usage:
#   sbatch sbatch_bench_inference_speed.sh
#
# Optional env overrides (examples):
#   NUM_TILES=50 BACKBONES="litept_b ptv3" sbatch sbatch_bench_inference_speed.sh
#   AMP=true sbatch sbatch_bench_inference_speed.sh
#
# Jean-Zay compute-accounting tags (IMAGINE wrapper):
#   https://github.com/Archiel19/compute-accounting

#SBATCH -A uhn@a100
#SBATCH -C a100
#SBATCH --comment=flair3d,explore,evaluate
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err

#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# A100 Jean-Zay: 8 CPU/GPU (gpu_p5). More CPUs = overcharge (e.g. 24 → billed as ~3 GPUs).
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=bench_inf

# Do not use `set -u`: conda activate.d scripts may reference unset vars.
set -eo pipefail

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept

NUM_TILES="${NUM_TILES:-200}"
NUM_WARMUP="${NUM_WARMUP:-10}"
SPLIT="${SPLIT:-test}"
CSV_MANIFEST="${CSV_MANIFEST:-data/flair3d_plus/raw/scene_split_manifest.csv}"
BACKBONES="${BACKBONES:-}"
AMP="${AMP:-false}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/stats/flair3d/inference_speed_bench/${SLURM_JOB_ID}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}

cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "CSV_MANIFEST: ${CSV_MANIFEST}"
    echo "SPLIT: ${SPLIT}"
    echo "NUM_TILES: ${NUM_TILES}  NUM_WARMUP: ${NUM_WARMUP}"
    echo "BACKBONES: ${BACKBONES:-<all 4>}"
    echo "AMP: ${AMP}"
    echo "OUT_DIR: ${OUT_DIR}"
    echo "EXTRA_ARGS: ${EXTRA_ARGS:-<none>}"
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

# Pointops built for A100 (train.sh prepends CODE_DIR but does not override an existing PYTHONPATH)
POINTOPS_PATH=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_a100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
export PYTHONPATH="${REPO_ROOT}:${POINTOPS_PATH}${PYTHONPATH:+:$PYTHONPATH}"

cmd=(
  python scripts/bench_inference_speed.py
  --csv-manifest "${CSV_MANIFEST}"
  --split "${SPLIT}"
  --num-tiles "${NUM_TILES}"
  --num-warmup "${NUM_WARMUP}"
  --device cuda:0
  --out-dir "${OUT_DIR}"
)

if [ -n "${BACKBONES}" ]; then
  # shellcheck disable=SC2206
  cmd+=(--backbones ${BACKBONES})
fi

if [ "${AMP}" = "true" ]; then
  cmd+=(--amp)
fi

if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi

START_TIME=$(date +%s)

echo "${cmd[*]}"
STATUS=0
"${cmd[@]}" || STATUS=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

{
    echo "Job finished at: $(date)"
    echo "Duration: ${DURATION} seconds"
    echo "Results under: ${OUT_DIR}"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log

echo "Done. Summary in ${JOB_DIR}/job_info.log and ${OUT_DIR}"
exit ${STATUS}
