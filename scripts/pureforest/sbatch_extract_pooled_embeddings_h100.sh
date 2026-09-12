#!/bin/bash

# Extract PureForest mean/max pooled embeddings for all MS-encoder backbones
# (1× H100). Writes under stats/pureforest/embeddings/<tag>/{train,val,test}.npz
#
# Models (encoder multiscale GridProbe configs):
#   - Sonata outdoor SSL (Malibu3D)
#   - LitePT-B / PTv3 / SpUNet / KPConvX Malibu3D multitask
#   - LitePT-B supervised ECLAIR from-scratch (job 1330042)
#
# Usage:
#   sbatch scripts/pureforest/sbatch_extract_pooled_embeddings_h100.sh
#   BATCH_SIZE=32 POINT_MAX=5000 sbatch scripts/pureforest/sbatch_extract_pooled_embeddings_h100.sh
#   SKIP_EXISTING=1 sbatch ...   # skip tags that already have train+val+test.npz
#
# Jean-Zay compute-accounting tags (IMAGINE wrapper):
#   https://github.com/Archiel19/compute-accounting

#SBATCH -A uhn@h100
#SBATCH -C h100
#SBATCH --comment=pureforest,explore,evaluate
#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/logs/slurm/%j/slurm.err

# 6 full-dataset forwards; leave headroom if a model is slow.
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# H100 Jean-Zay: 24 CPU/GPU (gpu_p6).
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

#SBATCH --job-name=pf_emb_extract

REPO_ROOT=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept
JOB_DIR=${REPO_ROOT}/logs/slurm/${SLURM_JOB_ID}
mkdir -p "${JOB_DIR}"

cp "$0" "${JOB_DIR}/script.slurm"

BATCH_SIZE="${BATCH_SIZE:-24}"
SPLITS="${SPLITS:-train val test}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/stats/pureforest/embeddings}"
DATA_ROOT="${DATA_ROOT:-}"          # empty -> use cfg data_root (data/pureforest)
POINT_MAX="${POINT_MAX:-}"          # empty -> full tile after GridSample (no SphereCrop)
SKIP_EXISTING="${SKIP_EXISTING:-0}"
PREECLAIR_WEIGHT="${PREECLAIR_WEIGHT:-${REPO_ROOT}/logs/slurm/1330042/model/model_best.pth}"

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "SPLITS=${SPLITS}"
    echo "OUT_ROOT=${OUT_ROOT}"
    echo "DATA_ROOT=${DATA_ROOT:-<cfg default>}"
    echo "POINT_MAX=${POINT_MAX:-<none>}"
    echo "SKIP_EXISTING=${SKIP_EXISTING}"
    echo "PREECLAIR_WEIGHT=${PREECLAIR_WEIGHT}"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    nvidia-smi
} > "${JOB_DIR}/job_info.log"

module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0

conda deactivate && while [ ! -z "$CONDA_DEFAULT_ENV" ]; do conda deactivate; done
conda activate pointcept_124

conda list > "${JOB_DIR}/conda_env.txt"

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

POINTOPS_PATH=/lustre/fswork/projects/rech/unv/usi32yh/Pointcept/pointops_build_h100/lib/python3.10/site-packages/pointops-1.0-py3.10-linux-x86_64.egg
export PYTHONPATH="${POINTOPS_PATH}:${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "${OUT_ROOT}"
SUMMARY="${JOB_DIR}/extract_summary.txt"
: > "${SUMMARY}"

extract_one() {
    local tag="$1"
    local config="$2"
    local weight="$3"
    local out_dir="${OUT_ROOT}/${tag}"

    if [ "${SKIP_EXISTING}" = "1" ] \
        && [ -f "${out_dir}/train.npz" ] \
        && [ -f "${out_dir}/val.npz" ] \
        && [ -f "${out_dir}/test.npz" ]; then
        echo "[skip] ${tag} (train/val/test.npz already present)" | tee -a "${SUMMARY}"
        return 0
    fi

    if [ ! -f "${weight}" ]; then
        echo "[FAIL] ${tag}: weight missing: ${weight}" | tee -a "${SUMMARY}"
        return 1
    fi

    echo "========== $(date)  extract ${tag} ==========" | tee -a "${SUMMARY}"
    local cmd=(
        python scripts/extract_pureforest_pooled_embeddings.py
        --config "${config}"
        --weight "${weight}"
        --output-dir "${out_dir}"
        --splits ${SPLITS}
        --batch-size "${BATCH_SIZE}"
    )
    if [ -n "${DATA_ROOT}" ]; then
        cmd+=(--data-root "${DATA_ROOT}")
    fi
    if [ -n "${POINT_MAX}" ]; then
        cmd+=(--point-max "${POINT_MAX}")
    fi

    echo "+ ${cmd[*]}" | tee -a "${SUMMARY}"
    if "${cmd[@]}"; then
        echo "[ok] ${tag} -> ${out_dir}" | tee -a "${SUMMARY}"
        return 0
    fi
    echo "[FAIL] ${tag} (rc=$?)" | tee -a "${SUMMARY}"
    return 1
}

START_TIME=$(date +%s)
FAILS=0

# tag | config | weight
extract_one sonata_outdoor_ms \
    configs/pureforest/cls-sonata-v1m2-pureforest-lin-grid-enc.py \
    "${REPO_ROOT}/ckpt/malibu3d/sonata_outdoor/epoch_120.pth" || FAILS=$((FAILS + 1))

extract_one litept_b_malibu3d_ms \
    configs/pureforest/cls-litept-b-v1m0-pureforest-lin-grid-enc.py \
    "${REPO_ROOT}/ckpt/malibu3d/litept_b_multitask/model_best.pth" || FAILS=$((FAILS + 1))

extract_one ptv3_malibu3d_ms \
    configs/pureforest/cls-ptv3-v1m0-pureforest-lin-grid-enc.py \
    "${REPO_ROOT}/ckpt/malibu3d/ptv3_multitask/model_best.pth" || FAILS=$((FAILS + 1))

extract_one spunet_malibu3d_ms \
    configs/pureforest/cls-spunet-v1m0-pureforest-lin-grid-enc.py \
    "${REPO_ROOT}/ckpt/malibu3d/spunet_multitask/model_best.pth" || FAILS=$((FAILS + 1))

extract_one kpconvx_malibu3d_ms \
    configs/pureforest/cls-kpconvx-v1m0-pureforest-lin-grid-enc.py \
    "${REPO_ROOT}/ckpt/malibu3d/kpconvx_multitask/model_best.pth" || FAILS=$((FAILS + 1))

extract_one litept_b_preECLAIR_ms \
    configs/pureforest/cls-litept-b-v1m0-pureforest-lin-grid-enc.py \
    "${PREECLAIR_WEIGHT}" || FAILS=$((FAILS + 1))

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

{
    echo "Finished at: $(date)"
    echo "Elapsed_s=${ELAPSED}"
    echo "FAILS=${FAILS}"
} | tee -a "${SUMMARY}" >> "${JOB_DIR}/job_info.log"

echo "Summary: ${SUMMARY}"
exit "${FAILS}"
