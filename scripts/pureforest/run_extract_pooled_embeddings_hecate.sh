#!/usr/bin/env bash
# Extract PureForest mean/max pooled embeddings for all MS-encoder backbones,
# locally on hecate (no Slurm) -- writes under
# stats/pureforest/embeddings/<tag>/{train,val,test}.npz
#
# Models (encoder multiscale GridProbe configs):
#   - Sonata outdoor SSL (Flair3D+ fork)
#   - Sonata indoor SSL (official Meta/HuggingFace release, coord/10)
#   - LitePT-B / PTv3 / SpUNet / KPConvX Malibu3D multitask
#   - LitePT-B supervised ECLAIR from-scratch + GradNorm/mono-task ablations
#   - LitePT-B noRGB ablation
#
# Usage (preferably inside tmux -- a full train+val+test pass over all 11
# tags is several hours):
#   bash scripts/pureforest/run_extract_pooled_embeddings_hecate.sh
#   SKIP_EXISTING=1 bash scripts/pureforest/run_extract_pooled_embeddings_hecate.sh
#   BATCH_SIZE=32 NUM_WORKERS=8 bash scripts/pureforest/run_extract_pooled_embeddings_hecate.sh
#
# Activate the pointcept env first (or set PY to its python directly), and
# make sure CUDA_VISIBLE_DEVICES / GPU below points at a working GPU -- on
# hecate, cuda:1 is known-broken for spconv ops (illegal memory access even
# on trivial input), always use cuda:0 (see project_hecate_gpu1_spconv_broken
# memory note).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PY="${PY:-python}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
# Ported from sbatch_extract_pooled_embeddings_h100.sh -- PureForest tiles are
# dense enough (see BATCH_SIZE note below) that consecutive dense batches
# fragment the CUDA caching allocator: an OOM can fire even with GBs nominally
# "free" (large "reserved but unallocated" in the error) because no single
# contiguous block is large enough. expandable_segments avoids that.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# A FIXED batch_size is not safe here, at any value -- do NOT go back to
# plain --batch-size and do NOT add --point-max to "fix" OOM instead (that
# changes the pooled features, defeating the point of full-tile extraction).
# Real (post coord-bug-fix) PureForest train tiles run far denser than these
# configs' point_max=5000 assumption (median ~92k pts after GridSample(0.1),
# not <5k), vary ~10x tile-to-tile, and PureForestDataset.get_data_list()
# sorts alphabetically, which clusters same-forest tiles together -- so a
# fixed tile-count batch can pack several of the split's largest tiles
# together. Confirmed 2026-09-22: batch_size=8 survived one sampled dense
# cluster but still OOM'd (38GB requested) on the true top-8-largest-tiles
# group (4.2M raw points combined) -- memory scales with total points/batch,
# not tile count, so POINT_BUDGET (--point-budget, First-Fit-Decreasing
# packing, see pack_indices_by_voxel_budget) replaces it entirely.
# 1,000,000 raw points/batch is a measured safe floor for the most
# memory-hungry backbone (Sonata/PT-v3m2: OK at ~1.2M combined, OOM at ~2.2M)
# with margin; BATCH_SIZE is now just the packer's max-tiles-per-batch cap
# (16 comfortably exceeds what the budget alone would ever pack at typical
# ~92k-pt tile density, ~10-11 tiles/batch).
POINT_BUDGET="${POINT_BUDGET:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SPLITS="${SPLITS:-train val test}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/stats/pureforest/embeddings}"
DATA_ROOT="${DATA_ROOT:-}"          # empty -> use cfg data_root (data/pureforest)
POINT_MAX="${POINT_MAX:-}"          # empty -> full tile after GridSample (no SphereCrop)
SKIP_EXISTING="${SKIP_EXISTING:-0}"
# See README's "did we actually measure a speedup" thread: 6 workers /
# prefetch_factor=1 was the plateau on this machine (40 cores, RTX A6000) --
# going higher (9, 12) gave no further gain and 12 measurably regressed
# (worker-spawn overhead). Stay at 6 unless you've re-benchmarked.
NUM_WORKERS="${NUM_WORKERS:-6}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"

mkdir -p "${OUT_ROOT}"
LOG_DIR="${REPO_ROOT}/logs/pureforest_embeddings_extract/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
SUMMARY="${LOG_DIR}/extract_summary.txt"
: > "${SUMMARY}"

{
    echo "GPU=${GPU} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
    echo "POINT_BUDGET=${POINT_BUDGET}"
    echo "BATCH_SIZE=${BATCH_SIZE} (max tiles/batch cap under POINT_BUDGET)"
    echo "SPLITS=${SPLITS}"
    echo "OUT_ROOT=${OUT_ROOT}"
    echo "DATA_ROOT=${DATA_ROOT:-<cfg default>}"
    echo "POINT_MAX=${POINT_MAX:-<none>}"
    echo "SKIP_EXISTING=${SKIP_EXISTING}"
    echo "NUM_WORKERS=${NUM_WORKERS}"
    echo "PREFETCH_FACTOR=${PREFETCH_FACTOR}"
    echo "Starting at: $(date)"
    echo "Host: $(hostname)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
} | tee "${LOG_DIR}/job_info.log"

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
        "${PY}" scripts/extract_pureforest_pooled_embeddings.py
        --config "${config}"
        --weight "${weight}"
        --output-dir "${out_dir}"
        --splits ${SPLITS}
        --batch-size "${BATCH_SIZE}"
        --point-budget "${POINT_BUDGET}"
        --num-workers "${NUM_WORKERS}"
        --prefetch-factor "${PREFETCH_FACTOR}"
    )
    if [ -n "${DATA_ROOT}" ]; then
        cmd+=(--data-root "${DATA_ROOT}")
    fi
    if [ -n "${POINT_MAX}" ]; then
        cmd+=(--point-max "${POINT_MAX}")
    fi

    echo "+ ${cmd[*]}" | tee -a "${SUMMARY}"
    if "${cmd[@]}" 2>&1 | tee "${LOG_DIR}/${tag}.log"; then
        echo "[ok] ${tag} -> ${out_dir}" | tee -a "${SUMMARY}"
        return 0
    fi
    echo "[FAIL] ${tag} (see ${LOG_DIR}/${tag}.log)" | tee -a "${SUMMARY}"
    return 1
}

START_TIME=$(date +%s)
FAILS=0

# tag | config | weight
extract_one sonata_outdoor_ms \
    configs/pureforest/cls-sonata-v1m2-pureforest-lin-grid-enc.py \
    "${REPO_ROOT}/ckpt/malibu3d/sonata_outdoor/epoch_120.pth" || FAILS=$((FAILS + 1))

# Official Meta/HuggingFace indoor release (not the Flair3D-fork outdoor ckpt
# above) -- coord_scale=1/10 baked into the config, see its docstring.
extract_one sonata_indoor_ms \
    configs/pureforest/cls-sonata-v1m1-pureforest-lin-grid-enc.py \
    "${REPO_ROOT}/ckpt/sonata/pretrain-sonata-v1m1-0-base.pth" || FAILS=$((FAILS + 1))

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

# GradNorm / mono-task / preECLAIR ablation checkpoints (same LitePT-B encoder,
# only the multitask-loss-balancing / head / pretrain-task setup differs -- see
# README_grid_then_seed.md's Frozen-backbone checkpoints block and
# ckpt/{1330042,1288597,1293025,1468317}/*.md). Local hecate layout is
# ckpt/<id>/model_best.pth -- NOT the Jean-Zay logs/slurm/<id>/model/ path used
# by sbatch_extract_pooled_embeddings_h100.sh.
PREECLAIR_WEIGHT="${PREECLAIR_WEIGHT:-${REPO_ROOT}/ckpt/1330042/model_best.pth}"
NOGNL_WEIGHT="${NOGNL_WEIGHT:-${REPO_ROOT}/ckpt/1288597/model_best.pth}"
MONOLC_WEIGHT="${MONOLC_WEIGHT:-${REPO_ROOT}/ckpt/1293025/model_best.pth}"
REALGN_WEIGHT="${REALGN_WEIGHT:-${REPO_ROOT}/ckpt/1468317/model_best.pth}"

extract_one litept_b_preECLAIR_ms \
    configs/pureforest/cls-litept-b-v1m0-pureforest-lin-grid-enc.py \
    "${PREECLAIR_WEIGHT}" || FAILS=$((FAILS + 1))

extract_one litept_b_noGNL_ms \
    configs/pureforest/cls-litept-b-v1m0-pureforest-lin-grid-enc.py \
    "${NOGNL_WEIGHT}" || FAILS=$((FAILS + 1))

extract_one litept_b_monoLC_ms \
    configs/pureforest/cls-litept-b-v1m0-pureforest-lin-grid-enc.py \
    "${MONOLC_WEIGHT}" || FAILS=$((FAILS + 1))

extract_one litept_b_realGN_ms \
    configs/pureforest/cls-litept-b-v1m0-pureforest-lin-grid-enc.py \
    "${REALGN_WEIGHT}" || FAILS=$((FAILS + 1))

# noRGB ablation: reference multitask checkpoint (malibu3d), colour replaced
# with the learned mask value on every forward -- see
# configs/pureforest/cls-litept-b-v1m0-pureforest-lin-grid-enc-norgb.py.
extract_one litept_b_malibu3d_norgb_ms \
    configs/pureforest/cls-litept-b-v1m0-pureforest-lin-grid-enc-norgb.py \
    "${REPO_ROOT}/ckpt/malibu3d/litept_b_multitask/model_best.pth" || FAILS=$((FAILS + 1))

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

{
    echo "Finished at: $(date)"
    echo "Elapsed_s=${ELAPSED}"
    echo "FAILS=${FAILS}"
} | tee -a "${SUMMARY}" >> "${LOG_DIR}/job_info.log"

echo "Summary: ${SUMMARY}"
exit "${FAILS}"
