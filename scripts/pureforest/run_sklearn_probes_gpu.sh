#!/usr/bin/env bash
# Chain PureForest linear probes on GPU (torch LBFGS) for all MS-encoder tags.
#
# Matches the probe block in
#   pointcept/datasets/preprocessing/pureforest/README.md
# but adds ``--device cuda`` (train/val tensors resident per aggregation).
#
# Usage (hecate / local, preferably inside tmux):
#   bash scripts/pureforest/run_sklearn_probes_gpu.sh
#   DEVICE=cuda:1 SKIP_EXISTING=1 bash scripts/pureforest/run_sklearn_probes_gpu.sh
#   EXTRA_ARGS='--fresh' bash scripts/pureforest/run_sklearn_probes_gpu.sh
#
# Env:
#   OUT          embeddings root (default: stats/pureforest/embeddings)
#   PROBE_ROOT   probe output root (default: stats/pureforest/sklearn_probe)
#   DEVICE       torch device (default: cuda)
#   SKIP_EXISTING=1  skip tags that already have metrics.json
#   EXTRA_ARGS   extra CLI flags forwarded to probe_pureforest_sklearn.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

OUT="${OUT:-stats/pureforest/embeddings}"
PROBE_ROOT="${PROBE_ROOT:-stats/pureforest/sklearn_probe}"
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TAGS=(
  sonata_outdoor_ms
  litept_b_malibu3d_ms
  ptv3_malibu3d_ms
  spunet_malibu3d_ms
  kpconvx_malibu3d_ms
  litept_b_preECLAIR_ms
)

echo "[probes_gpu] REPO_ROOT=${REPO_ROOT}"
echo "[probes_gpu] OUT=${OUT}  PROBE_ROOT=${PROBE_ROOT}  DEVICE=${DEVICE}"
echo "[probes_gpu] tags=${TAGS[*]}"
echo

n_ok=0
n_skip=0
n_fail=0

for tag in "${TAGS[@]}"; do
  emb_dir="${OUT}/${tag}"
  out_dir="${PROBE_ROOT}/${tag}"
  metrics="${out_dir}/metrics.json"

  if [[ ! -d "${emb_dir}" ]]; then
    echo "[probes_gpu] FAIL ${tag}: missing embeddings dir ${emb_dir}"
    n_fail=$((n_fail + 1))
    continue
  fi
  for split in train val test; do
    if [[ ! -f "${emb_dir}/${split}.npz" ]]; then
      echo "[probes_gpu] FAIL ${tag}: missing ${emb_dir}/${split}.npz"
      n_fail=$((n_fail + 1))
      continue 2
    fi
  done

  if [[ "${SKIP_EXISTING}" == "1" && -f "${metrics}" ]]; then
    echo "[probes_gpu] SKIP ${tag} (found ${metrics})"
    n_skip=$((n_skip + 1))
    continue
  fi

  echo "================================================================"
  echo "[probes_gpu] START ${tag}  $(date -Is)"
  echo "================================================================"
  # shellcheck disable=SC2086
  if python scripts/probe_pureforest_sklearn.py \
      --embeddings-dir "${emb_dir}" \
      --output-dir "${out_dir}" \
      --device "${DEVICE}" \
      -v \
      ${EXTRA_ARGS}; then
    echo "[probes_gpu] DONE ${tag}  $(date -Is)"
    n_ok=$((n_ok + 1))
  else
    echo "[probes_gpu] FAIL ${tag} (exit $?)  $(date -Is)"
    n_fail=$((n_fail + 1))
  fi
  echo
done

echo "[probes_gpu] summary: ok=${n_ok} skip=${n_skip} fail=${n_fail}"
if [[ "${n_fail}" -gt 0 ]]; then
  exit 1
fi
