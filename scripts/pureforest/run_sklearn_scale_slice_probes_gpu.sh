#!/usr/bin/env bash
# Chain PureForest linear probes over encoder-scale slices (grid B).
#
# For each MS-encoder tag and each slice in:
#   full, [:1]..[:L-1], [1:]..[L-1:]
# runs scripts/probe_pureforest_sklearn.py with --scale-slice (torch LBFGS).
# Channel blocks are read from each embeddings meta.json config.
#
# Usage (hecate / local, preferably inside tmux):
#   bash scripts/pureforest/run_sklearn_scale_slice_probes_gpu.sh
#   DEVICE=cuda:1 SKIP_EXISTING=1 bash scripts/pureforest/run_sklearn_scale_slice_probes_gpu.sh
#   EXTRA_ARGS='--aggs mean concat' bash scripts/pureforest/run_sklearn_scale_slice_probes_gpu.sh
#
# Env:
#   OUT          embeddings root (default: stats/pureforest/embeddings)
#   PROBE_ROOT   probe output root (default: stats/pureforest/sklearn_probe_scales)
#   DEVICE       torch device (default: cuda)
#   SKIP_EXISTING=1  skip (tag, slice) that already have metrics.json
#   EXTRA_ARGS   extra CLI flags forwarded to probe_pureforest_sklearn.py
#   TAGS         space-separated override of embedding tags

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

OUT="${OUT:-stats/pureforest/embeddings}"
PROBE_ROOT="${PROBE_ROOT:-stats/pureforest/sklearn_probe_scales}"
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -n "${TAGS:-}" ]]; then
  # shellcheck disable=SC2206
  TAG_ARR=(${TAGS})
else
  TAG_ARR=(
    sonata_outdoor_ms
    litept_b_malibu3d_ms
    ptv3_malibu3d_ms
    spunet_malibu3d_ms
    kpconvx_malibu3d_ms
    litept_b_preECLAIR_ms
  )
fi

n_levels_for_tag() {
  local emb_dir="$1"
  python - "$emb_dir" <<'PY'
import json
import sys
from pathlib import Path

from pointcept.utils.config import Config

emb_dir = Path(sys.argv[1])
meta_path = emb_dir / "meta.json"
if not meta_path.is_file():
    raise SystemExit(f"missing {meta_path}")
meta = json.loads(meta_path.read_text(encoding="utf-8"))
config = meta.get("config")
if not config:
    raise SystemExit(f"no config in {meta_path}")
cfg_path = Path(config)
if not cfg_path.is_file():
    cfg_path = Path(".") / config
cfg = Config.fromfile(str(cfg_path))
blocks = None
model_cfg = cfg.get("model")
if model_cfg is not None:
    blocks = model_cfg.get("channel_blocks")
if blocks is None:
    blocks = cfg.get("enc_channels")
if blocks is None:
    raise SystemExit(f"no channel_blocks/enc_channels in {cfg_path}")
print(len(tuple(blocks)))
PY
}

# Fill SLICES and SLUGS arrays for grid B given L encoder levels.
build_grid_b() {
  local L="$1"
  SLICES=(full)
  SLUGS=(full)
  local k
  for ((k = 1; k < L; k++)); do
    SLICES+=("[:${k}]")
    SLUGS+=("p0-${k}")
    SLICES+=("[${k}:]")
    SLUGS+=("s${k}-end")
  done
}

echo "[scale_probes] REPO_ROOT=${REPO_ROOT}"
echo "[scale_probes] OUT=${OUT}  PROBE_ROOT=${PROBE_ROOT}  DEVICE=${DEVICE}"
echo "[scale_probes] tags=${TAG_ARR[*]}"
echo

n_ok=0
n_skip=0
n_fail=0

for tag in "${TAG_ARR[@]}"; do
  emb_dir="${OUT}/${tag}"
  if [[ ! -d "${emb_dir}" ]]; then
    echo "[scale_probes] FAIL ${tag}: missing embeddings dir ${emb_dir}"
    n_fail=$((n_fail + 1))
    continue
  fi
  for split in train val test; do
    if [[ ! -f "${emb_dir}/${split}.npz" ]]; then
      echo "[scale_probes] FAIL ${tag}: missing ${emb_dir}/${split}.npz"
      n_fail=$((n_fail + 1))
      continue 2
    fi
  done

  if ! L="$(n_levels_for_tag "${emb_dir}")"; then
    echo "[scale_probes] FAIL ${tag}: could not resolve encoder levels"
    n_fail=$((n_fail + 1))
    continue
  fi
  build_grid_b "${L}"
  echo "[scale_probes] ${tag}: L=${L}  n_slices=${#SLICES[@]}"

  for i in "${!SLICES[@]}"; do
    slice_spec="${SLICES[$i]}"
    slug="${SLUGS[$i]}"
    out_dir="${PROBE_ROOT}/${tag}/scale_${slug}"
    metrics="${out_dir}/metrics.json"

    if [[ "${SKIP_EXISTING}" == "1" && -f "${metrics}" ]]; then
      echo "[scale_probes] SKIP ${tag} scale_${slug} (found ${metrics})"
      n_skip=$((n_skip + 1))
      continue
    fi

    echo "================================================================"
    echo "[scale_probes] START ${tag} scale=${slice_spec} slug=${slug}  $(date -Is)"
    echo "================================================================"
    # shellcheck disable=SC2086
    if python scripts/probe_pureforest_sklearn.py \
        --embeddings-dir "${emb_dir}" \
        --output-dir "${out_dir}" \
        --scale-slice "${slice_spec}" \
        --device "${DEVICE}" \
        -v \
        ${EXTRA_ARGS}; then
      echo "[scale_probes] DONE ${tag} scale_${slug}  $(date -Is)"
      n_ok=$((n_ok + 1))
    else
      echo "[scale_probes] FAIL ${tag} scale_${slug} (exit $?)  $(date -Is)"
      n_fail=$((n_fail + 1))
    fi
    echo
  done
done

echo "[scale_probes] summary: ok=${n_ok} skip=${n_skip} fail=${n_fail}"
if [[ "${n_fail}" -gt 0 ]]; then
  exit 1
fi
