#!/usr/bin/env bash
# Linear probes (C x aggregation grid, full encoder scale only) on Myria3D
# PureForest pooled embeddings. meta.json has no "config", so channel blocks
# are passed explicitly (32 128 256 512 = 928).
#
# Usage: DEVICE=cuda:0 bash scripts/pureforest/run_myria3d_probes_gpu.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"

EMB="${EMB:-/data/geist/myria3d/stats/pureforest_embeddings}"
ROOT="${ROOT:-stats/pureforest/sklearn_probe}"
DEVICE="${DEVICE:-cuda:0}"
BLOCKS="32 128 256 512"

run() { # out_dir [extra args]
  local out="$1"; shift
  "${PY:-python}" scripts/probe_pureforest_sklearn.py --embeddings-dir "${EMB}" \
    --output-dir "${out}" --device "${DEVICE}" --channel-blocks ${BLOCKS} -v "$@"
}

run "${ROOT}/myria3d_ms" --scale-slice full
echo ALL_DONE
