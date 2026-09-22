#!/usr/bin/env bash
# One-shot watcher: waits for the sonata_outdoor_ms embedding tag to finish
# in a running run_extract_pooled_embeddings_hecate.sh job, then launches its
# sklearn linear probe in the background -- in parallel with the still-running
# extraction of the NEXT model (sonata_indoor_ms and onward). Just a quick
# sanity check that perf looks as expected, not waiting for every backbone.
# If GPU memory gets tight enough to risk an OOM for either the probe or the
# extraction while both share the GPU, the probe wins: extraction is killed
# so probing can finish cleanly, then the extraction runner is relaunched
# (SKIP_EXISTING=1) to pick up the remaining/interrupted tags.
#
# Not meant to be re-run manually mid-flight; it's launched once per
# extraction run and exits after handing off (or after a hard failure).
#
# Usage:
#   SUMMARY=<path to extract_summary.txt of the running job> \
#   ORIG_PY=<python used to launch that job> ORIG_GPU=<GPU index> \
#     nohup bash scripts/pureforest/probe_sonata_when_ready.sh > watcher.log 2>&1 &

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SUMMARY="${SUMMARY:?Set SUMMARY=path/to/extract_summary.txt}"
PY="${ORIG_PY:-python}"
GPU="${ORIG_GPU:-0}"
EMB_ROOT="${REPO_ROOT}/stats/pureforest/embeddings"
PROBE_ROOT="${REPO_ROOT}/stats/pureforest/sklearn_probe"
WATCHER_LOG_DIR="$(dirname "${SUMMARY}")"

log() { echo "[probe_watcher $(date +%H:%M:%S)] $*"; }

wait_for_tag() {
    # Blocks until the tag has an [ok]/[FAIL]/[skip] line in SUMMARY.
    # Echoes "ok" / "fail" / "skip" to stdout.
    local tag="$1"
    while true; do
        if grep -qE "^\[ok\] ${tag} " "${SUMMARY}" 2>/dev/null; then
            echo "ok"; return 0
        fi
        if grep -qE "^\[FAIL\] ${tag}( |$)" "${SUMMARY}" 2>/dev/null; then
            echo "fail"; return 0
        fi
        if grep -qE "^\[skip\] ${tag} " "${SUMMARY}" 2>/dev/null; then
            echo "ok"; return 0
        fi
        sleep 30
    done
}

log "waiting for sonata_outdoor_ms to finish..."
outdoor_status=$(wait_for_tag sonata_outdoor_ms)
log "sonata_outdoor_ms -> ${outdoor_status}"

TAGS_TO_PROBE=()
[ "${outdoor_status}" = "ok" ] && TAGS_TO_PROBE+=(sonata_outdoor_ms)

if [ "${#TAGS_TO_PROBE[@]}" -eq 0 ]; then
    log "sonata_outdoor_ms failed -- nothing to probe, exiting."
    exit 1
fi
log "probing: ${TAGS_TO_PROBE[*]}"

run_probe() {
    local tag="$1"
    local out_dir="${PROBE_ROOT}/${tag}"
    local log_file="${WATCHER_LOG_DIR}/probe_${tag}.log"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" scripts/probe_pureforest_sklearn.py \
        --embeddings-dir "${EMB_ROOT}/${tag}" \
        --output-dir "${out_dir}" \
        --device cuda -v > "${log_file}" 2>&1
    echo $? > "${log_file}.rc"
}

PROBE_PIDS=()
for tag in "${TAGS_TO_PROBE[@]}"; do
    run_probe "${tag}" &
    PROBE_PIDS+=("$!")
    log "launched probe for ${tag} (pid $!) -> ${WATCHER_LOG_DIR}/probe_${tag}.log"
done

EXTRACTION_KILLED=0

probes_alive() {
    for pid in "${PROBE_PIDS[@]}"; do
        kill -0 "${pid}" 2>/dev/null && return 0
    done
    return 1
}

probe_oom_seen() {
    for tag in "${TAGS_TO_PROBE[@]}"; do
        if grep -qi "OutOfMemoryError\|CUDA out of memory" \
            "${WATCHER_LOG_DIR}/probe_${tag}.log" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

log "monitoring GPU headroom + probe logs while extraction continues in parallel..."
while probes_alive; do
    if [ "${EXTRACTION_KILLED}" -eq 0 ]; then
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null || echo 99999)
        if probe_oom_seen || [ "${free_mib}" -lt 1500 ]; then
            log "OOM risk detected (free=${free_mib}MiB or probe OOM'd) -- killing extraction to let probing finish."
            pkill -9 -f "run_extract_pooled_embeddings_hecate.sh" 2>/dev/null
            pkill -9 -f "extract_pureforest_pooled_embeddings.py" 2>/dev/null
            EXTRACTION_KILLED=1
        fi
    fi
    sleep 10
done

log "all probe(s) finished."
for tag in "${TAGS_TO_PROBE[@]}"; do
    rc=$(cat "${WATCHER_LOG_DIR}/probe_${tag}.log.rc" 2>/dev/null || echo "?")
    log "probe ${tag} exit code: ${rc}"
    if [ -f "${PROBE_ROOT}/${tag}/metrics.json" ]; then
        log "${tag} metrics: $(cat "${PROBE_ROOT}/${tag}/metrics.json")"
    fi
done

if [ "${EXTRACTION_KILLED}" -eq 1 ]; then
    log "resuming extraction for remaining tags (SKIP_EXISTING=1)..."
    nohup env PY="${PY}" GPU="${GPU}" SKIP_EXISTING=1 \
        bash scripts/pureforest/run_extract_pooled_embeddings_hecate.sh \
        > "${WATCHER_LOG_DIR}/resumed_extraction_stdout.log" 2>&1 &
    disown
    log "resumed extraction launched (pid $!), new log dir will appear under logs/pureforest_embeddings_extract/"
else
    log "extraction was never interrupted -- still running its own course."
fi

log "watcher done."
