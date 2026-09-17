#!/usr/bin/env bash

set -uo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: launch_cost_aware_after_maxrl.sh MAXRL_PID MAXRL_LOG" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MAXRL_PID=$1
MAXRL_LOG=$(realpath -m -- "$2")
FINAL_STEP=${MAXRL_FINAL_STEP:-230}
POLL_SECONDS=${MAXRL_HANDOFF_POLL_SECONDS:-15}
RUN_DIR=${REPO_ROOT}/outputs/run
LOG_DIR=${REPO_ROOT}/outputs/logs
CAPPED_EXPERIMENT=cost_aware_maxrl_Qwen3-1.7B-Base_math12k_cap4
CAPPED_CHECKPOINT_DIR=${REPO_ROOT}/outputs/checkpoints/Qwen3_MaxRL_Experiments/${CAPPED_EXPERIMENT}
CAPPED_LOG=${LOG_DIR}/qwen3_1_7b_math12k_cost_aware.log
CAPPED_ARCHIVE_LOG=${LOG_DIR}/qwen3_1_7b_math12k_cost_aware_archive.log
CAPPED_HF_PREFIX=zjhhhh/cost-aware-maxrl-qwen3-1.7b-base-math12k-cap4
HANDOFF_LOCK=${RUN_DIR}/qwen3_math12k_cost_aware_handoff.lock

if [[ ! "${MAXRL_PID}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: MAXRL_PID must be a positive integer" >&2
    exit 2
fi
command -v conda >/dev/null 2>&1 || {
    echo "error: conda is required" >&2
    exit 1
}
command -v flock >/dev/null 2>&1 || {
    echo "error: flock is required" >&2
    exit 1
}
[[ -x "${SCRIPT_DIR}/run_qwen3_1_7b_math12k_cost_aware.sh" ]] || {
    echo "error: capped inverse-cost launcher is not executable" >&2
    exit 1
}
[[ -x "${SCRIPT_DIR}/archive_checkpoints_to_hf.sh" ]] || {
    echo "error: checkpoint archiver is not executable" >&2
    exit 1
}

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
exec 9>"${HANDOFF_LOCK}"
if ! flock -n 9; then
    echo "Another MaxRL-to-cost-aware handoff watcher is already active" >&2
    exit 0
fi

log() {
    printf '[%(%Y-%m-%dT%H:%M:%SZ)T] %s\n' -1 "$*"
}

process_is_running() {
    local pid=$1
    local state
    state=$(ps -o stat= -p "${pid}" 2>/dev/null) || return 1
    [[ "${state//[[:space:]]/}" != Z* ]]
}

training_succeeded() {
    [[ -f "${MAXRL_LOG}" ]] || return 1
    grep -aEq \
        "step:${FINAL_STEP}([[:space:]-]|$)|training/global_step:${FINAL_STEP}\\.000|${FINAL_STEP}/${FINAL_STEP}" \
        "${MAXRL_LOG}"
}

log "Waiting for MaxRL PID ${MAXRL_PID} to finish"
while process_is_running "${MAXRL_PID}"; do
    sleep "${POLL_SECONDS}"
done

if ! training_succeeded; then
    log "MaxRL stopped without reaching step ${FINAL_STEP}; capped inverse-cost will not be launched"
    exit 1
fi

if [[ -f "${RUN_DIR}/qwen3_math12k_cost_aware.pid" ]]; then
    existing_pid=$(<"${RUN_DIR}/qwen3_math12k_cost_aware.pid")
    if [[ "${existing_pid}" =~ ^[1-9][0-9]*$ ]] && process_is_running "${existing_pid}"; then
        log "Capped inverse-cost is already running as PID ${existing_pid}"
        exit 0
    fi
fi

: >"${CAPPED_LOG}"
: >"${CAPPED_ARCHIVE_LOG}"
log "MaxRL completed; launching capped inverse-cost on GPUs 0-3"
(
    exec conda run --no-capture-output -n maxrl \
        env PYTHONNOUSERSITE=1 MAXRL_SKIP_ENV_SETUP=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
        MAXRL_LOSS_AGG_MODE=token-mean MAXRL_COST_REFERENCE_TOKENS=2048 \
        MAXRL_MAX_INVERSE_COST=4.0 \
        bash "${SCRIPT_DIR}/run_qwen3_1_7b_math12k_cost_aware.sh"
) >>"${CAPPED_LOG}" 2>&1 &
capped_pid=$!
printf '%s\n' "${capped_pid}" >"${RUN_DIR}/qwen3_math12k_cost_aware.pid"
log "Capped inverse-cost controller PID: ${capped_pid}; log: ${CAPPED_LOG}"

"${SCRIPT_DIR}/archive_checkpoints_to_hf.sh" \
    "${CAPPED_CHECKPOINT_DIR}" \
    "${CAPPED_HF_PREFIX}" \
    "${capped_pid}" \
    "${CAPPED_LOG}" \
    "${FINAL_STEP}" >>"${CAPPED_ARCHIVE_LOG}" 2>&1 &
archive_pid=$!
printf '%s\n' "${archive_pid}" >"${RUN_DIR}/qwen3_math12k_cost_aware_archive.pid"

wait "${capped_pid}"
capped_status=$?
log "Capped inverse-cost exited with status ${capped_status}"
wait "${archive_pid}"
archive_status=$?
log "Capped checkpoint archiver exited with status ${archive_status}"

if (( capped_status != 0 )); then
    exit "${capped_status}"
fi
exit "${archive_status}"
