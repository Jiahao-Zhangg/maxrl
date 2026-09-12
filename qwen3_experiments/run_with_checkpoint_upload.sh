#!/usr/bin/env bash

# Supervise an already configured training command and upload completed
# checkpoints, deleting local copies only after verified upload. Call after
# environment setup. FINAL_STEP=auto reads the trainer's actual epoch-derived
# step count; it does not add a step cap or change the optimizer schedule.
set -euo pipefail

if (( $# < 5 )) || [[ "$4" != "--" ]]; then
    echo "Usage: run_with_checkpoint_upload.sh CHECKPOINT_DIR HF_REPO_PREFIX FINAL_STEP_OR_auto -- COMMAND [ARGS...]" >&2
    exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CHECKPOINT_DIR=$(realpath -m -- "$1")
HF_REPO_PREFIX=$2
FINAL_STEP=$3
shift 4
PYTHON_BIN=${PYTHON_BIN:-python}
LOG_DIR=${CHECKPOINT_DIR}/logs
TRAINING_LOG=${LOG_DIR}/training.log
UPLOAD_LOG=${LOG_DIR}/checkpoint_upload.log
EXIT_STATUS_FILE=${LOG_DIR}/training.exit_status

[[ "${FINAL_STEP}" == auto || "${FINAL_STEP}" =~ ^[1-9][0-9]*$ ]] || {
    echo "FINAL_STEP must be a positive integer or auto." >&2
    exit 2
}
for executable in setsid flock; do
    command -v "${executable}" >/dev/null
done
if ! command -v hf >/dev/null && ! command -v huggingface-cli >/dev/null; then
    echo "Install the Hugging Face CLI, or set MAXRL_UPLOAD_CHECKPOINTS=0." >&2
    exit 1
fi
# Validate credentials without creating any remote repositories. Only saved
# checkpoints are published by the uploader once training has begun.
VALIDATION_STEP=${FINAL_STEP}
if [[ "${VALIDATION_STEP}" == auto ]]; then
    VALIDATION_STEP=1
fi
"${PYTHON_BIN}" - "${HF_REPO_PREFIX}-step_${VALIDATION_STEP}" <<'PY'
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import validate_repo_id

validate_repo_id(sys.argv[1])
print("HF checkpoint upload account:", HfApi().whoami()["name"])
PY

mkdir -p "${LOG_DIR}"
# Serialize launches using the same checkpoint directory and exit-status file.
exec 9>"${LOG_DIR}/checkpoint_upload.lock"
flock -n 9 || {
    echo "A training/upload process already owns ${CHECKPOINT_DIR}." >&2
    exit 1
}
rm -f -- "${EXIT_STATUS_FILE}"
TRAIN_PID=""
UPLOAD_PID=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    for pid in "${TRAIN_PID}" "${UPLOAD_PID}"; do
        if [[ -n "${pid}" ]]; then
            kill -TERM -- "-${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Checkpoints will upload to ${HF_REPO_PREFIX}-step_<N> (public); local copies are deleted only after verification."
echo "Checkpoint upload log: ${UPLOAD_LOG}"
# tee already replaces this per-attempt log. Clear it before the child starts
# so auto-detection cannot race with tee and consume a previous run's header.
: >"${TRAINING_LOG}"
# Wait for Python and tee together before publishing the actual exit status.
setsid bash -o pipefail -c 'log=$1; shift; "$@" 2>&1 | tee "$log"' \
    _ "${TRAINING_LOG}" "$@" &
TRAIN_PID=$!
if [[ "${FINAL_STEP}" == auto ]]; then
    echo "Waiting for the trainer's epoch-derived total step count."
    while true; do
        resolved_final_step=$(sed -nE 's/.*Total training steps: ([1-9][0-9]*)([^0-9].*)?$/\1/p' "${TRAINING_LOG}" | head -n 1)
        if [[ "${resolved_final_step}" =~ ^[1-9][0-9]*$ ]]; then
            FINAL_STEP=${resolved_final_step}
            echo "Checkpoint uploader final step: ${FINAL_STEP} (reported by trainer)."
            break
        fi
        if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
            training_status=0
            wait "${TRAIN_PID}" || training_status=$?
            TRAIN_PID=""
            printf '%s\n' "${training_status}" >"${EXIT_STATUS_FILE}.tmp"
            mv -- "${EXIT_STATUS_FILE}.tmp" "${EXIT_STATUS_FILE}"
            echo "Training ended before its total step count was available; retaining every local checkpoint." >&2
            if (( training_status != 0 )); then
                exit "${training_status}"
            fi
            exit 1
        fi
        sleep 1
    done
fi
PYTHON_BIN="${PYTHON_BIN}" MAXRL_TRAINING_EXIT_STATUS_FILE="${EXIT_STATUS_FILE}" \
    MAXRL_ARCHIVE_UPLOAD_LATEST=1 \
    setsid bash "${SCRIPT_DIR}/archive_checkpoints_to_hf.sh" \
    "${CHECKPOINT_DIR}" "${HF_REPO_PREFIX}" "$$" "${TRAINING_LOG}" "${FINAL_STEP}" \
    >>"${UPLOAD_LOG}" 2>&1 &
UPLOAD_PID=$!

training_status=0
wait "${TRAIN_PID}" || training_status=$?
TRAIN_PID=""
printf '%s\n' "${training_status}" >"${EXIT_STATUS_FILE}.tmp"
mv -- "${EXIT_STATUS_FILE}.tmp" "${EXIT_STATUS_FILE}"
echo "Training exited with status ${training_status}; waiting for checkpoint uploads."
upload_status=0
wait "${UPLOAD_PID}" || upload_status=$?
UPLOAD_PID=""
if (( training_status != 0 )); then
    exit "${training_status}"
fi
exit "${upload_status}"
