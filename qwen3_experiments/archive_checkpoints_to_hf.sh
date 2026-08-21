#!/usr/bin/env bash

set -uo pipefail

usage() {
    cat <<'EOF'
Usage: archive_checkpoints_to_hf.sh EXPERIMENT_DIR HF_REPO_PREFIX TRAINING_PID TRAINING_LOG FINAL_STEP

While training is running, uploads and removes every checkpoint except the
newest one. After a successful run, uploads and removes all remaining
checkpoints. If training fails, the newest local checkpoint is retained.

Set MAXRL_ARCHIVE_MIN_FREE_GIB to a positive integer to also archive the
newest completed checkpoint whenever free disk space falls below that limit.

Each global_step_N checkpoint is stored in a public Hub model repository named
HF_REPO_PREFIX-step_N, with global_step_N preserved as the top-level folder.
EOF
}

if [[ $# -ne 5 ]]; then
    usage >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
EXPERIMENT_DIR=$(realpath -m -- "$1")
HF_REPO_PREFIX=$2
TRAINING_PID=$3
TRAINING_LOG=$(realpath -m -- "$4")
FINAL_STEP=$5
POLL_SECONDS=${MAXRL_ARCHIVE_POLL_SECONDS:-60}
UPLOAD_WORKERS=${MAXRL_HF_UPLOAD_WORKERS:-4}
UPLOAD_LOCK=${MAXRL_HF_UPLOAD_LOCK:-${REPO_ROOT}/outputs/.hf_checkpoint_upload.lock}
MIN_FREE_GIB=${MAXRL_ARCHIVE_MIN_FREE_GIB:-0}

if [[ ! "${TRAINING_PID}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: TRAINING_PID must be a positive integer" >&2
    exit 2
fi
if [[ ! "${FINAL_STEP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: FINAL_STEP must be a positive integer" >&2
    exit 2
fi
if [[ ! "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: MAXRL_ARCHIVE_POLL_SECONDS must be a positive integer" >&2
    exit 2
fi
if [[ ! "${UPLOAD_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: MAXRL_HF_UPLOAD_WORKERS must be a positive integer" >&2
    exit 2
fi
if [[ ! "${MIN_FREE_GIB}" =~ ^[0-9]+$ ]]; then
    echo "error: MAXRL_ARCHIVE_MIN_FREE_GIB must be a nonnegative integer" >&2
    exit 2
fi
if [[ ! "${HF_REPO_PREFIX}" =~ ^[^/]+/[^/]+$ ]]; then
    echo "error: HF_REPO_PREFIX must have the form owner/name" >&2
    exit 2
fi

command -v hf >/dev/null 2>&1 || {
    echo "error: the Hugging Face 'hf' CLI is required" >&2
    exit 1
}
command -v flock >/dev/null 2>&1 || {
    echo "error: flock is required" >&2
    exit 1
}
python -c 'import huggingface_hub' >/dev/null 2>&1 || {
    echo "error: huggingface_hub is required by the active Python" >&2
    exit 1
}

mkdir -p "${EXPERIMENT_DIR}" "$(dirname -- "${UPLOAD_LOCK}")"

log() {
    printf '[%(%Y-%m-%dT%H:%M:%SZ)T] %s\n' -1 "$*"
}

training_is_running() {
    local state
    state=$(ps -o stat= -p "${TRAINING_PID}" 2>/dev/null) || return 1
    [[ "${state//[[:space:]]/}" != Z* ]]
}

training_succeeded() {
    [[ -f "${TRAINING_LOG}" ]] || return 1
    grep -aEq \
        "step:${FINAL_STEP}([[:space:]-]|$)|training/global_step:${FINAL_STEP}\\.000|${FINAL_STEP}/${FINAL_STEP}" \
        "${TRAINING_LOG}"
}

disk_space_is_low() {
    local free_bytes minimum_free_bytes
    (( MIN_FREE_GIB > 0 )) || return 1
    free_bytes=$(df -PB1 "${EXPERIMENT_DIR}" | awk 'NR == 2 {print $4}') || return 1
    [[ "${free_bytes}" =~ ^[0-9]+$ ]] || return 1
    minimum_free_bytes=$((MIN_FREE_GIB * 1024 * 1024 * 1024))
    (( free_bytes < minimum_free_bytes ))
}

list_checkpoints() {
    local path name
    [[ -d "${EXPERIMENT_DIR}" ]] || return 0
    while IFS= read -r path; do
        name=${path##*/}
        [[ "${name}" =~ ^global_step_[0-9]+$ ]] && printf '%s\n' "${path}"
    done < <(find "${EXPERIMENT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -print)
}

checkpoint_is_complete() {
    local checkpoint=$1
    local checkpoint_name checkpoint_step latest_step
    local model_count optim_count extra_count

    [[ -s "${checkpoint}/data.pt" && -d "${checkpoint}/actor" ]] || return 1
    checkpoint_name=${checkpoint##*/}
    checkpoint_step=${checkpoint_name#global_step_}
    [[ -s "${EXPERIMENT_DIR}/latest_checkpointed_iteration.txt" ]] || return 1
    latest_step=$(tr -d '[:space:]' <"${EXPERIMENT_DIR}/latest_checkpointed_iteration.txt")
    [[ "${latest_step}" =~ ^[0-9]+$ && "${checkpoint_step}" -le "${latest_step}" ]] || return 1

    model_count=$(find "${checkpoint}/actor" -maxdepth 1 -type f -size +0c -name 'model_world_size_*_rank_*.pt' | wc -l)
    optim_count=$(find "${checkpoint}/actor" -maxdepth 1 -type f -size +0c -name 'optim_world_size_*_rank_*.pt' | wc -l)
    extra_count=$(find "${checkpoint}/actor" -maxdepth 1 -type f -size +0c -name 'extra_state_world_size_*_rank_*.pt' | wc -l)
    [[ "${model_count}" -gt 0 && "${model_count}" -eq "${optim_count}" && "${model_count}" -eq "${extra_count}" ]]
}

make_repo_public() {
    local repo_id=$1
    python - "${repo_id}" <<'PY'
import sys

from huggingface_hub import HfApi

repo_id = sys.argv[1]
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
info = api.repo_info(repo_id=repo_id, repo_type="model")
if info.private:
    api.update_repo_settings(repo_id=repo_id, private=False)
PY
}

verify_upload() {
    local checkpoint=$1
    local repo_id=$2
    python - "${checkpoint}" "${repo_id}" <<'PY'
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

checkpoint = Path(sys.argv[1]).resolve()
repo_id = sys.argv[2]
prefix = checkpoint.name + "/"
local_files = {
    prefix + path.relative_to(checkpoint).as_posix(): path.stat().st_size
    for path in checkpoint.rglob("*")
    if path.is_file() and ".cache" not in path.relative_to(checkpoint).parts
}
if not local_files:
    raise SystemExit("checkpoint has no files to verify")

api = HfApi()
last_error = "repository metadata was unavailable"
for attempt in range(6):
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
        if info.private:
            raise RuntimeError("repository is private")
        remote_files = {item.rfilename: item.size for item in info.siblings}
        missing = sorted(set(local_files) - set(remote_files))
        wrong_size = sorted(
            path for path, size in local_files.items() if remote_files.get(path) != size
        )
        if not missing and not wrong_size:
            print(f"Verified {len(local_files)} files ({sum(local_files.values())} bytes)")
            raise SystemExit(0)
        last_error = f"missing={missing[:3]}, wrong_size={wrong_size[:3]}"
    except Exception as error:
        last_error = str(error)
    if attempt < 5:
        time.sleep(10)
raise SystemExit(f"remote verification failed: {last_error}")
PY
}

delete_verified_checkpoint() {
    local checkpoint=$1
    local checkpoint_name resolved expected

    checkpoint_name=${checkpoint##*/}
    [[ "${checkpoint_name}" =~ ^global_step_[0-9]+$ ]] || return 1
    resolved=$(realpath -- "${checkpoint}") || return 1
    expected="${EXPERIMENT_DIR}/${checkpoint_name}"
    [[ "${resolved}" == "${expected}" && -d "${resolved}" ]] || return 1

    find "${resolved}" -depth -delete
    [[ ! -e "${resolved}" ]]
}

clear_upload_metadata() {
    local metadata_dir="${EXPERIMENT_DIR}/.cache/huggingface"
    if [[ -d "${metadata_dir}" ]]; then
        find "${metadata_dir}" -depth -delete
        rmdir "${EXPERIMENT_DIR}/.cache" 2>/dev/null || true
    fi
}

archive_checkpoint() {
    local checkpoint=$1
    local checkpoint_name step repo_id
    local lock_fd

    checkpoint_name=${checkpoint##*/}
    step=${checkpoint_name#global_step_}
    repo_id="${HF_REPO_PREFIX}-step_${step}"

    if ! checkpoint_is_complete "${checkpoint}"; then
        log "Deferring incomplete checkpoint ${checkpoint_name}"
        return 1
    fi

    exec {lock_fd}>"${UPLOAD_LOCK}"
    log "Waiting for the shared Hugging Face upload lock for ${checkpoint_name}"
    flock "${lock_fd}"

    if [[ ! -d "${checkpoint}" ]]; then
        log "Checkpoint ${checkpoint_name} was already removed"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 0
    fi
    if ! checkpoint_is_complete "${checkpoint}"; then
        log "Checkpoint ${checkpoint_name} is no longer complete; retaining it"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi

    log "Uploading ${checkpoint_name} to public repo ${repo_id}"
    if ! make_repo_public "${repo_id}"; then
        log "Failed to create or publish ${repo_id}; retaining ${checkpoint_name}"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    # The pinned training environment deliberately sets PYTHONNOUSERSITE=1,
    # while this server's authenticated `hf` launcher is installed in the
    # user site. Unset only for the CLI subprocess so the trainer remains
    # isolated from user packages.
    if ! env -u PYTHONNOUSERSITE HF_XET_HIGH_PERFORMANCE=1 hf upload-large-folder \
        "${repo_id}" "${EXPERIMENT_DIR}" \
        --repo-type model \
        --include "${checkpoint_name}/**" \
        --exclude "${checkpoint_name}/.cache/**" \
        --num-workers "${UPLOAD_WORKERS}"; then
        log "Upload failed for ${checkpoint_name}; retaining it for retry"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    if ! verify_upload "${checkpoint}" "${repo_id}"; then
        log "Verification failed for ${checkpoint_name}; retaining it for retry"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    if ! delete_verified_checkpoint "${checkpoint}"; then
        log "Safety check prevented deletion of ${checkpoint_name}"
        flock -u "${lock_fd}"
        exec {lock_fd}>&-
        return 1
    fi
    clear_upload_metadata
    log "Archived and deleted ${checkpoint_name}; recover it from https://huggingface.co/${repo_id}"

    flock -u "${lock_fd}"
    exec {lock_fd}>&-
}

log "Monitoring ${EXPERIMENT_DIR} for training PID ${TRAINING_PID}"
while true; do
    mapfile -t all_checkpoints < <(list_checkpoints | sort -V)
    checkpoints=()
    for checkpoint in "${all_checkpoints[@]}"; do
        if checkpoint_is_complete "${checkpoint}"; then
            checkpoints+=("${checkpoint}")
        fi
    done

    if training_is_running; then
        retained_checkpoint_count=1
        if disk_space_is_low; then
            retained_checkpoint_count=0
            log "Free disk space is below ${MIN_FREE_GIB} GiB; archiving every completed checkpoint"
        fi
        if (( ${#checkpoints[@]} > retained_checkpoint_count )); then
            for ((index = 0; index < ${#checkpoints[@]} - retained_checkpoint_count; index++)); do
                if ! archive_checkpoint "${checkpoints[index]}"; then
                    break
                fi
            done
        fi
        sleep "${POLL_SECONDS}"
        continue
    fi

    if training_succeeded; then
        log "Training reached final step ${FINAL_STEP}; archiving all remaining checkpoints"
        archive_failed=0
        for checkpoint in "${checkpoints[@]}"; do
            if ! archive_checkpoint "${checkpoint}"; then
                archive_failed=1
                break
            fi
        done
        if (( ${#all_checkpoints[@]} != ${#checkpoints[@]} )); then
            log "At least one checkpoint is incomplete; retaining it"
            archive_failed=1
        fi
        if (( archive_failed == 0 )); then
            log "Checkpoint archival is complete"
            exit 0
        fi
        log "At least one checkpoint could not be archived; retrying in ${POLL_SECONDS}s"
        sleep "${POLL_SECONDS}"
        continue
    fi

    if (( ${#checkpoints[@]} > 1 )); then
        for ((index = 0; index < ${#checkpoints[@]} - 1; index++)); do
            if ! archive_checkpoint "${checkpoints[index]}"; then
                break
            fi
        done
    fi
    log "Training stopped before step ${FINAL_STEP}; retaining the newest checkpoint for recovery"
    exit 1
done
