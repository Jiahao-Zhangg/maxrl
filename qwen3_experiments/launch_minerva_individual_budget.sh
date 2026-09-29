#!/usr/bin/env bash
# Queue individual budgets after the compression run's nine-benchmark eval.
# Login only starts the detached compute-node controller.
# Optional: MINERVA_PARENT_RUN, MINERVA_RUN_DIR, MINERVA_PYTHON_BIN, MINERVA_SEED,
# MINERVA_SSH_KNOWN_HOSTS, MINERVA_CHECKPOINT_ONLY. --prepare-only does not launch a queue or use GPUs.
# --replace-waiting PATH supersedes an earlier, unstarted Minerva queue.
# MINERVA_DATASETS is a space-separated list; it defaults to minervamath.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
PARENT_RUN=${MINERVA_PARENT_RUN:-"${REPO_ROOT}/outputs/per_context_rb_l0_0_qwen3_1_7b_compression_bs32_32k_146103"}
SEED=${MINERVA_SEED:-0}
[[ ${SEED} =~ ^[0-9]+$ ]] || { echo "Invalid seed" >&2; exit 2; }
CHECKPOINT_ONLY=${MINERVA_CHECKPOINT_ONLY:-false}
[[ ${CHECKPOINT_ONLY} == true || ${CHECKPOINT_ONLY} == false ]] || { echo "Invalid checkpoint-only setting" >&2; exit 2; }
MODEL_SELECTION=five_models
MODEL_ARGS=()
if [[ ${CHECKPOINT_ONLY} == true ]]; then
    MODEL_SELECTION=checkpoint
    MODEL_ARGS=(--checkpoint-only)
fi
read -r -a BUDGET_DATASETS <<< "${MINERVA_DATASETS:-minervamath}"
DATASET_PREFIX=minerva
if [[ ${BUDGET_DATASETS[*]} != minervamath ]]; then
    DATASET_PREFIX=$(IFS=_; echo "${BUDGET_DATASETS[*]}")
fi
RUN_DIR=${MINERVA_RUN_DIR:-"${PARENT_RUN}/${DATASET_PREFIX}_individual_budget_seed${SEED}_${MODEL_SELECTION}"}
mapfile -t PARENT_CONFIG < <(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["job_id"]); print(p["python_bin"])' "${PARENT_RUN}/plan.json")
JOB_ID=${PARENT_CONFIG[0]}
PYTHON_BIN=${MINERVA_PYTHON_BIN:-${PARENT_CONFIG[1]}}
[[ ${JOB_ID} =~ ^[0-9]+$ ]] || { echo "Invalid allocation ID" >&2; exit 2; }
NODE=$(squeue --noheader --jobs="${JOB_ID}" --format='%N')
[[ ${NODE} =~ ^[a-zA-Z0-9-]+$ ]] || { echo "Expected one allocated compute node" >&2; exit 1; }
ACTION=launch
REPLACE_ARGS=()
if [[ ${1:-} == --prepare-only ]]; then
    ACTION=prepare
    shift
fi
if [[ ${1:-} == --replace-waiting && $# -eq 2 ]]; then
    REPLACE_ARGS=(--replace-waiting "$2")
    shift 2
fi
[[ $# -eq 0 ]] || { echo "Usage: $0 [--prepare-only] [--replace-waiting PATH]" >&2; exit 2; }
REMOTE_ARGS=(env "PYTHONPATH=${REPO_ROOT}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=)
for CACHE_VARIABLE in HF_HOME HF_TOKEN_PATH HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE; do
    if [[ -n ${!CACHE_VARIABLE:-} ]]; then
        REMOTE_ARGS+=("${CACHE_VARIABLE}=${!CACHE_VARIABLE}")
    fi
done
REMOTE_ARGS+=("${PYTHON_BIN}" -u -m qwen3_experiments.minerva_individual_budget "${ACTION}"
    --parent-run "${PARENT_RUN}" --output-root "${RUN_DIR}" --seed "${SEED}"
    --datasets "${BUDGET_DATASETS[@]}" "${MODEL_ARGS[@]}" "${REPLACE_ARGS[@]}")
printf -v REMOTE_COMMAND '%q ' "${REMOTE_ARGS[@]}"
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=15)
if [[ -n ${MINERVA_SSH_KNOWN_HOSTS:-} ]]; then
    SSH_OPTIONS+=(-o "UserKnownHostsFile=\"${MINERVA_SSH_KNOWN_HOSTS}\"" -o StrictHostKeyChecking=yes)
fi
exec ssh "${SSH_OPTIONS[@]}" "${NODE}" "${REMOTE_COMMAND}"
