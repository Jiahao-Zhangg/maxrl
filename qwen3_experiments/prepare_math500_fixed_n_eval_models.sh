#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
EVAL_ROOT=${MAXRL_EVAL_ROOT:-${REPO_ROOT}/outputs/eval_math500_step100}
CONDA_ENV=${MAXRL_EVAL_CONDA_ENV:-maxrl}

read -r -a MODELS <<< "${MAXRL_EVAL_MODELS:-fixed_n_rb_step50 fixed_n_rb_step100 fixed_n_rb_capped_step100 fixed_n_rb_capped_step150}"
declare -A REPOS=(
    [fixed_n_rb_step50]="zjhhhh/fixed-n-rb-cost-aware-marginrl-qwen3-1.7b-base-math12k-step_50"
    [fixed_n_rb_step100]="zjhhhh/fixed-n-rb-cost-aware-marginrl-qwen3-1.7b-base-math12k-step_100"
    [fixed_n_rb_capped_step100]="zjhhhh/fixed-n-rb-capped-cost-aware-marginrl-qwen3-1.7b-base-math12k-cap4-step_100"
    [fixed_n_rb_capped_step150]="zjhhhh/fixed-n-rb-capped-cost-aware-marginrl-qwen3-1.7b-base-math12k-cap4-step_150"
)
declare -A STEPS=(
    [fixed_n_rb_step50]=50
    [fixed_n_rb_step100]=100
    [fixed_n_rb_capped_step100]=100
    [fixed_n_rb_capped_step150]=150
)

mkdir -p "${EVAL_ROOT}/merged" "${EVAL_ROOT}/staging"

for model in "${MODELS[@]}"; do
    target_dir="${EVAL_ROOT}/merged/${model}"
    if [[ -s "${target_dir}/model.safetensors" && -s "${target_dir}/config.json" ]]; then
        printf 'Skipping merged model: %s\n' "${target_dir}"
        continue
    fi
    if [[ -e "${target_dir}" ]]; then
        printf 'error: refusing to overwrite incomplete target: %s\n' "${target_dir}" >&2
        exit 1
    fi

    step=${STEPS[$model]}
    checkpoint="global_step_${step}"
    staging_dir="${EVAL_ROOT}/staging/${model}"
    download_dir="${staging_dir}/download"
    actor_dir="${download_dir}/${checkpoint}/actor"
    mkdir -p "${staging_dir}"

    printf 'Downloading model shards for %s from %s\n' "${model}" "${REPOS[$model]}"
    env -u PYTHONNOUSERSITE \
        HF_HOME="${staging_dir}/hf_home" \
        hf download "${REPOS[$model]}" \
            --include "${checkpoint}/actor/*" \
            --exclude \
                "${checkpoint}/actor/optim_world_size_*" \
                "${checkpoint}/actor/extra_state_world_size_*" \
            --local-dir "${download_dir}" \
            --max-workers 4

    printf 'Merging %s into Hugging Face format\n' "${model}"
    (
        cd "${REPO_ROOT}"
        exec conda run --no-capture-output -n "${CONDA_ENV}" \
            env PYTHONNOUSERSITE=1 \
            python scripts/model_merger.py merge \
                --backend fsdp \
                --local_dir "${actor_dir}" \
                --target_dir "${target_dir}"
    )

    if [[ ! -s "${target_dir}/model.safetensors" || ! -s "${target_dir}/config.json" ]]; then
        printf 'error: merged model validation failed: %s\n' "${target_dir}" >&2
        exit 1
    fi

    # This staging path is created exclusively by this script. Remove it only
    # after the merged model has passed the checks above.
    expected_staging_dir="${EVAL_ROOT}/staging/${model}"
    resolved_staging_dir=$(realpath -- "${staging_dir}")
    if [[ "${resolved_staging_dir}" != "${expected_staging_dir}" ]]; then
        printf 'error: refusing to clean unexpected staging path: %s\n' \
            "${resolved_staging_dir}" >&2
        exit 1
    fi
    find "${resolved_staging_dir}" -depth -delete
    printf 'Prepared %s\n' "${target_dir}"
    df -h "${EVAL_ROOT}"
done
