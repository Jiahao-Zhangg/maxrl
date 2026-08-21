#!/usr/bin/env bash

set -euo pipefail

# Keep the Conda environment isolated from ~/.local Python packages.
export PYTHONNOUSERSITE=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MACHINE_ARCH=$(uname -m)

MAXRL_ENV_NAME=${MAXRL_ENV_NAME:-maxrl}
MAXRL_DATA_DIR=${MAXRL_DATA_DIR:-${REPO_ROOT}/data}
MAXRL_OUTPUT_DIR=${MAXRL_OUTPUT_DIR:-${REPO_ROOT}/outputs}
MAXRL_RAY_DIR=${MAXRL_RAY_DIR:-${MAXRL_OUTPUT_DIR}/ray}
MAXRL_SKIP_ENV_SETUP=${MAXRL_SKIP_ENV_SETUP:-0}
MAXRL_ENV_ONLY=${MAXRL_ENV_ONLY:-0}
MAXRL_REFRESH_DATA=${MAXRL_REFRESH_DATA:-0}
MAXRL_INSTALL_JOBS=${MAXRL_INSTALL_JOBS:-}
MAXRL_NVCC_THREADS=${MAXRL_NVCC_THREADS:-}
MAXRL_CMAKE_BUILD_TYPE=${MAXRL_CMAKE_BUILD_TYPE:-Release}
MAXRL_HOST_CC=${MAXRL_HOST_CC:-gcc}
MAXRL_HOST_CXX=${MAXRL_HOST_CXX:-g++}
MAXRL_CUDA_COMPILER_LAUNCHER=${MAXRL_CUDA_COMPILER_LAUNCHER:-}
MAXRL_REQUIREMENTS_FILE=${MAXRL_REQUIREMENTS_FILE:-}
MAXRL_PYTORCH_REQUIREMENTS_FILE=${MAXRL_PYTORCH_REQUIREMENTS_FILE:-}
MAXRL_BOOTSTRAP_REQUIREMENTS_FILE=${MAXRL_BOOTSTRAP_REQUIREMENTS_FILE:-}
MAXRL_SOURCE_REQUIREMENTS_FILE=${MAXRL_SOURCE_REQUIREMENTS_FILE:-}
MAXRL_EXPECTED_VLLM_VERSION=${MAXRL_EXPECTED_VLLM_VERSION:-0.8.4}
MAXRL_EXPECTED_VLLM_LOCAL_VERSION=${MAXRL_EXPECTED_VLLM_LOCAL_VERSION:-}

case "${MACHINE_ARCH}" in
    aarch64|arm64)
        MAXRL_INSTALL_JOBS=${MAXRL_INSTALL_JOBS:-1}
        MAXRL_NVCC_THREADS=${MAXRL_NVCC_THREADS:-1}
        MAXRL_EXPECTED_TORCH_VERSION=${MAXRL_EXPECTED_TORCH_VERSION:-2.6.0+cu126}
        MAXRL_EXPECTED_TORCH_CUDA=${MAXRL_EXPECTED_TORCH_CUDA:-12.6}
        MAXRL_EXPECTED_VLLM_LOCAL_VERSION=${MAXRL_EXPECTED_VLLM_LOCAL_VERSION:-cu126}
        MAXRL_CUDA_ARCH_LIST=${MAXRL_CUDA_ARCH_LIST:-9.0}
        MAXRL_FLASH_ATTN_CUDA_ARCHS=${MAXRL_FLASH_ATTN_CUDA_ARCHS:-90}
        MAXRL_CUDA_HOME=${MAXRL_CUDA_HOME:-/sw/user/cudatoolkits/installs/cuda-12.6.1}
        MAXRL_CUDA_COMPILER_LAUNCHER=${MAXRL_CUDA_COMPILER_LAUNCHER:-${SCRIPT_DIR}/cuda126_aarch64_compiler_launcher.sh}
        if [[ -z "${MAXRL_REQUIREMENTS_FILE}" ]]; then
            MAXRL_REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_qwen3_maxrl_cu126_aarch64.txt"
            MAXRL_PYTORCH_REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_qwen3_maxrl_cu126_aarch64_pytorch.txt"
            MAXRL_BOOTSTRAP_REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_qwen3_maxrl_cu126_aarch64_bootstrap.txt"
            MAXRL_SOURCE_REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_qwen3_maxrl_cu126_aarch64_sources.txt"
        fi
        ;;
    x86_64)
        MAXRL_INSTALL_JOBS=${MAXRL_INSTALL_JOBS:-4}
        MAXRL_NVCC_THREADS=${MAXRL_NVCC_THREADS:-2}
        MAXRL_EXPECTED_TORCH_VERSION=${MAXRL_EXPECTED_TORCH_VERSION:-2.6.0+cu124}
        MAXRL_EXPECTED_TORCH_CUDA=${MAXRL_EXPECTED_TORCH_CUDA:-12.4}
        MAXRL_CUDA_ARCH_LIST=${MAXRL_CUDA_ARCH_LIST:-}
        MAXRL_FLASH_ATTN_CUDA_ARCHS=${MAXRL_FLASH_ATTN_CUDA_ARCHS:-}
        MAXRL_CUDA_HOME=${MAXRL_CUDA_HOME:-${CUDA_HOME:-}}
        ;;
    *)
        MAXRL_INSTALL_JOBS=${MAXRL_INSTALL_JOBS:-4}
        MAXRL_NVCC_THREADS=${MAXRL_NVCC_THREADS:-2}
        MAXRL_EXPECTED_TORCH_VERSION=${MAXRL_EXPECTED_TORCH_VERSION:-2.6.0}
        MAXRL_EXPECTED_TORCH_CUDA=${MAXRL_EXPECTED_TORCH_CUDA:-}
        MAXRL_CUDA_ARCH_LIST=${MAXRL_CUDA_ARCH_LIST:-}
        MAXRL_FLASH_ATTN_CUDA_ARCHS=${MAXRL_FLASH_ATTN_CUDA_ARCHS:-}
        MAXRL_CUDA_HOME=${MAXRL_CUDA_HOME:-${CUDA_HOME:-}}
        ;;
esac

export MAXRL_EXPECTED_TORCH_VERSION MAXRL_EXPECTED_TORCH_CUDA
export MAXRL_EXPECTED_VLLM_VERSION MAXRL_EXPECTED_VLLM_LOCAL_VERSION

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

die() {
    echo "error: $*" >&2
    exit 1
}

without_other_cuda_toolkits() {
    local value=$1
    local entry
    local filtered=
    local -a entries=()
    IFS=: read -r -a entries <<< "${value}"
    for entry in "${entries[@]}"; do
        [[ -n "${entry}" ]] || continue
        if [[ "${entry}" != "${MAXRL_CUDA_HOME}"* ]] && \
            [[ "${entry}" == */cuda/* || \
               "${entry}" == */cuda-*/* || \
               "${entry}" == */math_libs/* ]]; then
            continue
        fi
        filtered="${filtered}${filtered:+:}${entry}"
    done
    printf '%s' "${filtered}"
}

configure_cuda_source_build() {
    [[ -x "${MAXRL_CUDA_HOME}/bin/nvcc" ]] || \
        die "CUDA compiler not found: ${MAXRL_CUDA_HOME}/bin/nvcc"
    local nvcc_version
    nvcc_version=$("${MAXRL_CUDA_HOME}/bin/nvcc" --version)
    [[ "${nvcc_version}" == *"release ${MAXRL_EXPECTED_TORCH_CUDA}"* ]] || \
        die "CUDA ${MAXRL_EXPECTED_TORCH_CUDA} compiler required under ${MAXRL_CUDA_HOME}"

    export CUDA_HOME="${MAXRL_CUDA_HOME}"
    export CUDA_PATH="${MAXRL_CUDA_HOME}"
    export CUDA_BIN_PATH="${MAXRL_CUDA_HOME}"
    export CUDACXX="${MAXRL_CUDA_HOME}/bin/nvcc"
    export CUDAToolkit_ROOT="${MAXRL_CUDA_HOME}"
    export CUDA_TOOLKIT_ROOT_DIR="${MAXRL_CUDA_HOME}"
    export PATH="${MAXRL_CUDA_HOME}/bin:${PATH}"
    local filtered_cpath
    local filtered_cmake_prefix_path
    local filtered_library_path
    local filtered_ld_library_path
    filtered_cpath=$(without_other_cuda_toolkits "${CPATH:-}")
    filtered_cmake_prefix_path=$(without_other_cuda_toolkits "${CMAKE_PREFIX_PATH:-}")
    filtered_library_path=$(without_other_cuda_toolkits "${LIBRARY_PATH:-}")
    filtered_ld_library_path=$(without_other_cuda_toolkits "${LD_LIBRARY_PATH:-}")
    export CPATH="${MAXRL_CUDA_HOME}/include${filtered_cpath:+:${filtered_cpath}}"
    export CMAKE_PREFIX_PATH="${MAXRL_CUDA_HOME}${filtered_cmake_prefix_path:+:${filtered_cmake_prefix_path}}"
    export LIBRARY_PATH="${MAXRL_CUDA_HOME}/lib64${filtered_library_path:+:${filtered_library_path}}"
    export LD_LIBRARY_PATH="${MAXRL_CUDA_HOME}/lib64${filtered_ld_library_path:+:${filtered_ld_library_path}}"
    export CC="${MAXRL_HOST_CC}"
    export CXX="${MAXRL_HOST_CXX}"
    if [[ -n "${MAXRL_CUDA_COMPILER_LAUNCHER}" ]]; then
        [[ -x "${MAXRL_CUDA_COMPILER_LAUNCHER}" ]] || \
            die "CUDA compiler launcher is not executable: ${MAXRL_CUDA_COMPILER_LAUNCHER}"
        export CMAKE_CUDA_COMPILER_LAUNCHER="${MAXRL_CUDA_COMPILER_LAUNCHER}"
    fi
}

has_distribution_version() {
    python - "$1" "$2" "$3" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

try:
    actual = Version(version(sys.argv[1]))
except PackageNotFoundError:
    raise SystemExit(1)
expected_local = sys.argv[3]
if expected_local:
    matches = (
        actual.base_version == sys.argv[2]
        and actual.local == expected_local
    )
else:
    matches = str(actual) == sys.argv[2]
raise SystemExit(not matches)
PY
}

source_requirement_is_satisfied() {
    case "$1" in
        triton\ @*)
            has_distribution_version triton 3.2.0 ""
            ;;
        vllm\ @*)
            has_distribution_version \
                vllm "${MAXRL_EXPECTED_VLLM_VERSION}" \
                "${MAXRL_EXPECTED_VLLM_LOCAL_VERSION}"
            ;;
        tensordict\ @*)
            has_distribution_version tensordict 0.6.2 ""
            ;;
        flash-attn==*)
            has_distribution_version flash-attn 2.7.4.post1 ""
            ;;
        flashinfer-python==*)
            has_distribution_version flashinfer-python 0.2.2.post1 ""
            ;;
        *)
            return 1
            ;;
    esac
}

activate_environment() {
    command -v conda >/dev/null 2>&1 || die "conda is required; load Conda and rerun this script"
    eval "$(conda shell.bash hook)"

    if ! conda run -n "${MAXRL_ENV_NAME}" python --version >/dev/null 2>&1; then
        conda create -y -n "${MAXRL_ENV_NAME}" python=3.10
    fi
    conda activate "${MAXRL_ENV_NAME}"
    python -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))' || \
        die "${MAXRL_ENV_NAME} exists but does not use Python 3.10; choose another MAXRL_ENV_NAME"
}

install_environment() {
    if [[ -n "${MAXRL_REQUIREMENTS_FILE}" ]]; then
        [[ -f "${MAXRL_REQUIREMENTS_FILE}" ]] || \
            die "requirements file not found: ${MAXRL_REQUIREMENTS_FILE}"
        if [[ -n "${MAXRL_BOOTSTRAP_REQUIREMENTS_FILE}" ]]; then
            [[ -f "${MAXRL_PYTORCH_REQUIREMENTS_FILE}" ]] || \
                die "PyTorch requirements file not found: ${MAXRL_PYTORCH_REQUIREMENTS_FILE}"
            [[ -f "${MAXRL_BOOTSTRAP_REQUIREMENTS_FILE}" ]] || \
                die "bootstrap requirements file not found: ${MAXRL_BOOTSTRAP_REQUIREMENTS_FILE}"
            [[ -f "${MAXRL_SOURCE_REQUIREMENTS_FILE}" ]] || \
                die "source requirements file not found: ${MAXRL_SOURCE_REQUIREMENTS_FILE}"
            configure_cuda_source_build
            python -m pip install --no-cache-dir \
                -r "${MAXRL_PYTORCH_REQUIREMENTS_FILE}"
            python -m pip install --no-cache-dir \
                -r "${MAXRL_BOOTSTRAP_REQUIREMENTS_FILE}"
            python -m pip install --no-cache-dir \
                -r "${MAXRL_REQUIREMENTS_FILE}"

            # Install native extensions separately so a later build failure does
            # not discard wheels that were already compiled successfully.
            local requirement
            while IFS= read -r requirement || [[ -n "${requirement}" ]]; do
                [[ "${requirement}" =~ ^[[:space:]]*($|#) ]] && continue
                if source_requirement_is_satisfied "${requirement}"; then
                    echo "Requirement already satisfied: ${requirement}"
                    continue
                fi
                MAX_JOBS="${MAXRL_INSTALL_JOBS}" \
                    NVCC_THREADS="${MAXRL_NVCC_THREADS}" \
                    TORCH_CUDA_ARCH_LIST="${MAXRL_CUDA_ARCH_LIST}" \
                    FLASH_ATTN_CUDA_ARCHS="${MAXRL_FLASH_ATTN_CUDA_ARCHS}" \
                    FLASH_ATTENTION_FORCE_BUILD=TRUE \
                    CMAKE_BUILD_TYPE="${MAXRL_CMAKE_BUILD_TYPE}" \
                    CMAKE_BUILD_PARALLEL_LEVEL="${MAXRL_INSTALL_JOBS}" \
                    VLLM_TARGET_DEVICE=cuda \
                    python -m pip install --no-cache-dir --no-build-isolation \
                    "${requirement}"
            done < "${MAXRL_SOURCE_REQUIREMENTS_FILE}"
        else
            python -m pip install --no-cache-dir -r "${MAXRL_REQUIREMENTS_FILE}"
        fi
        python -m pip install --no-deps -e "${REPO_ROOT}"
        return
    fi

    python -m pip install \
        pip==25.0.1 setuptools==75.8.0 wheel==0.45.1
    python -m pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        numpy==1.26.4 fsspec==2024.12.0 \
        --index-url https://download.pytorch.org/whl/cu124
    python -m pip install --no-cache-dir \
        vllm==0.8.4 tensordict==0.6.2 torchdata==0.11.0 \
        transformers==4.51.3 huggingface-hub==0.30.2 \
        "ray[cgraph]==2.43.0" cupy-cuda12x==13.3.0 \
        opencv-python-headless==4.11.0.86 "fastapi[standard]==0.115.12" \
        pydantic==2.11.3 protobuf==4.25.6 \
        opentelemetry-api==1.26.0 opentelemetry-sdk==1.26.0 \
        opentelemetry-proto==1.26.0 \
        opentelemetry-semantic-conventions==0.47b0
    python -m pip install --no-cache-dir \
        "transformers[hf_xet]==4.51.3" accelerate==1.6.0 datasets==3.5.0 \
        peft==0.17.0 hf-transfer==0.1.9 pyarrow==19.0.1 pandas==2.2.3 \
        codetiming==1.4.0 hydra-core==1.3.2 pylatexenc==2.10 \
        wandb==0.19.10 dill==0.3.8 pybind11==2.13.6 math-verify==0.9.0 \
        ninja==1.11.1.3 packaging==24.2 psutil==6.1.1 \
        nvidia-ml-py==12.575.51 optree==0.15.0

    if [[ "$(uname -m)" == "x86_64" ]]; then
        python -m pip install --no-cache-dir \
            "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
        python -m pip install --no-cache-dir \
            "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl"
    else
        MAX_JOBS="${MAXRL_INSTALL_JOBS}" python -m pip install --no-cache-dir \
            flash-attn==2.7.4.post1 --no-build-isolation
    fi
    python -m pip install --no-deps -e "${REPO_ROOT}"
}

validate_environment() {
    python -m pip check
    python - <<'PY'
import os
from importlib.metadata import version

from packaging.version import Version

import flash_attn
import flashinfer
import pkg_resources
import ray
import torch
import torch._inductor.runtime.triton_heuristics
import transformers
import vllm

expected = {
    "ray": "2.43.0",
    "torch": os.environ["MAXRL_EXPECTED_TORCH_VERSION"],
    "transformers": "4.51.3",
    "triton": "3.2.0",
    "vllm": os.environ["MAXRL_EXPECTED_VLLM_VERSION"],
}
actual = {package: version(package) for package in expected}
exact_packages = {"ray", "torch", "transformers", "triton"}
if any(actual[package] != expected[package] for package in exact_packages):
    raise SystemExit(f"Unexpected core package versions: {actual}")
vllm_version = Version(actual["vllm"])
if vllm_version.base_version != expected["vllm"]:
    raise SystemExit(f"Unexpected core package versions: {actual}")
expected_vllm_local = os.environ["MAXRL_EXPECTED_VLLM_LOCAL_VERSION"]
if expected_vllm_local and vllm_version.local != expected_vllm_local:
    raise SystemExit(
        f"Unexpected vLLM CUDA build: {actual['vllm']}; "
        f"expected local version {expected_vllm_local}"
    )
expected_torch_cuda = os.environ["MAXRL_EXPECTED_TORCH_CUDA"]
if expected_torch_cuda and torch.version.cuda != expected_torch_cuda:
    raise SystemExit(
        f"Unexpected PyTorch CUDA runtime: {torch.version.cuda}; "
        f"expected {expected_torch_cuda}"
    )
print("Validated core packages:", actual)
print("Validated PyTorch CUDA runtime:", torch.version.cuda)
PY
}

prepare_dataset() {
    local converter=$1
    local destination=$2

    mkdir -p "${destination}"
    if [[ "${MAXRL_REFRESH_DATA}" == "1" || ! -s "${destination}/train.parquet" || ! -s "${destination}/test.parquet" ]]; then
        python "${converter}" --local_dir "${destination}"
    else
        echo "Reusing prepared dataset in ${destination}"
    fi
}

if [[ "${MAXRL_SKIP_ENV_SETUP}" == "1" ]]; then
    echo "Skipping environment creation; using $(command -v python)"
else
    activate_environment
    install_environment
    validate_environment
fi

if [[ "${MAXRL_ENV_ONLY}" == "1" ]]; then
    echo "Environment ${MAXRL_ENV_NAME} is ready; skipping data preparation and training"
    exit 0
fi

cd "${REPO_ROOT}"

MATH12K_DIR=${MAXRL_DATA_DIR}/math12k
AIME25_DIR=${MAXRL_DATA_DIR}/aime25
MATH500_DIR=${MAXRL_DATA_DIR}/math500

prepare_dataset \
    "${REPO_ROOT}/examples/maxrl_data_preprocess/math12k.py" \
    "${MATH12K_DIR}"
prepare_dataset \
    "${REPO_ROOT}/examples/maxrl_data_preprocess/aime25.py" \
    "${AIME25_DIR}"
prepare_dataset \
    "${REPO_ROOT}/examples/maxrl_data_preprocess/math_500.py" \
    "${MATH500_DIR}"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the active Python environment")
if torch.cuda.device_count() != 4:
    raise SystemExit(
        f"Expected four visible GPUs from CUDA_VISIBLE_DEVICES, found {torch.cuda.device_count()}"
    )
print("Visible GPUs:", [torch.cuda.get_device_name(i) for i in range(4)])
PY

mkdir -p "${MAXRL_OUTPUT_DIR}" "${MAXRL_RAY_DIR}"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export SEED=79
export NCCL_DEBUG=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export NCCL_ALGO=RING
export NCCL_IGNORE_CPU_AFFINITY=1

FULL_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=256
NUM_PER_PROMPT_ROLLOUTS=16
MAX_RESPONSE_LENGTH=4096
MAX_PROMPT_LENGTH=1024
LEARNING_RATE=1e-6
REWARD_MANAGER=multi_thread
PER_GPU_MINI_BATCH_SIZE=4
NUM_PER_PROMPT_ROLLOUTS_VALIDATION=32
MAX_MODEL_LEN=32000
MAX_NUM_BATCHED_TOKENS=32000
TENSOR_MODEL_PARALLEL_SIZE=1
PPO_EPOCHS=1
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.2
GRAD_CLIP=0.3
KL_COEFF=0.0
TOTAL_EPOCHS=5
LOSS_AGG_MODE=${MAXRL_LOSS_AGG_MODE:-token-mean}

MODEL_PATH=Qwen/Qwen3-1.7B-Base
MODEL_NAME=Qwen3-1.7B-Base
ADVANTAGE_ESTIMATOR=${MAXRL_ADVANTAGE_ESTIMATOR:-maxrl}
PROJECT_NAME=${MAXRL_PROJECT_NAME:-Qwen3_MaxRL_Experiments}
EXPERIMENT_NAME=${MAXRL_EXPERIMENT_NAME:-${ADVANTAGE_ESTIMATOR}_${MODEL_NAME}_math12k}
CHECKPOINT_SAVE_PATH=${MAXRL_OUTPUT_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
TEST_DATASET_PATH="['${AIME25_DIR}/test.parquet','${MATH500_DIR}/test.parquet']"

ALGORITHM_OVERRIDES=()
if [[ "${ADVANTAGE_ESTIMATOR}" == "cost_aware_maxrl" ]]; then
    COST_REFERENCE_TOKENS=${MAXRL_COST_REFERENCE_TOKENS:-$((MAX_RESPONSE_LENGTH / 2))}
    MAX_INVERSE_COST=${MAXRL_MAX_INVERSE_COST:-4.0}
    ALGORITHM_OVERRIDES+=(
        "algorithm.cost_reference_tokens=${COST_REFERENCE_TOKENS}"
        "algorithm.max_inverse_cost=${MAX_INVERSE_COST}"
    )
    echo "Cost-aware MaxRL: reference_tokens=${COST_REFERENCE_TOKENS}, inverse_cost_cap=${MAX_INVERSE_COST}"
elif [[ "${ADVANTAGE_ESTIMATOR}" == "rb_cost_aware_maxrl" ]]; then
    RB_COST_MAX_TOKENS=${MAXRL_RB_COST_MAX_TOKENS:-${MAX_RESPONSE_LENGTH}}
    ALGORITHM_OVERRIDES+=("algorithm.rb_cost_max_tokens=${RB_COST_MAX_TOKENS}")
    echo "RB cost-aware MaxRL: cost_max_tokens=${RB_COST_MAX_TOKENS}, zero-success update=zero"
elif [[
    "${ADVANTAGE_ESTIMATOR}" == "fixed_n_rb_capped_cost_aware_marginrl"
    || "${ADVANTAGE_ESTIMATOR}" == "fixed_n_rb_capped_fixed_q_cost_aware_marginrl"
]]; then
    if [[ "${LOSS_AGG_MODE}" != "token-mean" && "${LOSS_AGG_MODE}" != "seq-mean-token-sum" ]]; then
        echo "error: ${ADVANTAGE_ESTIMATOR} supports token-mean or seq-mean-token-sum loss aggregation" >&2
        exit 1
    fi
    COST_REFERENCE_TOKENS=${MAXRL_COST_REFERENCE_TOKENS:-$((MAX_RESPONSE_LENGTH / 2))}
    MAX_INVERSE_COST=${MAXRL_MAX_INVERSE_COST:-4.0}
    ALGORITHM_OVERRIDES+=(
        "algorithm.cost_reference_tokens=${COST_REFERENCE_TOKENS}"
        "algorithm.max_inverse_cost=${MAX_INVERSE_COST}"
    )
    if [[ "${ADVANTAGE_ESTIMATOR}" == "fixed_n_rb_capped_fixed_q_cost_aware_marginrl" ]]; then
        FIXED_Q_HAT=${MAXRL_FIXED_Q_HAT:-2.0}
        ALGORITHM_OVERRIDES+=("algorithm.fixed_q_hat=${FIXED_Q_HAT}")
        echo "Fixed-q capped-cost fixed-N RB MarginRL: N=${NUM_PER_PROMPT_ROLLOUTS}, cost=max(tokens/${COST_REFERENCE_TOKENS},1/${MAX_INVERSE_COST}), q_hat=${FIXED_Q_HAT}"
    else
        echo "Capped-cost fixed-N RB MarginRL: N=${NUM_PER_PROMPT_ROLLOUTS}, cost=max(tokens/${COST_REFERENCE_TOKENS},1/${MAX_INVERSE_COST}), q_hat=M/sum(cost)"
    fi
elif [[ "${ADVANTAGE_ESTIMATOR}" == "fixed_n_rb_efficient_reasoning_cost_marginrl" ]]; then
    if [[ "${LOSS_AGG_MODE}" != "token-mean" && "${LOSS_AGG_MODE}" != "seq-mean-token-sum" ]]; then
        echo "error: ${ADVANTAGE_ESTIMATOR} supports token-mean or seq-mean-token-sum loss aggregation" >&2
        exit 1
    fi
    EFFICIENT_REASONING_EPSILON=${MAXRL_EFFICIENT_REASONING_EPSILON:-1e-7}
    ALGORITHM_OVERRIDES+=(
        "algorithm.efficient_reasoning_epsilon=${EFFICIENT_REASONING_EPSILON}"
    )
    echo "Efficient-Reasoning sigmoid-cost fixed-N RB MarginRL: N=${NUM_PER_PROMPT_ROLLOUTS}, normalization=all responses per prompt, q_hat=M/sum(cost), failure advantage=-q_hat*cost/(M+1)"
elif [[
    "${ADVANTAGE_ESTIMATOR}" == "fixed_n_rb_cost_aware_marginrl"
    || "${ADVANTAGE_ESTIMATOR}" == "fixed_n_rb_cost_aware_marginrl_success_gated"
]]; then
    if [[ "${LOSS_AGG_MODE}" != "token-mean" && "${LOSS_AGG_MODE}" != "seq-mean-token-sum" ]]; then
        echo "error: ${ADVANTAGE_ESTIMATOR} supports token-mean or seq-mean-token-sum loss aggregation" >&2
        exit 1
    fi
    if [[ "${ADVANTAGE_ESTIMATOR}" == "fixed_n_rb_cost_aware_marginrl_success_gated" ]]; then
        echo "Success-gated fixed-N RB cost-aware MarginRL: N=${NUM_PER_PROMPT_ROLLOUTS}, q_hat=M/sum(tokens), wrong-sample advantage=zero"
    else
        echo "Fixed-N RB cost-aware MarginRL: N=${NUM_PER_PROMPT_ROLLOUTS}, q_hat=M/sum(tokens), zero-success update=zero"
    fi
fi

echo "Training ${MODEL_NAME} with hiyouga/math12k for ${TOTAL_EPOCHS} epochs"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Advantage estimator: ${ADVANTAGE_ESTIMATOR}; loss aggregation: ${LOSS_AGG_MODE}"
echo "Checkpoints: ${CHECKPOINT_SAVE_PATH}"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "Note: the original logger configuration uses Weights & Biases; ensure 'wandb login' has been run."
fi

python -W ignore -m verl.trainer.main_ppo \
    algorithm.adv_estimator="${ADVANTAGE_ESTIMATOR}" \
    data.train_files="${MATH12K_DIR}/train.parquet" \
    data.val_files="${TEST_DATASET_PATH}" \
    data.train_batch_size="${FULL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PER_GPU_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEFF}" \
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    actor_rollout_ref.actor.grad_clip="${GRAD_CLIP}" \
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PER_GPU_MINI_BATCH_SIZE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n="${NUM_PER_PROMPT_ROLLOUTS}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PER_GPU_MINI_BATCH_SIZE}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n="${NUM_PER_PROMPT_ROLLOUTS_VALIDATION}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_ctrl.kl_coef="${KL_COEFF}" \
    "${ALGORITHM_OVERRIDES[@]}" \
    reward_model.reward_manager="${REWARD_MANAGER}" \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    trainer.val_on_last_step=True \
    "trainer.logger=['console','wandb']" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CHECKPOINT_SAVE_PATH}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=400 \
    trainer.max_critic_ckpt_to_keep=400 \
    trainer.test_freq=50 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    ray_init.ray_dir="${MAXRL_RAY_DIR}" \
    "$@"
