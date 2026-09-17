#!/usr/bin/env bash
# Reuse an existing allocation; never submit or cancel a holder.
set -euo pipefail
umask 077

[[ $# -ge 2 && $# -le 3 ]] || { echo "Usage: $0 JOB_ID OUTPUT_ROOT [NODE_LOCAL_SCRATCH]" >&2; exit 2; }
EVAL_JOB_ID=$1
EVAL_OUTPUT_ROOT=$2
EVAL_SCRATCH=${3:-}
[[ ${EVAL_JOB_ID} =~ ^[0-9]+$ ]] || exit 2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

if [[ ${SLURM_JOB_ID:-} != "${EVAL_JOB_ID}" ]]; then
    job_description=$(scontrol show job "${EVAL_JOB_ID}" -o)
    [[ ${job_description} == *"JobState=RUNNING"* && ${job_description} == *"UserId=${USER}("* ]] || {
        echo "Requested holder is not running and owned by the current user." >&2; exit 1;
    }
    exec srun --jobid="${EVAL_JOB_ID}" --overlap --nodes=1 --ntasks=1 \
        --cpus-per-task="${MAXRL_EVAL_CPUS:-96}" --gres=gpu:4 --network=no_vni \
        bash "$0" "${EVAL_JOB_ID}" "${EVAL_OUTPUT_ROOT}" "${EVAL_SCRATCH}"
fi

mkdir -p "${REPO_ROOT}/outputs/logs" "${EVAL_OUTPUT_ROOT}"
exec 9>"${REPO_ROOT}/outputs/logs/gpu_holder_${EVAL_JOB_ID}.launch.lock"
flock -n 9 || { echo "Another launcher owns this holder." >&2; exit 1; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z ${busy} ]] || { echo "GPUs are occupied; refusing to overlap." >&2; exit 1; }

if [[ -z ${EVAL_SCRATCH} ]]; then
    EVAL_SCRATCH=$(mktemp -d /tmp/maxrl-eval-matrix.XXXXXX)
fi
mkdir -p "${EVAL_SCRATCH}"
source "${MAXRL_EVAL_CONDA_SH:-/sw/user/python/miniforge3-pytorch-2.10.0/etc/profile.d/conda.sh}"
conda activate "${MAXRL_EVAL_CONDA_ENV:-maxrl}"
if [[ $(uname -m) == aarch64 ]]; then
    if type module >/dev/null 2>&1; then
        if module is-loaded gcc-native/14 >/dev/null 2>&1; then
            module swap gcc-native/14 gcc-native/13
        else
            module load gcc-native/13
        fi
    fi
    export CUDA_HOME=${MAXRL_EVAL_CUDA_HOME:-/sw/user/cudatoolkits/installs/cuda-12.6.1}
    export CUDA_PATH=${CUDA_HOME} TORCH_CUDA_ARCH_LIST=9.0
    export PATH=${CUDA_HOME}/bin:${PATH}
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export CC=$(command -v gcc) CXX=$(command -v g++)
    export CUDAHOSTCXX=${CXX} MAX_JOBS=4
fi
unset PYTHONPATH PYTHONHOME RAY_ADDRESS
export PYTHONPATH=${REPO_ROOT}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_XET_CACHE=${EVAL_SCRATCH}/xet HF_HUB_DISABLE_TELEMETRY=1
EVAL_CONFIG=${MAXRL_EVAL_CONFIG:-${SCRIPT_DIR}/math_eval_matrix.json}
EVAL_PLOT_PYTHON=${MAXRL_EVAL_PLOT_PYTHON:-/sw/user/python/miniforge3-pytorch-2.10.0/bin/python}
cd "${REPO_ROOT}"
echo "$(date --iso-8601=seconds) Preparing matrix on holder ${EVAL_JOB_ID}, scratch=${EVAL_SCRATCH}"
python -u "${SCRIPT_DIR}/prepare_math_eval_matrix.py" \
    --config "${EVAL_CONFIG}" --output-root "${EVAL_OUTPUT_ROOT}" --scratch "${EVAL_SCRATCH}"
echo "$(date --iso-8601=seconds) Starting four independent GPU evaluation workers."
EVAL_RETRY_ARGS=()
if [[ ${MAXRL_EVAL_RETRY_FAILED:-0} == 1 ]]; then
    EVAL_RETRY_ARGS+=(--retry-failed)
fi
if [[ -n ${MAXRL_EVAL_REUSE_EVAL1_FROM:-} ]]; then
    EVAL_RETRY_ARGS+=(--reuse-eval1-from "${MAXRL_EVAL_REUSE_EVAL1_FROM}")
fi
exec python -u "${SCRIPT_DIR}/run_math_eval_matrix.py" \
    --config "${EVAL_CONFIG}" --output-root "${EVAL_OUTPUT_ROOT}" \
    --scratch "${EVAL_SCRATCH}" --gpus 0 1 2 3 --plot-python "${EVAL_PLOT_PYTHON}" \
    "${EVAL_RETRY_ARGS[@]}"
