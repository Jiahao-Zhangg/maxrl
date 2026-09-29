#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${ER_RUNTIME_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
cd "${PROJECT_ROOT}"
CONDA_BASE=${CONDA_BASE:-/project/flame/jiahaoz4/miniconda3}
CONDA_ENV=${CONDA_ENV:-efficient_reasoning_official_hybrid}
RUN_NAME=${RUN_NAME:-rloo_qwen3_1.7b_compression_official_hybrid8_r32_n8_b128_32k_seed79}
RUN_DIR=${ER_TRAIN_DIR:-${PROJECT_ROOT}/outputs/${RUN_NAME}}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
MAX_SAMPLES=${MAX_SAMPLES:-3200}
GENERATE_MAX_LEN=${GENERATE_MAX_LEN:-32768}
PROMPT_MAX_LEN=${PROMPT_MAX_LEN:-1536}
SAVE_STEPS=${SAVE_STEPS:-20}
USE_WANDB=${USE_WANDB:-1}
ARCHIVE_CHECKPOINTS=${ARCHIVE_CHECKPOINTS:-1}
HF_REPO_ID=${HF_REPO_ID:-hi-todayis-jh/rloo-qwen3-1.7b-compression-official-hybrid8-bs32-n8-b128-32k-146103}
RM_PORT=${RM_PORT:-24378}
RESUME=${RESUME:-0}

TRAIN_CMD=(python -m openrlhf.cli.train_ppo_ray
    --pretrain "${ER_MODEL_PATH:-Qwen/Qwen3-1.7B}" --advantage_estimator rloo
    --actor_num_nodes 1 --actor_num_gpus_per_node 8
    --ref_num_nodes 1 --ref_num_gpus_per_node 8
    --vllm_num_engines 8 --vllm_tensor_parallel_size 1
    --colocate_all_models --vllm_enable_sleep --deepspeed_enable_sleep
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
    --vllm_sync_backend nccl --enforce_eager
    --remote_rm_url "http://127.0.0.1:${RM_PORT}/official_query"
    --prompt_data "${PROJECT_ROOT}/datasets/compression_thinking.jsonl"
    --prompt_data_probs 1.0 --input_key prompt --label_key label
    --max_samples "${MAX_SAMPLES}" --num_episodes 1 --max_epochs 1
    --rollout_batch_size "${ROLLOUT_BATCH_SIZE}" --n_samples_per_prompt "${N_SAMPLES_PER_PROMPT}"
    --train_batch_size "${TRAIN_BATCH_SIZE}" --micro_train_batch_size 1 --micro_rollout_batch_size 1
    --prompt_max_len "${PROMPT_MAX_LEN}" --generate_max_len "${GENERATE_MAX_LEN}"
    --temperature 1.0 --top_p 1.0 --actor_learning_rate 1e-6 --lr_warmup_ratio 0
    --adam_betas 0.9 0.999 --l2 0.01 --max_norm 0.3 --init_kl_coef 0.0
    --zero_stage 3 --bf16 --flash_attn --gradient_checkpointing --seed 79
    --eval_steps 1000000000 --save_steps "${SAVE_STEPS}"
    --max_ckpt_num 10000 --max_ckpt_mem 100000000
    --save_path "${RUN_DIR}/final_model" --ckpt_path "${RUN_DIR}/checkpoints"
    --wandb_project qwen3_compression_hybrid8_comparison --wandb_run_name "${RUN_NAME}")
if [[ "${USE_WANDB}" == 1 ]]; then
    TRAIN_CMD+=(--use_wandb enabled --wandb_org "${WANDB_ORG:-jiahaozhangg-carnegie-mellon-university}")
else
    TRAIN_CMD+=(--use_tensorboard "${RUN_DIR}/tensorboard")
fi
if [[ "${RESUME}" == 1 ]]; then TRAIN_CMD+=(--load_checkpoint); fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${TRAIN_CMD[@]}"
    printf '\n'
    exit 0
fi
[[ -n ${ER_PLAN:-} && -n ${ER_SHARED_ROOT:-} && -n ${ER_RUNTIME_ROOT:-} && -n ${ER_MODEL_PATH:-} ]] || {
    echo "Use launch_er_compression_compute.sh to prepare and queue this run." >&2; exit 2;
}
[[ ${RESUME} == 0 ]] || { echo "A resumed run needs an explicit rollout recovery plan." >&2; exit 2; }

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
unset TRANSFORMERS_CACHE ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/upstream:${PROJECT_ROOT}/reference:${PROJECT_ROOT}/reference/utils/latex2sympy"
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HF_HOME=${HF_HOME:-/project/flame/jiahaoz4/.cache/huggingface}
export RAY_ADDRESS=local RAY_TMPDIR=${RAY_TMPDIR:-/tmp/er_off_$$}
export TMPDIR=${RUN_DIR}/tmp OMP_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR=/project/flame/jiahaoz4/.cache/efficient_reasoning_official_hybrid/torch_extensions
export TRITON_CACHE_DIR=${RAY_TMPDIR}/triton MAX_JOBS=4
export WANDB_DIR=${RUN_DIR} WANDB_MODE=${WANDB_MODE:-online}
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=2147483648
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/checkpoints" "${RUN_DIR}/provenance" "${TMPDIR}" "${RAY_TMPDIR}"
if [[ "${RESUME}" != 1 && -f "${RUN_DIR}/trainer.pid" ]]; then
    echo "Run already started; choose a new RUN_NAME or set RESUME=1" >&2
    exit 1
fi
for gpu_id in 0 1 2 3 4 5 6 7; do
    if [[ -n $(nvidia-smi --id="${gpu_id}" --query-compute-apps=pid --format=csv,noheader) ]]; then
        echo "GPU ${gpu_id} is busy" >&2
        exit 1
    fi
done
rm -f "${RUN_DIR}/training_exit_status" "${RUN_DIR}/exit_status"
python - "${PROJECT_ROOT}" <<'PY'
import sys
from pathlib import Path
import openrlhf, torch, vllm, deepspeed
root = Path(sys.argv[1]).resolve()
assert Path(openrlhf.__file__).resolve().is_relative_to(root / "upstream")
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
print("Official OpenRLHF source:", openrlhf.__file__, flush=True)
print("Versions:", torch.__version__, vllm.__version__, deepspeed.__version__, flush=True)
PY
if [[ "${USE_WANDB}" == 1 ]]; then
    python - <<'PY'
import wandb
api = wandb.Api(timeout=30)
if not api.api_key:
    raise SystemExit("Run wandb login in the new environment before launching")
print("W&B account:", api.default_entity)
PY
fi
if [[ "${ARCHIVE_CHECKPOINTS}" == 1 ]]; then
    python - "${HF_REPO_ID}" <<'PY'
import sys
from huggingface_hub import HfApi
api=HfApi()
account=api.whoami()
assert account['name'] == sys.argv[1].split('/')[0]
print("Private archive prefix:", sys.argv[1], flush=True)
PY
fi
[[ -f "${PROJECT_ROOT}/datasets/compression_thinking.jsonl" ]] || { echo "Missing frozen compression inputs" >&2; exit 1; }
cp environment/source.json environment/version_diff.json "${RUN_DIR}/provenance/"
cp environment/upstream.patch "${RUN_DIR}/provenance/upstream.patch"
python - "${RUN_DIR}" "${HF_REPO_ID}" "${MAX_SAMPLES}" "${TRAIN_CMD[@]}" <<'PY'
import json, sys
from pathlib import Path
from qwen3_experiments.er_compression_eos import EOS_POLICY, EOS_REFERENCE_REPO, EOS_REFERENCE_REVISION
p=Path(sys.argv[1])
command=sys.argv[4:]
value=lambda name: command[command.index(name)+1]
rollout=int(value('--rollout_batch_size')); samples=int(value('--n_samples_per_prompt'))
train_batch=int(value('--train_batch_size')); count=int(sys.argv[3])
assert count % rollout == 0 and rollout*samples % train_batch == 0
config={'command':command,'hf_repo_prefix':sys.argv[2],'hf_archive_layout':'one_public_repo_per_checkpoint',
        'enable_thinking':True,'constant_learning_rate':1e-6,
        'expected_rollouts':count//rollout,'updates_per_rollout':rollout*samples//train_batch,
        'source':json.loads((p/'provenance/source.json').read_text()),'validation':False,
        'reward_scope':'full_text_raw_length_pool_and_force_eos_training','check_eos':True,'er_alpha':0.1,
        'eos_policy':EOS_POLICY,'eos_reference_repo':EOS_REFERENCE_REPO,'eos_reference_revision':EOS_REFERENCE_REVISION,
        'length_pool_requires_natural_response_eos':True,'force_eos':True,
        'all_training_rollouts_saved':True,'expected_responses':count*samples}
(p/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
print('Configuration:', config, flush=True)
PY

REWARD_PID=""; TRAIN_PID=""; ARCHIVE_PID=""
cleanup() {
    status=$?
    trap - EXIT INT TERM
    for pid in "${TRAIN_PID}" "${REWARD_PID}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -TERM -- "-${pid}" 2>/dev/null || true
        fi
    done
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid python -u -m integration.reward_bridge --port "${RM_PORT}" \
    --samples-per-prompt "${N_SAMPLES_PER_PROMPT}" --metrics-file "${RUN_DIR}/logs/reward_metrics.jsonl" \
    >"${RUN_DIR}/logs/reward.log" 2>&1 &
REWARD_PID=$!
echo "${REWARD_PID}" >"${RUN_DIR}/reward.pid"
reward_ready=0
for _ in $(seq 1 180); do
    kill -0 "${REWARD_PID}" 2>/dev/null || { echo "Reward server failed" >&2; exit 1; }
    if python -c "import socket; socket.create_connection(('127.0.0.1', ${RM_PORT}), timeout=1).close()" 2>/dev/null; then
        reward_ready=1; break
    fi
    sleep 1
done
[[ "${reward_ready}" == 1 ]] || { echo "Reward server startup timed out" >&2; exit 1; }
setsid "${TRAIN_CMD[@]}" >"${RUN_DIR}/logs/training.log" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" >"${RUN_DIR}/trainer.pid"
echo "Official training started, PID ${TRAIN_PID}; outputs ${RUN_DIR}"
if [[ "${ARCHIVE_CHECKPOINTS}" == 1 ]]; then
    python -u -m integration.archive_watch --run-dir "${RUN_DIR}" --repo-id "${HF_REPO_ID}" \
        --training-pid "${TRAIN_PID}" --expected-rollouts "$((MAX_SAMPLES / ROLLOUT_BATCH_SIZE))" \
        --updates-per-rollout "$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / TRAIN_BATCH_SIZE))" \
        >"${RUN_DIR}/logs/archive.log" 2>&1 &
    ARCHIVE_PID=$!
    echo "${ARCHIVE_PID}" >"${RUN_DIR}/archiver.pid"
fi
status=0
wait "${TRAIN_PID}" || status=$?
TRAIN_PID=""
echo "${status}" >"${RUN_DIR}/training_exit_status.tmp"
mv "${RUN_DIR}/training_exit_status.tmp" "${RUN_DIR}/training_exit_status"
if [[ ${status} == 0 ]]; then
    python -u -m qwen3_experiments.er_compression_compute upload-rollouts --plan "${ER_PLAN}" || status=1
fi
if [[ -n "${ARCHIVE_PID}" ]]; then wait "${ARCHIVE_PID}" || status=1; fi
echo "${status}" >"${RUN_DIR}/exit_status"
exit "${status}"
