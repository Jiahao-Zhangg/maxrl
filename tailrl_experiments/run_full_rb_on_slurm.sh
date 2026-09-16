#!/usr/bin/env bash
# One isolated attempt; Slurm cleans its own step before the controller retries.
set -euo pipefail
umask 077
[[ $# == 3 && "$1" =~ ^[0-9]+$ && "${SLURM_JOB_ID:-}" == "$1" ]] || exit 2
JOB_ID=$1
CKPT_STEP=$3
case "${CKPT_STEP}" in 2450|3000|3250|3350|3400|3450|3550) ;; *) exit 2 ;; esac
PLAN=$(realpath -- "$2")
CONTROL_DIR=$(dirname -- "${PLAN}")
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec 8>"${CONTROL_DIR}/node.lock"
flock -n 8 || exit 1
source "${CONDA_PROFILE:-/sw/user/python/miniforge3-pytorch-2.10.0/etc/profile.d/conda.sh}"
conda activate tailrl_rb
export CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MAX_JOBS=8
export TAILRL_RAY_TMPDIR="${TAILRL_RAY_TMPDIR:-/tmp/maze_rb_${JOB_ID}}"
export TMPDIR="${TAILRL_RAY_TMPDIR}/tmp" TRITON_CACHE_DIR="${TAILRL_RAY_TMPDIR}/triton"
unset PYTHONPATH PYTHONHOME RAY_ADDRESS
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}"
cd "${REPO_ROOT}"
mapfile -t paths < <(python - "${PLAN}" <<'PY'
import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text())
for k in ('experiment','state_dir','output_dir','cpu_ready'):print(p[k])
PY
)
[[ ${#paths[@]} == 4 && -f "${paths[3]}" ]] || exit 2
[[ ! -e "${CONTROL_DIR}/STOP" ]] || exit 0
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z "${busy}" ]] || { echo "GPUs are occupied; refusing to start." >&2; exit 1; }
args=(--experiment "${paths[0]}" --state-dir "${paths[1]}" --output-dir "${paths[2]}" --ckpt-step "${CKPT_STEP}")
if [[ ! -f "${CONTROL_DIR}/gpu_validation_passed" ]]; then
    timeout --kill-after=30s 15m python tailrl_experiments/check_text_maze_gpu.py --job-id "${JOB_ID}"
    timeout --kill-after=30s 5m python -m torch.distributed.run --standalone --nproc_per_node=4 \
        tailrl_experiments/check_text_maze_gpu.py --job-id "${JOB_ID}" --collective
    timeout --kill-after=30s 45m python -u tailrl_experiments/run_text_maze_rb_full.py "${args[@]}" --smoke
    printf '%s %s %s\n' "$(date --iso-8601=seconds)" "${JOB_ID}" "$(hostname)" \
        >"${CONTROL_DIR}/gpu_validation_passed"
fi
[[ ! -e "${CONTROL_DIR}/STOP" ]] || exit 0
exec python -u tailrl_experiments/run_text_maze_rb_full.py "${args[@]}"
