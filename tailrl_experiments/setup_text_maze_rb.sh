#!/usr/bin/env bash
# All downloads, dependencies, and results stay in the three supplied directories.
set -euo pipefail

MAZE_CHECKOUT=${1:?"Usage: setup_text_maze_rb.sh CHECKOUT STATE_DIR VENV [PYTHON_3_10]"}
MAZE_STATE_DIR=${2:?"STATE_DIR is required"}
MAZE_VENV=${3:?"VENV is required"}
MAZE_BASE_PYTHON=${4:-python3.10}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TAILRL_REVISION=5682c6ac03387355e017ce966693266bb148fa10

mkdir -p "${MAZE_STATE_DIR}"/{tmp,cache,runs,checkpoints,data}
export TMPDIR="${MAZE_STATE_DIR}/tmp"
export UV_CACHE_DIR="${MAZE_STATE_DIR}/cache/uv"
export HF_HOME="${MAZE_STATE_DIR}/cache/hf"
export PYTHONNOUSERSITE=1

if [ ! -d "${MAZE_CHECKOUT}/.git" ]; then
    git clone --filter=blob:none --sparse https://github.com/Zanette-Labs/TailRL.git "${MAZE_CHECKOUT}"
    git -C "${MAZE_CHECKOUT}" checkout --detach "${TAILRL_REVISION}"
    git -C "${MAZE_CHECKOUT}" sparse-checkout set experiments/text_maze
fi
if [ "$(git -C "${MAZE_CHECKOUT}" rev-parse HEAD)" != "${TAILRL_REVISION}" ]; then
    printf 'Expected TailRL revision %s; use a dedicated checkout.\n' "${TAILRL_REVISION}" >&2
    exit 1
fi
if [ ! -x "${MAZE_VENV}/bin/python" ]; then
    uv venv --python "${MAZE_BASE_PYTHON}" "${MAZE_VENV}"
fi
uv pip install --python "${MAZE_VENV}/bin/python" 'torch==2.6.0+cu124' \
    --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "${MAZE_VENV}/bin/python" -r "${SCRIPT_DIR}/requirements_text_maze.txt"

"${MAZE_VENV}/bin/python" - "${MAZE_STATE_DIR}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

state = Path(sys.argv[1])
snapshot_download(
    "max-rl/maze_v2_sft_ckpts_guanning", revision="145e5e2d1ddb160994eb6b2daabcf2362d5d00ce",
    local_dir=state / "checkpoints", allow_patterns=[f"ckpt-{s}/*" for s in (2450, 3350, 3550)],
)
for name in ("main_1.3M.jsonl.meta.json", "test.json"):
    hf_hub_download("max-rl/maze_17x17_diverse_1.3m", name, repo_type="dataset",
                    revision="9b9ed56991cb045ba4227d9120dad337085db439", local_dir=state / "data")
PY
EXPERIMENT="${MAZE_CHECKOUT}/experiments/text_maze"
"${MAZE_VENV}/bin/python" "${EXPERIMENT}/scripts/checkpoint_doctor.py" --all "${MAZE_STATE_DIR}/checkpoints"
"${MAZE_VENV}/bin/python" "${SCRIPT_DIR}/install_text_maze_adapter.py" --checkout "${MAZE_CHECKOUT}"
"${MAZE_VENV}/bin/python" "${SCRIPT_DIR}/prepare_text_maze_pilot.py" \
    --experiment "${EXPERIMENT}" --data-dir "${MAZE_STATE_DIR}/data"
(
    cd "${EXPERIMENT}"
    PYTHONPATH="${EXPERIMENT}" "${MAZE_VENV}/bin/python" -m pytest \
        "${SCRIPT_DIR}/tests/test_text_maze_rb_on_cpu.py" -q
)
uv pip freeze --python "${MAZE_VENV}/bin/python" > "${MAZE_STATE_DIR}/environment.lock.txt"
