# Repository Guidelines

## Workspace and Branch

- Work in the primary `maxrl` directory. The user-requested upgrade branch is `codex/verl-0.9.1-vllm-0.24-grpo`.
- Do not create or use temporary branches or sibling worktrees unless the user explicitly requests them.

## Training and Evaluation Environment

- Use `maxrl-code-verl091` for new training and evaluation: veRL 0.9.1 and vLLM 0.24.0.
- Run controllers, training, checkpoint uploads and evaluation on the assigned compute node, using frozen runtime copies.
- Coding GRPO defaults to the cleaned 3,200-row Nemotron/LeetCode dataset, batch 32, N=8, concurrency 16, thinking on, and 32k output tokens. Training sampling is 1/1/-1; evaluation is 0.6/0.95/20.
- Use the shared pinned LiveCodeBench grader for coding training and evaluation, binary reward, after-thinking code only, no EOS requirement. Log Polaris-compatible training accuracy bins globally and per data source.
- Save every 10 steps, verify public Hub uploads before deleting local checkpoints and completed rollout files, and preserve receipts for recovery.

## Hugging Face Uploads

- Project checkpoints and rollout datasets default to public Hugging Face repositories.
- The user has also requested that earlier project training repositories be public. Apply that preference to existing upload destinations unless the user explicitly requests a private repository later.

## Project Structure & Module Organization

Core Python code lives in `verl/`: trainers and Hydra-style YAML configuration are under `verl/trainer/`, distributed actors and rollout backends are under `verl/workers/`, and shared data, checkpoint, and metric helpers are under `verl/utils/`. Tests mirror these namespaces in `tests/`. Reusable training variants live in `recipe/`, while `examples/` contains preprocessing and launcher examples. Paper-specific entry points are grouped in `maze/`, `smollm/`, `imagenet/`, and `qwen3_experiments/`. Documentation, deployment helpers, and installation scripts live in `docs/`, `docker/`, and `scripts/`.

## Build, Test, and Development Commands

- `pip install -e .` installs the `verl` package in editable mode.
- `pytest tests/test_protocol_on_cpu.py tests/tools/test_base_tool_on_cpu.py` runs a quick CPU-only smoke set.
- `pytest tests/utils/test_*_on_cpu.py` runs CPU tests for one component; `pytest tests` exercises the full suite and generally requires GPUs and backend-specific dependencies.
- `ruff check verl tests examples recipe` checks imports and Python lint rules; add `--fix` only after reviewing the proposed changes.
- `make -C docs html` builds the Sphinx documentation into `docs/_build/html`.

Training is normally launched through a checked-in shell script, for example `bash smollm/smollm.sh`. Copy or edit paths and hardware settings deliberately; many published configurations assume multi-GPU machines.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python naming: `snake_case` for functions, modules, and variables; `PascalCase` for classes; `UPPER_CASE` for constants. Ruff configuration is defined in `pyproject.toml`, including sorted imports with `verl` treated as first-party. Keep configuration keys consistent with existing YAML and command-line override names. Prefer focused modules and avoid embedding machine-specific paths in Python code.

## Testing Guidelines

Use pytest and name files `test_*.py`; CPU-compatible files should end in `_on_cpu.py`. Place tests under the directory matching the changed `verl` namespace. Distributed, end-to-end, NPU, and standalone coverage belongs in the corresponding `tests/special_*` suite. Include regression tests for behavior changes and state required accelerator resources when reporting results.

## Commit & Pull Request Guidelines

History favors short, imperative subjects, sometimes with prefixes such as `fix:` or `refactor:`. Keep each commit scoped to one logical change. Pull requests should explain the motivation and implementation, link relevant issues, list exact validation commands, and document GPU type/count plus important configuration overrides. Include screenshots only for rendered documentation changes, and never commit secrets, local datasets, generated checkpoints, or personal filesystem paths.
