# veRL 0.9.1 / vLLM 0.24.0 GRPO benchmark

This branch updates the upstream veRL source and dependency metadata to release
`v0.9.1`, commit `1876b06d0a3e4e71e06230be10af14492ca8a75b`.
The complete project state before the update is commit `d8a375d`.
Project experiment files and modules that are absent from upstream are retained.
Upstream trainer and algorithm files follow the release; historical custom
MaxRL trainer/advantage changes are preserved in the parent commit and are not
automatically ported to the new trainer. This experiment validates GRPO.

The official `uv.lock` selects vLLM 0.24.0, PyTorch 2.11.0/CUDA 13.0,
Transformers 5.9.0, and Python 3.12. An independent environment is required.
NumPy is constrained to `<2.4` and locked at 2.3.5 because the release's
NumPy 2.4.6 pin violates mistral-common's requirement on Python 3.12.
Do not update an environment used by an existing training or evaluation queue.

## Installation

Use `uv sync --frozen --extra fsdp --extra vllm --python python3.12` with
`UV_PROJECT_ENVIRONMENT` and `UV_CACHE_DIR` pointing to the experiment's local
disk directories. The release's flash-attn wheelhouse URL returned HTTP 404
during this experiment. The reproducible fallback is:

```bash
uv sync --frozen --extra fsdp --extra vllm --python python3.12 --no-install-package flash-attn
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" ninja setuptools wheel psutil
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=32 NVCC_THREADS=2 \
  uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  --no-build-isolation --no-deps flash-attn==2.8.3
```

The source build needs a CUDA 13 compiler. `FLASH_ATTN_CUDA_ARCHS=90` targets
H100. Record the wheel/source hash and run an import and forward/backward check
before training. The inference engine ships its own FlashAttention 3 kernels;
the separate flash-attn 2.8.3 package serves the training model.
The source archive used here has SHA256
`1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d`.

## Controlled comparison

`grpo_verl091_benchmark.yaml` keeps the previous benchmark's original
Qwen3-1.7B, the same randomly selected first batch, seed 42, batch 32, N=8,
temperature/top-p/top-k 1/1/-1, thinking mode, 32,768 output tokens, TP=1,
eight H100 GPUs, 16 concurrent sequences per engine, and 0.7 memory fraction.
Actor-update entropy is disabled because its coefficient is zero, matching
the old actor's behavior. Entropy remains measured in the old-log-prob pass.
Enabling actor-update entropy in the new trainer unnecessarily retains its
backward graph and caused an 80 GB H100 OOM in the first complete rollout attempt.
`grpo_release_benchmark.py --prepare` verifies the full source dataset SHA256,
then materializes the baseline's 32 selected rows in a local benchmark parquet.
It inverts the actual StatefulDataLoader permutation to preserve prompt order,
and checks the real loader's first batch before writing the resolved config and
prompt receipt. `shuffle=True` remains enabled. This replay subset is only for
the one-step benchmark; ordinary training still uses the full source dataset.
Checking `list(sampler)` alone was insufficient: the new StatefulDataLoader
produced a different first batch despite the same seed. That exploratory run
is archived separately and excluded from the controlled comparison.

Inference enables Model Runner V2, FlashAttention 3, CUDA Graph,
async scheduling, chunked prefill, and prefix caching. The prefill batch token
limit is 8,192 instead of the older benchmark's 35,840. These settings form one
performance candidate; a single step does not establish a global optimum.
Weights and KV cache remain BF16. No speculative draft model is introduced.

The trainer uses synchronous GRPO updates. Its agent loop can grade completed
responses while other requests are still generating. Eight reward processes
with sixteen threads each bound sandbox concurrency to 128. The adapter
`lcb_verl_reward.py` uses the existing pinned LCB grader and sandbox environment:
binary reward, code after the thinking block only, no EOS gate, first failed
test stops execution, and only infrastructure errors receive one retry.

The rollout server's existing custom-server configuration is connected to
class loading so `grpo_release_server.py` can capture the resolved model-runner
selection, engine settings, and Prometheus counters at sleep boundaries. Model weights,
sampling, and training updates are not altered by these probes.

The plan records node, allocation, environment, model/data paths, and output
directories. All controller and GPU work runs on that compute node. Large
artifacts stay on local disk; logs and compact receipts are copied to the
configured persistent artifact directory. Checkpoint saving and evaluation
are disabled for this single-step experiment.

Compare output token count as well as generation and complete-step wall time.
The framework, PyTorch, scheduling and tokenizer versions also change, so
speed differences cannot be attributed solely to Model Runner V2. A first
step includes compilation costs, and seeds do not yield identical samples
across vLLM versions.

## Validation

The isolated environment passes the dependency compatibility check, a
FlashAttention GPU forward/backward check, and these 65 CPU tests:

```bash
python -m pytest -q tests/test_protocol_on_cpu.py tests/tools/test_base_tool_on_cpu.py \
  tests/utils/test_lcb_coding_on_cpu.py tests/utils/test_lcb_verl_reward_on_cpu.py \
  tests/utils/test_grpo_benchmark_batch_on_cpu.py
```

Run the experiment on its allocated compute node with the plan's Python:

```bash
python -m qwen3_experiments.grpo_release_benchmark --plan "$BENCHMARK_PLAN" --prepare
python -m qwen3_experiments.grpo_release_benchmark --plan "$BENCHMARK_PLAN"
```

The plan must identify an idle eight-GPU allocation, frozen runtime tree, local
model and data, pinned grading plan, and persistent artifact directory. The
controller refuses to launch if another compute process occupies a GPU.
