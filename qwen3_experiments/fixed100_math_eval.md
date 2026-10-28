# Fixed-100 compression math evaluation on another machine

This entry point evaluates the **L1-Max 100–4000 final checkpoint** by default,
then computes the same four evaluation families as the October 27 fixed-100
tables. It works on a Linux GPU machine without Slurm or any of the original
cluster paths. The model can also be changed explicitly.

The default model is
`hi-todayis-jh/l1-max-qwen3-1.7b-compression-100-4k-from80-mixed50-alpha3e-4-bs32-n8-32k-t1-146103-final`,
pinned to `b9bde5a3c7d66c3ff80612842bb702f5911e5a40`.

## Setup and one-GPU run

Use the `maxrl-code-verl091` environment if it is already available. Otherwise,
create an isolated Python 3.12 environment and install the generation requirements
below. These scripts do not require installing the full veRL training package.
The machine needs a CUDA driver compatible with the vLLM 0.24.0 installation.
Run generation on an assigned GPU; run grading and analysis on allocated CPUs.

```bash
git clone --branch codex/verl-0.9.1-vllm-0.24-grpo https://github.com/Jiahao-Zhangg/maxrl.git
cd maxrl
python3.12 -m venv .venv-math-eval
source .venv-math-eval/bin/activate
python -m pip install -r qwen3_experiments/requirements-fixed100-generation.txt

EVAL_RUN="$PWD/outputs/l1_max_100_4000_fixed100"
python -m qwen3_experiments.fixed100_math_eval prepare --run-dir "$EVAL_RUN"

python -m pip install --no-deps --target "$EVAL_RUN/grader_site" \
  -r qwen3_experiments/requirements-fixed100-grading.txt

CUDA_VISIBLE_DEVICES=0 python -m qwen3_experiments.fixed100_math_eval generate \
  --run-dir "$EVAL_RUN"

python -m qwen3_experiments.fixed100_math_eval grade --run-dir "$EVAL_RUN" --workers 8

# Read access to the private reference dataset is needed only for this comparison.
hf auth login
python -m qwen3_experiments.fixed100_math_eval analyze \
  --run-dir "$EVAL_RUN" --workers 4 --with-reference
```

If the repository already exists, fetch and check out the branch instead of
cloning again. `prepare` downloads only prompts and tokenizer/config files;
vLLM fetches the pinned model weights when generation starts. Reserve enough disk
space for the model cache, full text/token records and grades (20 GB is a useful
starting allowance for one 1.7B model). Generated artifacts are never deleted
automatically.

For an initial GPU smoke test, add `--limit-requests 8` to `generate`, then run it
again without that flag. Existing responses are verified and reused. This is a
partial generation check; grading deliberately refuses incomplete fixed-100 pools.

CUDA graphs are enabled (`enforce_eager=False`), with bf16, prefix caching and
chunked prefill. The default engine context is 40,960 tokens and the response cap
is 32,768 tokens. `prepare --concurrency 16` lowers the default 64 concurrent
requests if needed. `--gpu-memory-utilization` defaults to 0.85.

## Multiple GPUs

For four independent GPU replicas, create a **new** run with `prepare --shards 4`.
Run each shard on a different visible GPU, using the same run directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -m qwen3_experiments.fixed100_math_eval generate --run-dir "$EVAL_RUN" --shard-index 0
CUDA_VISIBLE_DEVICES=1 python -m qwen3_experiments.fixed100_math_eval generate --run-dir "$EVAL_RUN" --shard-index 1
CUDA_VISIBLE_DEVICES=2 python -m qwen3_experiments.fixed100_math_eval generate --run-dir "$EVAL_RUN" --shard-index 2
CUDA_VISIBLE_DEVICES=3 python -m qwen3_experiments.fixed100_math_eval generate --run-dir "$EVAL_RUN" --shard-index 3
```

Run these four commands in separate terminals/jobs for parallel execution. Wait
for all four to finish before grading. A shard lock prevents two processes from
writing the same shard concurrently. Sharding is fixed by the preparation plan;
relaunch the same shard to resume interrupted generation.

For tensor parallelism instead, set `prepare --tensor-parallel-size 2` and expose
two GPUs to each generation process, for example `CUDA_VISIBLE_DEVICES=0,1`.
The number of visible GPUs per shard must match the tensor parallel size.

Preparation copies the required code to `RUN/runtime`. Every later command
automatically executes that frozen copy and verifies its input hashes. Updating
the Git checkout therefore cannot change an in-progress evaluation. Keep the
run directory at the same absolute location while resuming grading, because
the grader receipt also records its isolated dependency paths.

## Other models and reference comparisons

For another checkpoint, provide `prepare --model OWNER/REPO --revision COMMIT
--model-key UNIQUE_KEY --label LABEL`, and use a new run directory. The checkpoint
must use the same Qwen3 tokenizer/chat template; all pinned thinking prompt token
IDs are checked before generation. A branch/tag revision is resolved and saved
as an immutable model commit.

`analyze --with-reference` downloads only the compact arrays and metadata from
the private dataset
`hi-todayis-jh/compression-math-big-eval-all-models-fixed100-20261027` at revision
`141398492e0d348652d817a80e55cb7b97c85589`, not the 24 GB of raw responses. It
recomputes all ten models on the same retained questions. An existing local
reference package can be passed with `--reference-dir /path/to/package` instead.
Omit both flags for standalone model results, which need no private Hub access.
**Standalone first-correct costs have a different solved cohort and must not be
compared directly with the nine-model paper table.**

## Exact evaluation protocol

- The source has 1,119 questions across Minerva Math, OlympiadBench, AMC22/23,
  AIME24, AIME25 and AIME26. Preparation excludes the same seven IDs as the
  consolidated paper tables: `amc22_23_0016`, `amc22_23_0028`, `amc22_23_0074`,
  `minervamath_0135`, `minervamath_0136`, `minervamath_0217`, `minervamath_0230`.
  It retains **1,112 questions and generates 111,200 responses**.
- Each question has exactly 100 independent response attempts, numbered 0–99.
  There are no adaptive top-ups. Original question positions and the historical
  BLAKE2b per-response seed formula are preserved. Sampling is temperature 0.6,
  top-p 0.95, top-k 20, thinking enabled, 32,768 output tokens. Seed compatibility
  does not imply bitwise-identical generation across GPU types or batch schedules.
- Full answers and token prefixes use the historical MathVerify 0.9.0 scorer,
  isolated from the newer SymPy used by PyTorch. Only text after the last
  `</think>` is graded, without an EOS gate. A missing/invalid closing tag is a
  failure. Each MathVerify call has a one-second deadline; timeout scores zero,
  without retry. The historical scorer is preserved byte-for-byte.
- **Per-question pass@budget:** independently shuffle each question's 100
  responses without replacement, 100 times. Budgets are 4k, 8k, 16k, 32k, 64k.
  The first successful boundary response earns remaining tokens / response length.
- **Shared-budget pass@budget:** allocate `N × B` output tokens within each
  benchmark. Stop-after-success permutes active unsolved questions in sweeps,
  with one next shuffled response per active question. Continue-after-success
  uniformly permutes all `N × 100` pairs, charging repeated successes but
  counting each solved question only once. Both use fractional boundary credit.
- **Tokens to first correct:** sum complete response costs through the first
  success in the same 100 per-question permutations. Restrict only this metric
  to questions every included model solves at least once. Also report the exact
  finite-pool expectation `sum(failed lengths)/(correct count + 1) + mean(correct lengths)`.
- **Capped single-attempt accuracy:** grade actual token prefixes at 1k, 2k,
  4k, 8k, 16k, 32k. Grades are binary, without fractional credit. Complete
  responses reuse their original grade; partially visible answers are regraded.
  Mean length uses the original responses, which were capped at 32,768 tokens.
- Budget/cap labels use 1k=1,024 tokens. Reported length/cost in thousands uses
  k=1,000 tokens. Macro averages six benchmarks equally; weighted averages
  questions equally. First-correct weights use the common-solved counts.
- Replay randomness is NumPy PCG64 with seed 42 and the same separate streams
  as the existing tables. If a newly evaluated pool cannot cover a requested
  budget, analysis records and excludes that question **for all models and
  budgets**, then reruns until all 100 shuffles are supported. It never treats
  an exhausted unseen response as a failure or silently skips it. Additional
  exclusions and the changed common-solved cohort are saved in `cohort.json`.
  Reported MC standard errors describe shuffle variation for the saved pools,
  not fresh-generation or benchmark-sampling uncertainty.

## Saved outputs

- `plan.json`, `questions.json`, `runtime/`: pinned inputs, seed identities and frozen code.
- `responses/*.json.gz`: every full response, exact token IDs, stop reason and provenance.
- `grades/*.json.gz`: full-answer and capped-prefix grades with response hashes.
- `samples.npz`, `grading_receipt.json`: complete fixed-100 arrays and their checksum.
- `analysis/four_evaluations_macro.md`, `analysis/four_evaluations_weighted.md`:
  ready-to-read tables; six additional files contain per-benchmark tables.
- `analysis/budget_metrics_long.csv`, `capped_accuracy_long.csv`,
  `tokens_to_first_correct.csv`, `trial_scores.npz`, `cohort.json`: numeric
  results, trial scores, exclusions and comparison provenance.

## CPU verification

```bash
python -m pytest -q tests/utils/test_fixed100_math_eval_on_cpu.py \
  tests/utils/test_capped_response_eval_on_cpu.py
```

The tests cover fractional boundary credit, deduplicated successes, exhausted
pools, paired seeds, resume provenance, shared exclusions, and binary prefixes.
They do not substitute for the small GPU generation smoke test on the target machine.
