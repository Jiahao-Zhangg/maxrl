# Nemotron coding mean@8

`nemotron_mean8_eval.py` extends the resumable compute-node evaluation controller
with the official NeMo Gym coding verifier and eight independently seeded
completions per question. It verifies frozen data, runtime and model hashes before
launch, and requires a successful preflight tied to the same plan hash.

The prepared experiment samples 3,200 rows uniformly without replacement from the
pinned default `train` split of `nvidia/Nemotron-RL-coding-competitive_coding`.
It preserves all original columns, messages and unit tests. A separate seed-42
sample chooses 100 positions within the uploaded subset. This is a training-data
difficulty diagnostic; those 100 questions are still part of the training subset.

The original Qwen3-1.7B generates eight responses per question, using thinking
mode, 32,768 output tokens, temperature 0.6, top-p 0.95 and top-k 20. Each expanded
request has seed `original_source_index * 8 + sample_index`. Eight GPU workers
run with tensor parallelism 1 and maximum concurrency 32. The context limit
includes the longest prompt plus the full output allowance.

Code is extracted only after the last `</think>` using the pinned upstream
`extract_code(..., LMStyle.OpenAIChat)`. A missing thinking close or missing
fenced final code receives zero. EOS is not an additional grading requirement.
`nemotron_code_sandbox_runner.py` executes upstream `check_correctness` and
`run_test` inside bubblewrap, with the official ten-second per-test timeout.
The original checker functions are selected from the pinned source without
initializing its Ray server. This preserves its comparison and timeout behavior.
Infrastructure errors are retried rather than counted as model failures.

Mean@8 is the mean of the 100 per-question fractions `correct_samples / 8`,
equivalently the number of correct completions divided by 800. It is distinct
from pass@8, which counts whether any sample passes. Final aggregation rejects
missing or repeated samples and mismatched question identities.

The compute-node guard maintains the supervisor and queue. Control records are
mirrored to home and node-local disk; reserve files allow status writes to recover
from a full filesystem. Responses, test payloads and grader scratch use node-local
disk. The final artifacts are `mean8_summary.json`, `per_question.csv`, and
`RESULTS.md`, including the distribution of correct samples out of eight.

Validation covers mean versus any-pass aggregation, incomplete and duplicate
samples, response identity, complete test coverage and infrastructure failures.
The launch preflight additionally checks correct, wrong, syntax-error and timeout
programs, float and integer comparisons, case handling, buffered stdin, and all
five recorded upstream example rollouts.

```bash
python -m pytest -q tests/utils/test_nemotron_mean8_eval_on_cpu.py
python -m qwen3_experiments.nemotron_mean8_eval launch --root /path/to/prepared/run
```

Sources: [dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-coding-competitive_coding),
[official grader](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/code_gen).
