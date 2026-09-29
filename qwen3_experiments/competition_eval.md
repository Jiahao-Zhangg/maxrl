# USACOBench and CodeContests evaluation

`competition_eval.py` runs a resumable comparison of two frozen checkpoints on a
single compute node. The plan names datasets, models, artifacts, the Slurm
allocation, the execution environment, and the ordered evaluation tasks. The
supervisor and queue run on that compute node. Generated responses and grading
scratch use its local disk; control records are mirrored to a separate filesystem.
Each destination has a run-owned disk reserve that can be released on ENOSPC.

The September 28 evaluation uses the same original Qwen3-1.7B and TACO GRPO
step-100 checkpoints as the completed LCB/TACO comparison. It uses one sample,
thinking enabled, 32,768 output tokens, temperature 0.6, top-p 0.95, top-k 20,
and a per-question seed equal to its frozen source index. Eight independent GPU
workers each use tensor parallelism 1 and maximum concurrency 32. Input token
IDs are checked to be identical with both checkpoints' tokenizers.

USACOBench uses all 307 problems in the paper's `usaco_subset307_dict.json`, the
official zero-shot `solve_prompt_fn`, and the unmodified `check_correctness`
function. The downloaded tests contain 3,695 input/output pairs. The wrapper only
supplies local paths, initializes the prediction file for early interpreter
failures, isolates execution, and summarizes the official result enum. It uses
the dataset's runtime and memory limits. There is no retrieval or reflection.

CodeContests uses all 165 test examples from `deepmind/code_contests`, including
285 public, 1,552 private, and 31,797 generated tests. It uses the original problem
descriptions with the previously frozen LCB generic standard-input prompt. This
is a declared prompt choice: CodeContests does not ship that chat template.
The official `SplitAndLowercase`, `ValuesMatch`, and `OutputsMatch` C++ functions
are compiled verbatim against the upstream-pinned Abseil version. They handle
case, whitespace and absolute numeric tolerance exactly as upstream does.
Execution uses bubblewrap and Python subprocesses, **not** upstream Sandbox2.
Per-problem dataset time/memory limits are used, with the upstream 32 MiB
interpreter memory allowance and CPU/wall-time accounting. Tests stop after the
first failure; passing requires all tests. No problems are filtered by outcomes.

Both datasets grade only code after `</think>`; a missing closing tag receives
zero, while EOS is not an additional requirement. The final-code extractor is
the same one used in the previous completed comparison. Infrastructure failures
raise an error and are retried instead of being silently counted as wrong answers.

The run directory contains `plan.json`, `preflight.json`, `launch.json`, mirrored
control records, per-question responses and grades, completion audits, and an
automatically updated `RESULTS.md`. Model revisions, dataset revisions, source
hashes, generated response hashes, and prompt provenance are recorded. Each
completed task also saves a hash for every grade and response.

Validation includes identity checks for resume, simulated ENOSPC on the home
destination, success requiring all tests, and infrastructure errors remaining
errors. Before launch, the sandbox was checked with correct, wrong, syntax-error
and timeout programs for both datasets, eight comparator cases, and seven
official solutions across all four USACO levels and three CodeContests problems.

```bash
python -m pytest -q tests/utils/test_competition_eval_on_cpu.py
python -m qwen3_experiments.competition_eval launch --root /path/to/prepared/run
```

Official sources: [USACOBench](https://github.com/princeton-nlp/USACO),
[CodeContests](https://github.com/google-deepmind/code_contests).
