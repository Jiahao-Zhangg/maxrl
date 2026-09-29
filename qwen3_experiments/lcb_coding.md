# Unified LiveCodeBench coding reward

`lcb_code` grades both stdin and `Solution` method problems using the **unchanged**
LiveCodeBench `run_test` from commit
`28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`, the version used for this project's
LCB evaluation. Its source SHA-256 is checked before execution. There is no
dispatch to the former NeMo or LeetCodeDataset graders.

The dataset is `hi-todayis-jh/Nemotron-LeetCode-coding-clean-3.2k`. Its LCB revision
keeps the same 3,200 IDs, prompts, source metadata, and 219,771 test cases.
`data_source` is `lcb_coding`; parse `reward_model.ground_truth` as JSON.

## Grading

- One native LCB `input_output` object: string arrays `inputs`, `outputs`, and
  `fn_name` (`null` for stdin; method name for functions).
- Require `</think>` and extract the final code block after it. Do not require EOS.
- Binary reward: all tests pass = 1; wrong answer, code exception, or test timeout
  = 0. Preserve LCB's fail-fast behavior. Negative error codes never count as true.
- Every test has a **10-second** limit. For LeetCode this replaces the previous
  **10-second whole-suite** limit. The host has a separate outer watchdog.
- Use LCB's native output comparisons. In particular, its stdin numeric comparison
  uses exact decimal equality and is not the NeMo fork's floating-point tolerance
  or case-insensitive YES/NO extension. This is a grader-policy migration, not a
  claim that every possible program receives the same score as the former graders.
- Sandbox startup, malformed result, and incomplete-success failures are retried
  once and then raised. They are not converted to training reward zero.

The LeetCode conversion reads the original test AST, checks every assertion and
argument against the starter signature, and rejects unsupported constructs.
576 questions use native JSON parameters and returns. For 14 questions, a thin
adapter converts tree/list inputs, structurally serializes node returns, or
represents infinities with an explicit JSON marker. It does not replace LCB's
comparison or timeout logic. Original LeetCode helper definitions are prepended
only inside the grader, not added to the model prompt. The adapter calls the
original method without replacing it, preserving recursive method calls.

## Prepare on the compute node

Use the existing coding environment with Python, NumPy, and `sortedcontainers`
installed. An existing working bubblewrap executable is required. The namespace
has no network, credentials, dataset directories, or shared project mounts.

```bash
python -m qwen3_experiments.prepare_lcb_grading \
  --root "$LCB_GRADING_ROOT" \
  --bubblewrap /path/to/bwrap
```

This downloads the pinned upstream checker and its license, copies the isolated
runner, and writes `grading_plan.json`. Point the existing veRL training config at
the resulting plan:

```yaml
reward_model:
  reward_manager: lcb_code
  reward_kwargs:
    grading_plan: /path/to/lcb-grading/grading_plan.json
    workers: 16
    check_eos: false
    score_after_thinking: true
```

This selects only the grading implementation. It does not launch training or
change the model, optimizer, rollout concurrency, or samples per prompt.

## Reproduce the conversion

```bash
python -m qwen3_experiments.prepare_lcb_coding_mix \
  --source /path/to/previous/train.parquet \
  --leetcode-source /path/to/LeetCodeDataset-train.jsonl \
  --destination /path/to/lcb/train.parquet
```

The converter refuses to replace an existing destination. It preserves every
question and test count, reloads the Parquet, and emits a conversion receipt.
The original construction/cleaning audit remains applicable to the unchanged
questions. LCB migration audits record the new Parquet hash separately.

`verify_lcb_coding_mix` runs all 590 retained LeetCode reference solutions,
negative controls for all 3,200 rows, native stdin/function/timeout/fail-fast
probes, and saved completions from the earlier difficulty evaluations. These are
CPU checks; a distributed GPU training run has not been launched by this migration.
