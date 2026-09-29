# LeetCodeDataset Medium/Hard mean@8

The queued diagnostic evaluates the original Qwen3-1.7B on 50 Medium and 50 Hard
questions from `codetm/LeetCodeDataset`, train split, pinned to revision
`364485ef1971b51136feb83bf38f3e3cf13a0558`. Sampling uses `random.Random(42)`,
uniformly without replacement, Medium first and then Hard in source-row order.
The eligible pools contain 1,397 Medium and 606 Hard questions. All selected
task IDs and full statement/starter-code pairs are unique.

The original `query` is the model's user message. The Qwen chat template enables
thinking; each question receives eight independent samples, with 32,768 output
tokens and temperature/top-p/top-k 0.6/0.95/20. Generation seeds are
`source_row_index * 8 + sample_index`. Only final code after `</think>` is graded;
there is no EOS requirement.

Execution uses the author's unmodified `eval_lcd.execution.check_correctness`
from `newfacade/LeetCodeDataset` revision
`182cd19fd53161efd70a1ef074fe1056b659bfa3`. Each question retains the original
`prompt`, `test`, and `entry_point`. The whole test suite of one answer has a
10-second deadline, matching the upstream standalone evaluation CLI default.
The checker runs inside the existing credential-free bubblewrap namespace.
These are the dataset's tests, not LeetCode's online judge.

The random sample is preserved despite three Hard reference solutions timing
out: `maximum-path-quality-of-a-graph`, `number-of-valid-words-for-each-puzzle`,
and `minimum-incompatibility`. The other 97 reference solutions pass. The main
metrics include all 50 questions in each group; the final JSON additionally
reports a diagnostic restricted to reference-passing questions. No model result
is used for selection or filtering.

The compute-node queue waits for the preceding ARC/AGC run to finish generation,
grading, and summary publication. It checks the predecessor's frozen plan,
completion audit, and hashes of every response and grade before permitting
generation. Persistent supervisor and guard processes recover interrupted work;
control state is mirrored to local scratch. The prior run is unchanged.

Final artifacts include `mean8_summary.json`, `per_question.csv`, and `RESULTS.md`.
Mean@8 is sample accuracy averaged across questions, rather than pass@8.

```bash
python -m qwen3_experiments.leetcode_mean8_eval launch --root <output_root>
```
