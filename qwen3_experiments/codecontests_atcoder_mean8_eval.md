# AtCoder train-difficulty evaluation

This adapter evaluates ARC and AGC questions from the pinned CodeContests train
split. The current experiment samples 50 unique questions from each family with
selection seed 42, with no task-letter filter, and generates eight independent
answers per question using the original Qwen3-1.7B.

The selection manifest records original source-row indices, full-statement
hashes, canonical AtCoder task IDs, and the metadata used to map task titles.
Sampling is uniform without replacement after deduplication. ARC has 124 eligible
questions and AGC has 271 in this source snapshot. Older ARC contests can begin
with task C; the task letter is not interpreted as a universal difficulty rating.

Generation uses thinking mode, 32,768 output tokens, temperature 0.6, top-p 0.95,
top-k 20, and the pinned official LCB standard-input prompt. The evaluator grades
only code after the thinking block and does not require EOS. It preserves all
provided public, private, and generated test pairs and uses the unmodified
compiled CodeContests output comparison with isolated bubblewrap execution and
the source time/memory limits. This is a diagnostic under the CodeContests tests,
not a submission to AtCoder's online judge. These selected rows have 320 public
tests, 9,815 generated tests, no private tests, and one public-only question.

The frozen plan and successful preflight are required before launch. The compute
node hosts generation, grading, queue supervision, and recovery. Interrupted work
resumes from verified response and grade files. The final report validates all
eight samples per question and both 50-question groups, then writes mean@8 and
the per-question 0–8 correct-sample histogram separately for ARC and AGC.

```bash
python -m qwen3_experiments.codecontests_atcoder_mean8_eval launch --root <output_root>
```
