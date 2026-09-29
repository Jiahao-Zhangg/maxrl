# CodeContests rating-stratified mean@8

`codecontests_rating_mean8_eval.py` reuses the persistent compute-node controller
and CodeContests grader for a training-difficulty diagnostic. The prepared plan
contains 50 train-split questions from each inclusive `cf_rating` interval:
800–1000, 1100–1300, and 1400–1600. Identical complete descriptions are deduplicated
after whitespace normalization, then sampled uniformly within each interval
without replacement using seed 42. The three groups have no shared questions.

The original Qwen3-1.7B generates eight completions per question with thinking
enabled, 32,768 output tokens, temperature 0.6, top-p 0.95, and top-k 20. Requests
use independent seeds `source_row_index * 8 + sample_index`. Original CodeContests
descriptions are placed in the pinned official LiveCodeBench standard-input chat
template, as in the existing CodeContests evaluation.

Only code after the final `</think>` is graded. There is no additional EOS gate.
Every provided public, private, and generated test is included; binary grading
stops at the first failure. Output comparison uses the compiled, unchanged
CodeContests `OutputsMatch` function. Python execution uses bubblewrap with
dataset time and memory limits; it does not use upstream Sandbox2. Verifier
infrastructure failures remain errors and are retried, rather than scored zero.

Test payloads are stored once per question, separately from the eight generation
requests. Frozen file hashes cover those payloads, prompts, selection metadata,
runtime, and checker; launch also verifies model files and a matching preflight.
The supervisor, queue, and outer guard run on the designated compute node and
mirror control state to home and node-local storage.

The report requires exactly eight distinct samples per question and the planned
number of questions per rating band. Mean@8 is the average per-question fraction
of correct completions, equivalent to correct completions divided by 400 in
each 50-question band. Final artifacts are `mean8_summary.json`,
`per_question.csv`, and `RESULTS.md`.

```bash
python -m pytest -q tests/utils/test_codecontests_rating_mean8_on_cpu.py
python -m qwen3_experiments.codecontests_rating_mean8_eval launch --root /path/to/prepared/run
```

Sources: [CodeContests dataset](https://huggingface.co/datasets/deepmind/code_contests)
and [official checker](https://github.com/google-deepmind/code_contests).
