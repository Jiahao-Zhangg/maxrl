# Compression model budget follow-up

`compression_budget_followup.py` adds prepared, immutable evaluation stages after
an existing audited budget run. The queue, its retrying supervisor, model caches,
and GPU launchers remain on the assigned compute allocation.

The September 25 allocation-146102 queue evaluates ER final, compression L+0
step100, then Qwen3-1.7B. Each model adds MATH500, OlympiadBench, AMC22+23, and
AIME24/25/26 at 8k, 16k, 32k, 48k, and 64k individual budgets: 90 new points.
The 15 completed Minerva points are reused after checking model revisions,
questions, settings, summary identities, and raw-artifact checksums. The combined
report contains all 105 model/dataset/budget points.

Generation and grading use the existing frozen `minerva_individual_budget.py`
worker and budget engine: thinking on, 0.6/0.95/20, seed 0, 32k per response,
after-thinking-only grading, no additional EOS requirement, and stop at the
first correct response within each question's own budget.

The queue requires successful predecessor status and a complete, matching audit.
It detects existing stage launchers and workers before spawning GPU work; the
underlying runner also holds the existing allocation locks and checks all GPUs.
Completed points are preserved across retries. Prepared weights are kept only
under this queue's node-local directory, with hashes matching the earlier Minerva
evaluation. After each stage's complete result audit and worker exit, its private
model cache is deleted. Shared base weights and other allocations are untouched.

Queue status and result directories reside outside the project filesystem.
Control writes retain a node-local mirror and retry storage quota errors. An
external supervisor retries unexpected queue exits. The queue verifies frozen
inputs again before every stage transition.

The active queue is discoverable through
`outputs/compression_budget_followups_146102_20260925/launch.json` and the original
L+4096 run's `additional_budget_queue.json`. Its `queue_status.json` tracks new
point counts; `reused_minerva.json` records the completed source results.
