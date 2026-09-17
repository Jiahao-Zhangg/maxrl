# Raw trajectories for the 541/700-point snapshot

The original completed evaluation trajectories are available as
[GitHub Release assets](https://github.com/Jiahao-Zhangg/maxrl/releases/tag/math-eval-matrix-20260916-raw-541),
separately from Git history. This is the same **partial** snapshot as the result
tables: Eval1 100/100, Eval2 300/300, and Eval3 141/300 budget/seed points.
Unfinished attempts are not included and missing evaluations are not zero scores.
The release contains **675,395 individual rollouts** and **348,805,957 generated
output tokens**, with about **599 MiB** of archives and integrity metadata.

## Downloads

| File | Completed points | Individual rollouts |
|---|---:|---:|
| [Eval1 archive](https://github.com/Jiahao-Zhangg/maxrl/releases/download/math-eval-matrix-20260916-raw-541/math-eval-matrix-20260916-eval1.tar) | 100 | 152,900 |
| [Eval2 archive](https://github.com/Jiahao-Zhangg/maxrl/releases/download/math-eval-matrix-20260916-raw-541/math-eval-matrix-20260916-eval2.tar) | 300 | 418,318 |
| [Eval3 archive](https://github.com/Jiahao-Zhangg/maxrl/releases/download/math-eval-matrix-20260916-raw-541/math-eval-matrix-20260916-eval3.tar) | 141 | 104,177 |

- [Full file/point manifest](https://github.com/Jiahao-Zhangg/maxrl/releases/download/math-eval-matrix-20260916-raw-541/raw_trajectories_manifest.json)
  records every included point, archive, file size, SHA256 checksum, rollout count,
  and generated-token count.
- [SHA256SUMS](https://github.com/Jiahao-Zhangg/maxrl/releases/download/math-eval-matrix-20260916-raw-541/SHA256SUMS)
  verifies the three archives and full manifest.

## Contents and integrity

All archives unpack into `math_eval_matrix_20260916/`, preserving the original
relative structure below `results/<model>/<dataset>/<eval>/budget_<B>/seed_<seed>/`:

- `summary.json`: completed-point metrics, identity, and raw-artifact receipts.
- `attempt_*/rollouts.jsonl.gz`: the original gzip file, **byte-for-byte unchanged**.
  Each JSON line includes response text, output token IDs/count, gold answer,
  correctness score, question ID/position, attempt/seed identifiers, and the
  protocol's cumulative token/budget counters.
- `attempt_*/prompts.json`: unchanged per-question counters. Despite the filename,
  this is a **statistics file, not a copy of the input prompt text**.

The published run manifests, original point receipts, and completed per-seed CSV
are also included. Full input dataset snapshots are not redistributed; dataset
revisions, question identifiers, and prompt construction are documented in the
[run manifest](run_manifest.json) and evaluation scripts.

The only summary redaction is `reused_from.output_root`, a machine-local path in
reused Eval1 metadata. Model outputs, token IDs, answers, scores, seed identities,
and budget ledgers are not edited. The release manifest provides both the
original summary hash (matching [point_receipts.csv](point_receipts.csv)) and the
exported summary hash, so a redacted summary need not match its original hash.

Before packaging, every source summary and raw file is checked against the
existing published receipts. The exporter scans text for credential-shaped data
and private source/home paths, checks saved rollout/token/correctness totals, and
verifies every archive member against its expected bytes. Original local results
are retained. No checkpoints, incomplete attempts, training rollouts, logs, or
credentials are included. This is an analysis archive, not a standalone resumable
inference run directory.

## Download and inspect

```bash
gh release download math-eval-matrix-20260916-raw-541 \
  --repo Jiahao-Zhangg/maxrl --dir math_eval_raw_20260916
cd math_eval_raw_20260916
sha256sum -c SHA256SUMS
tar -xf math-eval-matrix-20260916-eval1.tar
tar -xf math-eval-matrix-20260916-eval2.tar
tar -xf math-eval-matrix-20260916-eval3.tar
```

The inner `rollouts.jsonl.gz` files remain compressed and can be streamed using
Python's `gzip.open(path, "rt")` followed by `json.loads(line)`.

## Reproduce the export

The exporter performs no downloads, inference, uploads, or deletion. It requires
the original run directory and an empty destination:

```bash
python qwen3_experiments/export_math_eval_trajectories.py \
  --output-root /path/to/original/run \
  --snapshot-dir docs/evaluation_results/math_eval_matrix_20260916 \
  --export-dir /path/to/new/empty/export
python -m pytest tests/utils/test_math_eval_trajectory_export_on_cpu.py -q
```
