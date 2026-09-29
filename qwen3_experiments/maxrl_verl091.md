# MaxRL family on the veRL 0.9.1 release trainer

The advantage formulas and cost diagnostics are preserved from `d8a375d` in
`verl/trainer/ppo/maxrl_algos.py` and `maxrl_metrics.py`. They are registered with
the release estimator registry and connected to the V1 TransferQueue trainer.
Use the `maxrl-code-verl091` environment and `grpo_coding_release.py` launcher.

Let N be responses per prompt, r_i its binary reward, M the group's success
count, and c_i = L_i + L_0, where L_i includes thinking and final-answer tokens.

| Estimator | Successful response weight | Failed response weight |
|---|---|---|
| `maxrl` | (1 - M/N) / (M/N + 1e-6) | -(M/N) / (M/N + 1e-6) |
| `fixed_n_rb_offset_cost_aware_marginrl` | N/M - c_i/mean(c) | -M/(M+1) * c_i/mean(c) |
| `f_cov` | N/M - c_i/C * (H + 1/(K*(M+1))) | -c_i/C * H |

For f_cov, K is the full number of prompts, C is mean cost across all K*N
responses, and H = mean_k(M_k/(M_k+1)). MaxRL and per-context RB give all-failed
groups zero weight. f_cov can give them negative weight when other groups
succeed. An entirely unsuccessful batch gives zero weights for all three.
These are final optimizer weights: no extra multiplication by N or K, no GRPO
standard-deviation normalization, and no whitening. The base fixed-N raw-length
RB estimator remains registered as the shared implementation of the offset form.

`algorithm.cost_offset_tokens` controls L_0 (default 256; zero and 4096 are also
supported). V1 fills `algorithm.f_cov_num_prompts` from `data.train_batch_size`
and rejects a conflicting explicit value. All statistics are computed once on
the controller, before GPU microbatching. UID grouping survives load-balancing
reordering, and synthetic padding rows are excluded and receive zero advantage.
When the legacy dispatch selects a separate loss mask, costs still use the full
response mask. Cost metrics are forwarded into training/W&B logs unchanged.

Supported V1 training uses synchronous updates, complete fixed-N groups and one
output per rollout session. Partial groups, multiple outputs per session, prompt
filtering, reward KL shaping, and rejection-modified response masks fail explicitly
instead of silently changing the estimator. Historical Math12K shell launchers
and their old math reward-manager dependencies are outside this coding migration;
they are not valid release entry points. No environment packages are replaced.

## Queued original MaxRL run

Use `grpo_coding_release.py prepare --algorithm maxrl --n 16`, with the same
benchmark/data and holdout inputs as GRPO, distinct output/HF destinations, and
`--after-root` pointing to the frozen GRPO run. Start its frozen supervisor on
the same compute allocation. No running predecessor code or configuration is
changed.

The run starts from original Qwen3-1.7B and the pinned cleaned 3,200-row dataset:
batch 32, N=16 (512 rollouts per step), one epoch/100 updates, shuffle seed 42,
32k output, thinking enabled, training sampling 1/1/-1, concurrency 16, and the
same MRV2/FA3/Graph inference settings as GRPO. It retains accuracy bins,
checkpoint saves every ten steps, public verified uploads and local deletion,
and the shared after-thinking/no-EOS LCB grader with 128 grading workers.

The final step-100 checkpoint receives the same four holdouts with vLLM 0.24.0
and that same grader: LCB v6 (175), TACO non-SPJ test (782), USACOBench (307), and
CodeContests test (165). Evaluation is thinking on, 32k, 0.6/0.95/20, pass@1.

To prepare a cost variant later, use the same launcher with
`--algorithm fixed_n_rb_offset_cost_aware_marginrl --cost-offset-tokens 4096`
or `--algorithm f_cov --cost-offset-tokens 256`. These do not add runs to a
queue until a separate plan and supervisor are explicitly launched.

## Validation

Run in the release environment, with this checkout on PYTHONPATH and GPUs hidden:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest -q \
  tests/trainer/ppo/test_maxrl_v1_on_cpu.py \
  tests/trainer/ppo/test_cross_context_f_cov_on_cpu.py \
  tests/trainer/ppo/test_fixed_n_rb_offset_cost_aware_marginrl_on_cpu.py \
  tests/utils/test_grpo_coding_release_on_cpu.py \
  tests/utils/test_prompt_binning_on_cpu.py \
  -k 'not math12k and not launcher' -p no:cacheprovider
```

This covers independent formulas for N=2/8/16, 32- and 256-prompt f_cov batches,
all-fail/all-success groups, cost/loss masks, padding, real nested TensorDict
conversion and TransferQueue writeback, metrics, configuration guards, queue
dependency audits, and verified cleanup. It excludes historical Math12K launcher
tests, which require the old Hydra schema and math-grading environment. CPU
checks do not establish N=16 GPU throughput or memory usage; that is measured
when the queued run starts after the predecessor's evaluations.
