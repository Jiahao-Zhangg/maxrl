# MaxRL Gradient Estimators

## Shared notation

For one prompt, sample $N$ trajectories $y_i\sim\pi_\theta$. Let

\[
r_i\in\{0,1\},\qquad K=\sum_{i=1}^N r_i,\qquad
\bar r=K/N,
\]

and let $S_i=\nabla_\theta\log\pi_\theta(y_i\mid x)$ be the sum of the
trajectory's token score functions. The formulas below use the repository's
sequence-averaged convention

\[
\widehat g=\frac1N\sum_{i=1}^N A_iS_i.
\]

Consequently, practical advantages contain a factor of $N$. Broadcasting
$A_i$ over valid response tokens with `seq-mean-token-sum` realizes this
gradient directly. The fixed-$N$ launchers now default to `token-mean` to
match the original MaxRL runs. If an optimizer microbatch has mean valid
response length $\bar L$, the policy-gradient contribution becomes

\[
\widehat g_{\rm token\text{-}mean}
=\frac{1}{\bar L}\widehat g_{\rm seq\text{-}mean\text{-}token\text{-}sum}.
\]

Both modes retain the complete trajectory score $S_i$; `token-mean` adds a
microbatch-dependent scale. Set
`MAXRL_LOSS_AGG_MODE=seq-mean-token-sum` to recover the direct normalization.

## 1. Original MaxRL estimator

MaxRL optimizes the order-$N$ truncated-log success objective

\[
L_N(p)=-\sum_{k=1}^N\frac{(1-p)^k}{k},
\qquad p=\mathbb E[r].
\]

Its raw unbiased estimator is

\[
\widehat g_{\rm raw}
=\mathbf 1\{K>0\}\frac1K\sum_i r_iS_i.
\]

The theoretical zero-mean control variate subtracts the unconditional average
score $N^{-1}\sum_iS_i$. Its sequence-scale advantage is

\[
A_i^{\rm CV}=N\mathbf1\{K>0\}\frac{r_i}{K}-1.
\]

In particular, this unbiased form uses $A_i=-1$ when $K=0$. The repository's
practical MaxRL implementation instead uses

\[
A_i^{\rm MaxRL}
=\frac{r_i-\bar r}{\bar r+\varepsilon}
\approx
\begin{cases}
\displaystyle \frac{Nr_i}{K}-1,&K>0,\\
0,&K=0.
\end{cases}
\]

Ignoring $\varepsilon$, the implemented gradient is therefore

\[
\widehat g_{\rm MaxRL}^{\rm impl}
=\mathbf1\{K>0\}
\left(\frac1K\sum_i r_iS_i-\frac1N\sum_iS_i\right).
\]

This distinction matters: zeroing the whole group at $K=0$ is not identical
to subtracting the unconditional control variate. The displayed practical
formula exactly describes the repository behavior. Under the iid binary-reward
model and with $\varepsilon\to0$, its expectation is
$\nabla_\theta L_{N-1}(p)$ rather than the raw estimator's
$\nabla_\theta L_N(p)$. The code uses
`(r_i - mean_reward) / (mean_reward + 1e-6)`.

## 2. Capped inverse-cost estimator

This heuristic reweights successful trajectories by inverse token cost:

\[
c_i=\frac{L_i}{L_{\max}/2},\qquad
w_i=\min\!\left(\frac1{c_i},4\right).
\]

The requested target estimator and practical advantage are

\[
\widehat g_{\rm IC}
=\mathbf 1\{K>0\}\frac1K\sum_i r_iw_iS_i,
\]

\[
A_i^{\rm IC}=
\begin{cases}
\displaystyle \frac{Nr_iw_i}{K}-1,&K>0,\\
0,&K=0.
\end{cases}
\]

Thus the implemented sequence-averaged gradient is

\[
\widehat g_{\rm IC}^{\rm impl}
=\mathbf1\{K>0\}
\left(\frac1K\sum_i r_iw_iS_i-\frac1N\sum_iS_i\right).
\]

The gated $-1$ follows the repository's practical MaxRL convention. The
denominator remains
$K=\sum_i r_i$, not $\sum_i r_iw_i$. Thus correct, shorter trajectories
receive larger positive updates, while the cap prevents very short responses
from producing extreme weights.

## 3. Rao–Blackwellized cost-aware estimator

Define a bounded cost probability

\[
\kappa_i=\frac{L_i}{L_{\max}}\in[0,1]
\]

and independent auxiliary variables
$a_j\sim\operatorname{Bernoulli}(\kappa_j)$. With
$K_{-i}=\sum_{j\ne i}a_j$, the exact Rao–Blackwell coefficient is

\[
\beta_i=\kappa_i\,\mathbb E\!\left[\frac1{1+K_{-i}}\right].
\]

It is computed by a leave-one-out Poisson-binomial DP. Starting from $d_0=1$,
each $j\ne i$ updates

\[
d_k^{\rm new}=(1-\kappa_j)d_k+\kappa_jd_{k-1},
\qquad
\beta_i=\kappa_i\sum_k\frac{d_k}{k+1}.
\]

Without success gating,

\[
\sum_i\left[\mathbf1\{K>0\}\frac{r_i}{K}-\beta_i\right]S_i
\]

is unbiased for the finite-$N$ objective $L_N(p)-L_N(q)$, where
$q=\mathbb E[\kappa]$. The implemented variant deliberately gates the entire
update when every rollout fails:

\[
A_i^{\rm RB}=
\begin{cases}
\displaystyle \frac{Nr_i}{K}-N\beta_i,&K>0,\\
0,&K=0.
\end{cases}
\]

There is **no additional $-1$** in this variant. Zeroing the cost term at
$K=0$ avoids learning solely from response length before a success is found,
but makes the implemented estimator biased relative to
$L_N(p)-L_N(q)$.

## 4. Fixed-$N$ RB estimator with capped normalized cost

This estimator preserves the fixed-rollout RB update while replacing raw token
cost with

\[
\widetilde c_i=
\max\!\left(\frac{L_i}{L_{\mathrm{ref}}},\frac1{w_{\max}}\right),
\qquad
L_{\mathrm{ref}}=L_{\max}/2,
\qquad
w_{\max}=4.
\]

For $M=\sum_i r_i$ and the detached same-batch rate
$\widehat q=M/\sum_i\widetilde c_i$, its raw trajectory advantage is

\[
A_i^{\rm capped\text{-}RB}=
\begin{cases}
\displaystyle \frac{1-\widehat q\widetilde c_i}{M},&r_i=1,\\[2mm]
\displaystyle -\frac{\widehat q\widetilde c_i}{M+1},&r_i=0.
\end{cases}
\]

The optimizer receives $N A_i^{\rm capped\text{-}RB}$ to preserve the fixed
rollout-group convention. Under the default `token-mean` aggregation, the
result is additionally divided by the optimizer microbatch's mean response
length as described above. With a 4096-token response limit, all responses of
512 tokens or fewer have the same effective cost $1/4$. This removes the
incentive to become progressively shorter inside that range while retaining
cost pressure above it. If $M=0$, then $\widehat q=0$ and the group update is
zero.

### Additive response-length cost ($L_0=256$)

The `fixed_n_rb_offset_cost_aware_marginrl` variant uses $c_i=L_i+L_0$,
with `algorithm.cost_offset_tokens=256`. Here $L_i$ counts the full response
tokens, including EOS when generated, using the same response mask as the
hard-clip Math12K run. It applies no cost clipping or reference-length normalization.
For each prompt, use all $N$ rollouts to compute

\[
M=\sum_i r_i,\qquad \widehat p=M/N,\qquad
\overline L=\frac1N\sum_i L_i,\qquad
\widehat q=\frac{M}{\sum_i(L_i+L_0)}.
\]

After the existing multiplication of raw coefficients by $N$, the optimizer's
trajectory advantages are

\[
A_i^{\rm offset\text{-}RB}=\begin{cases}
\displaystyle \frac1{\widehat p}-\frac{L_i+L_0}{\overline L+L_0},&r_i=1,\\[2mm]
\displaystyle -\frac{M}{M+1}\frac{L_i+L_0}{\overline L+L_0},&r_i=0.
\end{cases}
\]

All-failure groups receive zero advantage. The token-mean loss reduction is
unchanged. Unlike the hard floor $\max(L_i,256)$, additive cost continues to
distinguish response lengths below 256 tokens. An offset of zero recovers the
raw response-length estimator. Diagnostics use `fixed_n_rb_offset_marginrl/`;
costs are reported in tokens plus offset, and `cap_ratio` is always zero.

`qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_offset_marginrl.sh`
reuses the hard-clip launcher's `run_qwen3_1_7b_math12k.sh` entry point. The
defaults remain Qwen/Qwen3-1.7B-Base, hiyouga/math12k, 256 prompts × 16 responses
per step, a 4,096-token output budget, token-mean loss, learning rate $10^{-6}$,
zero KL, and seed 79. This launcher stops at step 150 with a five-epoch ceiling
and saves/evaluates every 50 steps, producing checkpoints at steps 50, 100, and 150.
The same prompt preprocessing, AIME25/Math500 validation, and `multi_thread`
Math-Verify grader are retained, including the original 1-second item and
10-second batch timeouts, no EOS reward gate, and no forced terminal EOS.
Apart from the 150-step limit, automatic checkpoint uploads, and rollout archival, only the cost
estimator and experiment/output names change relative to the
hard-clip-256 configuration (`MAXRL_COST_REFERENCE_TOKENS=2048`,
`MAXRL_MAX_INVERSE_COST=8`). The additive offset defaults to 256 tokens and can
be set with `MAXRL_COST_OFFSET_TOKENS`.

Checkpoint uploads are enabled by default with `MAXRL_UPLOAD_CHECKPOINTS=1`.
The shared uploader publishes each completed checkpoint, including the newest
one during training, to a public model repository under
`zjhhhh/fixed-n-rb-offset-cost-aware-marginrl-qwen3-1.7b-base-math12k-offset256-token-mean-step_<N>`.
These are the original FSDP checkpoints, including optimizer and data-loader
state. Local checkpoint directories are deleted only after all remote file
names and sizes have been verified. Failed uploads remain local for retry.
The launcher waits for the final upload and records the actual training exit
status; a failed run is not treated as completed solely because a final-step
message appeared in its log. Upload logs are saved at
`<checkpoint directory>/logs/checkpoint_upload.log`.

Set `MAXRL_CHECKPOINT_HF_REPO_PREFIX` to change the upload destination, or
`MAXRL_UPLOAD_CHECKPOINTS=0` to save checkpoints only locally. When uploads are enabled,
configure the final step and checkpoint location via
`MAXRL_TOTAL_TRAINING_STEPS`, `MAXRL_OUTPUT_DIR`, and `MAXRL_EXPERIMENT_NAME`
so the trainer and uploader use the same settings. After uploaded checkpoints
have been deleted locally, restore one before resuming it.

Training rollouts are also saved by default (`MAXRL_SAVE_ROLLOUT_DATASET=1`),
with all responses from each step in a compressed JSONL shard. The trainer
uploads and verifies this dataset at training exit. Its public HF destination
defaults to `${MAXRL_CHECKPOINT_HF_REPO_PREFIX}-rollouts`; override it with
`MAXRL_ROLLOUT_DATASET_HF_REPO`. Rollout saving/uploading can be disabled
independently with `MAXRL_SAVE_ROLLOUT_DATASET=0`.

For an already-running trainer that writes legacy `<step>.jsonl` files through
`trainer.rollout_data_dir`, `qwen3_experiments/upload_training_rollouts_to_hf.py`
can attach without restarting training. It waits for the expected full row
count, stages compressed shards, incrementally uploads and verifies them, and
retains source files on both success and failure. Its state and lock live
outside the staged dataset directory. Upload errors are retried, and a
terminated trainer is distinguished from a reused process ID.

The prepared launcher is:

```bash
bash qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_offset_marginrl.sh
```

### Cross-context plug-in advantage (`f_cov`)

For a complete batch of $K$ distinct prompts with $N$ responses per prompt,
define $M_k=\sum_i R_{k,i}$, $c_{k,i}=L_{k,i}+L_0$, and

$$
\bar c_{\mathrm{all}}=\frac{1}{KN}\sum_{k,i}c_{k,i},
\qquad H=\frac{1}{K}\sum_k\frac{M_k}{M_k+1}.
$$

The final optimizer advantage is

$$
A_{k,i}^{f_{\mathrm{cov}}}=
\begin{cases}
\displaystyle\frac{N}{M_k}-\frac{c_{k,i}}{\bar c_{\mathrm{all}}}
\left(H+\frac{1}{K(M_k+1)}\right), & R_{k,i}=1,\\[6pt]
\displaystyle-\frac{c_{k,i}}{\bar c_{\mathrm{all}}}H, & R_{k,i}=0.
\end{cases}
$$

All costs, counts, and advantages are detached. These coefficients already
have the optimizer scaling: no additional multiplication by $N$ or $K$, or
advantage whitening, is applied. Statistics are computed on the complete
driver batch before the actor splits it into GPU microbatches. The standard
PPO token-mean loss and optimizer settings are retained.

The implementation uses `algorithm.adv_estimator=f_cov`,
`algorithm.cost_offset_tokens=256`, and `algorithm.f_cov_num_prompts=256`.
It rejects incomplete prompt batches and unequal response counts. Entirely
failed prompts can receive negative advantages when another prompt succeeds;
an entirely failed batch receives zero. For $K=1$ the formula reduces to the
previous per-prompt additive-cost estimator. Metrics use the `f_cov/` prefix,
including `H`, `global_cost_mean`, success counts, and final advantage scales.

`qwen3_experiments/run_qwen3_1_7b_math12k_f_cov_offset_marginrl.sh` starts from
Qwen/Qwen3-1.7B-Base on Math12K with the same grader, EOS behavior, 256 × 16
rollouts, and training settings as the additive-cost baseline. It trains for
150 steps, saves/evaluates every 50 steps, uploads and verifies checkpoints
before deleting them locally, and saves every training rollout for HF dataset
upload at the end of training.

`python -m qwen3_experiments.queue_math12k_after_run --plan <plan.json>
--state-file <state.json>` queues that launcher behind a specific predecessor
PID and process start time. It requires successful predecessor training,
verified checkpoint/rollout uploads, checkpoint cleanup, and two idle checks
on the selected GPUs. It rechecks GPU availability after network checks and
records a launch before spawning it to prevent duplicate launches. `--check`
validates the plan without starting training; a `STOP` file beside the queue
state cancels the pending launch.

## 5. Fixed-$N$ RB estimator with fixed $\widehat q=2$

This ablation keeps the capped normalized cost from Section 4,

\[
\widetilde c_i=\max\!\left(2L_i/L_{\max},1/4\right),
\]

but replaces the same-batch rate estimate by the detached constant
$\widehat q=2$. Its trajectory advantages are

\[
A_i=
\begin{cases}
\displaystyle \frac{1-2\widetilde c_i}{M}, & r_i=1,\\[2mm]
\displaystyle -\frac{2\widetilde c_i}{M+1}, & r_i=0.
\end{cases}
\]

The success branch is only evaluated when $M>0$. Unlike the plug-in variant,
an all-failure group has $A_i=-2\widetilde c_i$ and therefore retains a cost
update. This is a fixed-rate ablation, not a same-batch estimate of
$p/\mathbb E[c]$.

## 6. Fixed-$N$ RB estimator with Efficient-Reasoning sigmoid cost

This variant replaces the capped normalized cost from Section 4 with the
Efficient-Reasoning length function. For each prompt group $g$, let $\mu_g$ and
$\sigma_g$ be the population mean and standard deviation of the response
lengths across all $N$ rollouts in that group. Every rollout receives cost

\[
c_i^{\rm ER}=\operatorname{sigmoid}\!\left(
    \frac{L_i-\mu_g}{\sigma_g+10^{-7}}
\right).
\]

The fixed-$N$ estimator still computes its detached, per-prompt plug-in rate
from all $N$ costs,

\[
M_g=\sum_{i\in g}r_i,
\qquad
\widehat q_g=\frac{M_g}{\sum_{j\in g}c_j^{\rm ER}}.
\]

Its raw trajectory advantage is

\[
A_i^{\rm ER}=
\begin{cases}
\displaystyle \frac{1-\widehat q_g c_i^{\rm ER}}{M_g},&r_i=1,\\[2mm]
\displaystyle -\frac{\widehat q_g c_i^{\rm ER}}{M_g+1},&r_i=0.
\end{cases}
\]

The optimizer again receives $N A_i^{\rm ER}$. An all-failure group has
$\widehat q_g=0$ and receives zero advantage. Only the sigmoid length function
is borrowed from Efficient Reasoning; there is no $\alpha$ coefficient in this
variant, and the Fixed-$N$ RB plug-in estimate remains in use.

The success-gated Efficient-Reasoning variant keeps the same costs,
$\widehat q_g$, and successful-response branch, but assigns every wrong answer
zero raw advantage:

\[
A_i^{\rm ER\text{-}gated}=
\begin{cases}
\displaystyle \frac{1-\widehat q_g c_i^{\rm ER}}{M_g},&r_i=1,\\[2mm]
0,&r_i=0.
\end{cases}
\]

Its optimizer advantage is $N A_i^{\rm ER\text{-}gated}$. In particular,
wrong answers remain exactly zero after the fixed-$N$ multiplier.

### Training metrics for early-EOS debugging

Both Efficient-Reasoning fixed-$N$ variants log the following W&B keys. The
standard variant uses the prefix `fixed_n_rb_er_cost_marginrl/`; the gated
variant uses `fixed_n_rb_er_cost_marginrl_success_gated/`. Here $A_i$ is the
detached trajectory advantage supplied to PPO, including the fixed-rollout
multiplier:

\[
A_i=A_i^{\rm optimizer}=N A_i^{\rm raw},
\]

\[
\begin{aligned}
\mathrm{early\_eos\_rate}
&=\frac1B\sum_i\mathbf1[L_i\le2],\\
\mathrm{early\_eos\_fail\_rate}
&=\frac{\sum_i\mathbf1[r_i=0,L_i\le2]}
        {\sum_i\mathbf1[r_i=0]},\\
\mathrm{mean\_len\_fail}
&=\frac{\sum_i(1-r_i)L_i}{\sum_i(1-r_i)},\\
\mathrm{mean\_len\_success}
&=\frac{\sum_i r_iL_i}{\sum_i r_i},\\
\mathrm{adv\_short\_fail}
&=\frac{\sum_iA_i\mathbf1[r_i=0,L_i\le2]}
        {\sum_i\mathbf1[r_i=0,L_i\le2]},\\
\mathrm{adv\_normal\_fail}
&=\frac{\sum_iA_i\mathbf1[r_i=0,L_i>2]}
        {\sum_i\mathbf1[r_i=0,L_i>2]},\\
\mathrm{adv\_success}
&=\frac{\sum_iA_i\mathbf1[r_i=1]}{\sum_i\mathbf1[r_i=1]},\\
\mathrm{frac\_negative\_adv\_success}
&=\frac{\sum_i\mathbf1[r_i=1,A_i<0]}{\sum_i\mathbf1[r_i=1]}.
\end{aligned}
\]

Metrics conditioned on an empty subset are logged as `0.0`. The characteristic
early-EOS failure pattern is
$\mathrm{adv\_short\_fail}\approx0$ while
$\mathrm{adv\_normal\_fail}\ll0$, together with a rising
$\mathrm{early\_eos\_fail\_rate}$ and falling
$\mathrm{mean\_len\_fail}$. For the success-gated variant,
`adv_short_fail` and `adv_normal_fail` should both remain exactly zero; the
length and EOS metrics still reveal whether failed generations are collapsing.

### Shortest-rollout trace

Shortest-rollout logging is disabled by default because the rollout-dataset
export already preserves every generated response. Set
`MAXRL_SAVE_SHORTEST_ROLLOUT=true` to additionally append the globally shortest
response from every training round to
`<checkpoint_dir>/debug/shortest_rollouts.jsonl`. Each JSON object contains the
training step, batch position, prompt UID and decoded prompt, decoded response,
response-token count, scalar grader reward, and binary correctness. Selection
is over the complete driver batch; equal-length responses are resolved by the
first batch position.

## Implementation map

| Estimator | Registered name | Launcher |
| --- | --- | --- |
| Original MaxRL | `maxrl` | `qwen3_experiments/run_qwen3_1_7b_math12k.sh` |
| Capped inverse-cost | `cost_aware_maxrl` | `qwen3_experiments/run_qwen3_1_7b_math12k_cost_aware.sh` |
| Rao–Blackwellized, success-gated | `rb_cost_aware_maxrl` | `qwen3_experiments/run_qwen3_1_7b_math12k_rb_cost_aware.sh` |
| Fixed-$N$ RB | `fixed_n_rb_cost_aware_marginrl` | `qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_marginrl.sh` |
| Fixed-$N$ RB, failures gated | `fixed_n_rb_cost_aware_marginrl_success_gated` | `qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_marginrl_success_gated.sh` |
| Fixed-$N$ RB, capped normalized cost | `fixed_n_rb_capped_cost_aware_marginrl` | `qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_capped_marginrl.sh` |
| Fixed-$N$ RB, additive $L+256$ cost | `fixed_n_rb_offset_cost_aware_marginrl` | `qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_offset_marginrl.sh` |
| Cross-context plug-in, additive $L+256$ cost | `f_cov` | `qwen3_experiments/run_qwen3_1_7b_math12k_f_cov_offset_marginrl.sh` |
| Fixed-$N$ RB, capped cost and fixed $\widehat q$ | `fixed_n_rb_capped_fixed_q_cost_aware_marginrl` | `qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_capped_fixed_q_marginrl.sh` |
| Fixed-$N$ RB, Efficient-Reasoning cost | `fixed_n_rb_efficient_reasoning_cost_marginrl` | `qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_er_cost_marginrl.sh` |
| Fixed-$N$ RB, Efficient-Reasoning cost, failures gated | `fixed_n_rb_efficient_reasoning_cost_marginrl_success_gated` | `qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_er_cost_marginrl_success_gated.sh` |

The Math12K fixed-$N$ launchers default to `token-mean`, five epochs, and Math12K,
matching the loss reduction used by original MaxRL and capped inverse-cost
MaxRL. They also include the aggregation mode in their default W&B run and
checkpoint names. `seq-mean-token-sum` remains available through
`MAXRL_LOSS_AGG_MODE`; earlier fixed-$N$ checkpoints created before this
configuration change used that mode.

## References

- [Maximum Likelihood Reinforcement Learning paper](https://arxiv.org/abs/2602.02710)
- [Official MaxRL project explanation](https://zanette-labs.github.io/MaxRL/)
- [Detailed inverse-cost design](cost_aware_maxrl_plan.md)
- [Detailed Rao–Blackwell derivation](<Rao–Blackwellized Cost-Aware MaxRL.md>)
