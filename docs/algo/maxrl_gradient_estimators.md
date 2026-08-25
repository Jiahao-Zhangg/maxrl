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

Both Efficient-Reasoning launchers also append the globally shortest response
from every training round to
`<checkpoint_dir>/debug/shortest_rollouts.jsonl`. Each JSON object contains the
training step, batch position, prompt UID and decoded prompt, decoded response,
response-token count, scalar grader reward, and binary correctness. Selection
is over the complete driver batch; equal-length responses are resolved by the
first batch position. Set `MAXRL_SAVE_SHORTEST_ROLLOUT=false` to disable this
trace for an ER launch.

## Implementation map

| Estimator | Registered name | Launcher |
| --- | --- | --- |
| Original MaxRL | `maxrl` | `qwen3_experiments/run_qwen3_1_7b_math12k.sh` |
| Capped inverse-cost | `cost_aware_maxrl` | `qwen3_experiments/run_qwen3_1_7b_math12k_cost_aware.sh` |
| Rao–Blackwellized, success-gated | `rb_cost_aware_maxrl` | `qwen3_experiments/run_qwen3_1_7b_math12k_rb_cost_aware.sh` |
| Fixed-$N$ RB | `fixed_n_rb_cost_aware_marginrl` | `qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_marginrl.sh` |
| Fixed-$N$ RB, failures gated | `fixed_n_rb_cost_aware_marginrl_success_gated` | `qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_marginrl_success_gated.sh` |
| Fixed-$N$ RB, capped normalized cost | `fixed_n_rb_capped_cost_aware_marginrl` | `qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_capped_marginrl.sh` |
| Fixed-$N$ RB, capped cost and fixed $\widehat q$ | `fixed_n_rb_capped_fixed_q_cost_aware_marginrl` | `qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_capped_fixed_q_marginrl.sh` |
| Fixed-$N$ RB, Efficient-Reasoning cost | `fixed_n_rb_efficient_reasoning_cost_marginrl` | `qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_er_cost_marginrl.sh` |
| Fixed-$N$ RB, Efficient-Reasoning cost, failures gated | `fixed_n_rb_efficient_reasoning_cost_marginrl_success_gated` | `qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_er_cost_marginrl_success_gated.sh` |

The fixed-$N$ launchers default to `token-mean`, five epochs, and Math12K,
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
