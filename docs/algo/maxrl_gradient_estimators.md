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
gradient directly.

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

## Implementation map

| Estimator | Registered name | Launcher |
| --- | --- | --- |
| Original MaxRL | `maxrl` | `qwen3_experiments/run_qwen3_1_7b_math12k.sh` |
| Capped inverse-cost | `cost_aware_maxrl` | `qwen3_experiments/run_qwen3_1_7b_math12k_cost_aware.sh` |
| Rao–Blackwellized, success-gated | `rb_cost_aware_maxrl` | `qwen3_experiments/run_qwen3_1_7b_math12k_rb_cost_aware.sh` |

The two cost-aware launchers use `seq-mean-token-sum`, five epochs, Math12K,
and separate W&B experiment and checkpoint names. The original launcher keeps
the repository's `token-mean` default unless `MAXRL_LOSS_AGG_MODE` is set.

## References

- [Maximum Likelihood Reinforcement Learning paper](https://arxiv.org/abs/2602.02710)
- [Official MaxRL project explanation](https://zanette-labs.github.io/MaxRL/)
- [Detailed inverse-cost design](cost_aware_maxrl_plan.md)
- [Detailed Rao–Blackwell derivation](<Rao–Blackwellized Cost-Aware MaxRL.md>)
