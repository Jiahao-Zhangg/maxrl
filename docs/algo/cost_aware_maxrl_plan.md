# Cost-Aware MaxRL Implementation Plan

## Standard MaxRL: Advantage and Gradient

For one prompt $x$, let the policy's success probability be

$$
p_\theta(x) = \mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[r(y)],
\qquad
S_i = \nabla_\theta\log\pi_\theta(y_i\mid x),
$$

where $r_i\in\{0,1\}$. With $N$ rollouts, MaxRL optimizes the order-$N$
truncation of the log-likelihood expansion:

$$
J_{\mathrm{MaxRL}}^{(N)}(x)
= -\sum_{k=1}^{N}\frac{(1-p_\theta(x))^k}{k}.
$$

Its population gradient is

$$
\nabla_\theta J_{\mathrm{MaxRL}}^{(N)}(x)
= \frac{1-(1-p)^N}{p}\,\nabla_\theta p
= \frac{1-(1-p)^N}{p}\,\mathbb{E}[rS].
$$

For sampled rewards, let $K=\sum_i r_i$ and
$\bar r=K/N$. The practical estimator is

$$
\widehat g_N(x) =
\begin{cases}
\displaystyle\frac{1}{K}\sum_{i=1}^{N}r_iS_i, & K>0,\\
0, & K=0.
\end{cases}
$$

This is unbiased for the gradient above: a batch contains a success with
probability $1-(1-p)^N$, while the conditional mean score of successful
trajectories is $\mathbb{E}[S\mid r=1]=\mathbb{E}[rS]/p$.

To express this estimator as a sequence-averaged PPO loss, MaxRL uses the
zero-mean score-function control variate
$V_N=N^{-1}\sum_i S_i$. The resulting scalar trajectory advantage is

$$
\boxed{A_i^{\mathrm{MaxRL}}
= \frac{r_i-\bar r}{\bar r}
= \frac{N r_i}{K}-1}, \qquad K>0,
$$

with all advantages set to zero when $K=0$. Thus a successful trajectory has
advantage $N/K-1$, while a failed trajectory has advantage $-1$. Broadcasting
this scalar across its response tokens gives

$$
\frac{1}{N}\sum_{i=1}^{N}A_i^{\mathrm{MaxRL}}S_i
= \frac{1}{K}\sum_{i=1}^{N}r_iS_i
- \frac{1}{N}\sum_{i=1}^{N}S_i.
$$

The last term has expectation zero, so it changes variance but not the expected
gradient. The current implementation in `verl/trainer/ppo/core_algos.py` uses
`(r_i - mean_reward) / (mean_reward + epsilon)` and then masks padding. Its
`epsilon=1e-6` makes the $K>0$ result negligibly smaller than the exact formula
and produces zero when every rollout fails.

With `loss_agg_mode=seq-mean-token-sum`, each $S_i$ is the sum of token score
functions and the equation above is matched directly. For a minibatch of $B$
trajectories, the current `token-mean` mode instead multiplies that gradient by
$B/\sum_i L_i$. This preserves the fixed-minibatch direction but makes its
scale depend on generated lengths.

See the [MaxRL paper](https://arxiv.org/abs/2602.02710) and
[official project explanation](https://zanette-labs.github.io/MaxRL/) for the
objective and unbiased-estimator derivation.

## Cost-Aware Goal

For each prompt, sample $N$ trajectories. Let $r_i \in \{0,1\}$ be the math
grader reward, $L_i$ the number of generated tokens (from
`response_mask.sum(-1)`), and $K = \sum_i r_i$. Define

$$
L_{\mathrm{ref}} = \frac{L_{\max}}{2}, \qquad
c_i = \frac{L_i}{L_{\mathrm{ref}}}.
$$

The requested per-prompt policy-gradient estimator is

$$
G = \frac{1}{K}\sum_{i=1}^{N}\frac{r_i}{c_i}S_i,
\qquad
S_i = \nabla_\theta \log \pi_\theta(y_i\mid x).
$$

This variant favors correct, shorter trajectories. Scaling by
$L_{\mathrm{ref}}$ fixes the overall gradient scale but does not change the
relative preference between two response lengths.

## Advantage

The actor averages losses over the $N$ trajectories, so use the raw
trajectory advantage

$$
A_i^{\mathrm{raw}} = \frac{N}{K}\frac{r_i}{c_i}.
$$

Use the MaxRL control variate in practice:

$$
\boxed{A_i = \frac{N r_i L_{\max}}{2K L_i} - 1}
$$

when $K>0$, and set every advantage in the group to zero when $K=0$.
The `-1` term does not change the expected gradient because
$\mathbb{E}[S_i]=0$, but it supplies the usual MaxRL baseline and reduces
variance. Broadcast each trajectory's scalar advantage over its valid response
tokens and keep padding at zero.

Do **not** divide by $\sum_i r_i/c_i$. That would normalize the
cost-weighted rewards and optimize a different estimator. Do not apply group
standard-deviation normalization either.

For example, with $N=16$, $K=4$, and $L_{\max}=4096$, a correct
1,024-token response has $c_i=0.5$ and $A_i=7$; an incorrect response has
$A_i=-1$.

## Stability Policy

Inverse length can be very large for unusually short outputs. Implement an
optional cap without silently enabling it:

```text
inverse_cost = (cost_reference_tokens / response_length).clamp(max=max_inverse_cost)
```

The exact estimator uses no cap. For the first training comparison, also test
`max_inverse_cost=4.0`. Log the capped fraction because a cap intentionally
changes the objective. Clamp response length to at least one before division,
and detach lengths, rewards, and advantages from autograd.

## Repository Changes

1. Add `COST_AWARE_MAXRL = "cost_aware_maxrl"` and a registered estimator in
   `verl/trainer/ppo/core_algos.py`. Read these settings from the algorithm
   config:

   ```yaml
   algorithm:
     adv_estimator: cost_aware_maxrl
     cost_reference_tokens: 2048  # data.max_response_length / 2
     max_inverse_cost: null       # exact estimator; use 4.0 for capped ablation
   ```

2. Use `actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum`. This makes
   the actor's sequence-average gradient match $G$. The current `token-mean`
   mode introduces an additional batch token-count scale.
3. Add a separate Qwen3 Math12K launcher so the running MaxRL experiment and
   its checkpoints remain unchanged.
4. Log mean response length, mean successful-response length, inverse cost,
   cap fraction, zero-success group fraction, and advantage min/mean/max.

## Tests and Experiment

Add CPU tests in `tests/trainer/ppo/test_core_algos_on_cpu.py` for mixed
lengths, $K=0$, all-success groups, multiple prompt groups, padding, and the
optional cap. Numerically verify that

$$
\frac{1}{N}\sum_i (A_i+1)S_i
= \frac{1}{K}\sum_i \frac{r_i}{c_i}S_i.
$$

Then compare the existing MaxRL run, exact cost-aware MaxRL, and the capped
variant with the same model, data, seed, rollout count, and five epochs. Report
validation accuracy together with response length and correct answers per
generated token; accuracy alone would hide the cost tradeoff.
