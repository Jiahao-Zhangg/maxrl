# Cost-Aware MaxRL Implementation Plan

## Goal

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
