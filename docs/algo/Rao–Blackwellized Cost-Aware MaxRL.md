# MaxRL and Rao–Blackwellized Cost-Aware MaxRL

Fix a context \(x\). Sample

\[
z_1,\ldots,z_N \overset{\mathrm{iid}}{\sim} m_\theta(\cdot\mid x),
\]

and define

\[
S_i := \nabla_\theta \log m_\theta(z_i\mid x),
\qquad
r_i := r(z_i,x)\in\{0,1\}.
\]

Let

\[
p_\theta(x)
:=
\mathbb E_{z\sim m_\theta(\cdot\mid x)}[r(z,x)]
\]

denote the probability of success.

Define the order-\(N\) truncated-log function

\[
L_N(u)
:=
-\sum_{k=1}^N \frac{(1-u)^k}{k}.
\]

Its derivative is

\[
L_N'(u)
=
\sum_{j=0}^{N-1}(1-u)^j
=
\frac{1-(1-u)^N}{u}.
\]

As \(N\to\infty\),

\[
L_N(u)\to \log u.
\]

---

## 1. Original MaxRL

### Objective

The finite-compute MaxRL objective is

\[
J_{\mathrm{MaxRL}}^{(N)}(\theta)
=
\mathbb E_{x\sim\rho}
\left[
L_N(p_\theta(x))
\right].
\]

For a fixed context,

\[
\nabla_\theta L_N(p_\theta(x))
=
\frac{1-(1-p_\theta(x))^N}{p_\theta(x)}
\nabla_\theta p_\theta(x).
\]

As \(N\to\infty\),

\[
J_{\mathrm{MaxRL}}^{(N)}(\theta)
\longrightarrow
\mathbb E_x[\log p_\theta(x)].
\]

Thus MaxRL interpolates between ordinary expected-reward RL at
\(N=1\) and maximum likelihood as \(N\to\infty\).

### Gradient estimator

Let

\[
K_r:=\sum_{i=1}^N r_i.
\]

The basic MaxRL estimator is

\[
A_N
:=
\mathbf 1\{K_r>0\}
\frac{1}{K_r}
\sum_{i=1}^N r_iS_i.
\]

It satisfies

\[
\boxed{
\mathbb E[A_N\mid x]
=
\nabla_\theta L_N(p_\theta(x)).
}
\]

Equivalently,

\[
A_N
=
\sum_{i=1}^N
\mathbf 1\{K_r>0\}
\frac{r_i}{K_r}S_i.
\]

Hence the raw per-rollout MaxRL coefficient is

\[
A_i^{\mathrm{raw}}
=
\mathbf 1\{K_r>0\}\frac{r_i}{K_r}.
\]

### Variance-reduced MaxRL advantage

MaxRL additionally uses the zero-mean score control variate

\[
V_N
=
\frac1N\sum_{i=1}^N S_i,
\qquad
\mathbb E[V_N]=0.
\]

Therefore

\[
\widetilde g_{\mathrm{MaxRL}}
=
A_N-V_N
=
\sum_{i=1}^N
\left(
\mathbf 1\{K_r>0\}\frac{r_i}{K_r}
-\frac1N
\right)S_i.
\]

When \(K_r>0\), defining the empirical success rate

\[
\bar r=\frac{K_r}{N},
\]

this becomes

\[
\widetilde g_{\mathrm{MaxRL}}
=
\sum_{i=1}^N
\frac{r_i-\bar r}{N\bar r}S_i.
\]

Thus the practical MaxRL advantage is

\[
\boxed{
A_i^{\mathrm{MaxRL}}
=
\frac{r_i-\bar r}{N\bar r}
=
\frac{r_i}{K_r}-\frac1N,
\qquad K_r>0.
}
\]

In the standard implementation, the entire update is set to zero when
\(K_r=0\).

---

## 2. Cost-Aware MaxRL

Each rollout additionally has a realized cost

\[
C_i:=C(z_i,x)\in[0,C_{\max}(x)],
\]

where \(C_{\max}(x)\) is independent of \(\theta\), and \(C(z,x)\) has no
explicit \(\theta\)-dependence for fixed \((z,x)\).

Define

\[
c_\theta(x)
:=
\mathbb E[C(z,x)],
\]

and the normalized expected cost

\[
q_\theta(x)
:=
\frac{c_\theta(x)}{C_{\max}(x)}
=
\mathbb E
\left[
\frac{C(z,x)}{C_{\max}(x)}
\right].
\]

For convenience, for each sampled rollout define

\[
\kappa_i
:=
\frac{C_i}{C_{\max}(x)}
\in[0,1].
\]

### Objective

The finite-compute cost-aware objective is

\[
\boxed{
J_{\mathrm{CA}}^{(N)}(\theta)
=
\mathbb E_{x\sim\rho}
\left[
L_N(p_\theta(x))
-
L_N(q_\theta(x))
\right].
}
\]

Its gradient for a fixed context is

\[
\nabla_\theta J_{\mathrm{CA}}^{(N)}(x)
=
L_N'(p_\theta(x))\nabla_\theta p_\theta(x)
-
L_N'(q_\theta(x))\nabla_\theta q_\theta(x).
\]

Equivalently,

\[
\nabla_\theta J_{\mathrm{CA}}^{(N)}(x)
=
\frac{1-(1-p_\theta)^N}{p_\theta}\nabla_\theta p_\theta
-
\frac{1-(1-q_\theta)^N}{q_\theta}\nabla_\theta q_\theta.
\]

As \(N\to\infty\),

\[
J_{\mathrm{CA}}^{(N)}(\theta)
\longrightarrow
\mathbb E_x
\left[
\log p_\theta(x)-\log q_\theta(x)
\right].
\]

Since

\[
q_\theta(x)
=
\frac{c_\theta(x)}{C_{\max}(x)},
\]

and \(C_{\max}(x)\) does not depend on \(\theta\), this has the same
gradient as

\[
\boxed{
\mathbb E_x
\left[
\log \frac{p_\theta(x)}{c_\theta(x)}
\right].
}
\]

Hence the infinite-compute objective favors high success probability and
low expected rollout cost.

---

### Rao–Blackwellized gradient estimator

The reward side is exactly the original MaxRL estimator,

\[
A_N
=
\mathbf 1\{K_r>0\}
\frac1{K_r}
\sum_{i=1}^N r_iS_i.
\]

For the cost side, define

\[
K_{-i}
:=
\sum_{j\neq i} a_j,
\qquad
a_j\sim\mathrm{Bernoulli}(\kappa_j),
\]

where the Bernoulli variables are used only to define the
Rao–Blackwell expectation.

Define

\[
\boxed{
\beta_i
:=
\kappa_i
\mathbb E
\left[
\frac{1}{1+K_{-i}}
\,\middle|\,
z_{1:N}
\right].
}
\]

Equivalently,

\[
\boxed{
\beta_i
=
\kappa_i
\int_0^1
\prod_{j\neq i}
\left(
1-\kappa_j+\kappa_j t
\right)\,dt.
}
\]

An equivalent permutation-marginal representation is

\[
\boxed{
\beta_i
=
\kappa_i
\mathbb E_{\sigma}
\left[
\prod_{j\prec_\sigma i}(1-\kappa_j)
\right],
}
\]

where \(\sigma\) is a uniformly random permutation of the \(N\)
rollouts.

The Rao–Blackwellized cost-aware estimator is therefore

\[
\boxed{
\widehat g_{\mathrm{CA}}^{(N)}
=
A_N
-
\sum_{i=1}^N\beta_iS_i.
}
\]

Expanding \(A_N\),

\[
\boxed{
\widehat g_{\mathrm{CA}}^{(N)}
=
\sum_{i=1}^N
\left[
\mathbf 1\{K_r>0\}\frac{r_i}{K_r}
-
\beta_i
\right]S_i.
}
\]

It is unbiased for the finite-compute cost-aware objective:

\[
\boxed{
\mathbb E[
\widehat g_{\mathrm{CA}}^{(N)}
\mid x
]
=
\nabla_\theta
\left[
L_N(p_\theta(x))
-
L_N(q_\theta(x))
\right].
}
\]

The Rao–Blackwellization integrates out the auxiliary cost-acceptance
randomness and therefore does not change the expected gradient while
removing that source of Monte-Carlo variance.

### Cost-aware advantage

The estimator already has standard policy-gradient form

\[
\widehat g_{\mathrm{CA}}^{(N)}
=
\sum_{i=1}^N
A_i^{\mathrm{CA}} S_i.
\]

Therefore the rollout-level advantage is simply

\[
\boxed{
A_i^{\mathrm{CA}}
=
\mathbf 1\{K_r>0\}\frac{r_i}{K_r}
-
\beta_i.
}
\]

Substituting the Rao–Blackwell coefficient,

\[
\boxed{
A_i^{\mathrm{CA}}
=
\mathbf 1\{K_r>0\}\frac{r_i}{K_r}
-
\frac{C_i}{C_{\max}(x)}
\mathbb E
\left[
\frac{1}{1+K_{-i}}
\,\middle|\,
z_{1:N}
\right].
}
\]

Equivalently,

\[
\boxed{
A_i^{\mathrm{CA}}
=
\mathbf 1\{K_r>0\}\frac{r_i}{K_r}
-
\frac{C_i}{C_{\max}(x)}
\int_0^1
\prod_{j\neq i}
\left(
1-\frac{C_j}{C_{\max}(x)}
+
\frac{C_j}{C_{\max}(x)}t
\right)dt.
}
\]

Thus each rollout receives two pieces of credit:

\[
\text{advantage}
=
\underbrace{
\text{success-conditioned MaxRL credit}
}_{+\;r_i/K_r}
-
\underbrace{
\text{cost-marginal credit}
}_{-\;\beta_i}.
\]

A successful rollout is pushed up according to its share of the
successful samples, while an expensive rollout is pushed down according
to its Rao–Blackwellized marginal contribution to the finite-\(N\)
cost objective.