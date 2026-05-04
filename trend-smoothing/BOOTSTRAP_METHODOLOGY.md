# Bootstrap Methodology — Sharpe-Difference Test

This document details the procedure used by `lw_sharpe_diff_bootstrap()` in
`trend.py` for testing whether one return series has a higher Sharpe ratio
than another. The test follows Ledoit & Wolf (2008) using a Politis–Romano
(1994) stationary block bootstrap and Politis–Wolf (2003) centering.

## Procedure

**Inputs**
- `ret_a`, `ret_b`: monthly return series (same date index)
- `mean_block_size = 12` (months — Geometric block-length parameter)
- `n_boot = 10,000`
- `seed = 42`
- `alternative = "greater"` (one-sided H1: SR(a) > SR(b))

**Pre-processing**

Pair `ret_a` and `ret_b` by date and drop rows where either is NaN. Convert
to numpy arrays `a`, `b` of common length `n`.

---

### Step 1 — Observed statistic

Compute annualized Sharpe ratios and their difference:

```
SR_a   = mean(a) / std(a) · √12
SR_b   = mean(b) / std(b) · √12
Δ_obs  = SR_a − SR_b
```

---

### Step 2 — Politis–Romano stationary bootstrap (n_boot iterations)

Block lengths are not fixed; they are drawn from `Geometric(p)` where
`p = 1 / mean_block_size`. This produces *stationary* resampled series
(in expectation) and is the variant Ledoit–Wolf (2008) recommend.

For each iteration `i ∈ {1, …, n_boot}`:

  **a)** Draw `n` Bernoulli(p) flags `is_new[t]` (with `is_new[0] = True`):
  - `is_new[t] = True`  ⇒ start a new block at position `t`
  - `is_new[t] = False` ⇒ continue the current block

  **b)** Draw `n` candidate start positions `new_starts[t] ∼ Uniform{0, …, n−1}`.

  **c)** Construct the index sequence by walking forward:
  - At each `is_new` position: pick a fresh random start
  - Otherwise: `idx[t] = (idx[t−1] + 1) mod n` (circular continuation)

  Vectorized form:
  ```
  block_id    = cumsum(is_new) − 1
  block_starts_compact = new_starts[is_new]
  starts_per_step = block_starts_compact[block_id]
  offset[t]   = t − last_position_where_is_new[t]
  idx[t]      = (starts_per_step[t] + offset[t]) mod n
  ```

  **d)** Apply the **same** indices to both series (paired resampling — preserves
  contemporaneous correlation between a and b):
  ```
  a* = a[idx]
  b* = b[idx]
  ```

  **e)** Compute bootstrap Sharpe difference:
  ```
  SR_a* = mean(a*) / std(a*) · √12
  SR_b* = mean(b*) / std(b*) · √12
  Δ*[i] = SR_a* − SR_b*
  ```

This yields a length-`n_boot` array of bootstrap statistics `{Δ*[i]}`.

---

### Step 3 — Centering for null hypothesis (Politis–Wolf 2003)

The raw bootstrap distribution is centered approximately at `Δ_obs`. To test
`H0: Δ = 0`, we shift it to be centered at zero:

```
Δ*_centered[i] = Δ*[i] − Δ_obs
```

Now `{Δ*_centered}` represents the distribution of the test statistic
**under H0**.

---

### Step 4 — One-sided p-value

The p-value is the probability under H0 of observing a Sharpe-difference at
least as large as `Δ_obs`:

```
p = (1 / n_boot) · |{ i : Δ*_centered[i] ≥ Δ_obs }|
  = mean(Δ*_centered ≥ Δ_obs)
```

For two-sided: `mean(|Δ*_centered| ≥ |Δ_obs|)`.
For "less" alternative: `mean(Δ*_centered ≤ Δ_obs)`.

Reject H0 at significance α if `p < α`.

#### Why threshold `Δ_obs` (not `0`)?

`Δ*_centered` is constructed to be approximately centered at 0; comparing
`mean(Δ*_centered ≥ 0)` would yield ~0.5 for any data — not a test. We need
the *tail probability* of the H0 distribution **at the observed value**:

> "If H0 were true, how often would we see a difference at least as
> extreme as Δ_obs?"

That is, we ask whether `Δ_obs` falls in the tail of the H0 distribution.
The smaller this probability, the less likely the observed effect is
attributable to chance alone under H0.

#### Why threshold sign `≥` (not `≤`)?

Pre-registered alternative is one-sided "greater" (`H1: Δ > 0`); the test
asks whether the rolling-best protocol *improves* over the baseline. The
right-tail probability `mean(Δ*_centered ≥ Δ_obs)` is the relevant quantity.

`mean(Δ*_centered ≤ Δ_obs)` would test the opposite direction
(`H1: Δ < 0`, deterioration), which contradicts the pre-registered H1.

#### What a small p-value means (and does not)

A small p-value (e.g. 0.01) means:

> Under H0 (true Δ = 0), the data we observed (or something more extreme)
> would occur with probability 0.01.

This is a statement about the **data given H0**, not about Δ given the data.
It does not say "Δ > 0 with 99% probability"; that is a Bayesian credible
statement, not a frequentist one. The correct interpretation is:

> "We reject H0 at the 5% level; the data are consistent with H1: Δ > 0."

---

### Step 5 — 95% confidence interval for Δ

Percentile interval on the **uncentered** bootstrap distribution
(estimating the true distribution of Δ̂, not the H0 distribution):

```
CI_lower = percentile(Δ*, 2.5)
CI_upper = percentile(Δ*, 97.5)
```

If the CI excludes 0, the test rejects at α = 0.05 (two-sided).

---

## Key design choices

| Choice | Value | Rationale |
|---|---|---|
| Block length | Geometric mean = 12 | Captures one full annual cycle in TSMOM lookback |
| Resampling | Paired (a, b together) | Preserves contemporaneous correlation; otherwise the variance of Δ would be inflated |
| Centering | Subtract `Δ_obs` | Politis–Wolf (2003) — gives valid H0 distribution without HAC variance estimator |
| Statistic | Raw `Δ` (non-studentized) | Simpler than HAC-studentized; both reported in robustness section |
| n_boot | 10,000 | Standard; gives p-value resolution ~1e-4 |
| seed | 42 | Locked in pre-registration for exact reproducibility |

## Block-length distribution (sanity check)

A representative draw with `n = 596`, `mean_block_size = 12`:

```
n_blocks         = 41
mean length      = 14.5  (vs target 12)
median length    = 11
max length       = 103
```

The Geometric tail produces occasional long blocks (~100 months), which is
correct behavior — these capture rare-but-important long-range dependence.

## References

- **Politis, D. N., & Romano, J. P. (1994).** "The Stationary Bootstrap."
  *Journal of the American Statistical Association*, 89(428), 1303–1313.
- **Politis, D. N., & Wolf, M. (2003).** "Subsampling vs. bootstrap." *etc.*
- **Ledoit, O., & Wolf, M. (2008).** "Robust performance hypothesis testing
  with the Sharpe ratio." *Journal of Empirical Finance*, 15(5), 850–859.
