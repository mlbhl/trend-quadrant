# Pre-Registration — Trend Smoothing on FF49

**Status**: DRAFT (not yet timestamped)
**Author**: Byounghyo Lim
**Intended timestamp venue**: SSRN
**Lock date**: TBD (must be before any OOS computation)

---

## 1. Research Question

Does augmenting the canonical 12-month time-series momentum signal with a
short-horizon (1, 2, or 3-month) component — entering with either positive
weight (continuation) or negative weight (reversal) — yield a statistically
significant improvement in OOS risk-adjusted performance over the static 12M
TSMOM baseline?

## 2. Universe and Sample

- **Universe**: Fama–French 49 industry portfolios, value-weighted, monthly
- **Source**: Ken French Data Library
- **Raw data range**: 1926-07 ~ 2026-02 (1196 monthly observations)
- **Risk-free rate**: 1-month Treasury bill, monthly, from Ken French DL (RF column)
- **Staggered universe**: 9 of 49 industries (Soda, Hlth, Rubbr, FabPr, Guns,
  Gold, Softw, Paper, plus one) start later than 1926-07; all 49 are alive
  by 1969-07. Industries with insufficient signal/vol history are excluded
  from the EW portfolio that month (handled in `strategy.py`).
- **Effective backtest start**: 1929-07 (after 36-month vol-scaling warm-up
  for industries with 1926-07 first-valid)
- **Train (warm-up) for selection**: 1929-07 ~ 1976-06 (47 years)
- **OOS evaluation**: 1976-07 ~ 2026-02 (49.7 years)
- **Final spec re-estimation**: June 2025; last evaluated returns:
  July 2025 ~ February 2026 (8 months partial)

If raw data extent advances (e.g. data refreshed before lock), endpoints
update accordingly; the relative protocol (49y warm-up cutoff at 1976-06,
annual re-estimation in June) is what is locked.

## 3. Signal Construction (sign-smoothing)

For industry `i` at month `t`:

```
x_slow_{i,t} = ∏_{s=t-12}^{t-1} (1 + r_{i,s}) − 1     # 12-month compound
x_fast_{i,t} = ∏_{s=t-k}^{t-1}  (1 + r_{i,s}) − 1     # k ∈ {1, 2, 3}

s_{i,t} = w_slow · sign(x_slow_{i,t}) + w_fast · sign(x_fast_{i,t})
```

where `w_slow = 1.0` (anchor) and `w_fast ∈ G_w`. The composite score `s` ∈
{±(1+|w_fast|), ±(1−|w_fast|)} encodes the BBT 4-state regime as relative
conviction (full conviction when slow & fast agree, reduced conviction when
they conflict, sign-flipped when `w_fast < 0`). Only past returns are used.

## 4. Position and Portfolio

### 4a. Per-asset position (score × asset vol-scaling)

```
σ_{i,t} = std(r_{i,t-36 : t-1}) × √12              # 36M annualized vol
p_{i,t} = s_{i,t} · σ_asset_target / σ_{i,t}        # σ_asset_target = 10%
```

Industries lacking 12 prior months of returns or 36 prior months of vol are
excluded that month.

### 4b. Equal-weight portfolio aggregation

```
w_{i,t}^pre = p_{i,t} / N_active(t)
R_p,t^pre   = Σ_i w_{i,t}^pre · r_{i,t}
```

### 4c. Portfolio vol targeting (multiplicative leverage)

Realized portfolio vol is computed on the pre-leverage return series:

```
σ_p,t = std(R_p,t-36 : t-1^pre) × √12
ℓ_t   = clip(σ_port_target / σ_p,t, upper = 3.0)        # σ_port_target = 10%
```

Final positions and returns:

```
w_{i,t} = w_{i,t}^pre · ℓ_t
R_p,t   = R_p,t^pre · ℓ_t
```

Leverage cap of 3.0× protects against blow-ups in low-vol periods.

## 5. Spec Grid (locked)

```
G_fast    = {1, 2, 3}
G_w       = {-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
              0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9, 1.0}
                                                                 # 21 values
```

Total specs per re-estimation date: 3 × 21 = **63**.
Baseline (nested): `w_fast = 0` (pure 12M TSMOM). When `w_fast = 0`, all three
`fast` values produce identical output and are counted as one effective spec.

## 6. Transaction Cost

- 20 bps round-trip × one-way turnover (0.5 · Σ_i |Δw_{i,t}|)
- Robustness: also report tcost ∈ {0, 10, 40} bps

## 7. Walk-Forward Protocol (two parallel windows)

We run two walk-forward protocols and report each independently:

**Protocol A — Expanding window**
```
Train: 1929-07 ~ June of year T   (all data up to decision date)
```

**Protocol B — Rolling 30-year window**
```
Train: (June of year T − 30y) ~ June of year T
```

Common loop (both protocols):
```
For each year T ∈ {1976, 1977, ..., 2025}:
  For each spec s ∈ 63 specs:
    Compute monthly net returns over train window
    Compute net Sharpe excess of 1M Tbill
  s*(T) = argmax_s Sharpe_s(train)
  Apply s*(T) to OOS period: July T ~ June (T+1)
                              (final year ends at 2026-02 partial)
```

Each protocol yields its own OOS monthly return series (~597 months). Both
are evaluated against the same static baseline.

## 8. Hypotheses and Tests

### Primary (two parallel tests, reported independently)

For each protocol P ∈ {Expanding, Rolling 30y}:

```
H0_P: SR_OOS(rolling-best, P) = SR_OOS(static w_fast = 0)
H1_P: SR_OOS(rolling-best, P) > SR_OOS(static w_fast = 0)
```

- **Test**: Ledoit–Wolf (2008) Sharpe-difference test, implemented via
  Politis–Romano (1994) stationary block bootstrap (paired resampling of
  (a,b) returns to preserve contemporaneous correlation).
- **Mean block length**: 12 months (Geometric block-length distribution)
- **Bootstrap iterations**: 10,000
- **Centering**: bootstrap distribution centered at observed Δ (Politis–Wolf
  2003 convention) to enforce H0 in test statistic.
- **Random seed**: 42
- **Significance**: one-sided p < 0.05
- **Statistic**: raw Δ Sharpe (non-studentized). Studentized variant with
  HAC-corrected standard error (Andrews 1991 bandwidth) reported as
  robustness check; p-values from both are reported.

The two tests are reported as **separate findings** (not corrected for joint
testing). Each protocol stands on its own; rejecting H0_A but not H0_B (or
vice versa) is interpreted descriptively, not via family-wise control. This
choice is acknowledged honestly in the paper text — we are not claiming
"adding fast helps under either protocol with FWER control"; we are reporting
each protocol's result on its own merits.

### Secondary

For each protocol P ∈ {Expanding, Rolling 30y}, applied to its best-spec
history:

```
H0: P(continuation) = P(reversal) = P(neutral) = 1/3

where:
  continuation = best w_fast > 0
  reversal     = best w_fast < 0
  neutral      = best w_fast = 0   (pure 12M TSMOM is optimal that year)
```

- **Test**: multinomial χ² goodness-of-fit (df = 2)
- **Significance**: p < 0.05
- Reports which mode (continuation / reversal / neutral) dominates each
  protocol's OOS history. Reported per protocol; no joint-test correction.

### Tertiary (descriptive — labeled exploratory)

- Time series of best `(fast, w_fast)` over 1976–2024
- Transition matrix entropy vs uniform baseline
- Autocorrelation of integer `fast` and continuous `w_fast`

### Exploratory (labeled exploratory; NO inferential claim)

- Best `(fast, w_fast)` regressed on contemporaneous covariates:
  - NBER recession dummy
  - VIX (1990+)
  - Cross-sectional dispersion of FF49 returns
  - Aggregate market vol
- Multinomial logistic on `fast`, OLS on `w_fast`

## 9. Robustness Checks (locked)

| Dimension | Variants |
|---|---|
| Tcost | 0, 10, 20 (main), 40 bps |
| Re-estimation freq | annual (main), quarterly, triennial |
| Sub-period split | full (main), 1977–1999, 2000–2024 |
| Asset / port vol target | 10% / 10% (main), 10% / 8%, 10% / 12% |

Each robustness variant reports the same primary test result; all reported in a
single robustness table. **No secondary spec selection** based on robustness
outcomes.

## 10. Reporting Commitment

- Primary test result is reported regardless of outcome (positive, null, or
  negative).
- All 63 in-sample Sharpe values for each year are made available as a
  supplementary CSV.
- All code, data-fetching scripts, and random seeds are released on GitHub
  upon submission.

## 11. Anti-Snooping Commitments

- The OOS pipeline (`src/walkforward.py`) is run **exactly once** for each
  configuration in this document. Re-runs after seeing OOS results are
  forbidden and would invalidate this pre-registration.
- No post-hoc additions to the grid `G_fast` or `G_w` are permitted.
- No post-hoc changes to the selection criterion are permitted.

## 12. Computational Reproducibility

- Python 3.11
- Exact pinned dependencies in `requirements.txt`, generated via:
  ```
  pip freeze > requirements.txt
  ```
  at the lock date. Replicators run `pip install -r requirements.txt` in a
  fresh virtualenv to reproduce.
- Random seeds: bootstrap = 42, all other deterministic
- Single-machine deterministic execution; no GPU / threading non-determinism

---

**Sign-off**:

I commit to publishing the primary OOS test result as defined above,
regardless of whether it supports my prior hypothesis.

— Byounghyo Lim
Date of lock: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
SSRN timestamp ID: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
