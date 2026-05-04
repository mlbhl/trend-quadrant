# Trend Smoothing — Does Adding a Fast Signal Improve 12-Month TSMOM?

**Status**: pre-registration draft / scaffolding

## Research Question

Does augmenting the canonical 12-month time-series momentum (TSMOM) signal with
short-horizon (1–3 month) information yield a statistically significant
improvement in OOS risk-adjusted performance?

The fast component enters with **either sign**:
- `w_fast > 0` → continuation (BBT-style trend acceleration)
- `w_fast = 0` → pure 12M TSMOM baseline (Moskowitz–Ooi–Pedersen 2012)
- `w_fast < 0` → short-term reversal (Lehmann 1990, Jegadeesh 1990)

## Universe & Sample

- Fama–French 49 industry portfolios, value-weighted, 1927-07 ~ 2024-12
- Warm-up: 1927-07 ~ 1976-06 (49y); OOS: 1976-07 ~ 2024-12 (~48y)

## Design (locked in `PRE_REGISTRATION.md`)

- `slow = 12`, `fast ∈ {1,2,3}`, `w_fast ∈ [-1.0, 1.0]` step 0.1 → **63 specs/year**
- Annual walk-forward (expanding train, max in-sample net Sharpe ex 1M Tbill)
- Position: `sign(z) × σ_target / σ_i`, EW across active industries
- Tcost: 20bps round-trip × one-way turnover

## Layout

```
trend.py                         all functions (data / signals / backtest /
                                 walk-forward / stats / tests / plots)

notebooks/
  01_replicate.ipynb               data download + baseline TSMOM sanity
  02_walkforward.ipynb             PRIMARY OOS test (run once)
  03_robustness.ipynb              tcost / freq / train / sub-periods
  04_dynamics.ipynb                best-spec time series + sign test
```

`data/` and `results/` are created at runtime and gitignored.

## Reproducibility

- Pre-registration timestamped on SSRN before any OOS analysis
- All grid searches confined to in-sample window
- `02_walkforward.ipynb` is run exactly once per spec configuration
- Bootstrap seed = 42 (recorded in `PRE_REGISTRATION.md`)
