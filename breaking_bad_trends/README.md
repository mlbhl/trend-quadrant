# Breaking Bad Trends — Replication

Standalone replication of:

> Goulding, Harvey & Mazzoleni (2023). *Breaking Bad Trends.* Financial Analysts Journal.

## Scope

Per-asset time-series momentum (TSMOM) with 4-state cycle-conditional blending. This is an **independent research track**, separate from:

- `../utils.py` — existing cross-sectional quadrant / quantile framework
- `../disagreement/` — K_eff voter-ensemble paper

Eventual plan: once this standalone replication is validated, merge the TSMOM state machinery with the cross-sectional framework (Path B step 2).

## Files

| File | Description |
|---|---|
| `Breaking Bad Trends.pdf` | Source paper |
| `tsmom.py` | Replication module (signals, states, mixing params, backtest) |
| `tsmom_backtest.ipynb` | Research notebook (yfinance proxies, 1998–2024, frictionless) |
| `tsmom_realtime.ipynb` | Production notebook — driven by `config.py`, tcost-aware |
| `config.py` | Production parameters: universe, lookbacks, vol target, tcost |

## Method summary

**Signals (eq. 1–2):** slow/fast lookback means of prior monthly excess returns.
`x_slow(m) = mean(r_{m-1..m-k_slow})`, `x_fast(m) = mean(r_{m-1..m-k_fast})`. Main spec `k_slow=12, k_fast=2`.

**4-state classification (eq. 4):**

| State | slow | fast |
|---|---|---|
| Bull | ≥ 0 | ≥ 0 |
| Correction | ≥ 0 | < 0 |
| Bear | < 0 | < 0 |
| Rebound | < 0 | ≥ 0 |

Turning points ≡ Correction ∪ Rebound (sign disagreement).

**Static strategy (eq. 5):** `r_static = sign(x_slow) · r`. Ignores fast entirely. Bull and Correction both get +1; Bear and Rebound both get −1.

**Dynamic strategy (eq. 7):**
```
r_dyn = r,                                              if Bull
      = -r,                                             if Bear
      = (1 - a_co)·r_slow + a_co·r_fast,                if Correction
      = (1 - a_re)·r_slow + a_re·r_fast,                if Rebound
```
Bull/Bear replicate static. Correction/Rebound blend via asset-specific mixing params.

**Mixing parameter estimation (eq. 8–10):** expanding-window estimator from historical post-Correction/Rebound returns, clamped to `[0, 1]`, shrinking to 0.5 when noisy. Updated every 30 months; minimum 12 obs per state.

**Vol scaling:** two orthogonal axes — *mode* × *level*.

*Mode* (`vol_mode`):
- `ex_post` — full-period realized vol rescaling (paper Figures 2, 4). Not tradeable, comparison only.
- `realtime` — rolling 36-month target-vol with `shift(1)` (tradeable).

*Level* (`scale_level`):
- `portfolio` — **paper default.** Raw per-asset trend returns → equal-weight 1/N → single scalar (or rolling series) rescales the portfolio to `target_vol`. Risk contribution is dominated by high-vol assets (commodities, equity) since each enters the portfolio at its natural vol.
- `asset` — **risk-parity style.** Each asset's trend returns are first scaled to `target_vol` independently, then equal-weighted. Each asset contributes roughly equal ex-ante risk. Bond ETFs become meaningful contributors; commodity dominance is attenuated. Portfolio realized vol under this mode is typically *below* `target_vol` due to diversification — no second portfolio-level rescale is applied by default.

**No look-ahead:** signals use `.rolling().mean().shift(1)`; mixing params estimated from history only.

**Weight timing convention:** `wt_dyn` (and all weight DataFrames) are indexed by month-end date. The index date is the month **whose return the weight is applied to**, not the month the signal was computed from. Concretely, for row `2026-03-31`:

| Component | Value at row 2026-03-31 | Based on data through |
|---|---|---|
| `x_slow` signal | `rolling(12).mean().shift(1)` | **2025-02** (shift pushes 2026-02 rolling mean → 2026-03 row) |
| `x_fast` signal | `rolling(2).mean().shift(1)` | **2025-02** |
| Position (`pos`) | `sign(x_slow)` or dynamic blend | 2025-02 signals |
| Vol scale (`s`) | `rolling(36).std()` → `.shift(1)` | **2026-02** (portfolio vol through Feb, shifted to Mar row) |
| **Weight** | `pos / N_t × s` | Determined at **end of Feb** |
| **Applied to** | `returns.loc[2026-03-31]` | **March return** (Feb-end → Mar-end price change) |

Interpretation: weight at index `m` = position entered at start of month `m` (based on end-of-`m−1` information), held through month `m`, earning month `m`'s return. The last row in a live run reflects the **current month's position** (partial return if mid-month). Note: next month's signal and position are already computable from current data — the rolling mean at the last row (before `shift(1)`) incorporates up to the current partial month's return (e.g., at 2026-04-30 it uses 2025-05 through 2026-04 including the partial April return), which after `shift(1)` becomes the May signal. However, the backtest framework only produces `w × r` pairs, so since no next-month return row exists in the DataFrame, the next month's weight simply has nowhere to appear.

**Transaction costs & turnover:**
- Per-asset notional weights `w_i(t) = scale(t) · (1/N_t) · pos_i(t)` are built via `build_weights()`, where `N_t` is the number of assets with valid position AND return at time t (assets not yet available get weight 0, not 1/N_total). Consistent with the scaled portfolio returns (`Σ w_i r_i ≡ port_scaled`).
- **One-way turnover:** `turnover(t) = 0.5 · Σ_i |Δw_i(t)|`. A full flip of a 1/N_t weight contributes `1/N_t` per month.
- **Round-trip cost:** `TCOST_BPS` in `config.py` is interpreted as round-trip bps applied to one-way turnover: `cost(t) = turnover(t) · TCOST_BPS / 1e4`. A full 1/N flip at 20 bps round-trip costs `(1/N) · 20 bps`.
- `tsmom_backtest.ipynb` passes `tcost_bps=0` (frictionless research). `tsmom_realtime.ipynb` uses `cfg.TCOST_BPS` and reports annualized one-way turnover and tcost drag per strategy.

## Notebook sections

1. Data — yfinance proxies (11 equity, 3 bond, 7 commodity ETFs)
2. Backtest — full-sample run at 10% vol target
3. Cumulative returns (Figure E.1 equivalent)
4. State decomposition — portfolio-level (state mode) + asset-level (per-asset states)
5. Turning point frequency vs performance (Figure 2)
6. Sub-period comparison (Table E.1)
7. Mixing parameter time series — cross-asset mean + per-asset (user-selected tickers)
8. State frequency heatmap
9. Vol scaling comparison (raw / ex-post / realtime)
10. Fast lookback sensitivity (1M / 2M / 3M)
11. Asset positions & portfolio weights

## Usage

```python
from tsmom import run_backtest, print_summary

result = run_backtest(
    returns,                # monthly excess returns (DataFrame)
    k_slow=12, k_fast=2,
    target_vol=0.10,
    vol_mode="ex_post",     # or "realtime"
    scale_level="portfolio",  # or "asset" (risk-parity style)
    tcost_bps=20.0,         # round-trip bps; 0 = frictionless
)
print_summary(result)
```

`result` keys:
- `port_static`, `port_dynamic` — portfolio return series (**net of tcost**)
- `port_static_gross`, `port_dynamic_gross` — pre-tcost series
- `pos_static`, `pos_dynamic` — per-asset positions in [-1, +1]
- `weights_static`, `weights_dynamic` — per-asset notional weights on eval window
- `turnover_static`, `turnover_dynamic` — one-way monthly turnover series
- `states` — per-asset state labels
- `a_co`, `a_re` — per-asset mixing parameter time series
- `stats` — annualized return/vol/Sharpe/MDD + `{static,dynamic}_turnover` (ann. 1-way) + `tcost_bps`
- `decomp_static`, `decomp_dynamic` — portfolio-level state decomposition (via state mode)
- `decomp_static_asset`, `decomp_dynamic_asset` — per-asset decomposition (dict keyed by ticker)

## Deviations from paper

- **Universe:** paper uses 43 Barchart futures (1990–2022); we use 21 yfinance ETF proxies (1998–2024). ETF tracking error and later start date affect level comparability.
- **Total return, not excess-of-cash:** paper footnote 10 defines trailing k-month return as average monthly returns *in excess of cash*. Our notebook feeds `monthly_prices.pct_change()` (total returns) directly. Impact:
  - Signal signs rarely flip (cash rate vs. monthly-return scale is small), so state classification is largely preserved
  - Mixing parameter estimates (`a_co`, `a_re`) slightly biased by `E[r|state]` being shifted up by the cash rate
  - Reported returns and Sharpes are **level-biased upward** in high-rate periods (1998–2008, 2022–2024); negligible in ZIRP (2009–2021)
  - Not a look-ahead issue — purely a level/definition deviation
- **Portfolio decomposition:** paper labels state per-asset and buckets that asset's return. Our portfolio-level `decomp_*` uses monthly state mode across assets as a coarse approximation; per-asset `decomp_*_asset` matches the paper's asset-level aggregation.
- **Mixing param update cadence:** paper details in Appendix C; we use 30-month expanding-window refresh with 12-obs minimum per state.

## Implementation audit (look-ahead / logic check)

Last audited: 2026-04-11. All checks passed.

- **Signals:** `rolling(k).mean().shift(1)` at row m yields `(r_{m-1}+...+r_{m-k})/k`, matching eq.(1)
- **Static / dynamic returns:** position from strictly-historical signals, multiplied by realized row-m return
- **Mixing param estimation:** expanding window uses `iloc[:i]` — strictly before current date
- **Realtime vol target:** `rolling().std()` then `scale.shift(1)` — no look-ahead
- **Ex-post vol scale:** intentionally uses full-sample std; by design, for Figure reproduction only, not tradeable
- No logic bugs detected in state classification, mixing formula (eq. 8–10), or dynamic blending (eq. 7)

## Known extreme values

Some assets end up with `a_co=0` or `a_re=1` after clipping — this is the estimator's intended behavior when historical post-turning-point returns strongly support one direction. It means "fully lean into slow/fast" for that asset. Inspect `result["a_co"]` / `result["a_re"]` time series to verify extremes are not driven by sparse samples.
