"""Trend smoothing on FF49 — single module.

Sections (in order):
  1. Data loaders          — fetch_ff49(), fetch_tbill_1m()
  2. Signals               — compute_x(), composite_z()
  3. Strategy              — vol_scaled_position(), portfolio_return()
  4. Backtest              — backtest_spec(spec) -> monthly net returns
  5. Selection             — best_spec_in_sample()
  6. Walk-forward          — walkforward_oos()
  7. Stats                 — sharpe_ex_rf(), drawdown(), summarize()
  8. Tests                 — lw_sharpe_diff_bootstrap(), hansen_spa()
  9. Plots                 — plot_equity(), plot_best_spec_ts()

All functions are stateless. Configuration (universe, sample, grid) lives in
PRE_REGISTRATION.md and is passed as arguments.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# 1. Data loaders
# ---------------------------------------------------------------------------

FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FF49_URL = f"{FF_BASE}/49_Industry_Portfolios_CSV.zip"
FF_FACTORS_URL = f"{FF_BASE}/F-F_Research_Data_Factors_CSV.zip"


def _download_zipped_csv(url: str, cache_dir: Path | str = "data") -> str:
    """Download a Ken French CSV zip, cache it, return raw CSV text."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    fname = cache / Path(url).name
    if not fname.exists():
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        fname.write_bytes(r.content)
    with zipfile.ZipFile(fname) as zf:
        inner = zf.namelist()[0]
        return zf.read(inner).decode("latin-1")


def _parse_ff_block(csv_text: str, block: int = 0) -> pd.DataFrame:
    """Parse a block from a Ken French CSV.

    Ken French CSVs contain free-text preamble + multiple tables (monthly,
    annual, value-weighted, equal-weighted) separated by blank lines. We
    locate "true" tables by looking for a header line immediately followed by
    rows starting with YYYYMM.

    block=0 returns the first such table (typically value-weighted monthly).
    """
    import re
    lines = csv_text.splitlines()
    blocks: list[tuple[int, int]] = []  # (header_idx, end_idx_exclusive)
    i = 0
    while i < len(lines) - 1:
        nxt = lines[i + 1].lstrip()
        # Header is the line immediately above a YYYYMM row
        if re.match(r"^\d{6}\s*,", nxt):
            header_idx = i
            j = i + 1
            while j < len(lines) and re.match(r"^\d{4,6}\s*,", lines[j].lstrip()):
                j += 1
            blocks.append((header_idx, j))
            i = j
        else:
            i += 1

    if not blocks:
        raise ValueError("No data blocks found in FF CSV")

    h_idx, end_idx = blocks[block]
    chunk = "\n".join(lines[h_idx:end_idx])
    df = pd.read_csv(io.StringIO(chunk))
    # First column is the date
    df = df.rename(columns={df.columns[0]: "date"})
    df = df[df["date"].astype(str).str.match(r"^\s*\d{6}\s*$")].copy()
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(),
                                 format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    return df


def fetch_ff49(cache_dir: Path | str = "data") -> pd.DataFrame:
    """FF49 value-weighted monthly returns (decimal). Replaces -99.99 / -999 sentinels with NaN."""
    csv = _download_zipped_csv(FF49_URL, cache_dir=cache_dir)
    df = _parse_ff_block(csv, block=0)   # block 0 = value-weighted
    df = df.mask(df < -0.999)            # FF sentinel = -99.99% / -999.99%
    df.columns = [c.strip() for c in df.columns]
    return df


def fetch_tbill_1m(cache_dir: Path | str = "data") -> pd.Series:
    """1-month Treasury bill (RF column from F-F factors), monthly decimal."""
    csv = _download_zipped_csv(FF_FACTORS_URL, cache_dir=cache_dir)
    df = _parse_ff_block(csv, block=0)
    return df["RF"].rename("rf")


# ---------------------------------------------------------------------------
# 2. Signals  (sign-smoothing)
# ---------------------------------------------------------------------------

def compute_x(returns: pd.DataFrame, k: int) -> pd.DataFrame:
    """k-month cumulative compound return per asset, evaluated at month t using
    returns from t-k to t-1 (no look-ahead).
    """
    cum = lambda x: (1 + x).prod() - 1
    return returns.rolling(k).apply(cum, raw=True).shift(1)


def composite_score(x_slow: pd.DataFrame, x_fast: pd.DataFrame,
                     w_slow: float, w_fast: float) -> pd.DataFrame:
    """Sign-smoothed score: s = w_slow · sign(x_slow) + w_fast · sign(x_fast).

    Range: ±(w_slow + |w_fast|) (full conviction) or ±(w_slow − |w_fast|)
    (regime conflict). Encodes BBT 4-state regime as relative conviction.
    """
    return w_slow * np.sign(x_slow) + w_fast * np.sign(x_fast)


# ---------------------------------------------------------------------------
# 3. Strategy  (asset vol-scale → EW → portfolio vol-target)
# ---------------------------------------------------------------------------

def realized_vol(returns: pd.DataFrame | pd.Series,
                 lookback: int = 36) -> pd.DataFrame | pd.Series:
    """Annualized realized vol (rolling std × √12), evaluated at t using
    returns t-lookback to t-1.
    """
    return returns.rolling(lookback).std().shift(1) * np.sqrt(12)


def conviction_position(score: pd.DataFrame, sigma_ann: pd.DataFrame,
                         asset_target: float = 0.10) -> pd.DataFrame:
    """Per-asset position = score × asset_target / σ_i. Direction & magnitude
    both come from `score` (no extra sign() applied)."""
    return score * (asset_target / sigma_ann)


def portfolio_weights_pre(positions: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight portfolio weights = p_i / N_active. Zero where p NaN."""
    n_active = positions.notna().sum(axis=1).replace(0, np.nan)
    return positions.div(n_active, axis=0).fillna(0.0)


def apply_portfolio_vol_target(weights_pre: pd.DataFrame,
                                returns: pd.DataFrame, *,
                                target: float = 0.10, lookback: int = 36,
                                lev_cap: float = 3.0
                                ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Multiplicative leverage to hit annualized portfolio vol target.

    Returns (weights_scaled, gross_return_scaled, leverage_series).
    """
    gross_pre = (weights_pre * returns.fillna(0)).sum(axis=1, min_count=1)
    sigma_p = realized_vol(gross_pre, lookback)
    lev = (target / sigma_p.replace(0, np.nan)).clip(upper=lev_cap)
    weights = weights_pre.mul(lev.fillna(0), axis=0)
    gross = gross_pre * lev
    return weights, gross, lev


def turnover_from_weights(weights: pd.DataFrame) -> pd.Series:
    """One-way turnover: 0.5 · Σ_i |Δw_{i,t}|. Entry from cash on first row."""
    dw = weights.diff()
    if len(weights) > 0:
        dw.iloc[0] = weights.iloc[0]
    return 0.5 * dw.abs().sum(axis=1)


# ---------------------------------------------------------------------------
# 4. Backtest
# ---------------------------------------------------------------------------

def backtest_spec(returns: pd.DataFrame, rf: pd.Series | None = None, *,
                   slow: int = 12, fast: int = 2,
                   w_slow: float = 1.0, w_fast: float = 0.0,
                   asset_vol_target: float = 0.10, asset_vol_lookback: int = 36,
                   port_vol_target: float = 0.10, port_vol_lookback: int = 36,
                   port_lev_cap: float = 3.0,
                   tcost_bps: float = 20.0) -> dict:
    """Run one (slow, fast, w_slow, w_fast) spec end-to-end.

    Pipeline:
      excess returns → x_slow, x_fast → score (sign-smoothed)
      → per-asset position = score × σ_asset_tgt / σ_i
      → EW portfolio weights = p_i / N_active
      → portfolio vol-target leverage ℓ_t = clip(σ_port_tgt / σ_p, ≤ lev_cap)
      → final weights, gross & net returns, turnover, cost
    """
    if rf is not None:
        returns = returns.sub(rf, axis=0)

    x_slow = compute_x(returns, slow)
    x_fast = compute_x(returns, fast)
    score = composite_score(x_slow, x_fast, w_slow, w_fast)
    sigma_i = realized_vol(returns, asset_vol_lookback)
    pos = conviction_position(score, sigma_i, asset_target=asset_vol_target)

    valid = pos.dropna(how="all").index
    pos = pos.loc[valid].copy()
    rets = returns.loc[valid]

    weights_pre = portfolio_weights_pre(pos)
    weights, gross, lev = apply_portfolio_vol_target(
        weights_pre, rets,
        target=port_vol_target, lookback=port_vol_lookback,
        lev_cap=port_lev_cap,
    )

    to = turnover_from_weights(weights)
    cost = to * tcost_bps / 1e4
    net = gross - cost

    return {
        "gross_return": gross,
        "net_return": net,
        "weights": weights,
        "weights_pre": weights_pre,
        "score": score.loc[valid],
        "leverage": lev,
        "turnover": to,
        "cost": cost,
        "spec": {"slow": slow, "fast": fast,
                  "w_slow": w_slow, "w_fast": w_fast,
                  "asset_vol_target": asset_vol_target,
                  "asset_vol_lookback": asset_vol_lookback,
                  "port_vol_target": port_vol_target,
                  "port_vol_lookback": port_vol_lookback,
                  "port_lev_cap": port_lev_cap,
                  "tcost_bps": tcost_bps,
                  "excess_of_rf": rf is not None},
    }


# ---------------------------------------------------------------------------
# 7. Stats   (defined before §5 because selection uses sharpe)
# ---------------------------------------------------------------------------

def sharpe_annualized(returns: pd.Series, freq: int = 12) -> float:
    """Mean / std × √freq. NaN if n < 2 or std = 0."""
    s = returns.dropna()
    if len(s) < 2 or s.std() == 0:
        return np.nan
    return s.mean() / s.std() * np.sqrt(freq)


def drawdown(returns: pd.Series) -> pd.Series:
    """Drawdown series from peak equity."""
    eq = (1 + returns.fillna(0)).cumprod()
    return eq / eq.cummax() - 1


def summarize(returns: pd.Series, freq: int = 12) -> pd.Series:
    """CAGR, Vol, Sharpe, MDD, HitRate, Months."""
    s = returns.dropna()
    n = len(s)
    if n == 0:
        return pd.Series({"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan,
                          "MDD": np.nan, "HitRate": np.nan, "Months": 0})
    cagr = (1 + s).prod() ** (freq / n) - 1
    vol = s.std() * np.sqrt(freq)
    return pd.Series({
        "CAGR": cagr, "Vol": vol,
        "Sharpe": sharpe_annualized(s, freq=freq),
        "MDD": drawdown(s).min(),
        "HitRate": (s > 0).mean(),
        "Months": n,
    })


# ---------------------------------------------------------------------------
# 5. Selection
# ---------------------------------------------------------------------------

# Pre-registered grid (LOCKED — see PRE_REGISTRATION.md §5)
GRID_SLOW = (12,)
GRID_FAST = (1, 2, 3)
GRID_W_FAST = tuple(round(i * 0.1, 2) for i in range(-10, 11))   # -1.0..+1.0, no -0.0


def all_specs() -> list[dict]:
    """All 3 × 21 = 63 pre-registered specs (w_slow=1.0 anchor)."""
    return [
        {"slow": s, "fast": f, "w_slow": 1.0, "w_fast": float(wf)}
        for s in GRID_SLOW for f in GRID_FAST for wf in GRID_W_FAST
    ]


def spec_label(spec: dict) -> str:
    return f"fast={spec['fast']},wf={spec['w_fast']:+.1f}"


def parse_spec_label(label: str) -> dict:
    parts = label.split(",")
    return {"slow": 12, "w_slow": 1.0,
            "fast": int(parts[0].split("=")[1]),
            "w_fast": float(parts[1].split("=")[1])}


def grid_backtest(returns: pd.DataFrame, rf: pd.Series, *,
                   asset_vol_target: float = 0.10, asset_vol_lookback: int = 36,
                   port_vol_target: float = 0.10, port_vol_lookback: int = 36,
                   port_lev_cap: float = 3.0,
                   tcost_bps: float = 20.0) -> pd.DataFrame:
    """Run all 63 specs once on the full sample.

    Returns DataFrame: index = date, columns = spec_label, values = net return.
    """
    out: dict[str, pd.Series] = {}
    for s in all_specs():
        res = backtest_spec(
            returns, rf=rf,
            slow=s["slow"], fast=s["fast"],
            w_slow=s["w_slow"], w_fast=s["w_fast"],
            asset_vol_target=asset_vol_target,
            asset_vol_lookback=asset_vol_lookback,
            port_vol_target=port_vol_target,
            port_vol_lookback=port_vol_lookback,
            port_lev_cap=port_lev_cap,
            tcost_bps=tcost_bps,
        )
        out[spec_label(s)] = res["net_return"]
    return pd.DataFrame(out)


def best_spec_in_sample(grid_returns: pd.DataFrame,
                         train_end: pd.Timestamp) -> tuple[str, pd.Series]:
    """Argmax in-sample net Sharpe on returns up to train_end (inclusive).

    Returns (best_label, all_sharpes).
    """
    train = grid_returns.loc[:train_end]
    sharpes = train.apply(lambda c: sharpe_annualized(c, freq=12))
    return sharpes.idxmax(), sharpes


# ---------------------------------------------------------------------------
# 6. Walk-forward
# ---------------------------------------------------------------------------

def walkforward_oos(grid_returns: pd.DataFrame, *,
                     warmup_end: str | pd.Timestamp = "1976-06-30",
                     train_window: str | int = "expanding"
                     ) -> tuple[pd.Series, pd.DataFrame]:
    """Annual walk-forward (June month-end re-estimation).

    Args:
        grid_returns: full-sample grid output (date × spec_label).
        warmup_end:   last date of initial training (first decision = warmup_end).
        train_window: "expanding" (use all data ≤ d) or int (rolling N years
                      ending at d).

    For each decision date d ∈ {warmup_end, +1y, +2y, ...} while d < sample_end:
      * Train on grid_returns[start : d] where start = data_start (expanding)
        or d − N years (rolling N).
      * Pick best spec by in-sample net Sharpe.
      * Apply that spec for d < date <= d + 1y.

    Returns:
        oos_returns: monthly OOS net returns (concat of yearly chunks)
        spec_history: DataFrame indexed by decision_date with cols
                      'spec', 'sharpe_train', 'fast', 'w_fast', 'oos_months'
    """
    warmup_end = pd.Timestamp(warmup_end)
    sample_end = grid_returns.index.max()

    decision_dates = []
    d = warmup_end
    while d < sample_end:
        decision_dates.append(d)
        d = d + pd.DateOffset(years=1)

    yearly_oos = []
    history = []
    for d in decision_dates:
        if train_window == "expanding":
            train = grid_returns.loc[:d]
        elif isinstance(train_window, int):
            start = d - pd.DateOffset(years=train_window)
            train = grid_returns.loc[start:d]
        else:
            raise ValueError(f"train_window must be 'expanding' or int years; got {train_window!r}")

        sharpes = train.apply(lambda c: sharpe_annualized(c, freq=12))
        best = sharpes.idxmax()
        next_d = d + pd.DateOffset(years=1)
        chunk = grid_returns.loc[(grid_returns.index > d) &
                                  (grid_returns.index <= next_d), best]
        if len(chunk) == 0:
            continue
        chunk = chunk.rename("rolling_best")
        yearly_oos.append(chunk)

        parsed = parse_spec_label(best)
        history.append({"decision_date": d, "spec": best,
                        "sharpe_train": sharpes[best],
                        "fast": parsed["fast"], "w_fast": parsed["w_fast"],
                        "oos_months": len(chunk)})

    oos_returns = pd.concat(yearly_oos) if yearly_oos else pd.Series(dtype=float)
    spec_history = pd.DataFrame(history).set_index("decision_date")
    return oos_returns, spec_history


# ---------------------------------------------------------------------------
# 8. Tests
# ---------------------------------------------------------------------------

def _stationary_bootstrap_indices(n: int, mean_block_size: int,
                                    rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap indices, length n.

    At each step t > 0, with probability p = 1/mean_block_size start a new
    random block; otherwise continue the previous block (circular).
    """
    p = 1.0 / mean_block_size
    is_new = rng.random(n) < p
    is_new[0] = True
    new_starts = rng.integers(0, n, size=n)
    # offset within current block (0 at each new-start position)
    last_new = np.where(is_new, np.arange(n), -1)
    last_new = np.maximum.accumulate(last_new)
    offset = np.arange(n) - last_new
    # map each step to the start position of its block
    block_id = np.cumsum(is_new) - 1
    starts_compact = new_starts[is_new]
    return (starts_compact[block_id] + offset) % n


def lw_sharpe_diff_bootstrap(ret_a: pd.Series, ret_b: pd.Series, *,
                              mean_block_size: int = 12,
                              n_boot: int = 10_000,
                              seed: int = 42,
                              alternative: str = "greater",
                              freq: int = 12) -> dict:
    """Ledoit-Wolf (2008) style Sharpe-difference test via Politis-Romano (1994)
    stationary block bootstrap.

    H0: SR(a) − SR(b) = 0
    H1: SR(a) − SR(b) > 0  (default; "greater" alternative)

    The bootstrap is **paired** (same indices applied to both series) to
    preserve contemporaneous correlation between a and b. Block lengths are
    drawn from a Geometric(p = 1/mean_block_size) distribution, yielding
    stationary resampled series.

    The test centers the bootstrap distribution at the observed difference
    (Politis-Wolf 2003 convention) so that the bootstrap samples Δ* − Δ_obs
    represent the H0 null distribution.

    Returns:
        dict — sr_a, sr_b, delta, p_value, ci_low_95, ci_high_95, n_obs,
                 n_boot, mean_block_size, alternative.
    """
    rng = np.random.default_rng(seed)
    df = pd.concat([ret_a.rename("a"), ret_b.rename("b")], axis=1).dropna()
    a = df["a"].to_numpy(); b = df["b"].to_numpy()
    n = len(a)
    if n < 2 * mean_block_size:
        raise ValueError(f"n={n} too small for mean_block_size={mean_block_size}")

    sr_a = a.mean() / a.std() * np.sqrt(freq)
    sr_b = b.mean() / b.std() * np.sqrt(freq)
    delta_obs = sr_a - sr_b

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = _stationary_bootstrap_indices(n, mean_block_size, rng)
        ab = a[idx]; bb = b[idx]
        sra_b = ab.mean() / ab.std() * np.sqrt(freq)
        srb_b = bb.mean() / bb.std() * np.sqrt(freq)
        deltas[i] = sra_b - srb_b

    deltas_centered = deltas - delta_obs
    if alternative == "greater":
        p_value = (deltas_centered >= delta_obs).mean()
    elif alternative == "less":
        p_value = (deltas_centered <= delta_obs).mean()
    elif alternative == "two-sided":
        p_value = (np.abs(deltas_centered) >= abs(delta_obs)).mean()
    else:
        raise ValueError(f"unknown alternative: {alternative!r}")

    return {
        "sr_a": float(sr_a), "sr_b": float(sr_b),
        "delta": float(delta_obs),
        "p_value": float(p_value),
        "ci_low_95": float(np.percentile(deltas, 2.5)),
        "ci_high_95": float(np.percentile(deltas, 97.5)),
        "n_obs": int(n), "n_boot": int(n_boot),
        "mean_block_size": int(mean_block_size),
        "alternative": alternative,
        "bootstrap_dist": deltas,                # raw (uncentered) Δ*
        "bootstrap_dist_centered": deltas_centered,  # H0 distribution
    }


def multinomial_chi2_test(observed: list | np.ndarray,
                           expected_probs: list | np.ndarray | None = None) -> dict:
    """Multinomial χ² goodness-of-fit (df = k-1).

    Returns dict — chi2, p_value, df, observed, expected.
    """
    from scipy.stats import chisquare
    obs = np.asarray(observed, dtype=float)
    n = obs.sum()
    k = len(obs)
    if expected_probs is None:
        exp_p = np.ones(k) / k
    else:
        exp_p = np.asarray(expected_probs, dtype=float)
    exp = n * exp_p
    chi2, p = chisquare(obs, exp)
    return {"chi2": float(chi2), "p_value": float(p), "df": k - 1,
            "observed": obs.astype(int), "expected": exp,
            "n_total": int(n)}


def best_spec_sign_test(spec_history: pd.DataFrame, eps: float = 1e-9) -> dict:
    """Multinomial χ² on (continuation, reversal, neutral) counts of best w_fast."""
    wf = spec_history["w_fast"].to_numpy()
    n_cont = int((wf > eps).sum())
    n_rev = int((wf < -eps).sum())
    n_neu = int((np.abs(wf) <= eps).sum())
    out = multinomial_chi2_test([n_cont, n_rev, n_neu])
    out["categories"] = ["continuation", "reversal", "neutral"]
    return out


# ---------------------------------------------------------------------------
# 9. Plots
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


_DEFAULT_COLORS = ("#3498db", "#e74c3c", "#7f8c8d", "#2ecc71", "#f39c12")


def plot_equity_with_drawdown(returns_dict: dict, *,
                                title: str | None = None,
                                figsize: tuple = (11, 6),
                                colors: list | tuple | None = None,
                                log_scale: bool = True):
    """Two-panel: cumulative equity (top) + drawdown (bottom) for multiple strategies.

    returns_dict — {label: pd.Series of monthly returns}
    """
    if colors is None:
        colors = _DEFAULT_COLORS

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.5]})
    for i, (name, r) in enumerate(returns_dict.items()):
        s = r.dropna()
        if len(s) == 0:
            continue
        eq = (1 + s).cumprod()
        dd = eq / eq.cummax() - 1
        c = colors[i % len(colors)]
        axes[0].plot(eq.index, eq.values, label=name, color=c, lw=1.4)
        axes[1].fill_between(dd.index, dd.values, 0, color=c, alpha=0.25)
        axes[1].plot(dd.index, dd.values, color=c, lw=0.8)

    axes[0].set_ylabel("Equity" + (" (log)" if log_scale else ""))
    if log_scale:
        axes[0].set_yscale("log")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[0].grid(alpha=0.3, which="both")
    if title:
        axes[0].set_title(title, fontsize=13)

    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    return fig, axes


def plot_best_spec_ts(spec_histories: dict, *, figsize: tuple = (11, 6)):
    """Time series of best (fast, w_fast), one column per protocol.

    spec_histories — {protocol_name: spec_history DataFrame from walkforward_oos}
    """
    n = len(spec_histories)
    fig, axes = plt.subplots(2, n, figsize=figsize, sharex=True, squeeze=False)
    for i, (name, h) in enumerate(spec_histories.items()):
        years = h.index.year
        sc = axes[0, i].scatter(years, h["fast"], c=h["w_fast"], cmap="RdBu_r",
                                 vmin=-1, vmax=1, s=50, edgecolor="black",
                                 linewidth=0.4)
        axes[0, i].set_yticks([1, 2, 3])
        axes[0, i].set_ylabel("fast (months)" if i == 0 else "")
        axes[0, i].set_title(name, fontsize=11)
        axes[0, i].grid(alpha=0.3)

        axes[1, i].plot(years, h["w_fast"], "o-", color="#3498db", ms=3, lw=1)
        axes[1, i].axhline(0, color="black", lw=0.6)
        axes[1, i].axhspan(0, 1.05, alpha=0.05, color="#2ecc71")  # continuation tint
        axes[1, i].axhspan(-1.05, 0, alpha=0.05, color="#e74c3c")  # reversal tint
        axes[1, i].set_ylim(-1.05, 1.05)
        axes[1, i].set_ylabel("w_fast" if i == 0 else "")
        axes[1, i].set_xlabel("Decision year")
        axes[1, i].grid(alpha=0.3)

    # Shared colorbar for top row
    fig.colorbar(sc, ax=axes[0, :].tolist(), label="w_fast", fraction=0.04, pad=0.02)
    return fig, axes


def plot_bootstrap_dist(boot_result: dict, *,
                         title: str | None = None,
                         figsize: tuple = (8, 4),
                         bins: int = 60):
    """Histogram of centered bootstrap Δ* (H0 distribution) with observed Δ marked."""
    deltas_c = boot_result.get("bootstrap_dist_centered")
    if deltas_c is None:
        raise ValueError("boot_result missing 'bootstrap_dist_centered'; "
                          "rerun lw_sharpe_diff_bootstrap.")
    delta_obs = boot_result["delta"]
    p = boot_result["p_value"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(deltas_c, bins=bins, color="#bdc3c7", edgecolor="white",
             alpha=0.8, label="H₀ distribution (centered Δ*)")
    ax.axvline(delta_obs, color="#e74c3c", lw=2,
                label=f"observed Δ = {delta_obs:+.4f}")
    ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("Sharpe difference")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title(title or f"P-R bootstrap Sharpe-diff (one-sided p = {p:.3f})", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_rolling_sharpe(returns_dict: dict, *,
                         window: int = 60,
                         figsize: tuple = (11, 4),
                         colors: list | tuple | None = None):
    """Rolling annualized Sharpe over a `window`-month window for multiple strategies."""
    if colors is None:
        colors = _DEFAULT_COLORS
    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, r) in enumerate(returns_dict.items()):
        s = r.dropna()
        rs = (s.rolling(window).mean() / s.rolling(window).std()) * np.sqrt(12)
        ax.plot(rs.index, rs.values, label=name,
                 color=colors[i % len(colors)], lw=1.2)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel(f"Rolling {window}M Sharpe")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_w_fast_histogram(spec_histories: dict, *, figsize: tuple = (10, 4)):
    """Stacked bar of w_fast frequency per protocol — visualizes secondary test."""
    n = len(spec_histories)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True, squeeze=False)
    bins = np.linspace(-1.05, 1.05, 22)
    for i, (name, h) in enumerate(spec_histories.items()):
        ax = axes[0, i]
        ax.hist(h["w_fast"], bins=bins, color="#3498db",
                 edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", lw=0.6, ls="--")
        ax.set_xlabel("w_fast (best per year)")
        ax.set_title(name, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("Years")
    plt.tight_layout()
    return fig, axes
