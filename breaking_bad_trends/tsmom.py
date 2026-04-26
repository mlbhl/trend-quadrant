"""Time-series momentum: static/dynamic trend following (Breaking Bad Trends replication).

Paper: Goulding, Harvey & Mazzoleni (2023, FAJ)
- Per-asset independent long/short (TSMOM)
- 4-state regime classification (Bull/Correction/Bear/Rebound)
- Dynamic strategy: fast/slow blending at turning points via mixing params
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Momentum signals (eq. 1-2)
# ---------------------------------------------------------------------------

def momentum_signals(returns: pd.DataFrame,
                     k_slow: int = 12,
                     k_fast: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute slow/fast momentum signals from monthly excess returns.

    Cumulative compound return convention (consistent with utils.calc_trend_regime):
    x_slow(m) = prod(1+r_{m-1}) * ... * (1+r_{m-k_slow}) - 1
    x_fast(m) = prod(1+r_{m-1}) * ... * (1+r_{m-k_fast}) - 1

    Note: BBT paper eq.(1)-(2) uses arithmetic mean; we deviate to use cumulative
    compound returns (MOP 2012 / standard TSMOM convention) for cross-notebook
    consistency with the quadrant dashboard.

    Args:
        returns: monthly returns (index=date, columns=assets)
        k_slow: slow lookback (months)
        k_fast: fast lookback (months)

    Returns:
        (x_slow, x_fast) DataFrames, same shape as returns
    """
    cum = lambda x: (1 + x).prod() - 1
    x_slow = returns.rolling(k_slow).apply(cum, raw=True).shift(1)
    x_fast = returns.rolling(k_fast).apply(cum, raw=True).shift(1)
    return x_slow, x_fast


# ---------------------------------------------------------------------------
# 4-state regime classification (eq. 4)
# ---------------------------------------------------------------------------

BULL = "Bull"
CORRECTION = "Correction"
BEAR = "Bear"
REBOUND = "Rebound"


def classify_states(x_slow: pd.DataFrame,
                    x_fast: pd.DataFrame) -> pd.DataFrame:
    """Classify 4-state regime from slow/fast signal signs.

    Bull:       slow >= 0, fast >= 0
    Correction: slow >= 0, fast < 0
    Bear:       slow < 0,  fast < 0
    Rebound:    slow < 0,  fast >= 0

    Returns:
        DataFrame (same shape), values are state strings
    """
    states = pd.DataFrame(np.nan, index=x_slow.index, columns=x_slow.columns, dtype=object)
    valid = x_slow.notna() & x_fast.notna()
    slow_pos = (x_slow >= 0) & valid
    fast_pos = (x_fast >= 0) & valid
    slow_neg = (x_slow < 0) & valid
    fast_neg = (x_fast < 0) & valid
    states[slow_pos & fast_pos] = BULL
    states[slow_pos & fast_neg] = CORRECTION
    states[slow_neg & fast_neg] = BEAR
    states[slow_neg & fast_pos] = REBOUND
    return states


# ---------------------------------------------------------------------------
# Static trend returns (eq. 5-6)
# ---------------------------------------------------------------------------

def static_trend_returns(returns: pd.DataFrame,
                         x_slow: pd.DataFrame) -> pd.DataFrame:
    """Static trend-following returns: r_trend(m) = sign(x_slow(m)) * r(m).

    Lookback is implicit in x_slow (computed upstream in momentum_signals).
    """
    sign_slow = np.sign(x_slow).replace(0, 1)  # treat 0 as long
    return sign_slow * returns


def fast_trend_returns(returns: pd.DataFrame,
                       x_fast: pd.DataFrame) -> pd.DataFrame:
    """Fast signal trend-following returns."""
    sign_fast = np.sign(x_fast).replace(0, 1)
    return sign_fast * returns


# ---------------------------------------------------------------------------
# Dynamic mixing parameter estimation (eq. 8-10)
# ---------------------------------------------------------------------------

def _estimate_mixing_params(returns_hist: pd.Series,
                            states_hist: pd.Series) -> tuple[float, float]:
    """Estimate aCo, aRe from single asset historical data.

    eq. 8: aCo = 0.5 * (1 - (1/C) * E[r|Co] / E[r²|Co])
    eq. 9: aRe = 0.5 * (1 + (1/C) * E[r|Re] / E[r²|Re])
    eq. 10: C = freq(Bu)/(freq(Bu or Be)) * E[r|Bu]/E[r²|Bu or Be]
               - freq(Be)/(freq(Bu or Be)) * E[r|Be]/E[r²|Bu or Be]

    Requires min 12 months per state. Returns 0.5 (neutral) if insufficient.
    """
    MIN_OBS = 12

    mask_bu = states_hist == BULL
    mask_be = states_hist == BEAR
    mask_co = states_hist == CORRECTION
    mask_re = states_hist == REBOUND

    n_bu = mask_bu.sum()
    n_be = mask_be.sum()
    n_co = mask_co.sum()
    n_re = mask_re.sum()

    if n_bu < MIN_OBS or n_be < MIN_OBS or n_co < MIN_OBS or n_re < MIN_OBS:
        return 0.5, 0.5

    r_bu = returns_hist[mask_bu]
    r_be = returns_hist[mask_be]
    r_co = returns_hist[mask_co]
    r_re = returns_hist[mask_re]

    # eq. 10: C
    freq_bu = n_bu / (n_bu + n_be) if (n_bu + n_be) > 0 else 0.5
    freq_be = n_be / (n_bu + n_be) if (n_bu + n_be) > 0 else 0.5

    avg_r2_bube = np.mean(np.concatenate([r_bu.values, r_be.values]) ** 2)
    if avg_r2_bube < 1e-12:
        return 0.5, 0.5

    C = (freq_bu * r_bu.mean() - freq_be * r_be.mean()) / avg_r2_bube
    if abs(C) < 1e-12:
        return 0.5, 0.5

    # eq. 8: aCo
    avg_r2_co = (r_co ** 2).mean()
    if avg_r2_co < 1e-12:
        a_co = 0.5
    else:
        a_co = 0.5 * (1 - (1 / C) * r_co.mean() / avg_r2_co)

    # eq. 9: aRe
    avg_r2_re = (r_re ** 2).mean()
    if avg_r2_re < 1e-12:
        a_re = 0.5
    else:
        a_re = 0.5 * (1 + (1 / C) * r_re.mean() / avg_r2_re)

    # clamp to [0, 1]
    a_co = np.clip(a_co, 0.0, 1.0)
    a_re = np.clip(a_re, 0.0, 1.0)

    return float(a_co), float(a_re)


def estimate_mixing_params(returns: pd.DataFrame,
                           states: pd.DataFrame,
                           update_every: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate mixing params for all assets × dates via expanding window.

    Updates every 30 months; holds prior estimate in between.

    Returns:
        (aCo, aRe) DataFrames, same shape as returns
    """
    a_co_df = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    a_re_df = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    dates = returns.index
    for col in returns.columns:
        last_update = -update_every  # estimate at first eligible point
        cur_aco, cur_are = 0.5, 0.5

        for i, date in enumerate(dates):
            if i - last_update >= update_every:
                if i >= 24:  # min 24 months history
                    cur_aco, cur_are = _estimate_mixing_params(
                        returns[col].iloc[:i],
                        states[col].iloc[:i],
                    )
                    last_update = i

            a_co_df.loc[date, col] = cur_aco
            a_re_df.loc[date, col] = cur_are

    return a_co_df, a_re_df


# ---------------------------------------------------------------------------
# Dynamic trend returns (eq. 7)
# ---------------------------------------------------------------------------

def dynamic_trend_positions(x_slow: pd.DataFrame,
                            x_fast: pd.DataFrame,
                            states: pd.DataFrame,
                            a_co: pd.DataFrame,
                            a_re: pd.DataFrame) -> pd.DataFrame:
    """Compute state-conditional dynamic positions (direction/magnitude before multiplying returns).

    Bull:       +1
    Bear:       -1
    Correction: (1-aCo)*sign(slow) + aCo*sign(fast)
    Rebound:    (1-aRe)*sign(slow) + aRe*sign(fast)

    Returns:
        DataFrame (same shape), values in [-1, +1]
    """
    sign_slow = np.sign(x_slow).replace(0, 1)
    sign_fast = np.sign(x_fast).replace(0, 1)

    pos = pd.DataFrame(np.nan, index=x_slow.index, columns=x_slow.columns)

    is_bull = states == BULL
    is_bear = states == BEAR
    is_corr = states == CORRECTION
    is_rebo = states == REBOUND

    pos[is_bull] = 1.0
    pos[is_bear] = -1.0

    # Correction/Rebound: use 0.5 (neutral) if a_co/a_re is NaN
    a_co_filled = a_co.fillna(0.5)
    a_re_filled = a_re.fillna(0.5)
    pos[is_corr] = (1 - a_co_filled[is_corr]) * sign_slow[is_corr] + a_co_filled[is_corr] * sign_fast[is_corr]
    pos[is_rebo] = (1 - a_re_filled[is_rebo]) * sign_slow[is_rebo] + a_re_filled[is_rebo] * sign_fast[is_rebo]

    return pos


def dynamic_trend_returns(returns: pd.DataFrame,
                          x_slow: pd.DataFrame,
                          x_fast: pd.DataFrame,
                          states: pd.DataFrame,
                          a_co: pd.DataFrame,
                          a_re: pd.DataFrame) -> pd.DataFrame:
    """State-conditional dynamic blending trend-following returns.

    Bull:       r
    Bear:       -r
    Correction: (1 - aCo) * r_slow + aCo * r_fast
    Rebound:    (1 - aRe) * r_slow + aRe * r_fast
    """
    sign_slow = np.sign(x_slow).replace(0, 1)
    sign_fast = np.sign(x_fast).replace(0, 1)

    r_slow = sign_slow * returns
    r_fast = sign_fast * returns

    r_dyn = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    is_bull = states == BULL
    is_bear = states == BEAR
    is_corr = states == CORRECTION
    is_rebo = states == REBOUND

    r_dyn[is_bull] = returns[is_bull]
    r_dyn[is_bear] = -returns[is_bear]

    a_co_filled = a_co.fillna(0.5)
    a_re_filled = a_re.fillna(0.5)
    r_dyn[is_corr] = (1 - a_co_filled[is_corr]) * r_slow[is_corr] + a_co_filled[is_corr] * r_fast[is_corr]
    r_dyn[is_rebo] = (1 - a_re_filled[is_rebo]) * r_slow[is_rebo] + a_re_filled[is_rebo] * r_fast[is_rebo]

    return r_dyn


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def equal_weight_portfolio(asset_returns: pd.DataFrame) -> pd.Series:
    """Equal-weight multi-asset portfolio returns."""
    return asset_returns.mean(axis=1)


def ex_post_vol_scale(portfolio_returns: pd.Series,
                      target_vol: float = 0.10) -> pd.Series:
    """Ex-post vol scaling (paper Figure 2, 4 method).

    Rescales by full-period realized vol to match target_vol.
    For comparison/visualization only. Not a real-time strategy.
    """
    realized_vol = portfolio_returns.std() * np.sqrt(12)
    if realized_vol < 1e-12:
        return portfolio_returns
    return portfolio_returns * (target_vol / realized_vol)


def vol_target_portfolio(portfolio_returns: pd.Series,
                         target_vol: float = 0.10,
                         lookback: int = 36) -> pd.Series:
    """Real-time target vol scaling (tradeable).

    Lever/delever via rolling realized vol. shift(1) prevents look-ahead.
    """
    realized_vol = portfolio_returns.rolling(lookback).std() * np.sqrt(12)
    scale = target_vol / realized_vol.replace(0, np.nan)
    scale = scale.clip(upper=3.0)
    return (portfolio_returns * scale.shift(1)).dropna()


def asset_vol_scale(asset_returns: pd.DataFrame,
                    target_vol: float = 0.10,
                    mode: str = "ex_post",
                    lookback: int = 36) -> pd.DataFrame:
    """Per-asset vol scaling (risk-parity style).

    Each asset is independently levered/delevered to target_vol before
    portfolio aggregation. When combined with equal-weight averaging, this
    approximates equal risk contribution per asset.

    Args:
        asset_returns: per-asset trend returns (index=date, columns=assets)
        target_vol: per-asset target vol (annualized)
        mode: "ex_post" (full-period std) or "realtime" (rolling, tradeable)
        lookback: months for realtime rolling std

    Returns:
        Scaled DataFrame, same shape. Realtime mode leaves first `lookback`
        rows as NaN.
    """
    if mode == "ex_post":
        asset_vol = asset_returns.std() * np.sqrt(12)
        scale = target_vol / asset_vol.replace(0, np.nan)
        return asset_returns.mul(scale, axis=1)
    elif mode == "realtime":
        asset_vol = asset_returns.rolling(lookback).std() * np.sqrt(12)
        scale = target_vol / asset_vol.replace(0, np.nan)
        scale = scale.clip(upper=3.0).shift(1)
        return asset_returns * scale
    else:
        raise ValueError(f"unknown mode: {mode}")


def build_weights(pos: pd.DataFrame,
                  asset_returns: pd.DataFrame,
                  target_vol: float | None = 0.10,
                  vol_mode: str = "realtime",
                  scale_level: str = "portfolio",
                  vol_lookback: int = 36) -> pd.DataFrame:
    """Per-asset notional weights consistent with ``scaled_portfolio`` aggregation.

    w_i(t) = scale_factor(t) * (1/N_t) * pos_i(t)

    N_t is the number of assets with valid position AND return at time t,
    so assets that are not yet available are excluded rather than diluting
    the portfolio.

    Satisfies sum_i w_i(t) * r_i(t) == scaled portfolio return for the matching
    (target_vol, vol_mode, scale_level, vol_lookback) combination. Inputs must be
    pre-sliced to the evaluation window so scale computations match the returns.
    """
    active = pos.notna() & asset_returns.notna()
    n_t = active.sum(axis=1).replace(0, np.nan)
    base = pos.where(active, 0.0).div(n_t, axis=0)
    if target_vol is None:
        return base

    if scale_level == "asset":
        asset_trend = pos * asset_returns
        if vol_mode == "ex_post":
            v = asset_trend.std() * np.sqrt(12)
            s = target_vol / v.replace(0, np.nan)
            return base.mul(s, axis=1)
        elif vol_mode == "realtime":
            v = asset_trend.rolling(vol_lookback).std() * np.sqrt(12)
            s = (target_vol / v.replace(0, np.nan)).clip(upper=3.0).shift(1)
            return base * s
        else:
            raise ValueError(f"unknown vol_mode: {vol_mode}")

    # portfolio level
    port_raw = (pos * asset_returns).mean(axis=1)
    if vol_mode == "ex_post":
        v = port_raw.std() * np.sqrt(12)
        s = target_vol / v if v > 1e-12 else 1.0
        return base * s
    elif vol_mode == "realtime":
        v = port_raw.rolling(vol_lookback).std() * np.sqrt(12)
        s = (target_vol / v.replace(0, np.nan)).clip(upper=3.0).shift(1)
        return base.mul(s, axis=0)
    else:
        raise ValueError(f"unknown vol_mode: {vol_mode}")


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """One-way turnover: 0.5 * Σ|Δw_i| across assets per period.

    Uses the one-way (single-leg) convention — a full flip of a 1/N weight
    produces turnover of 1/N (not 2/N). Pair this with a round-trip tcost in
    ``apply_tcost``: cost = turnover × tcost_bps_roundtrip / 1e4.

    First valid row counted as entry from 0 (initial trade-in).
    """
    dw = weights.diff()
    first = weights.dropna(how="all").index
    if len(first) > 0:
        dw.loc[first[0]] = weights.loc[first[0]]
    return 0.5 * dw.abs().sum(axis=1, skipna=True)


def apply_tcost(port_return: pd.Series,
                turnover: pd.Series,
                tcost_bps: float) -> pd.Series:
    """Subtract tcost_bps * turnover(t) from portfolio returns.

    ``tcost_bps`` is interpreted as round-trip cost (buy + sell) per unit of
    one-way turnover; ``compute_turnover`` already halves to one-way volume,
    so a full flip of a 1/N weight costs (1/N) * tcost_bps / 1e4.
    """
    if tcost_bps == 0 or turnover is None:
        return port_return
    cost = turnover * (tcost_bps / 1e4)
    cost = cost.reindex(port_return.index).fillna(0.0)
    return port_return - cost


def annualized_turnover(turnover: pd.Series) -> float:
    """Mean monthly turnover × 12 (annualized sum of |Δw|)."""
    return float(turnover.mean() * 12)


def scaled_portfolio(asset_returns: pd.DataFrame,
                     target_vol: float | None = 0.10,
                     vol_mode: str = "realtime",
                     scale_level: str = "portfolio",
                     vol_lookback: int = 36) -> pd.Series:
    """Unified vol-scaled equal-weight aggregation.

    Used to build dynamic/static/buy-and-hold portfolios under the same
    scaling convention, so performance is directly comparable.

    Args:
        asset_returns: per-asset strategy returns (DataFrame)
        target_vol: annualized target vol; None = no scaling
        vol_mode: "ex_post" or "realtime"
        scale_level: "portfolio" or "asset"
        vol_lookback: lookback months for realtime rolling std

    Returns:
        Scaled portfolio return Series (1/N equal-weight aggregation).
    """
    if target_vol is None:
        return asset_returns.mean(axis=1)

    if scale_level == "asset":
        scaled = asset_vol_scale(asset_returns, target_vol=target_vol,
                                 mode=vol_mode, lookback=vol_lookback)
        return scaled.mean(axis=1)

    # portfolio level
    port = asset_returns.mean(axis=1)
    if vol_mode == "realtime":
        return vol_target_portfolio(port, target_vol=target_vol, lookback=vol_lookback)
    return ex_post_vol_scale(port, target_vol=target_vol)


# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------

def annualized_stats(returns: pd.Series) -> dict:
    """Annualized return, vol, Sharpe."""
    mu = returns.mean() * 12
    sigma = returns.std() * np.sqrt(12)
    sharpe = mu / sigma if sigma > 0 else np.nan
    return {"ann_return": mu, "ann_vol": sigma, "sharpe": sharpe}


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown."""
    cumret = (1 + returns).cumprod()
    peak = cumret.cummax()
    dd = (cumret - peak) / peak
    return dd.min()


def state_decomposition(returns: pd.Series,
                        states: pd.DataFrame) -> pd.DataFrame:
    """Return decomposition by state (replicates paper Figure 4).

    For multi-asset, aggregates states via mode (most frequent state per month).
    """
    if isinstance(states, pd.DataFrame) and states.shape[1] > 1:
        # most frequent state per month
        state_mode = states.mode(axis=1)[0]
    else:
        state_mode = states.iloc[:, 0] if isinstance(states, pd.DataFrame) else states

    common_idx = returns.index.intersection(state_mode.index)
    returns = returns.loc[common_idx]
    state_mode = state_mode.loc[common_idx]

    result = {}
    for s in [BULL, CORRECTION, BEAR, REBOUND]:
        mask = state_mode == s
        r_state = returns[mask]
        result[s] = {
            "count": int(mask.sum()),
            "ann_return": r_state.mean() * 12 if len(r_state) > 0 else 0.0,
            "total_contrib": r_state.sum(),
        }

    # aggregate turning points
    tp_mask = (state_mode == CORRECTION) | (state_mode == REBOUND)
    r_tp = returns[tp_mask]
    result["Turning Points"] = {
        "count": int(tp_mask.sum()),
        "ann_return": r_tp.mean() * 12 if len(r_tp) > 0 else 0.0,
        "total_contrib": r_tp.sum(),
    }

    return pd.DataFrame(result).T


def asset_state_decomposition(asset_returns: pd.DataFrame,
                              states: pd.DataFrame) -> dict:
    """Per-asset return decomposition by state.

    For each asset, buckets that asset's trend returns by its own state label
    (no cross-asset mode aggregation). Returns a dict: {ticker: DataFrame}.
    """
    common_idx = asset_returns.index.intersection(states.index)
    common_cols = asset_returns.columns.intersection(states.columns)
    asset_returns = asset_returns.loc[common_idx, common_cols]
    states = states.loc[common_idx, common_cols]

    out = {}
    for col in common_cols:
        r = asset_returns[col]
        st = states[col]
        rec = {}
        for s in [BULL, CORRECTION, BEAR, REBOUND]:
            mask = st == s
            r_s = r[mask].dropna()
            rec[s] = {
                "count": int(len(r_s)),
                "ann_return": r_s.mean() * 12 if len(r_s) > 0 else 0.0,
                "total_contrib": r_s.sum(),
            }
        tp_mask = (st == CORRECTION) | (st == REBOUND)
        r_tp = r[tp_mask].dropna()
        rec["Turning Points"] = {
            "count": int(len(r_tp)),
            "ann_return": r_tp.mean() * 12 if len(r_tp) > 0 else 0.0,
            "total_contrib": r_tp.sum(),
        }
        out[col] = pd.DataFrame(rec).T
    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_backtest(returns: pd.DataFrame,
                 k_slow: int = 12,
                 k_fast: int = 2,
                 eval_start: str | None = None,
                 target_vol: float | None = 0.10,
                 vol_mode: str = "ex_post",
                 scale_level: str = "portfolio",
                 vol_lookback: int = 36,
                 tcost_bps: float = 0.0) -> dict:
    """Run static/dynamic trend-following backtest.

    Args:
        returns: monthly excess returns (index=date, columns=assets)
        k_slow: slow lookback
        k_fast: fast lookback
        eval_start: evaluation start date (after warm-up). None = full sample.
        target_vol: target vol. None = no scaling.
        vol_mode: "ex_post" (paper figures), "realtime" (tradeable rolling target)
        scale_level: "portfolio" (paper default, single scalar on 1/N portfolio)
                     or "asset" (risk-parity: each asset scaled independently before
                     equal-weight aggregation)

    Returns:
        dict with keys: static, dynamic, states, params, stats
    """
    x_slow, x_fast = momentum_signals(returns, k_slow=k_slow, k_fast=k_fast)
    states = classify_states(x_slow, x_fast)

    # static positions & returns
    pos_static = np.sign(x_slow).replace(0, 1)  # +1 or -1
    r_static = static_trend_returns(returns, x_slow)

    # dynamic mixing param estimation
    a_co, a_re = estimate_mixing_params(returns, states, update_every=30)

    # dynamic positions & returns
    pos_dynamic = dynamic_trend_positions(x_slow, x_fast, states, a_co, a_re)
    r_dynamic = dynamic_trend_returns(returns, x_slow, x_fast, states, a_co, a_re)

    # --- asset-level vol scaling path: scale each asset BEFORE portfolio aggregation ---
    if scale_level == "asset" and target_vol is not None:
        r_static = asset_vol_scale(r_static, target_vol=target_vol,
                                   mode=vol_mode, lookback=vol_lookback)
        r_dynamic = asset_vol_scale(r_dynamic, target_vol=target_vol,
                                    mode=vol_mode, lookback=vol_lookback)

    # portfolio (equal-weight: 1/N per asset)
    port_static = equal_weight_portfolio(r_static)
    port_dynamic = equal_weight_portfolio(r_dynamic)

    # evaluation period filter
    if eval_start is not None:
        mask = port_static.index >= pd.Timestamp(eval_start)
        port_static = port_static[mask]
        port_dynamic = port_dynamic[mask]
        states_eval = states.loc[mask]
    else:
        # remove warm-up (k_slow + 1 months)
        port_static = port_static.iloc[k_slow + 1:]
        port_dynamic = port_dynamic.iloc[k_slow + 1:]
        states_eval = states.iloc[k_slow + 1:]

    # --- portfolio-level vol scaling path ---
    if scale_level == "portfolio" and target_vol is not None:
        if vol_mode == "realtime":
            scale_fn = lambda s: vol_target_portfolio(s, target_vol=target_vol,
                                                      lookback=vol_lookback)
        else:
            scale_fn = lambda s: ex_post_vol_scale(s, target_vol=target_vol)
        port_static_scaled = scale_fn(port_static)
        port_dynamic_scaled = scale_fn(port_dynamic)
    else:
        # asset-level (already scaled) or no scaling
        port_static_scaled = port_static.dropna()
        port_dynamic_scaled = port_dynamic.dropna()

    # --- weights / turnover / tcost ---
    # Build weights on the pre-portfolio-scale window (same slice that
    # vol_target_portfolio consumed internally) so the rolling(lookback)
    # alignment matches the scaled return series, then reindex to eval idx.
    pre_idx = port_static.index   # post warm-up, pre portfolio-scale
    returns_pre = returns.loc[pre_idx]
    pos_static_pre = pos_static.loc[pre_idx]
    pos_dynamic_pre = pos_dynamic.loc[pre_idx]

    weights_static = build_weights(
        pos_static_pre, returns_pre,
        target_vol=target_vol, vol_mode=vol_mode,
        scale_level=scale_level, vol_lookback=vol_lookback,
    )
    weights_dynamic = build_weights(
        pos_dynamic_pre, returns_pre,
        target_vol=target_vol, vol_mode=vol_mode,
        scale_level=scale_level, vol_lookback=vol_lookback,
    )

    # Restrict to the scaled-return evaluation window
    eval_idx = port_static_scaled.index
    weights_static = weights_static.loc[eval_idx]
    weights_dynamic = weights_dynamic.loc[eval_idx]

    turnover_static = compute_turnover(weights_static)
    turnover_dynamic = compute_turnover(weights_dynamic)

    port_static_net = apply_tcost(port_static_scaled, turnover_static, tcost_bps)
    port_dynamic_net = apply_tcost(port_dynamic_scaled, turnover_dynamic, tcost_bps)

    # statistics (net of tcost)
    stats = {
        "static": annualized_stats(port_static_net),
        "dynamic": annualized_stats(port_dynamic_net),
        "static_mdd": max_drawdown(port_static_net),
        "dynamic_mdd": max_drawdown(port_dynamic_net),
        "static_turnover": annualized_turnover(turnover_static),
        "dynamic_turnover": annualized_turnover(turnover_dynamic),
        "tcost_bps": float(tcost_bps),
    }

    # state decomposition (multi-asset portfolio level via state mode)
    decomp_static = state_decomposition(port_static, states_eval)
    decomp_dynamic = state_decomposition(port_dynamic, states_eval)

    # asset-level decomposition (per-asset state labels)
    r_static_eval = r_static.loc[states_eval.index]
    r_dynamic_eval = r_dynamic.loc[states_eval.index]
    decomp_static_asset = asset_state_decomposition(r_static_eval, states_eval)
    decomp_dynamic_asset = asset_state_decomposition(r_dynamic_eval, states_eval)

    return {
        "port_static": port_static_net,        # net of tcost
        "port_dynamic": port_dynamic_net,      # net of tcost
        "port_static_gross": port_static_scaled,
        "port_dynamic_gross": port_dynamic_scaled,
        "pos_static": pos_static,      # per-asset static position (+1/-1)
        "pos_dynamic": pos_dynamic,     # per-asset dynamic position ([-1, +1])
        "weights_static": weights_static,
        "weights_dynamic": weights_dynamic,
        "turnover_static": turnover_static,
        "turnover_dynamic": turnover_dynamic,
        "states": states,
        "a_co": a_co,
        "a_re": a_re,
        "stats": stats,
        "decomp_static": decomp_static,
        "decomp_dynamic": decomp_dynamic,
        "decomp_static_asset": decomp_static_asset,
        "decomp_dynamic_asset": decomp_dynamic_asset,
    }


def print_summary(result: dict) -> None:
    """Print backtest summary."""
    s = result["stats"]

    print("=" * 60)
    print("Time-Series Momentum Backtest Results")
    print("=" * 60)

    print(f"\n{'':20s} {'Static':>12s} {'Dynamic':>12s}")
    print("-" * 44)
    print(f"{'Ann. Return':20s} {s['static']['ann_return']:>11.2%} {s['dynamic']['ann_return']:>11.2%}")
    print(f"{'Ann. Vol':20s} {s['static']['ann_vol']:>11.2%} {s['dynamic']['ann_vol']:>11.2%}")
    print(f"{'Sharpe':20s} {s['static']['sharpe']:>11.3f} {s['dynamic']['sharpe']:>11.3f}")
    print(f"{'Max Drawdown':20s} {s['static_mdd']:>11.2%} {s['dynamic_mdd']:>11.2%}")
    print(f"{'Ann. Turnover':20s} {s['static_turnover']:>11.2f} {s['dynamic_turnover']:>11.2f}")
    print(f"{'Tcost (bps)':20s} {s['tcost_bps']:>11.1f} {s['tcost_bps']:>11.1f}")

    print("\nReturn Decomposition by State (Static)")
    print(result["decomp_static"].to_string())

    print("\nReturn Decomposition by State (Dynamic)")
    print(result["decomp_dynamic"].to_string())
