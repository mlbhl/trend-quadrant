import yfinance as yf
import pandas as pd
import numpy as np


def fetch_prices(tickers, start, end, proxy=None):
    """복수 티커의 일별 수정주가를 반환한다."""
    df = yf.download(tickers, start=start, end=end, proxy=proxy, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    else:
        df.columns = tickers if isinstance(tickers, list) and len(tickers) == 1 else [tickers]
    return df


def fetch_index(ticker, start, end, proxy=None):
    """단일 티커의 일별 종가를 Series로 반환한다."""
    df = yf.download(ticker, start=start, end=end, proxy=proxy, auto_adjust=True)
    return df["Close"].squeeze()


def calc_monthly_returns(prices):
    """일별 주가 → 월말 리샘플링 → 월별 수익률."""
    monthly = prices.resample("ME").last()
    returns = monthly.pct_change().dropna()
    return returns


def calc_trend_regime(monthly_returns, fast_months=1, slow_months=12):
    """매월말 Fast/Slow 트렌드 판정 및 4분면 라벨 부여.

    Args:
        monthly_returns: DataFrame (index=date, columns=tickers)
        fast_months: Fast 룩백 (개월)
        slow_months: Slow 룩백 (개월)

    Returns:
        DataFrame with columns: ticker, date, ret_fast, ret_slow, fast, slow, regime
    """
    if isinstance(monthly_returns, pd.Series):
        monthly_returns = monthly_returns.to_frame()

    records = []
    for col in monthly_returns.columns:
        s = monthly_returns[col]
        ret_fast_s = s.rolling(fast_months).apply(lambda x: (1 + x).prod() - 1, raw=True)
        ret_slow_s = s.rolling(slow_months).apply(lambda x: (1 + x).prod() - 1, raw=True)
        for date in s.index:
            rf = ret_fast_s.loc[date] if date in ret_fast_s.index else np.nan
            rs = ret_slow_s.loc[date] if date in ret_slow_s.index else np.nan
            if pd.isna(rf) or pd.isna(rs):
                continue
            fast = "+" if rf >= 0 else "-"
            slow = "+" if rs >= 0 else "-"
            if slow == "+" and fast == "+":
                regime = "Bullish"
            elif slow == "+" and fast == "-":
                regime = "Correction"
            elif slow == "-" and fast == "-":
                regime = "Bearish"
            else:
                regime = "Rebound"
            records.append({
                "ticker": col,
                "date": date,
                "ret_fast": rf,
                "ret_slow": rs,
                "fast": fast,
                "slow": slow,
                "regime": regime,
            })
    return pd.DataFrame(records)


def calc_regime_stats(monthly_returns, regimes):
    """국면별 익월 동일가중 수익률로 평균수익률, 변동성, 샤프비율 산출.

    Args:
        monthly_returns: DataFrame (index=date, columns=tickers)
        regimes: DataFrame from calc_trend_regime

    Returns:
        DataFrame(index=regime, columns=[mean, std, sharpe])
    """
    if isinstance(monthly_returns, pd.Series):
        monthly_returns = monthly_returns.to_frame()

    dates = sorted(regimes["date"].unique())
    regime_next_returns = {r: [] for r in ["Bullish", "Correction", "Bearish", "Rebound"]}

    for date in dates:
        mask = regimes["date"] == date
        for regime_name in regime_next_returns:
            tickers_in_regime = regimes.loc[mask & (regimes["regime"] == regime_name), "ticker"].tolist()
            if not tickers_in_regime:
                continue
            valid = [t for t in tickers_in_regime if t in monthly_returns.columns]
            if not valid:
                continue
            # 익월 수익률
            future_dates = monthly_returns.index[monthly_returns.index > date]
            if len(future_dates) == 0:
                continue
            next_date = future_dates[0]
            next_ret = monthly_returns.loc[next_date, valid].mean()
            regime_next_returns[regime_name].append(next_ret)

    stats = {}
    for regime_name, rets in regime_next_returns.items():
        if len(rets) == 0:
            stats[regime_name] = {"mean": np.nan, "std": np.nan, "sharpe": np.nan}
            continue
        arr = np.array(rets)
        m = arr.mean()
        s = arr.std()
        sharpe = m / s if s > 0 else np.nan
        stats[regime_name] = {"mean": m, "std": s, "sharpe": sharpe}

    return pd.DataFrame(stats).T


def calc_trend_quintile(monthly_returns, regimes, w_slow=0.5, w_fast=0.5, n_quantiles=5):
    """Slow/Fast 순위 가중평균으로 퀀타일 분류 후 익월 통계 산출.

    Args:
        monthly_returns: DataFrame (index=date, columns=tickers)
        regimes: DataFrame from calc_trend_regime
        w_slow: Slow trend 순위 가중치
        w_fast: Fast trend 순위 가중치
        n_quantiles: 분위 수 (default 5)

    Returns:
        quintile_latest: 최신 월말 기준 티커별 퀀타일 배정 DataFrame
        quintile_stats: DataFrame(index=Q1..Q5, columns=[mean, std, sharpe])
    """
    if isinstance(monthly_returns, pd.Series):
        monthly_returns = monthly_returns.to_frame()

    dates = sorted(regimes["date"].unique())
    quintile_next_returns = {f"Q{i}": [] for i in range(1, n_quantiles + 1)}
    latest_date = dates[-1]
    quintile_latest = None

    for date in dates:
        snap = regimes[regimes["date"] == date].copy()
        if len(snap) < n_quantiles:
            continue

        # 순위: ascending이므로 rank 1 = 가장 낮은 수익률 = worst
        snap["rank_slow"] = snap["ret_slow"].rank(method="average")
        snap["rank_fast"] = snap["ret_fast"].rank(method="average")
        snap["rank_composite"] = w_slow * snap["rank_slow"] + w_fast * snap["rank_fast"]
        # 동률은 같은 Q에 배정: percentile rank(method="average") 후 ceil 버킷팅
        pct = snap["rank_composite"].rank(method="average", pct=True)
        bucket = np.ceil(pct * n_quantiles).clip(1, n_quantiles).astype(int)
        snap["quintile"] = "Q" + bucket.astype(str)

        if date == latest_date:
            quintile_latest = snap

        # 익월 수익률
        future_dates = monthly_returns.index[monthly_returns.index > date]
        if len(future_dates) == 0:
            continue
        next_date = future_dates[0]

        for q in quintile_next_returns:
            tickers_in_q = snap.loc[snap["quintile"] == q, "ticker"].tolist()
            valid = [t for t in tickers_in_q if t in monthly_returns.columns]
            if not valid:
                continue
            next_ret = monthly_returns.loc[next_date, valid].mean()
            quintile_next_returns[q].append(next_ret)

    stats = {}
    for q, rets in quintile_next_returns.items():
        if len(rets) == 0:
            stats[q] = {"mean": np.nan, "std": np.nan, "sharpe": np.nan}
            continue
        arr = np.array(rets)
        m = arr.mean()
        s = arr.std()
        stats[q] = {"mean": m, "std": s, "sharpe": m / s if s > 0 else np.nan}

    return quintile_latest, pd.DataFrame(stats).T


def calc_momentum_signals(monthly_returns, fast_months=1, slow_months=12):
    """가변 룩백 기간의 Fast/Slow 모멘텀 시그널을 산출한다.

    Returns:
        DataFrame with columns: ticker, date, ret_fast, ret_slow
    """
    if isinstance(monthly_returns, pd.Series):
        monthly_returns = monthly_returns.to_frame()

    records = []
    for col in monthly_returns.columns:
        s = monthly_returns[col]
        ret_fast = s.rolling(fast_months).apply(lambda x: (1 + x).prod() - 1, raw=True)
        ret_slow = s.rolling(slow_months).apply(lambda x: (1 + x).prod() - 1, raw=True)
        for date in s.index:
            rf = ret_fast.loc[date] if date in ret_fast.index else np.nan
            rs = ret_slow.loc[date] if date in ret_slow.index else np.nan
            if pd.isna(rf) or pd.isna(rs):
                continue
            records.append({"ticker": col, "date": date, "ret_fast": rf, "ret_slow": rs})
    return pd.DataFrame(records)


def calc_quintile_from_signals(monthly_returns, signals, w_slow=0.5, w_fast=0.5, n_quantiles=5):
    """모멘텀 시그널 기반 퀀타일 분류 후 익월 통계 산출.

    Args:
        monthly_returns: DataFrame (index=date, columns=tickers)
        signals: DataFrame from calc_momentum_signals (ret_fast, ret_slow)
        w_slow, w_fast: 순위 가중치
        n_quantiles: 분위 수

    Returns:
        quintile_latest, quintile_stats
    """
    if isinstance(monthly_returns, pd.Series):
        monthly_returns = monthly_returns.to_frame()

    dates = sorted(signals["date"].unique())
    quintile_next_returns = {f"Q{i}": [] for i in range(1, n_quantiles + 1)}
    latest_date = dates[-1]
    quintile_latest = None

    for date in dates:
        snap = signals[signals["date"] == date].copy()
        if len(snap) < n_quantiles:
            continue

        snap["rank_slow"] = snap["ret_slow"].rank(method="average")
        snap["rank_fast"] = snap["ret_fast"].rank(method="average")
        snap["rank_composite"] = w_slow * snap["rank_slow"] + w_fast * snap["rank_fast"]
        # 동률은 같은 Q에 배정: percentile rank(method="average") 후 ceil 버킷팅
        pct = snap["rank_composite"].rank(method="average", pct=True)
        bucket = np.ceil(pct * n_quantiles).clip(1, n_quantiles).astype(int)
        snap["quintile"] = "Q" + bucket.astype(str)

        if date == latest_date:
            quintile_latest = snap

        future_dates = monthly_returns.index[monthly_returns.index > date]
        if len(future_dates) == 0:
            continue
        next_date = future_dates[0]

        for q in quintile_next_returns:
            tickers_in_q = snap.loc[snap["quintile"] == q, "ticker"].tolist()
            valid = [t for t in tickers_in_q if t in monthly_returns.columns]
            if not valid:
                continue
            next_ret = monthly_returns.loc[next_date, valid].mean()
            quintile_next_returns[q].append(next_ret)

    stats = {}
    for q, rets in quintile_next_returns.items():
        if len(rets) == 0:
            stats[q] = {"mean": np.nan, "std": np.nan, "sharpe": np.nan}
            continue
        arr = np.array(rets)
        m = arr.mean()
        s = arr.std()
        stats[q] = {"mean": m, "std": s, "sharpe": m / s if s > 0 else np.nan}

    return quintile_latest, pd.DataFrame(stats).T


def grid_search_quintile(monthly_returns, fast_list=(1, 2, 3),
                         slow_list=(6, 9, 10, 11, 12), weight_step=0.1):
    """Fast/Slow 룩백 + 가중치 조합을 그리드 서치하여 Q5-Q1 샤프 스프레드 최대화.

    Args:
        monthly_returns: DataFrame (index=date, columns=tickers)
        fast_list: Fast 룩백 후보 (개월)
        slow_list: Slow 룩백 후보 (개월)
        weight_step: 가중치 단위 (default 0.1 = 10%)

    Returns:
        results: DataFrame with all combos sorted by sharpe_spread desc
    """
    if isinstance(monthly_returns, pd.Series):
        monthly_returns = monthly_returns.to_frame()

    weights = np.arange(0, 1 + weight_step / 2, weight_step)
    weights = np.round(weights, 2)

    results = []

    for fast_m in fast_list:
        for slow_m in slow_list:
            if fast_m >= slow_m:
                continue
            signals = calc_momentum_signals(monthly_returns, fast_months=fast_m, slow_months=slow_m)
            if len(signals) == 0:
                continue

            for w_s in weights:
                w_f = round(1 - w_s, 2)
                _, qstats = calc_quintile_from_signals(
                    monthly_returns, signals, w_slow=w_s, w_fast=w_f
                )
                s_q5 = qstats.loc["Q5", "sharpe"] if "Q5" in qstats.index else np.nan
                s_q1 = qstats.loc["Q1", "sharpe"] if "Q1" in qstats.index else np.nan
                spread = s_q5 - s_q1 if not (pd.isna(s_q5) or pd.isna(s_q1)) else np.nan

                # Monotonicity check: Q1 < Q2 < Q3 < Q4 < Q5 in sharpe
                sharpes = []
                for qi in range(1, 6):
                    label = f"Q{qi}"
                    sharpes.append(qstats.loc[label, "sharpe"] if label in qstats.index else np.nan)

                if any(pd.isna(s) for s in sharpes):
                    mono_violations = np.nan
                    is_monotonic = False
                else:
                    mono_violations = sum(
                        1 for i in range(4) if sharpes[i] >= sharpes[i + 1]
                    )
                    is_monotonic = mono_violations == 0

                results.append({
                    "fast_months": fast_m,
                    "slow_months": slow_m,
                    "w_slow": w_s,
                    "w_fast": w_f,
                    "sharpe_Q1": sharpes[0],
                    "sharpe_Q2": sharpes[1],
                    "sharpe_Q3": sharpes[2],
                    "sharpe_Q4": sharpes[3],
                    "sharpe_Q5": sharpes[4],
                    "sharpe_spread": spread,
                    "mono_violations": mono_violations,
                    "is_monotonic": is_monotonic,
                })

    df = pd.DataFrame(results).sort_values(
        ["mono_violations", "sharpe_spread"],
        ascending=[True, False],
    ).reset_index(drop=True)
    return df
