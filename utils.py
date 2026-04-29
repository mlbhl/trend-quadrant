import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


REGIME_COLORS = {
    "Bullish": "#2ecc71",
    "Correction": "#f39c12",
    "Bearish": "#e74c3c",
    "Rebound": "#3498db",
}


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
    """일별 주가 → 월말 리샘플링 → 월별 수익률.

    티커별 상장일이 다를 수 있으므로, 모든 티커가 NaN인 행만 제거한다.
    개별 티커의 NaN은 다운스트림 함수(calc_trend_regime 등)가 per-ticker로 건너뛴다.
    """
    monthly = prices.resample("ME").last()
    returns = monthly.pct_change().dropna(how="all")
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


def grid_search_regime(monthly_returns, fast_list=(1, 2, 3),
                       slow_list=(6, 9, 10, 11, 12)):
    """Fast/Slow 룩백 그리드 서치하여 4-state regime 샤프 단조성 탐색.

    이상적 패턴: Bullish > Rebound ≈ Correction > Bearish (샤프).
    위반 카운트 (각 1점, 최대 4):
        bull > rebound, bull > correction, rebound > bear, correction > bear

    Args:
        monthly_returns: DataFrame (index=date, columns=tickers)
        fast_list: Fast 룩백 후보 (개월)
        slow_list: Slow 룩백 후보 (개월)

    Returns:
        DataFrame sorted by (regime_violations asc, bull_bear_spread desc)
    """
    if isinstance(monthly_returns, pd.Series):
        monthly_returns = monthly_returns.to_frame()

    results = []
    for fast_m in fast_list:
        for slow_m in slow_list:
            if fast_m >= slow_m:
                continue
            regimes = calc_trend_regime(monthly_returns,
                                        fast_months=fast_m, slow_months=slow_m)
            if len(regimes) == 0:
                continue
            stats = calc_regime_stats(monthly_returns, regimes)
            sh = {r: stats.loc[r, "sharpe"] if r in stats.index else np.nan
                  for r in ["Bullish", "Correction", "Bearish", "Rebound"]}
            bull, corr, bear, rebd = sh["Bullish"], sh["Correction"], sh["Bearish"], sh["Rebound"]

            if any(pd.isna(v) for v in (bull, corr, bear, rebd)):
                violations = np.nan
                is_monotonic = False
            else:
                v = 0
                if not (bull > rebd): v += 1
                if not (bull > corr): v += 1
                if not (rebd > bear): v += 1
                if not (corr > bear): v += 1
                violations = v
                is_monotonic = (v == 0)

            spread = (bull - bear) if not (pd.isna(bull) or pd.isna(bear)) else np.nan

            results.append({
                "fast_months": fast_m,
                "slow_months": slow_m,
                "sharpe_Bullish": bull,
                "sharpe_Rebound": rebd,
                "sharpe_Correction": corr,
                "sharpe_Bearish": bear,
                "bull_bear_spread": spread,
                "regime_violations": violations,
                "is_monotonic": is_monotonic,
            })

    df = pd.DataFrame(results).sort_values(
        ["regime_violations", "bull_bear_spread"],
        ascending=[True, False],
    ).reset_index(drop=True)
    return df


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


def _quadrant_axes_setup(ax, x_abs, y_abs, slow_months, fast_months, title):
    ax.set_xlim(-x_abs, x_abs)
    ax.set_ylim(-y_abs, y_abs)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Slow Trend ({slow_months}M Return)", fontsize=11)
    ax.set_ylabel(f"Fast Trend ({fast_months}M Return)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    # Quadrant background shading
    ax.axhspan(0, y_abs, xmin=0.5, xmax=1.0, alpha=0.04, color=REGIME_COLORS["Bullish"])
    ax.axhspan(-y_abs, 0, xmin=0.5, xmax=1.0, alpha=0.04, color=REGIME_COLORS["Correction"])
    ax.axhspan(-y_abs, 0, xmin=0.0, xmax=0.5, alpha=0.04, color=REGIME_COLORS["Bearish"])
    ax.axhspan(0, y_abs, xmin=0.0, xmax=0.5, alpha=0.04, color=REGIME_COLORS["Rebound"])

    # Quadrant labels
    ax.text( x_abs * 0.95,  y_abs * 0.92, "Bullish",    ha="right", va="top",    fontsize=13, color=REGIME_COLORS["Bullish"], alpha=0.6, weight="bold")
    ax.text( x_abs * 0.95, -y_abs * 0.92, "Correction", ha="right", va="bottom", fontsize=13, color=REGIME_COLORS["Correction"], alpha=0.6, weight="bold")
    ax.text(-x_abs * 0.95, -y_abs * 0.92, "Bearish",    ha="left",  va="bottom", fontsize=13, color=REGIME_COLORS["Bearish"], alpha=0.6, weight="bold")
    ax.text(-x_abs * 0.95,  y_abs * 0.92, "Rebound",    ha="left",  va="top",    fontsize=13, color=REGIME_COLORS["Rebound"], alpha=0.6, weight="bold")
    ax.grid(alpha=0.2)


def plot_quadrant_scatter(latest, snap_date, ticker_names=None,
                          fast_months=1, slow_months=12, ax=None, figsize=(8, 8)):
    """Chart A: 4분면 스냅샷 산포도. 각 티커는 (ret_slow, ret_fast) 위치에 점으로 표시."""
    from adjustText import adjust_text

    if ticker_names is None:
        ticker_names = {}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for regime_name, color in REGIME_COLORS.items():
        subset = latest[latest["regime"] == regime_name]
        ax.scatter(subset["ret_slow"], subset["ret_fast"],
                   color=color, label=regime_name, s=150, zorder=5,
                   edgecolors="white", linewidths=1.5)

    x_abs = max(abs(latest["ret_slow"].min()), abs(latest["ret_slow"].max())) * 1.6
    y_abs = max(abs(latest["ret_fast"].min()), abs(latest["ret_fast"].max())) * 1.6

    title = f"Trend Cycle Quadrant — {pd.Timestamp(snap_date).strftime('%Y-%m-%d')}"
    _quadrant_axes_setup(ax, x_abs, y_abs, slow_months, fast_months, title)

    texts = []
    for _, row in latest.iterrows():
        name = ticker_names.get(row["ticker"], row["ticker"])
        texts.append(ax.text(row["ret_slow"], row["ret_fast"], name,
                             fontsize=9, fontweight="bold", zorder=10))
    adjust_text(texts, ax=ax,
                force_text=(2.0, 2.0),
                force_points=(1.5, 1.5),
                expand=(2.0, 2.0),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

    plt.tight_layout()
    return fig, ax


def plot_quadrant_trajectory(regimes, ticker, ticker_names=None,
                              fast_months=1, slow_months=12,
                              start_date=None, end_date=None,
                              lookback_months=None,
                              ax=None, figsize=(8, 8), cmap="viridis"):
    """Chart A2: 특정 티커의 (ret_slow, ret_fast) 궤적을 시간순 라인으로 표시.

    Args:
        regimes: calc_trend_regime 결과 DataFrame
        ticker: 대상 티커
        ticker_names: 표시명 매핑
        start_date: 궤적의 시작 시점 (None=lookback_months 또는 전체)
        end_date: 궤적의 마지막 시점 (None=최신)
        lookback_months: end_date 기준 과거 기간 (start_date가 지정되면 무시)
    """
    if ticker_names is None:
        ticker_names = {}

    sub = regimes[regimes["ticker"] == ticker].sort_values("date").copy()
    if sub.empty:
        raise ValueError(f"No data for ticker '{ticker}' in regimes")

    if end_date is not None:
        sub = sub[sub["date"] <= pd.Timestamp(end_date)]
    if start_date is not None:
        sub = sub[sub["date"] >= pd.Timestamp(start_date)]
    elif lookback_months is not None and not sub.empty:
        cutoff = sub["date"].max() - pd.DateOffset(months=lookback_months)
        sub = sub[sub["date"] >= cutoff]
    if sub.empty:
        raise ValueError(f"No trajectory points after applying date filters for '{ticker}'")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    xs = sub["ret_slow"].values
    ys = sub["ret_fast"].values
    n = len(xs)

    # Symmetric axes covering the trajectory
    x_abs = max(abs(xs.min()), abs(xs.max())) * 1.4 if n > 0 else 0.1
    y_abs = max(abs(ys.min()), abs(ys.max())) * 1.4 if n > 0 else 0.1
    name = ticker_names.get(ticker, ticker)
    title = f"Trajectory: {name} ({ticker}) — {sub['date'].min().strftime('%Y-%m')} → {sub['date'].max().strftime('%Y-%m')}"
    _quadrant_axes_setup(ax, x_abs, y_abs, slow_months, fast_months, title)

    # Connecting line
    ax.plot(xs, ys, color="gray", linewidth=1.0, alpha=0.5, zorder=3)

    # Time-graded scatter points
    cmap_obj = plt.get_cmap(cmap)
    colors_t = cmap_obj(np.linspace(0.15, 0.95, n))
    ax.scatter(xs, ys, c=colors_t, s=60, zorder=5,
               edgecolors="white", linewidths=0.8)

    # Mark start and end
    ax.scatter(xs[0], ys[0], facecolor="white", edgecolor="black",
               s=140, zorder=6, linewidths=1.5, label="start")
    ax.scatter(xs[-1], ys[-1], facecolor="black", edgecolor="white",
               s=180, zorder=7, linewidths=1.5, label="end")

    ax.annotate(sub["date"].iloc[0].strftime("%Y-%m"),
                (xs[0], ys[0]), xytext=(8, 8), textcoords="offset points",
                fontsize=9, color="black")
    ax.annotate(f"{name}\n{sub['date'].iloc[-1].strftime('%Y-%m')}",
                (xs[-1], ys[-1]), xytext=(10, 10), textcoords="offset points",
                fontsize=10, fontweight="bold", color="black")

    plt.tight_layout()
    return fig, ax


def plot_regime_stats(regime_stats, fast_months=1, slow_months=12, figsize=(10, 5)):
    """Chart B: 국면별 익월 평균/변동성/샤프 3패널 막대그래프."""
    regime_order = ["Bullish", "Correction", "Bearish", "Rebound"]
    rs = regime_stats.loc[[r for r in regime_order if r in regime_stats.index]]
    bar_colors = [REGIME_COLORS[r] for r in rs.index]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].bar(rs.index, rs["mean"], color=bar_colors, edgecolor="white")
    axes[0].set_title("Next-Month Avg Return")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=2))
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(rs.index, rs["std"], color=bar_colors, edgecolor="white")
    axes[1].set_title("Next-Month Volatility")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=2))
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(rs.index, rs["sharpe"], color=bar_colors, edgecolor="white")
    axes[2].set_title("Next-Month Sharpe Ratio")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle(
        f"Regime-Based Next-Month Statistics (Fast={fast_months}M, Slow={slow_months}M)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    return fig, axes
