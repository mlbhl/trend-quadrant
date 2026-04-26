"""Production configuration for realtime TSMOM strategy.

Edit values below to change universe, lookbacks, or vol targeting.
All parameters are consumed by tsmom_realtime.ipynb.
"""

# ---------------------------------------------------------------------------
# Investment universe (yfinance tickers)
# ---------------------------------------------------------------------------
TICKERS: dict[str, str] = {
    "SMH":       "Semiconductors",
    "IGV":       "Software",
    "XAR":       "Aerospace",
    "XBI":       "Biotech",
    "XME":       "Metals&Mining",
    "GDX":       "GoldMiners",
    "XOP":       "Oil&Gas",
    "PAVE":      "InfraDev",
    "MGK":       "Growth",
    "MGV":       "Value",
    "IWM":       "Small",
    "SCHD":      "Dividend",
    "USMV":      "MinVol",
    "MTUM":      "Momentum",
    "QUAL":      "Quality",
    "372330.KS": "HangSengTech",
    "487230.KS": "AI-Power",
    "BOTZ":      "Robot",
    "SKYY":      "CloudComputing",
    "ICLN":      "CleanEnergy",
    "AIQ":       "AI-Tech"
}

# ---------------------------------------------------------------------------
# Sample period
# ---------------------------------------------------------------------------
START_DATE: str = "2006-01-01"   # ETF universe fully available from ~2006
END_DATE: str | None = None       # None = latest available

# ---------------------------------------------------------------------------
# Trend signal lookbacks (months)
# ---------------------------------------------------------------------------
K_SLOW: int = 12
K_FAST: int = 2

# ---------------------------------------------------------------------------
# Vol scaling
# ---------------------------------------------------------------------------
VOL_SCALING: bool = True
VOL_MODE: str = "realtime"        # "realtime" for production (tradeable)
SCALE_LEVEL: str = "portfolio"    # "portfolio" or "asset"
TARGET_VOL: float = 0.10          # annualized
VOL_LOOKBACK: int = 36            # months of rolling std

# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------
# Round-trip cost in basis points, applied to one-way turnover
# (0.5 * sum of |Δw_i|). A full position flip of a 1/N weight (one-way
# turnover = 1/N) costs (1/N) * TCOST_BPS/1e4. Set to 0 for a frictionless
# backtest. Typical ETF universe: 10–30 bps round-trip.
TCOST_BPS: float = 20.0
