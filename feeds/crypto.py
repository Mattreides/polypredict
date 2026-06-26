"""
polypredict/feeds/crypto.py

OHLCV feed + feature engineering for short-timeframe crypto direction modeling.

Data source: Kraken public OHLC REST API — no key, US-accessible.
(The original used Binance via ccxt; Binance now returns HTTP 451 for US IPs,
so this version uses Kraken over plain requests. Set EXCHANGE below to switch.)
"""

import requests
import numpy as np
import pandas as pd

UA = "polypredict/1.0 (research; contact: you@example.com)"

KRAKEN_OHLC = "https://api.kraken.com/0/public/OHLC"

# Kraken OHLC interval is in MINUTES. These are the natively supported steps.
TF_MIN = {"15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}

# Friendly symbol -> Kraken pair. Kraken uses XBT for BTC.
SYMBOL_MAP = {
    "BTC/USD": "XBTUSD",
    "ETH/USD": "ETHUSD",
    "SOL/USD": "SOLUSD",
    "XRP/USD": "XRPUSD",
    "DOGE/USD": "DOGEUSD",
    "LTC/USD": "LTCUSD",
    "ADA/USD": "ADAUSD",
}

DEFAULT_SYMBOLS = list(SYMBOL_MAP.keys())
DEFAULT_TIMEFRAMES = list(TF_MIN.keys())


def fetch_ohlcv(symbol: str = "BTC/USD", timeframe: str = "1h", limit: int = 720) -> pd.DataFrame:
    """Return a DataFrame [time, open, high, low, close, volume] of recent candles."""
    pair = SYMBOL_MAP.get(symbol, symbol.replace("/", ""))
    interval = TF_MIN.get(timeframe, 60)
    r = requests.get(
        KRAKEN_OHLC,
        params={"pair": pair, "interval": interval},
        headers={"User-Agent": UA},
        timeout=20,
    )
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        raise RuntimeError("Kraken error: " + ", ".join(j["error"]))
    result = j["result"]
    key = next(k for k in result if k != "last")
    rows = result[key]  # [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    for c in ["open", "high", "low", "close", "vwap", "volume"]:
        df[c] = df[c].astype(float)
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    return df[["time", "open", "high", "low", "close", "volume"]].tail(limit).reset_index(drop=True)


# ── indicator helpers (pure pandas/numpy, no TA-Lib) ───────────────────────────
def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _stoch(df: pd.DataFrame, n: int = 14, d: int = 3):
    ll = df["low"].rolling(n).min()
    hh = df["high"].rolling(n).max()
    k = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    return k.fillna(50), k.rolling(d).mean().fillna(50)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff()).fillna(0)
    return (sign * df["volume"]).cumsum()


FEATURE_COLS = [
    "rsi14", "rsi6", "macd_n", "stoch_k", "stoch_d",
    "ema9_21", "ema21_50", "px_ema9", "px_ema21",
    "bb_width", "bb_pos", "atr_pct",
    "vol_ratio", "obv_slope",
    "ret1", "ret2", "ret3", "ret5", "ret10",
]


def build_features(df: pd.DataFrame):
    """Return (X, y, feat_df). 19 dimensionless features; y = next-bar up (1/0)."""
    c = df["close"]
    f = pd.DataFrame(index=df.index)

    # momentum
    f["rsi14"] = _rsi(c, 14) / 100
    f["rsi6"] = _rsi(c, 6) / 100
    ema12, ema26 = _ema(c, 12), _ema(c, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    f["macd_n"] = (macd - signal) / c            # normalized by price -> scale-invariant
    k, d = _stoch(df)
    f["stoch_k"] = k / 100
    f["stoch_d"] = d / 100

    # trend
    ema9, ema21, ema50 = _ema(c, 9), _ema(c, 21), _ema(c, 50)
    f["ema9_21"] = (ema9 - ema21) / c
    f["ema21_50"] = (ema21 - ema50) / c
    f["px_ema9"] = (c - ema9) / c
    f["px_ema21"] = (c - ema21) / c

    # volatility
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    f["bb_width"] = (4 * sd) / mid
    f["bb_pos"] = (c - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)
    f["atr_pct"] = _atr(df) / c

    # volume
    f["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    obv = _obv(df)
    f["obv_slope"] = (obv - obv.shift(5)) / df["volume"].rolling(20).mean().replace(0, np.nan)

    # price-lag returns
    for lag in (1, 2, 3, 5, 10):
        f[f"ret{lag}"] = c.pct_change(lag)

    f = f[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    y = (c.shift(-1) > c).astype(int)            # 1 if next bar closes higher

    valid = f.dropna().index
    valid = valid[valid <= df.index[-1]]
    X = f.loc[valid]
    y_full = y.loc[valid]
    return X, y_full, f
