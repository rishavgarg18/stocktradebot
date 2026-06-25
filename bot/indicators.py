"""Technical indicators implemented with pandas/numpy (no native deps).

All functions take a daily OHLCV DataFrame (columns: Open, High, Low, Close,
Volume) and return Series aligned to its index. `enrich()` computes the full
set used by the strategy engine and returns a widened copy of the frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift()
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_smooth = true_range(df).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr_smooth
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr_smooth
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(close, window)
    std = close.rolling(window).std()
    return mid + num_std * std, mid, mid - num_std * std


def roc(close: pd.Series, period: int) -> pd.Series:
    return close.pct_change(period) * 100


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with every indicator column the strategy needs."""
    out = df.copy()
    close = out["Close"]

    out["sma20"] = sma(close, 20)
    out["sma50"] = sma(close, 50)
    out["sma200"] = sma(close, 200)
    out["ema21"] = ema(close, 21)

    out["rsi14"] = rsi(close, 14)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(close)
    out["atr14"] = atr(out, 14)
    out["adx14"] = adx(out, 14)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = bollinger(close)

    out["roc20"] = roc(close, 20)
    out["roc63"] = roc(close, 63)   # ~3 months
    out["roc126"] = roc(close, 126)  # ~6 months

    out["vol_avg20"] = sma(out["Volume"], 20)
    out["obv"] = obv(out)
    out["obv_sma20"] = sma(out["obv"], 20)

    out["high_52w"] = close.rolling(252, min_periods=60).max()
    out["low_52w"] = close.rolling(252, min_periods=60).min()
    out["pct_from_52w_high"] = (close / out["high_52w"] - 1) * 100

    return out
