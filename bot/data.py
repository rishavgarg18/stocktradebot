"""OHLCV data layer backed by Yahoo Finance with an on-disk cache.

Daily bars are cached per-symbol as pickled DataFrames and refreshed when
stale (default: 12 hours), so a full Nifty 500 rescan after the first run
costs almost nothing.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from .storage import CACHE_BASE
from .universe import to_yahoo

CACHE_DIR = CACHE_BASE / "ohlcv"
CACHE_TTL_SECONDS = 12 * 3600
HISTORY_PERIOD = "2y"       # ~500 trading days downloaded per symbol
MIN_ANALYSIS_BARS = 22      # ~1 calendar month of trading sessions
RECENT_WINDOW = 22          # recent-regime checks use last month, not last week
BATCH_SIZE = 50

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(symbol: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("&", "_AND_").replace("-", "_").replace("^", "IDX_")
    return CACHE_DIR / f"{safe}.pkl"


def _is_fresh(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def _clean(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df[OHLCV_COLS].dropna(subset=["Close"])
    if len(df) < 60:  # not enough history to compute anything meaningful
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def get_ltp(symbol: str) -> float | None:
    """Latest price for open-position marks — always hits Yahoo, not the 12h cache."""
    ticker = symbol if symbol.startswith("^") else to_yahoo(symbol)
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi:
            for key in ("lastPrice", "last_price", "regularMarketPrice"):
                val = fi.get(key) if hasattr(fi, "get") else getattr(fi, key, None)
                if val and float(val) > 0:
                    return round(float(val), 2)
        df = t.history(period="5d", auto_adjust=True)
        if df is not None and not df.empty:
            return round(float(df["Close"].iloc[-1]), 2)
    except Exception:
        pass
    df = get_history(symbol, force_refresh=True)
    if df is not None and not df.empty:
        return round(float(df["Close"].iloc[-1]), 2)
    return None


def get_history(symbol: str, force_refresh: bool = False) -> pd.DataFrame | None:
    """Daily OHLCV for one NSE symbol (or an index ticker like ^NSEI)."""
    path = _cache_path(symbol)
    if not force_refresh and _is_fresh(path):
        return pd.read_pickle(path)

    ticker = symbol if symbol.startswith("^") else to_yahoo(symbol)
    try:
        df = yf.Ticker(ticker).history(period=HISTORY_PERIOD, auto_adjust=True)
    except Exception:
        df = None
    df = _clean(df) if df is not None else None
    if df is not None:
        df.to_pickle(path)
        return df
    # fall back to stale cache rather than nothing
    if path.exists():
        return pd.read_pickle(path)
    return None


def get_histories(
    symbols: list[str], force_refresh: bool = False, progress_cb=None
) -> dict[str, pd.DataFrame]:
    """Bulk download. Uses yfinance batch download for symbols missing from cache."""
    out: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for sym in symbols:
        path = _cache_path(sym)
        if not force_refresh and _is_fresh(path):
            out[sym] = pd.read_pickle(path)
        else:
            missing.append(sym)

    done = len(out)
    if progress_cb:
        progress_cb(done, len(symbols))

    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i : i + BATCH_SIZE]
        tickers = [to_yahoo(s) for s in batch]
        try:
            raw = yf.download(
                tickers,
                period=HISTORY_PERIOD,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception:
            raw = None

        for sym, ticker in zip(batch, tickers):
            df = None
            if raw is not None and not raw.empty:
                try:
                    sub = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
                    df = _clean(sub)
                except KeyError:
                    df = None
            if df is not None:
                df.to_pickle(_cache_path(sym))
                out[sym] = df
            elif _cache_path(sym).exists():
                out[sym] = pd.read_pickle(_cache_path(sym))
            done += 1
            if progress_cb:
                progress_cb(done, len(symbols))

    return out
