"""Stock universes for NSE.

Index constituent lists are fetched from NSE's public archive CSVs and cached
on disk for 7 days. If NSE is unreachable, a bundled Nifty 50 list is used as
fallback so the bot always works.
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import time
from pathlib import Path

import requests

from .storage import CACHE_BASE

CACHE_DIR = CACHE_BASE
CACHE_TTL_SECONDS = 7 * 24 * 3600

INDEX_CSV_URLS = {
    "nifty50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "nifty100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "nifty200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "nifty500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "smallcap250": "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    "microcap250": "https://archives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
}

# Fallback so the bot still runs if NSE blocks/changes the archive endpoint.
NIFTY50_FALLBACK = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY",
    "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN",
    "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

NIFTY_INDEX_TICKER = "^NSEI"  # Nifty 50 index on Yahoo Finance


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"universe_{name}.json"


def _read_cache(name: str) -> list[str] | None:
    path = _cache_path(name)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        if time.time() - payload["fetched_at"] < CACHE_TTL_SECONDS:
            return payload["symbols"]
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _write_cache(name: str, symbols: list[str]) -> None:
    _cache_path(name).write_text(
        json.dumps({"fetched_at": time.time(), "symbols": symbols})
    )


def _download(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.SSLError:
        # NSE serves an incomplete TLS chain that Python's certifi store
        # rejects; the system curl validates via the OS keychain instead.
        result = subprocess.run(
            ["curl", "-sf", "--max-time", "20", "-A", HEADERS["User-Agent"], url],
            capture_output=True, text=True, check=True,
        )
        return result.stdout


def _fetch_from_nse(name: str) -> list[str]:
    text = _download(INDEX_CSV_URLS[name])
    reader = csv.DictReader(io.StringIO(text))
    symbols = [row["Symbol"].strip() for row in reader if row.get("Symbol")]
    if len(symbols) < 30:
        raise ValueError(f"suspiciously small universe from NSE: {len(symbols)}")
    return symbols


def get_universe(name: str = "nifty500") -> list[str]:
    """Return NSE symbols (without .NS suffix) for the given index."""
    name = name.lower()
    if name not in INDEX_CSV_URLS:
        raise ValueError(f"unknown universe {name!r}; options: {list(INDEX_CSV_URLS)}")

    cached = _read_cache(name)
    if cached:
        return cached

    try:
        symbols = _fetch_from_nse(name)
        _write_cache(name, symbols)
        return symbols
    except Exception:
        return list(NIFTY50_FALLBACK)


def get_multibagger_universe() -> list[str]:
    """Nifty 500 + Microcap 250 — broad coverage for sub-₹50 hunting."""
    seen: set[str] = set()
    out: list[str] = []
    for name in ("nifty500", "microcap250"):
        for sym in get_universe(name):
            if sym not in seen:
                seen.add(sym)
                out.append(sym)
    return out


def to_yahoo(symbol: str) -> str:
    return f"{symbol}.NS"


def from_yahoo(ticker: str) -> str:
    return ticker.removesuffix(".NS")
