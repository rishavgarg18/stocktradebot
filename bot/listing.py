"""Listing age lookup — cached first-trade date from Yahoo max history."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

from .storage import CACHE_BASE
from .universe import to_yahoo

LISTING_CACHE_DIR = CACHE_BASE / "listing_age"
LISTING_TTL_SECONDS = 30 * 24 * 3600


def _cache_path(symbol: str) -> Path:
    LISTING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("&", "_AND_").replace("-", "_")
    return LISTING_CACHE_DIR / f"{safe}.json"


def get_listing_age_years(symbol: str) -> float | None:
    """Years since first available daily bar on Yahoo (proxy for listing age)."""
    symbol = symbol.upper()
    path = _cache_path(symbol)
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            if time.time() - payload["fetched_at"] < LISTING_TTL_SECONDS:
                return payload.get("age_years")
        except (json.JSONDecodeError, KeyError):
            pass

    age_years: float | None = None
    first_date: str | None = None
    try:
        hist = yf.Ticker(to_yahoo(symbol)).history(period="max", auto_adjust=True)
        if hist is not None and not hist.empty:
            first = hist.index[0].to_pydatetime().replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            age_years = round((now - first).days / 365.25, 1)
            first_date = str(first.date())
    except Exception:
        pass

    path.write_text(json.dumps({
        "fetched_at": time.time(),
        "age_years": age_years,
        "first_date": first_date,
    }))
    return age_years
