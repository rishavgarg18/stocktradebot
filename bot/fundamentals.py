"""Company fundamentals via Yahoo Finance, with a quality gate.

Fundamentals barely move a stock over a 2-day hold, but they matter as a
*quality filter*: dip-buying works on profitable, growing companies whose
dips get bought back, and fails on junk that dips because it deserves to.

`.info` is one HTTP call per ticker, so results are cached for 7 days and
we only fetch fundamentals for stocks that already pass the technical screen.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yfinance as yf

from .storage import CACHE_BASE
from .universe import to_yahoo

CACHE_DIR = CACHE_BASE / "fundamentals"
CACHE_TTL_SECONDS = 7 * 24 * 3600

FIELDS = [
    "sector",
    "industry",
    "marketCap",
    "returnOnEquity",
    "profitMargins",
    "operatingMargins",
    "revenueGrowth",
    "earningsGrowth",
    "debtToEquity",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "bookValue",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "freeCashflow",
]


def _cache_path(symbol: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("&", "_AND_").replace("-", "_")
    return CACHE_DIR / f"{safe}.json"


def get_fundamentals(symbol: str) -> dict:
    path = _cache_path(symbol)
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            if time.time() - payload["fetched_at"] < CACHE_TTL_SECONDS:
                return payload["data"]
        except (json.JSONDecodeError, KeyError):
            pass

    data: dict = {}
    try:
        info = yf.Ticker(to_yahoo(symbol)).info or {}
        data = {k: info.get(k) for k in FIELDS}
    except Exception:
        pass

    path.write_text(json.dumps({"fetched_at": time.time(), "data": data}))
    return data


def quality_check(symbol: str) -> tuple[bool, list[str], list[str]]:
    """Return (passed, positives, negatives) for the quality gate.

    Pass = at least 2 quality positives and no hard fail. Missing data is
    treated leniently (some NSE tickers have sparse Yahoo fundamentals).
    """
    f = get_fundamentals(symbol)
    positives: list[str] = []
    negatives: list[str] = []
    hard_fail = False

    roe = f.get("returnOnEquity")
    if roe is not None:
        if roe >= 0.12:
            positives.append(f"ROE {roe * 100:.0f}%")
        elif roe < 0:
            negatives.append(f"Negative ROE ({roe * 100:.0f}%)")
            hard_fail = True

    margin = f.get("profitMargins")
    if margin is not None:
        if margin >= 0.05:
            positives.append(f"Profit margin {margin * 100:.0f}%")
        elif margin < 0:
            negatives.append("Loss-making company")
            hard_fail = True

    rev_g = f.get("revenueGrowth")
    if rev_g is not None:
        if rev_g > 0:
            positives.append(f"Revenue growing {rev_g * 100:.0f}% YoY")
        elif rev_g < -0.10:
            negatives.append(f"Revenue shrinking {abs(rev_g) * 100:.0f}% YoY")

    earn_g = f.get("earningsGrowth")
    if earn_g is not None and earn_g > 0:
        positives.append(f"Earnings growing {earn_g * 100:.0f}% YoY")

    # Debt check skipped for financials, where high leverage is the business.
    sector = (f.get("sector") or "").lower()
    dte = f.get("debtToEquity")
    if dte is not None and "financial" not in sector and dte > 200:
        negatives.append(f"High debt/equity ({dte:.0f}%)")

    if not positives and not negatives:
        return False, [], ["No fundamental data — skipped (need verified financials)"]

    passed = not hard_fail and len(positives) >= 2 and len(negatives) <= 1
    return passed, positives, negatives
