"""Multibagger hunter — sub-₹50 stocks with strong fundamentals & growth tailwinds.

This is a *research screener*, not a trading signal. It looks for small, cheap
stocks that combine:
  - Price below ₹50 (early-stage valuation territory)
  - Profitable, growing fundamentals (ROE, revenue, margins)
  - Sectors with structural demand (auto, defense, renewables, manufacturing…)
  - Technical accumulation (not dead money)

Hard anti-trap filters reject loss-makers, shrinking revenue, extreme debt,
illiquid shells, and distressed PSU banks. No screener guarantees the next
MRF or Maruti — use this as a shortlist for deep manual research.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .fundamentals import get_fundamentals
from .indicators import enrich, roc, sma

MAX_PRICE = 50.0
MIN_PRICE = 5.0
MIN_MARKET_CAP = 500_000_000  # ₹50 Cr — filters pure shell penny stocks
MIN_AVG_TURNOVER = 10_000_000  # ₹1 Cr/day average traded value
QUALIFIED_SCORE = 70
WATCH_SCORE = 55
MAX_RESULTS = 15

# Sector tailwinds — industries that benefited early Maruti/MRF-type stories
GROWTH_SECTOR_KEYWORDS: dict[str, float] = {
    "auto": 10,
    "automobile": 10,
    "auto components": 10,
    "electric": 9,
    "renewable": 9,
    "solar": 9,
    "defence": 9,
    "defense": 9,
    "aerospace": 9,
    "capital goods": 8,
    "industrial": 8,
    "engineering": 8,
    "chemical": 7,
    "specialty": 7,
    "pharma": 7,
    "healthcare": 7,
    "electronic": 8,
    "semiconductor": 9,
    "technology": 6,
    "consumer": 6,
    "fmcg": 5,
    "infrastructure": 7,
    "power": 6,
    "metals": 5,
    "mining": 5,
    "textile": 4,
    "real estate": 3,
    "financial": 2,
    "bank": 1,
    "insurance": 2,
}

# Yahoo sector names → base theme score
SECTOR_SCORE: dict[str, float] = {
    "consumer cyclical": 8,
    "industrials": 9,
    "basic materials": 6,
    "healthcare": 7,
    "technology": 7,
    "consumer defensive": 5,
    "energy": 6,
    "utilities": 5,
    "communication services": 4,
    "financial services": 2,
    "real estate": 3,
}


@dataclass
class MultibaggerSignal:
    symbol: str
    close: float
    score: float
    tier: str  # qualified | watch
    market_cap_cr: float | None
    roe_pct: float | None
    revenue_growth_pct: float | None
    earnings_growth_pct: float | None
    profit_margin_pct: float | None
    debt_to_equity: float | None
    sector: str
    industry: str
    pct_from_52w_high: float | None
    roc_6m_pct: float | None
    turnover_cr: float | None
    pillar_scores: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    as_of: str = ""


def _theme_score(sector: str, industry: str) -> tuple[float, list[str]]:
    """Score 0-10 based on sector/industry tailwinds."""
    text = f"{sector} {industry}".lower()
    best = 0.0
    hits: list[str] = []
    for kw, pts in GROWTH_SECTOR_KEYWORDS.items():
        if kw in text and pts > best:
            best = pts
            hits = [f"Growth theme: {kw}"]
    base = SECTOR_SCORE.get(sector.lower(), 4)
    score = min(10, max(best, base * 0.6))
    if score >= 7 and not hits:
        hits.append(f"Favourable sector: {sector or industry}")
    return score, hits


def _fundamental_score(f: dict) -> tuple[float, list[str], list[str], list[str]]:
    """Score 0-50 from fundamentals. Returns (score, reasons, warnings, red_flags)."""
    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []
    red_flags: list[str] = []

    roe = f.get("returnOnEquity")
    if roe is not None:
        if roe >= 0.18:
            score += 12
            reasons.append(f"Strong ROE {roe * 100:.0f}%")
        elif roe >= 0.12:
            score += 8
            reasons.append(f"Good ROE {roe * 100:.0f}%")
        elif roe >= 0.08:
            score += 4
        elif roe < 0:
            red_flags.append(f"Negative ROE ({roe * 100:.0f}%)")

    margin = f.get("profitMargins")
    if margin is not None:
        if margin >= 0.10:
            score += 10
            reasons.append(f"Healthy margins {margin * 100:.0f}%")
        elif margin >= 0.05:
            score += 6
        elif margin >= 0.02:
            score += 2
        elif margin < 0:
            red_flags.append("Loss-making (negative margins)")

    rev_g = f.get("revenueGrowth")
    if rev_g is not None:
        if rev_g >= 0.20:
            score += 12
            reasons.append(f"Revenue +{rev_g * 100:.0f}% YoY")
        elif rev_g >= 0.10:
            score += 8
            reasons.append(f"Revenue growing {rev_g * 100:.0f}% YoY")
        elif rev_g >= 0.03:
            score += 4
        elif rev_g < -0.05:
            red_flags.append(f"Revenue shrinking {abs(rev_g) * 100:.0f}% YoY")

    earn_g = f.get("earningsGrowth")
    if earn_g is not None:
        if earn_g >= 0.25:
            score += 8
            reasons.append(f"Earnings +{earn_g * 100:.0f}% YoY")
        elif earn_g >= 0.10:
            score += 5
        elif earn_g < -0.20:
            warnings.append(f"Earnings fell {abs(earn_g) * 100:.0f}% YoY")

    dte = f.get("debtToEquity")
    sector = (f.get("sector") or "").lower()
    if dte is not None and "financial" not in sector:
        if dte <= 50:
            score += 4
            reasons.append("Low debt")
        elif dte <= 100:
            score += 2
        elif dte > 250:
            red_flags.append(f"High debt/equity ({dte:.0f}%)")

    fcf = f.get("freeCashflow")
    if fcf is not None and fcf > 0:
        score += 4
        reasons.append("Positive free cash flow")

    return min(score, 50), reasons, warnings, red_flags


def _technical_score(df: pd.DataFrame) -> tuple[float, list[str], list[str]]:
    """Score 0-25 from price action / accumulation."""
    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []

    if df is None or len(df) < 120:
        return 0, [], ["Insufficient price history"]

    e = enrich(df)
    row = e.iloc[-1]
    close = float(row.Close)

    r6 = float(roc(e["Close"], 126).iloc[-1]) if len(e) > 126 else 0
    if r6 > 30:
        score += 8
        reasons.append(f"Strong 6M momentum +{r6:.0f}%")
    elif r6 > 10:
        score += 5
        reasons.append(f"Positive 6M momentum +{r6:.0f}%")
    elif r6 > 0:
        score += 2
    elif r6 < -20:
        warnings.append(f"Weak 6M trend ({r6:.0f}%)")

    if row.sma200 == row.sma200:
        if close > row.sma200:
            score += 6
            reasons.append("Above 200 DMA (long-term uptrend)")
        elif close > row.sma200 * 0.90:
            score += 3
            reasons.append("Near 200 DMA — possible base")

    if row.sma50 == row.sma50 and close > row.sma50:
        score += 4
        reasons.append("Above 50 DMA")

    if row.obv > row.obv_sma20:
        score += 4
        reasons.append("Volume accumulation (OBV rising)")

    hi_52 = float(e["High"].iloc[-252:].max()) if len(e) >= 252 else float(e["High"].max())
    pct_from_high = (close / hi_52 - 1) * 100
    if -40 <= pct_from_high <= -10:
        score += 3
        reasons.append(f"{abs(pct_from_high):.0f}% below 52W high — room to run")
    elif pct_from_high > -5:
        warnings.append("Near 52-week high — limited upside near-term")

    return min(score, 25), reasons, warnings


def _hard_fail(
    f: dict,
    close: float,
    turnover: float,
    red_flags: list[str],
) -> list[str]:
    """Return list of hard failure reasons (empty = passes)."""
    fails: list[str] = list(red_flags)

    mcap = f.get("marketCap")
    if mcap is not None and mcap < MIN_MARKET_CAP:
        fails.append(f"Market cap too small (₹{mcap / 1e7:.0f} Cr)")

    if close < MIN_PRICE:
        fails.append(f"Extreme penny (< ₹{MIN_PRICE:.0f})")

    if turnover < MIN_AVG_TURNOVER:
        fails.append("Illiquid (low daily turnover)")

    sector = (f.get("sector") or "").lower()
    industry = (f.get("industry") or "").lower()
    roe = f.get("returnOnEquity")
    margin = f.get("profitMargins")

    # Distressed PSU-style banks under ₹50 are rarely multibaggers
    if ("bank" in sector or "bank" in industry) and close < 50:
        if roe is not None and roe < 0.10:
            fails.append("Low-quality PSU/distressed bank")
        if margin is not None and margin < 0.15:
            fails.append("Thin-margin bank — not a growth compounder")

    if margin is not None and margin < 0:
        fails.append("Company is loss-making")

    if roe is not None and roe < 0:
        fails.append("Negative return on equity")

    rev_g = f.get("revenueGrowth")
    if rev_g is not None and rev_g < -0.10:
        fails.append("Revenue in serious decline")

    # Need at least some fundamental data
    if roe is None and margin is None and rev_g is None:
        fails.append("No verified fundamental data")

    return fails


def evaluate_multibagger(symbol: str, df: pd.DataFrame) -> MultibaggerSignal | None:
    """Evaluate one symbol. Returns None if price >= MAX_PRICE or hard fail."""
    if df is None or len(df) < 60:
        return None

    close = float(df["Close"].iloc[-1])
    if close >= MAX_PRICE or close < MIN_PRICE:
        return None

    turnover_series = df["Close"] * df["Volume"]
    turnover = float(sma(turnover_series, 20).iloc[-1])
    if turnover != turnover:
        turnover = float(turnover_series.iloc[-20:].mean())

    f = get_fundamentals(symbol)
    fund_score, fund_reasons, fund_warn, red_flags = _fundamental_score(f)
    theme_score, theme_reasons = _theme_score(f.get("sector") or "", f.get("industry") or "")
    tech_score, tech_reasons, tech_warn = _technical_score(df)

    fails = _hard_fail(f, close, turnover, red_flags)
    if fails:
        return None

    total = fund_score + theme_score * 1.5 + tech_score  # theme weighted ~15 pts max
    total = round(min(100, total), 1)

    tier = "qualified" if total >= QUALIFIED_SCORE else "watch" if total >= WATCH_SCORE else None
    if tier is None:
        return None

    hi_52 = f.get("fiftyTwoWeekHigh")
    pct_from_high = None
    if hi_52 and hi_52 > 0:
        pct_from_high = round((close / hi_52 - 1) * 100, 1)

    r6 = float(roc(df["Close"], 126).iloc[-1]) if len(df) > 126 else None
    mcap = f.get("marketCap")

    sig = MultibaggerSignal(
        symbol=symbol,
        close=round(close, 2),
        score=total,
        tier=tier,
        market_cap_cr=round(mcap / 1e7, 1) if mcap else None,
        roe_pct=round(f["returnOnEquity"] * 100, 1) if f.get("returnOnEquity") is not None else None,
        revenue_growth_pct=round(f["revenueGrowth"] * 100, 1) if f.get("revenueGrowth") is not None else None,
        earnings_growth_pct=round(f["earningsGrowth"] * 100, 1) if f.get("earningsGrowth") is not None else None,
        profit_margin_pct=round(f["profitMargins"] * 100, 1) if f.get("profitMargins") is not None else None,
        debt_to_equity=round(f["debtToEquity"], 1) if f.get("debtToEquity") is not None else None,
        sector=f.get("sector") or "—",
        industry=f.get("industry") or "—",
        pct_from_52w_high=pct_from_high,
        roc_6m_pct=round(r6, 1) if r6 is not None else None,
        turnover_cr=round(turnover / 1e7, 2),
        pillar_scores={
            "fundamentals": round(fund_score, 1),
            "growth_theme": round(theme_score * 1.5, 1),
            "technical": round(tech_score, 1),
        },
        reasons=fund_reasons + theme_reasons + tech_reasons,
        warnings=fund_warn + tech_warn,
        as_of=str(df.index[-1].date()),
    )
    return sig


def scan_multibagger(
    data: dict[str, pd.DataFrame],
    *,
    qualified_only: bool = False,
) -> tuple[list[MultibaggerSignal], dict]:
    """Scan universe for sub-₹50 multibagger candidates."""
    price_pass = 0
    scored = 0
    qualified = 0
    watch = 0
    results: list[MultibaggerSignal] = []

    for sym, df in data.items():
        if df is None or df.empty:
            continue
        close = float(df["Close"].iloc[-1])
        if close >= MAX_PRICE:
            continue
        price_pass += 1

        sig = evaluate_multibagger(sym, df)
        if sig is None:
            continue
        scored += 1
        if sig.tier == "qualified":
            qualified += 1
        else:
            watch += 1
        if qualified_only and sig.tier != "qualified":
            continue
        results.append(sig)

    results.sort(key=lambda s: s.score, reverse=True)
    results = results[:MAX_RESULTS]

    diagnostics = {
        "max_price": MAX_PRICE,
        "min_market_cap_cr": MIN_MARKET_CAP / 1e7,
        "symbols_scanned": len(data),
        "sub50_price_pass": price_pass,
        "scored": scored,
        "qualified": qualified,
        "watch": watch,
        "returned": len(results),
    }
    return results, diagnostics
