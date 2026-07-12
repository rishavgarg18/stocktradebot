"""Multibagger hunter — sub-₹100 stocks filtered by quality, value & catalyst proxies.

Primary gate: share price under ₹100, market cap ₹500–30,000 Cr, listed within 10 years.
Quality + valuation + governance scoring; optional event catalyst layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .catalysts import detect_catalysts
from .fundamentals import get_fundamentals
from .indicators import enrich, roc, sma
from .listing import get_listing_age_years

MAX_PRICE = 100.0
MIN_PRICE = 5.0
MIN_MARKET_CAP = 5_000_000_000       # ₹500 Cr
MAX_MARKET_CAP = 300_000_000_000     # ₹30,000 Cr
MAX_LISTING_AGE_YEARS = 10.0       # exclude listings older than 10 years
MIN_AVG_TURNOVER = 10_000_000       # ₹1 Cr/day
QUALIFIED_SCORE = 72
WATCH_SCORE = 58
MAX_RESULTS = 15

GOVERNANCE_BLOCKLIST = frozenset({
    "PCJEWELLER", "YESBANK", "UCOBANK", "CENTRALBK", "IOB", "SUZLON",
    "RCOM", "RPOWER", "RTNPOWER", "IDEA", "JPPOWER", "JPASSOCIAT",
    "GTL", "DHFL", "RELCAPITAL", "RELINFRA", "ALOKINDS", "VAKRANGEE",
})

SECTOR_MAX_FORWARD_PE: dict[str, float] = {
    "technology": 40, "healthcare": 35, "consumer cyclical": 32,
    "industrials": 30, "basic materials": 22, "energy": 18,
    "utilities": 18, "financial services": 18, "consumer defensive": 28,
    "communication services": 25, "real estate": 20,
}
DEFAULT_MAX_FORWARD_PE = 30.0

GROWTH_SECTOR_KEYWORDS: dict[str, float] = {
    "auto": 10, "automobile": 10, "auto components": 10, "auto parts": 10,
    "electric": 9, "renewable": 9, "solar": 9, "defence": 9, "defense": 9,
    "aerospace": 9, "capital goods": 8, "industrial": 8, "engineering": 8,
    "chemical": 7, "specialty": 7, "pharma": 7, "healthcare": 7,
    "electronic": 8, "semiconductor": 9, "technology": 6, "consumer": 6,
    "fmcg": 5, "infrastructure": 7, "power": 6, "metals": 5, "mining": 5,
    "textile": 4, "real estate": 3, "financial": 2, "bank": 1, "insurance": 2,
}

SECTOR_SCORE: dict[str, float] = {
    "consumer cyclical": 8, "industrials": 9, "basic materials": 6,
    "healthcare": 7, "technology": 7, "consumer defensive": 5,
    "energy": 6, "utilities": 5, "communication services": 4,
    "financial services": 2, "real estate": 3,
}


@dataclass
class MultibaggerSignal:
    symbol: str
    close: float
    score: float
    tier: str
    market_cap_cr: float | None
    roe_pct: float | None
    revenue_growth_pct: float | None
    earnings_growth_pct: float | None
    profit_margin_pct: float | None
    debt_to_equity: float | None
    forward_pe: float | None
    peg_ratio: float | None
    insider_pct: float | None
    sector: str
    industry: str
    pct_from_52w_high: float | None
    roc_6m_pct: float | None
    turnover_cr: float | None
    catalyst_score: float = 0.0
    catalyst_alerts: list[str] = field(default_factory=list)
    volume_ratio: float | None = None
    pillar_scores: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    as_of: str = ""


def _mcap_cr(mcap: float | None) -> float | None:
    return round(mcap / 1e7, 1) if mcap else None


def _compute_peg(f: dict) -> float | None:
    peg = f.get("pegRatio")
    if peg is not None and peg > 0:
        return float(peg)
    fpe = f.get("forwardPE")
    eg = f.get("earningsGrowth")
    if fpe and eg and eg > 0.05:
        return round(fpe / (eg * 100), 2)
    return None


def _theme_score(sector: str, industry: str) -> tuple[float, list[str]]:
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


def _quality_score(f: dict) -> tuple[float, list[str], list[str], list[str]]:
    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []
    red_flags: list[str] = []

    roe = f.get("returnOnEquity")
    if roe is not None:
        if roe >= 0.18:
            score += 8
            reasons.append(f"Strong ROE {roe * 100:.0f}%")
        elif roe >= 0.12:
            score += 5
            reasons.append(f"Good ROE {roe * 100:.0f}%")
        elif roe >= 0.08:
            score += 2
        elif roe < 0:
            red_flags.append(f"Negative ROE ({roe * 100:.0f}%)")

    margin = f.get("profitMargins")
    if margin is not None:
        if margin >= 0.10:
            score += 7
            reasons.append(f"Healthy margins {margin * 100:.0f}%")
        elif margin >= 0.05:
            score += 4
        elif margin < 0:
            red_flags.append("Loss-making")

    rev_g = f.get("revenueGrowth")
    if rev_g is not None:
        if rev_g >= 0.15:
            score += 8
            reasons.append(f"Revenue +{rev_g * 100:.0f}% YoY")
        elif rev_g >= 0.08:
            score += 5
        elif rev_g < -0.05:
            red_flags.append(f"Revenue shrinking {abs(rev_g) * 100:.0f}% YoY")

    earn_g = f.get("earningsGrowth")
    if earn_g is not None and earn_g >= 0.20:
        score += 4
        reasons.append(f"Earnings +{earn_g * 100:.0f}% YoY")

    fcf = f.get("freeCashflow")
    if fcf is not None and fcf > 0:
        score += 3
        reasons.append("Positive free cash flow")

    dte = f.get("debtToEquity")
    sector = (f.get("sector") or "").lower()
    if dte is not None and "financial" not in sector and dte > 250:
        red_flags.append(f"High debt/equity ({dte:.0f}%)")

    return min(score, 30), reasons, warnings, red_flags


def _valuation_score(f: dict) -> tuple[float, list[str], list[str], list[str]]:
    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []
    red_flags: list[str] = []

    sector = (f.get("sector") or "").lower()
    roe = f.get("returnOnEquity")
    max_pe = SECTOR_MAX_FORWARD_PE.get(sector, DEFAULT_MAX_FORWARD_PE)

    fpe = f.get("forwardPE")
    if fpe is not None and fpe > 0:
        if fpe <= max_pe * 0.65:
            score += 8
            reasons.append(f"Attractive forward PE {fpe:.0f}")
        elif fpe <= max_pe:
            score += 5
        elif fpe > max_pe * 1.4:
            red_flags.append(f"Expensive forward PE {fpe:.0f}")

    peg = _compute_peg(f)
    if peg is not None:
        if peg <= 1.2:
            score += 8
            reasons.append(f"PEG {peg:.1f}")
        elif peg <= 2.0:
            score += 4
        elif peg > 3.5:
            warnings.append(f"Rich PEG {peg:.1f}")

    ev = f.get("enterpriseToEbitda")
    if ev is not None and 0 < ev <= 15:
        score += 4
        reasons.append(f"EV/EBITDA {ev:.0f}x")

    pb = f.get("priceToBook")
    if pb is not None and roe and roe >= 0.15 and pb <= 6:
        score += 3

    return min(score, 25), reasons, warnings, red_flags


def _governance_score(symbol: str, f: dict) -> tuple[float, list[str], list[str], list[str]]:
    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []
    red_flags: list[str] = []

    if symbol in GOVERNANCE_BLOCKLIST:
        red_flags.append("Governance blocklist")

    insider = f.get("heldPercentInsiders")
    if insider is not None:
        pct = insider * 100
        if pct >= 40:
            score += 6
            reasons.append(f"Promoter holding {pct:.0f}%")
        elif pct >= 25:
            score += 3
        elif pct < 10:
            warnings.append(f"Low insider holding {pct:.0f}%")

    cr = f.get("currentRatio")
    if cr is not None:
        if cr >= 1.2:
            score += 3
        elif cr < 0.8:
            red_flags.append(f"Weak liquidity (CR {cr:.1f})")

    inst = f.get("heldPercentInstitutions")
    if inst is not None and 5 <= inst * 100 <= 45:
        score += 2

    return min(score, 15), reasons, warnings, red_flags


def _technical_score(df: pd.DataFrame) -> tuple[float, list[str], list[str]]:
    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []

    if df is None or len(df) < 120:
        return 0, [], ["Insufficient price history"]

    e = enrich(df)
    row = e.iloc[-1]
    close = float(row.Close)

    r6 = float(roc(e["Close"], 126).iloc[-1]) if len(e) > 126 else 0
    if 5 <= r6 <= 40:
        score += 5
        reasons.append(f"6M momentum +{r6:.0f}%")
    elif -5 <= r6 < 5:
        score += 2
    elif r6 < -25:
        warnings.append(f"Weak 6M trend ({r6:.0f}%)")

    if row.sma200 == row.sma200 and close > row.sma200:
        score += 5
        reasons.append("Above 200 DMA")
    elif row.sma200 == row.sma200 and close > row.sma200 * 0.88:
        score += 3

    if row.obv > row.obv_sma20:
        score += 4
        reasons.append("OBV accumulation")

    hi = float(e["High"].iloc[-252:].max()) if len(e) >= 252 else float(e["High"].max())
    pct = (close / hi - 1) * 100
    if -45 <= pct <= -12:
        score += 3

    return min(score, 20), reasons, warnings


def _hard_fail(
    symbol: str,
    f: dict,
    close: float,
    turnover: float,
    red_flags: list[str],
    listing_age_years: float | None,
) -> list[str]:
    fails = list(red_flags)

    mcap = f.get("marketCap")
    if mcap is not None:
        if mcap < MIN_MARKET_CAP:
            fails.append(f"Too small (₹{_mcap_cr(mcap):.0f} Cr)")
        elif mcap > MAX_MARKET_CAP:
            fails.append(f"Too large (₹{_mcap_cr(mcap):.0f} Cr > ₹{MAX_MARKET_CAP / 1e7:.0f} Cr cap)")

    if listing_age_years is not None and listing_age_years > MAX_LISTING_AGE_YEARS:
        fails.append(f"Listed {listing_age_years:.0f}y ago (> {MAX_LISTING_AGE_YEARS:.0f}y)")

    sector = (f.get("sector") or "").lower()
    industry = (f.get("industry") or "").lower()
    if sector == "basic materials" and any(k in industry for k in ("steel", "iron", "mining", "coal")):
        fails.append("Commodity miner/metal — cyclical, not multibagger profile")

    if close >= MAX_PRICE:
        fails.append(f"Price ≥ ₹{MAX_PRICE:.0f}")
    if close < MIN_PRICE:
        fails.append(f"Penny stock (< ₹{MIN_PRICE:.0f})")
    if turnover < MIN_AVG_TURNOVER:
        fails.append("Illiquid")

    if "bank" in sector or "bank" in industry:
        fails.append("Banks excluded")

    margin = f.get("profitMargins")
    if margin is not None and margin < 0:
        fails.append("Loss-making")

    roe = f.get("returnOnEquity")
    if roe is not None and roe < 0:
        fails.append("Negative ROE")

    rev_g = f.get("revenueGrowth")
    if rev_g is not None and rev_g < -0.08:
        fails.append("Revenue declining")

    if roe is None and margin is None and rev_g is None:
        fails.append("No fundamental data")

    if symbol in GOVERNANCE_BLOCKLIST:
        fails.append("Governance blocklist")

    return fails


def evaluate_multibagger(
    symbol: str,
    df: pd.DataFrame,
    *,
    catalyst_enabled: bool = True,
) -> MultibaggerSignal | None:
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
    listing_age = get_listing_age_years(symbol)
    qual, qual_r, qual_w, qual_f = _quality_score(f)
    val, val_r, val_w, val_f = _valuation_score(f)
    gov, gov_r, gov_w, gov_f = _governance_score(symbol, f)
    theme, theme_r = _theme_score(f.get("sector") or "", f.get("industry") or "")
    tech, tech_r, tech_w = _technical_score(df)

    cat = detect_catalysts(df) if catalyst_enabled else None
    cat_score = cat.score if cat else 0.0
    cat_alerts = cat.alerts if cat else []

    fails = _hard_fail(symbol, f, close, turnover, qual_f + val_f + gov_f, listing_age)
    if fails:
        return None

    total = qual + val + gov + theme + tech + cat_score
    total = round(min(100, total), 1)

    tier = "qualified" if total >= QUALIFIED_SCORE else "watch" if total >= WATCH_SCORE else None
    if tier is None:
        return None

    hi_52 = f.get("fiftyTwoWeekHigh")
    pct_hi = round((close / hi_52 - 1) * 100, 1) if hi_52 and hi_52 > 0 else None
    r6 = float(roc(df["Close"], 126).iloc[-1]) if len(df) > 126 else None
    mcap = f.get("marketCap")
    peg = _compute_peg(f)

    return MultibaggerSignal(
        symbol=symbol,
        close=round(close, 2),
        score=total,
        tier=tier,
        market_cap_cr=_mcap_cr(mcap),
        roe_pct=round(f["returnOnEquity"] * 100, 1) if f.get("returnOnEquity") is not None else None,
        revenue_growth_pct=round(f["revenueGrowth"] * 100, 1) if f.get("revenueGrowth") is not None else None,
        earnings_growth_pct=round(f["earningsGrowth"] * 100, 1) if f.get("earningsGrowth") is not None else None,
        profit_margin_pct=round(f["profitMargins"] * 100, 1) if f.get("profitMargins") is not None else None,
        debt_to_equity=round(f["debtToEquity"], 1) if f.get("debtToEquity") is not None else None,
        forward_pe=round(f["forwardPE"], 1) if f.get("forwardPE") is not None else None,
        peg_ratio=peg,
        insider_pct=round(f["heldPercentInsiders"] * 100, 1) if f.get("heldPercentInsiders") is not None else None,
        sector=f.get("sector") or "—",
        industry=f.get("industry") or "—",
        pct_from_52w_high=pct_hi,
        roc_6m_pct=round(r6, 1) if r6 is not None else None,
        turnover_cr=round(turnover / 1e7, 2),
        catalyst_score=round(cat_score, 1),
        catalyst_alerts=cat_alerts,
        volume_ratio=cat.volume_ratio if cat else None,
        pillar_scores={
            "quality": round(qual, 1),
            "valuation": round(val, 1),
            "governance": round(gov, 1),
            "growth_theme": round(theme, 1),
            "technical": round(tech, 1),
            "catalyst": round(cat_score, 1),
        },
        reasons=qual_r + val_r + gov_r + theme_r + tech_r + cat_alerts,
        warnings=qual_w + val_w + gov_w + tech_w,
        as_of=str(df.index[-1].date()),
    )


def scan_multibagger(
    data: dict[str, pd.DataFrame],
    *,
    qualified_only: bool = False,
    catalyst_enabled: bool = True,
) -> tuple[list[MultibaggerSignal], dict]:
    price_pass = 0
    scored = 0
    qualified = 0
    watch = 0
    with_catalyst = 0
    results: list[MultibaggerSignal] = []

    for sym, df in data.items():
        if df is None or df.empty:
            continue
        close = float(df["Close"].iloc[-1])
        if close >= MAX_PRICE or close < MIN_PRICE:
            continue
        price_pass += 1

        sig = evaluate_multibagger(sym, df, catalyst_enabled=catalyst_enabled)
        if sig is None:
            continue
        scored += 1
        if sig.catalyst_score > 0:
            with_catalyst += 1
        if sig.tier == "qualified":
            qualified += 1
        else:
            watch += 1
        if qualified_only and sig.tier != "qualified":
            continue
        results.append(sig)

    results.sort(key=lambda s: (s.score, s.catalyst_score), reverse=True)
    results = results[:MAX_RESULTS]

    diagnostics = {
        "max_price": MAX_PRICE,
        "min_market_cap_cr": MIN_MARKET_CAP / 1e7,
        "max_market_cap_cr": MAX_MARKET_CAP / 1e7,
        "max_listing_age_years": MAX_LISTING_AGE_YEARS,
        "symbols_scanned": len(data),
        "under100_price_pass": price_pass,
        "scored": scored,
        "with_catalyst_signal": with_catalyst,
        "qualified": qualified,
        "watch": watch,
        "returned": len(results),
        "catalyst_enabled": catalyst_enabled,
    }
    return results, diagnostics
