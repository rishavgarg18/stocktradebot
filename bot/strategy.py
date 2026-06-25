"""Composite swing-trading strategy engine.

Scores each stock 0-100 across four pillars drawn from classic trading
literature, then emits a Signal with a full trade plan:

  Trend (30)     - Weinstein/Minervini: price above rising 50 & 200 DMA,
                   moving averages stacked, ADX confirming the trend.
  Momentum (25)  - O'Neil/CANSLIM: near 52-week high, outperforming the
                   Nifty over 3 months, positive 6-month momentum.
  Timing (25)    - Buy pullbacks within uptrends: RSI reset and recovering,
                   MACD turning up, price not over-extended above the mean.
  Volume (20)    - Wyckoff: accumulation (OBV rising), demand on up days.

BUY requires a high composite score AND the trend gate (never buy a
downtrend, no matter how good everything else looks). SELL fires on trend
breaks and distribution. Stops are ATR-based (Van Tharp), targets at 2R/3R,
and position size is computed for a fixed 1% account risk per trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .indicators import enrich

DEFAULT_CAPITAL = 1_000_000  # INR, used for position sizing illustration
RISK_PER_TRADE = 0.01        # 1% of capital risked per trade
BUY_THRESHOLD = 70
SELL_THRESHOLD = 35
RECENT_WINDOW = 22           # ~1 calendar month of trading days for recent checks


@dataclass
class Signal:
    symbol: str
    action: str                  # BUY | SELL | HOLD | WATCH
    score: float
    close: float
    entry: float | None = None
    stop_loss: float | None = None
    target1: float | None = None
    target2: float | None = None
    risk_per_share: float | None = None
    qty_for_1pct_risk: int | None = None
    rs_vs_nifty_3m: float | None = None  # percentage points vs ^NSEI
    pct_from_52w_high: float | None = None
    rsi: float | None = None
    adx: float | None = None
    pillar_scores: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    as_of: str = ""


def _trend_pillar(row) -> tuple[float, list[str]]:
    score, reasons = 0.0, []
    if row.Close > row.sma200:
        score += 8
        reasons.append("Price above 200 DMA (long-term uptrend)")
    if row.Close > row.sma50:
        score += 6
        reasons.append("Price above 50 DMA")
    if row.sma50 > row.sma200:
        score += 6
        reasons.append("50 DMA above 200 DMA (golden alignment)")
    if row.sma50_slope > 0:
        score += 4
        reasons.append("50 DMA rising")
    if row.sma200_slope > 0:
        score += 2
        reasons.append("200 DMA rising")
    if row.adx14 >= 20:
        score += 4
        reasons.append(f"ADX {row.adx14:.0f} confirms trend strength")
    return min(score, 30), reasons


def _momentum_pillar(row, nifty_roc63: float) -> tuple[float, list[str]]:
    score, reasons = 0.0, []
    if row.pct_from_52w_high > -15:
        score += 8
        reasons.append(f"Within {abs(row.pct_from_52w_high):.0f}% of 52-week high")
    elif row.pct_from_52w_high > -25:
        score += 4
    rs = row.roc63 - nifty_roc63
    if rs > 0:
        score += 8 if rs > 5 else 5
        reasons.append(f"Outperforming Nifty by {rs:.1f}pp over 3 months")
    if row.roc126 > 0:
        score += 5
        reasons.append("Positive 6-month momentum")
    if 50 <= row.rsi14 <= 75:
        score += 4
        reasons.append(f"RSI {row.rsi14:.0f} in bullish zone")
    return min(score, 25), reasons


def _timing_pillar(row, df: pd.DataFrame) -> tuple[float, list[str]]:
    score, reasons = 0.0, []
    recent = df.iloc[-RECENT_WINDOW:]
    # Pullback-and-recover: touched the 21 EMA / 50 DMA zone recently,
    # now closing back above the 21 EMA.
    touched_support = (recent["Low"] <= recent["ema21"]).any()
    if touched_support and row.Close > row.ema21:
        score += 8
        reasons.append("Pullback to support, now reclaiming 21 EMA")
    if row.macd_hist > 0 and df["macd_hist"].iloc[-2] <= row.macd_hist:
        score += 6
        reasons.append("MACD histogram positive and improving")
    prev_rsi = df["rsi14"].iloc[-5:-1]
    if (prev_rsi < 50).any() and row.rsi14 > 50:
        score += 5
        reasons.append("RSI recovering above 50")
    extension = (row.Close - row.sma20) / row.atr14 if row.atr14 else 0
    if extension < 2:
        score += 6
        reasons.append("Not over-extended above 20 DMA")
    return min(score, 25), reasons


def _volume_pillar(row, df: pd.DataFrame) -> tuple[float, list[str]]:
    score, reasons = 0.0, []
    if row.obv > row.obv_sma20:
        score += 8
        reasons.append("OBV above its average (accumulation)")
    recent = df.iloc[-RECENT_WINDOW:]
    up_days = recent[recent["Close"] > recent["Close"].shift()]
    down_days = recent[recent["Close"] < recent["Close"].shift()]
    if len(up_days) and len(down_days):
        if up_days["Volume"].mean() > down_days["Volume"].mean():
            score += 7
            reasons.append("Up-day volume exceeds down-day volume")
    elif len(up_days):
        score += 7
    if row.Volume > 1.2 * row.vol_avg20 and row.Close > df["Close"].iloc[-2]:
        score += 5
        reasons.append("Today's up-move on above-average volume")
    return min(score, 20), reasons


def _sell_checks(row, df: pd.DataFrame) -> list[str]:
    flags = []
    if row.Close < row.sma200:
        flags.append("Price broke below 200 DMA")
    if (
        len(df) > 21
        and row.sma50 < row.sma200
        and df["sma50"].iloc[-21] >= df["sma200"].iloc[-21]
    ):
        flags.append("Fresh death cross (50 DMA under 200 DMA)")
    if row.Close < row.sma50 and row.macd_hist < 0 and row.rsi14 < 45:
        flags.append("Below 50 DMA with negative MACD and weak RSI")
    if row.pct_from_52w_high < -25:
        flags.append(f"{abs(row.pct_from_52w_high):.0f}% below 52-week high")
    recent = df.iloc[-RECENT_WINDOW:]
    down_days = recent[recent["Close"] < recent["Close"].shift()]
    if len(down_days) >= 12 and down_days["Volume"].mean() > row.vol_avg20:
        flags.append("Heavy-volume distribution over the last month")
    return flags


def evaluate(
    symbol: str,
    df: pd.DataFrame,
    nifty_roc63: float = 0.0,
    capital: float = DEFAULT_CAPITAL,
) -> Signal | None:
    """Evaluate one stock. df is raw OHLCV; indicators are computed here."""
    if df is None or len(df) < 210:
        return None

    e = enrich(df)
    e["sma50_slope"] = e["sma50"].diff(5)
    e["sma200_slope"] = e["sma200"].diff(10)
    e = e.dropna(subset=["sma200", "atr14"])
    if e.empty:
        return None

    row = e.iloc[-1]

    t_score, t_reasons = _trend_pillar(row)
    m_score, m_reasons = _momentum_pillar(row, nifty_roc63)
    e_score, e_reasons = _timing_pillar(row, e)
    v_score, v_reasons = _volume_pillar(row, e)
    score = t_score + m_score + e_score + v_score
    sell_flags = _sell_checks(row, e)

    trend_gate = row.Close > row.sma200 and row.sma50 > row.sma200

    if score >= BUY_THRESHOLD and trend_gate and not sell_flags:
        action = "BUY"
    elif sell_flags and (len(sell_flags) >= 2 or score <= SELL_THRESHOLD):
        action = "SELL"
    elif score >= BUY_THRESHOLD - 10 and trend_gate:
        action = "WATCH"
    else:
        action = "HOLD"

    sig = Signal(
        symbol=symbol,
        action=action,
        score=round(score, 1),
        close=round(float(row.Close), 2),
        rs_vs_nifty_3m=round(float(row.roc63 - nifty_roc63), 2),
        pct_from_52w_high=round(float(row.pct_from_52w_high), 2),
        rsi=round(float(row.rsi14), 1),
        adx=round(float(row.adx14), 1),
        pillar_scores={
            "trend": round(t_score, 1),
            "momentum": round(m_score, 1),
            "timing": round(e_score, 1),
            "volume": round(v_score, 1),
        },
        reasons=t_reasons + m_reasons + e_reasons + v_reasons,
        warnings=sell_flags,
        as_of=str(e.index[-1].date()),
    )

    if action in ("BUY", "WATCH"):
        entry = float(row.Close)
        swing_low = float(e["Low"].iloc[-10:].min())
        stop = min(entry - 2 * float(row.atr14), swing_low * 0.995)
        stop = round(stop, 2)
        risk = entry - stop
        if risk <= 0:
            return sig
        sig.entry = round(entry, 2)
        sig.stop_loss = stop
        sig.target1 = round(entry + 2 * risk, 2)
        sig.target2 = round(entry + 3 * risk, 2)
        sig.risk_per_share = round(risk, 2)
        sig.qty_for_1pct_risk = int((capital * RISK_PER_TRADE) / risk)

    return sig


def nifty_roc63_from(df: pd.DataFrame | None) -> float:
    if df is None or len(df) < 64:
        return 0.0
    return float(df["Close"].pct_change(63).iloc[-1] * 100)
