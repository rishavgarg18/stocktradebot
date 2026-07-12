"""Event-driven catalyst proxies from OHLCV only.

Detects recent market activity that may indicate news, orders, or sentiment
shifts — does not know *why*; flags symbols worth manual follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .indicators import sma

VOLUME_SURGE_RATIO = 3.0
GAP_UP_PCT = 2.0
HIGH_VOL_RATIO = 2.0
BREAKOUT_LOOKBACK = 60
MAX_CATALYST_SCORE = 10


@dataclass
class CatalystResult:
    score: float = 0.0
    alerts: list[str] = field(default_factory=list)
    volume_ratio: float | None = None
    gap_pct: float | None = None


def detect_catalysts(df: pd.DataFrame) -> CatalystResult:
    """Score 0–10 from recent volume/gap/breakout signals on latest bar."""
    if df is None or len(df) < 25:
        return CatalystResult()

    score = 0.0
    alerts: list[str] = []
    vol = df["Volume"].astype(float)
    avg_vol = float(sma(vol, 20).iloc[-1])
    if avg_vol != avg_vol or avg_vol <= 0:
        avg_vol = float(vol.iloc[-21:-1].mean()) or 1.0

    last = df.iloc[-1]
    prev = df.iloc[-2]
    last_vol = float(last.Volume)
    vol_ratio = last_vol / avg_vol

    open_px = float(last.Open)
    prev_close = float(prev.Close)
    gap_pct = (open_px / prev_close - 1) * 100 if prev_close > 0 else 0.0
    close_px = float(last.Close)
    high_vol = vol_ratio >= HIGH_VOL_RATIO

    # Volume surge
    if vol_ratio >= VOLUME_SURGE_RATIO:
        pts = 4 if vol_ratio >= 5 else 3
        score += pts
        alerts.append(f"Volume {vol_ratio:.1f}× avg — check for news/catalyst")

    # Gap-up on volume (announcement-day pattern)
    if gap_pct >= GAP_UP_PCT and high_vol and close_px >= open_px * 0.995:
        score += 3
        alerts.append(f"Gap-up +{gap_pct:.1f}% on heavy volume")

    # Breakout on volume
    lookback = min(BREAKOUT_LOOKBACK, len(df) - 1)
    prior_high = float(df["High"].iloc[-lookback - 1 : -1].max())
    if close_px >= prior_high * 0.998 and high_vol:
        score += 3
        alerts.append(f"{lookback}d high breakout on {vol_ratio:.1f}× volume")

    return CatalystResult(
        score=min(score, MAX_CATALYST_SCORE),
        alerts=alerts,
        volume_ratio=round(vol_ratio, 2),
        gap_pct=round(gap_pct, 2),
    )
