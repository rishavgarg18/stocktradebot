"""Backtest the strategy rules on a single stock's history.

Replays the same composite-score entry rules day by day:
  - enter at next day's open when a BUY fires
  - exit at the ATR stop, the 3R target, a SELL signal, or a 60-bar time stop
Reports win rate, profit factor, average R-multiple, and max drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .indicators import enrich
from .strategy import (
    BUY_THRESHOLD,
    _momentum_pillar,
    _sell_checks,
    _timing_pillar,
    _trend_pillar,
    _volume_pillar,
)

MAX_HOLD_BARS = 60
MIN_WARMUP = 210


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    entry: float
    exit: float
    stop: float
    pnl_pct: float
    r_multiple: float
    bars_held: int
    exit_reason: str


@dataclass
class BacktestResult:
    symbol: str
    trades: list[Trade] = field(default_factory=list)
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_r: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    buy_and_hold_pct: float = 0.0

    @property
    def n_trades(self) -> int:
        return len(self.trades)


def _score_at(e: pd.DataFrame, i: int, nifty_roc63: float) -> tuple[float, bool, list[str]]:
    window = e.iloc[: i + 1]
    row = window.iloc[-1]
    t, _ = _trend_pillar(row)
    m, _ = _momentum_pillar(row, nifty_roc63)
    g, _ = _timing_pillar(row, window)
    v, _ = _volume_pillar(row, window)
    gate = row.Close > row.sma200 and row.sma50 > row.sma200
    return t + m + g + v, gate, _sell_checks(row, window)


def run_backtest(
    symbol: str, df: pd.DataFrame, nifty_df: pd.DataFrame | None = None
) -> BacktestResult | None:
    if df is None or len(df) < MIN_WARMUP + 30:
        return None

    e = enrich(df)
    e["sma50_slope"] = e["sma50"].diff(5)
    e["sma200_slope"] = e["sma200"].diff(10)
    e = e.dropna(subset=["sma200", "atr14", "sma50_slope", "sma200_slope"])
    if len(e) < 60:
        return None

    # Rolling 3-month Nifty return, aligned to this stock's dates.
    if nifty_df is not None and len(nifty_df) > 64:
        nifty_roc = (nifty_df["Close"].pct_change(63) * 100).reindex(
            e.index, method="ffill"
        ).fillna(0)
    else:
        nifty_roc = pd.Series(0.0, index=e.index)

    result = BacktestResult(symbol=symbol)
    in_pos = False
    entry = stop = target = 0.0
    entry_i = 0

    i = 30  # warmup so window-based checks have enough bars
    while i < len(e) - 1:
        row = e.iloc[i]
        if not in_pos:
            score, gate, flags = _score_at(e, i, float(nifty_roc.iloc[i]))
            if score >= BUY_THRESHOLD and gate and not flags:
                entry = float(e["Open"].iloc[i + 1])  # fill at next open
                swing_low = float(e["Low"].iloc[max(0, i - 9) : i + 1].min())
                stop = min(entry - 2 * float(row.atr14), swing_low * 0.995)
                risk = entry - stop
                if risk > 0:
                    target = entry + 3 * risk
                    entry_i = i + 1
                    in_pos = True
                    i += 1
                    continue
        else:
            bar = e.iloc[i]
            exit_price, reason = None, ""
            if float(bar.Low) <= stop:
                exit_price, reason = stop, "stop"
            elif float(bar.High) >= target:
                exit_price, reason = target, "target"
            elif i - entry_i >= MAX_HOLD_BARS:
                exit_price, reason = float(bar.Close), "time"
            else:
                flags = _sell_checks(bar, e.iloc[: i + 1])
                if len(flags) >= 2:
                    exit_price, reason = float(bar.Close), "sell-signal"
            if exit_price is not None:
                risk = entry - stop
                result.trades.append(
                    Trade(
                        entry_date=str(e.index[entry_i].date()),
                        exit_date=str(e.index[i].date()),
                        entry=round(entry, 2),
                        exit=round(exit_price, 2),
                        stop=round(stop, 2),
                        pnl_pct=round((exit_price / entry - 1) * 100, 2),
                        r_multiple=round((exit_price - entry) / risk, 2),
                        bars_held=i - entry_i,
                        exit_reason=reason,
                    )
                )
                in_pos = False
        i += 1

    trades = result.trades
    if trades:
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]
        result.win_rate = round(100 * len(wins) / len(trades), 1)
        gross_win = sum(t.pnl_pct for t in wins)
        gross_loss = abs(sum(t.pnl_pct for t in losses))
        result.profit_factor = round(gross_win / gross_loss, 2) if gross_loss else float("inf")
        result.avg_r = round(sum(t.r_multiple for t in trades) / len(trades), 2)

        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for t in trades:
            equity *= 1 + t.pnl_pct / 100
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
        result.total_return_pct = round((equity - 1) * 100, 2)
        result.max_drawdown_pct = round(max_dd * 100, 2)

    first_close = float(e["Close"].iloc[0])
    result.buy_and_hold_pct = round((float(e["Close"].iloc[-1]) / first_close - 1) * 100, 2)
    return result
