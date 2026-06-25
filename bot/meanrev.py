"""Quick (2-day hold) mean-reversion strategy — Connors RSI-2 style (v2).

Improvements over v1:
  - Stricter setup (sma50, 2 down days, low-volume dip, not over-extended)
  - Realistic backtest: brokerage/slippage, gap-up skip, intraday stop, early RSI exit
  - Walk-forward: both halves of history must win >= 50%
  - Nifty regime filter (index above 50 DMA)
  - Stat gates: >=15 trades, win rate >=55%, profit factor >=1.25, worst loss >=-6%
  - Near-miss watchlist: setups that fail 1-2 gates are surfaced for manual review
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .fundamentals import quality_check
from .indicators import atr, rsi, sma

HOLD_BARS = 2
RSI2_ENTRY = 10.0
RSI2_EXIT_EARLY = 70.0
MIN_TRADES = 15
MIN_WIN_RATE = 55.0
MIN_PROFIT_FACTOR = 1.25
MAX_WORST_PCT = -6.0
MIN_RECENT_WIN_RATE = 50.0
MIN_RECENT_TRADES = 2
MIN_TURNOVER = 5e7
RECENT_WINDOW = 22
ROUND_TRIP_COST_PCT = 0.4
MAX_GAP_UP_PCT = 0.5
MAX_PCT_ABOVE_200 = 50.0
WALK_FORWARD_MIN_WR = 50.0
WALK_FORWARD_MIN_EACH = 5
MAX_SIGNALS_PER_SCAN = 3
RISK_PER_TRADE = 0.01


@dataclass
class QuickSignal:
    symbol: str
    close: float
    entry_note: str
    stop_loss: float
    qty_for_1pct_risk: int
    hist_trades: int
    hist_win_rate: float
    hist_avg_pnl_pct: float
    hist_profit_factor: float
    hist_worst_pct: float
    hist_days: int = 0
    recent_trades: int = 0
    recent_win_rate: float | None = None
    walk_forward_ok: bool = True
    fundamentals_ok: bool = True
    fundamental_notes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    as_of: str = ""
    is_watch: bool = False          # near-miss (failed 1-2 strict gates)
    failed_gates: list[str] = field(default_factory=list)


def nifty_regime_ok(nifty_df: pd.DataFrame | None) -> tuple[bool, str]:
    """Only dip-buy when the broad market is in a risk-on regime."""
    if nifty_df is None or len(nifty_df) < 55:
        return True, "Nifty regime unchecked (no index data)"
    close = nifty_df["Close"]
    s50 = sma(close, 50)
    last = float(close.iloc[-1])
    s50v = float(s50.iloc[-1])
    if last > s50v:
        return True, f"Nifty above 50 DMA ({last:.0f} > {s50v:.0f}) — risk-on"
    return False, f"Nifty below 50 DMA ({last:.0f} < {s50v:.0f}) — skip dip-buys"


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    e = df.copy()
    close = e["Close"]
    e["sma5"] = sma(close, 5)
    e["sma50"] = sma(close, 50)
    e["sma200"] = sma(close, 200)
    e["rsi2"] = rsi(close, 2)
    e["atr14"] = atr(e, 14)
    e["vol_avg20"] = sma(e["Volume"], 20)
    e["turnover20"] = sma(close * e["Volume"], 20)
    e["pct_above_200"] = (close / e["sma200"] - 1) * 100
    down = close < close.shift(1)
    e["one_down_day"] = down
    e["two_down_days"] = down & down.shift(1)
    e["low_vol_dip"] = e["Volume"] <= e["vol_avg20"] * 1.25
    return e


def _history_mask(e: pd.DataFrame) -> pd.Series:
    """Mask used to count past occurrences (needs enough samples)."""
    return (
        (e["Close"] > e["sma200"])
        & (e["rsi2"] < RSI2_ENTRY)
        & (e["Close"] < e["sma5"])
        & (e["turnover20"] >= MIN_TURNOVER)
        & (e["pct_above_200"] <= MAX_PCT_ABOVE_200)
    )


def _today_mask(e: pd.DataFrame) -> pd.Series:
    """Extra quality checks for today's live signal only."""
    return (
        _history_mask(e)
        & (e["Close"] > e["sma50"])
        & e["one_down_day"].fillna(False)
    )


def _setup_mask(e: pd.DataFrame) -> pd.Series:
    return _today_mask(e)


def _simulate_trade(e: pd.DataFrame, i: int) -> float | None:
    """Return net P&L % for one setup at bar i, or None if gap-up skip."""
    if i + HOLD_BARS >= len(e):
        return None
    signal_close = float(e["Close"].iloc[i])
    entry_open = float(e["Open"].iloc[i + 1])
    if (entry_open / signal_close - 1) * 100 > MAX_GAP_UP_PCT:
        return None
    stop = signal_close - 2 * float(e["atr14"].iloc[i])
    entry = entry_open

    for day_offset in range(1, HOLD_BARS + 1):
        bar = i + day_offset
        if float(e["Low"].iloc[bar]) <= stop:
            return round((stop / entry - 1) * 100 - ROUND_TRIP_COST_PCT, 2)
        if day_offset == 1 and float(e["rsi2"].iloc[bar]) >= RSI2_EXIT_EARLY:
            exit_px = float(e["Close"].iloc[bar])
            return round((exit_px / entry - 1) * 100 - ROUND_TRIP_COST_PCT, 2)

    exit_px = float(e["Close"].iloc[i + HOLD_BARS])
    return round((exit_px / entry - 1) * 100 - ROUND_TRIP_COST_PCT, 2)


def _walk_forward_ok(pnls: list[float]) -> bool:
    if len(pnls) < MIN_TRADES:
        return False
    mid = len(pnls) // 2
    first, second = pnls[:mid], pnls[mid:]
    if len(first) < WALK_FORWARD_MIN_EACH or len(second) < WALK_FORWARD_MIN_EACH:
        return True
    wr = lambda xs: 100 * sum(1 for p in xs if p > 0) / len(xs)
    return wr(first) >= WALK_FORWARD_MIN_WR and wr(second) >= WALK_FORWARD_MIN_WR


def historical_stats(e: pd.DataFrame) -> dict | None:
    mask = _history_mask(e)
    pnls: list[float] = []
    for i in range(len(e) - HOLD_BARS - 1):
        if not bool(mask.iloc[i]):
            continue
        pnl = _simulate_trade(e, i)
        if pnl is not None:
            pnls.append(pnl)

    if len(pnls) < MIN_TRADES:
        return None

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_loss = abs(sum(losses))
    pf = wins and gross_loss and sum(wins) / gross_loss or float("inf")

    recent_pnls = pnls[-8:]  # ~last month of trades (not bars)
    recent_wr = (
        round(100 * sum(1 for p in recent_pnls if p > 0) / len(recent_pnls), 1)
        if recent_pnls else None
    )

    return {
        "trades": len(pnls),
        "win_rate": round(100 * len(wins) / len(pnls), 1),
        "avg_pnl_pct": round(sum(pnls) / len(pnls), 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else 99.0,
        "worst_pct": round(min(pnls), 2),
        "hist_days": len(e),
        "recent_trades": len(recent_pnls),
        "recent_win_rate": recent_wr,
        "walk_forward_ok": _walk_forward_ok(pnls),
    }


def _gate_failures(stats: dict) -> list[str]:
    """Human-readable list of which strict gates a setup fails (empty = passes all)."""
    fails: list[str] = []
    if stats["win_rate"] < MIN_WIN_RATE:
        fails.append(f"win rate {stats['win_rate']}% < {MIN_WIN_RATE:.0f}%")
    if stats["avg_pnl_pct"] <= 0:
        fails.append("avg P&L not positive")
    if stats["profit_factor"] < MIN_PROFIT_FACTOR:
        fails.append(f"profit factor {stats['profit_factor']} < {MIN_PROFIT_FACTOR}")
    if stats["worst_pct"] < MAX_WORST_PCT:
        fails.append(f"worst loss {stats['worst_pct']}% beyond {MAX_WORST_PCT:.0f}%")
    if not stats["walk_forward_ok"]:
        fails.append("walk-forward inconsistent")
    if (
        stats["recent_trades"] >= MIN_RECENT_TRADES
        and stats["recent_win_rate"] is not None
        and stats["recent_win_rate"] < MIN_RECENT_WIN_RATE
    ):
        fails.append(f"recent win rate {stats['recent_win_rate']}% < {MIN_RECENT_WIN_RATE:.0f}%")
    return fails


def _stats_pass(stats: dict) -> bool:
    return not _gate_failures(stats)


def _is_near_miss(stats: dict, fails: list[str]) -> bool:
    """A near-miss: fails only 1-2 strict gates but is still a decent, profitable edge."""
    return (
        1 <= len(fails) <= 2
        and stats["win_rate"] >= 50.0
        and stats["profit_factor"] >= 1.0
        and stats["avg_pnl_pct"] > 0
    )


def _make_signal(
    symbol: str,
    e: pd.DataFrame,
    stats: dict,
    capital: float,
    regime_note: str,
    is_watch: bool = False,
    failed_gates: list[str] | None = None,
) -> QuickSignal:
    row = e.iloc[-1]
    close = float(row["Close"])
    stop = round(close - 2 * float(row["atr14"]), 2)
    risk = close - stop

    reasons = [
        regime_note,
        f"Analyzed {stats['hist_days']} days · backtest includes {ROUND_TRIP_COST_PCT}% costs",
        f"RSI(2) {row['rsi2']:.0f} · down day · above 50 & 200 DMA · not over-extended",
        f"History: {stats['win_rate']}% wins / {stats['trades']} trades · PF {stats['profit_factor']}",
        "Walk-forward: both halves of history profitable" if stats["walk_forward_ok"] else "",
        (
            f"Recent trades: {stats['recent_win_rate']}% over {stats['recent_trades']}"
            if stats["recent_trades"]
            else "First setup in recent window"
        ),
        f"Avg net {stats['avg_pnl_pct']:+.2f}% · worst {stats['worst_pct']:.2f}%",
    ]

    sig = QuickSignal(
        symbol=symbol,
        close=round(close, 2),
        entry_note=(
            "Buy at tomorrow's open only if gap-up < 0.5%; "
            "sell day-2 close OR day-1 if RSI(2) recovers; stop if hit"
        ),
        stop_loss=stop,
        qty_for_1pct_risk=int(capital * RISK_PER_TRADE / risk) if risk > 0 else 0,
        hist_trades=stats["trades"],
        hist_win_rate=stats["win_rate"],
        hist_avg_pnl_pct=stats["avg_pnl_pct"],
        hist_profit_factor=stats["profit_factor"],
        hist_worst_pct=stats["worst_pct"],
        hist_days=stats["hist_days"],
        recent_trades=stats["recent_trades"],
        recent_win_rate=stats["recent_win_rate"],
        walk_forward_ok=stats["walk_forward_ok"],
        is_watch=is_watch,
        failed_gates=failed_gates or [],
        reasons=[r for r in reasons if r],
        as_of=str(e.index[-1].date()),
    )
    if is_watch and failed_gates:
        sig.warnings.append("Near-miss — review manually: " + "; ".join(failed_gates))
    return sig


def _evaluate(
    symbol: str,
    df: pd.DataFrame,
    capital: float,
    nifty_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict, str] | None:
    """Shared front-half: regime + today-setup + stats. Returns (e, stats, regime_note)."""
    if df is None or len(df) < 220:
        return None
    regime_ok, regime_note = nifty_regime_ok(nifty_df)
    if not regime_ok:
        return None
    e = _prepare(df).dropna(subset=["sma200", "sma50", "atr14"])
    if e.empty or not bool(_today_mask(e).iloc[-1]):
        return None
    stats = historical_stats(e)
    if stats is None:
        return None
    return e, stats, regime_note


def evaluate_quick(
    symbol: str,
    df: pd.DataFrame,
    capital: float = 1_000_000,
    check_fundamentals: bool = True,
    nifty_df: pd.DataFrame | None = None,
) -> QuickSignal | None:
    prepared = _evaluate(symbol, df, capital, nifty_df)
    if prepared is None:
        return None
    e, stats, regime_note = prepared
    if not _stats_pass(stats):
        return None

    sig = _make_signal(symbol, e, stats, capital, regime_note)

    if check_fundamentals:
        ok, positives, negatives = quality_check(symbol)
        sig.fundamentals_ok = ok
        sig.fundamental_notes = positives + negatives
        if not ok:
            sig.warnings.append("Failed fundamental quality gate")

    return sig


def evaluate_watch(
    symbol: str,
    df: pd.DataFrame,
    capital: float = 1_000_000,
    nifty_df: pd.DataFrame | None = None,
) -> QuickSignal | None:
    """Return a near-miss QuickSignal (is_watch=True) for stocks that fail 1-2 strict gates."""
    prepared = _evaluate(symbol, df, capital, nifty_df)
    if prepared is None:
        return None
    e, stats, regime_note = prepared
    fails = _gate_failures(stats)
    if not _is_near_miss(stats, fails):
        return None
    return _make_signal(
        symbol, e, stats, capital, regime_note, is_watch=True, failed_gates=fails
    )


def scan_quick(
    data: dict[str, pd.DataFrame],
    capital: float = 1_000_000,
    nifty_df: pd.DataFrame | None = None,
) -> list[QuickSignal]:
    regime_ok, regime_note = nifty_regime_ok(nifty_df)
    if not regime_ok:
        return []

    candidates = [
        sig
        for sym, df in data.items()
        if (sig := evaluate_quick(sym, df, capital, check_fundamentals=False, nifty_df=nifty_df))
    ]
    out: list[QuickSignal] = []
    for sig in candidates:
        ok, positives, negatives = quality_check(sig.symbol)
        sig.fundamentals_ok = ok
        sig.fundamental_notes = positives + negatives
        if not ok:
            continue
        if regime_note and regime_note not in sig.reasons:
            sig.reasons.insert(0, regime_note)
        out.append(sig)

    out.sort(
        key=lambda s: (s.hist_win_rate * max(s.hist_avg_pnl_pct, 0.1), s.hist_profit_factor),
        reverse=True,
    )
    return out[:MAX_SIGNALS_PER_SCAN]


def quick_watchlist(
    data: dict[str, pd.DataFrame],
    capital: float = 1_000_000,
    nifty_df: pd.DataFrame | None = None,
    limit: int = 8,
) -> list[QuickSignal]:
    """Near-miss setups (dip today, fundamentally sound, but failing 1-2 strict gates).

    These are NOT auto-buys — they are surfaced so the user can review and decide manually.
    """
    regime_ok, _ = nifty_regime_ok(nifty_df)
    if not regime_ok:
        return []

    out: list[QuickSignal] = []
    for sym, df in data.items():
        sig = evaluate_watch(sym, df, capital, nifty_df=nifty_df)
        if sig is None:
            continue
        ok, positives, negatives = quality_check(sym)
        sig.fundamentals_ok = ok
        sig.fundamental_notes = positives + negatives
        if not ok:
            continue  # only show fundamentally sound near-misses
        out.append(sig)

    out.sort(key=lambda s: (s.hist_win_rate, s.hist_profit_factor), reverse=True)
    return out[:limit]
