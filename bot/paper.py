"""Paper trading ledger — pseudo buy/sell with P&L tracking.

State is persisted as JSON under `.cache/paper_portfolio.json` so trades
survive server restarts.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .data import get_history, get_ltp
from .indicators import enrich, rsi
from .meanrev import HOLD_BARS, RSI2_EXIT_EARLY
from .storage import CACHE_BASE, kv_enabled, kv_get, kv_set, storage_kind
from .strategy import _sell_checks

LEDGER_PATH = CACHE_BASE / "paper_portfolio.json"
KV_KEY = "paper_portfolio"
_lock = threading.Lock()

SYMBOL_COOLDOWN_DAYS = 5
MAX_OPEN_QUICK = 3
SWING_MAX_HOLD_BARS = 60


@dataclass
class PaperTrade:
    id: str
    symbol: str
    qty: int
    entry_price: float
    entry_date: str
    strategy: str  # swing | quick
    stop_loss: float | None = None
    target1: float | None = None
    target2: float | None = None
    status: str = "open"  # open | closed
    exit_price: float | None = None
    exit_date: str | None = None
    pnl: float | None = None
    pnl_pct: float | None = None
    notes: str = ""


@dataclass
class PaperPortfolio:
    starting_capital: float = 20_000
    cash: float = 20_000
    trades: list[PaperTrade] = field(default_factory=list)


def _raw_to_portfolio(raw: dict) -> PaperPortfolio:
    trades = [PaperTrade(**t) for t in raw.get("trades", [])]
    return PaperPortfolio(
        starting_capital=raw.get("starting_capital", 20_000),
        cash=raw.get("cash", raw.get("starting_capital", 20_000)),
        trades=trades,
    )


def _load() -> PaperPortfolio:
    if kv_enabled():
        try:
            raw = kv_get(KV_KEY)
            return _raw_to_portfolio(raw) if raw else PaperPortfolio()
        except Exception:
            return PaperPortfolio()

    if not LEDGER_PATH.exists():
        return PaperPortfolio()
    try:
        return _raw_to_portfolio(json.loads(LEDGER_PATH.read_text()))
    except (json.JSONDecodeError, TypeError, KeyError):
        return PaperPortfolio()


def _save(portfolio: PaperPortfolio) -> None:
    payload = {
        "starting_capital": portfolio.starting_capital,
        "cash": round(portfolio.cash, 2),
        "trades": [asdict(t) for t in portfolio.trades],
    }
    if kv_enabled():
        try:
            kv_set(KV_KEY, payload)
            return
        except Exception:
            pass  # fall through to file so the request still succeeds
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(payload, indent=2))


def latest_price(symbol: str, *, live: bool = False) -> float | None:
    """Return LTP. Pass live=True to bypass the OHLCV disk cache."""
    symbol = symbol.upper()
    if live:
        return get_ltp(symbol)
    df = get_history(symbol)
    if df is None or df.empty:
        return None
    return round(float(df["Close"].iloc[-1]), 2)


def _open_trades(portfolio: PaperPortfolio) -> list[PaperTrade]:
    return [t for t in portfolio.trades if t.status == "open"]


def _closed_trades(portfolio: PaperPortfolio) -> list[PaperTrade]:
    return [t for t in portfolio.trades if t.status == "closed"]


def _last_closed_date(portfolio: PaperPortfolio, symbol: str) -> date | None:
    symbol = symbol.upper()
    closed = [t for t in _closed_trades(portfolio) if t.symbol == symbol and t.exit_date]
    if not closed:
        return None
    return max(datetime.strptime(t.exit_date, "%Y-%m-%d").date() for t in closed)


def _close_trade(trade: PaperTrade, exit_px: float, reason: str) -> None:
    exit_px = round(exit_px, 2)
    proceeds = round(exit_px * trade.qty, 2)
    cost_basis = round(trade.entry_price * trade.qty, 2)
    trade.status = "closed"
    trade.exit_price = exit_px
    trade.exit_date = str(date.today())
    trade.pnl = round(proceeds - cost_basis, 2)
    trade.pnl_pct = round((exit_px / trade.entry_price - 1) * 100, 2)
    suffix = f" | auto: {reason}"
    trade.notes = (trade.notes + suffix).strip()


def _quick_exit(trade: PaperTrade, df: pd.DataFrame) -> tuple[float, str] | None:
    e = df.copy()
    e["rsi2"] = rsi(e["Close"], 2)
    entry_ts = pd.Timestamp(trade.entry_date)
    after = e.loc[e.index >= entry_ts]
    if len(after) < 2:
        return None

    stop = trade.stop_loss
    for offset in range(len(after)):
        bar = after.iloc[offset]
        if stop is not None and float(bar.Low) <= stop:
            return stop, "stop"
        if offset == 1 and float(bar["rsi2"]) >= RSI2_EXIT_EARLY:
            return float(bar.Close), "rsi-exit"
        if offset >= HOLD_BARS:
            return float(bar.Close), "day2-close"
    return None


def _swing_exit(trade: PaperTrade, df: pd.DataFrame) -> tuple[float, str] | None:
    e = enrich(df)
    e["sma50_slope"] = e["sma50"].diff(5)
    e["sma200_slope"] = e["sma200"].diff(10)
    e = e.dropna(subset=["sma200", "atr14"])
    if e.empty:
        return None

    entry_ts = pd.Timestamp(trade.entry_date)
    mask = e.index >= entry_ts
    if not mask.any():
        return None
    start_i = int(e.index.get_indexer([e.index[mask][0]])[0])

    stop = trade.stop_loss or 0.0
    t1 = trade.target1
    t2 = trade.target2

    for i in range(start_i, len(e)):
        bar = e.iloc[i]
        bars_held = i - start_i
        if stop and float(bar.Low) <= stop:
            return stop, "stop"
        if t2 and float(bar.High) >= t2:
            return t2, "target2"
        if t1 and float(bar.High) >= t1:
            return t1, "target1"
        if bars_held >= SWING_MAX_HOLD_BARS:
            return float(bar.Close), "time"
        flags = _sell_checks(bar, e.iloc[: i + 1])
        if len(flags) >= 2:
            return float(bar.Close), "sell-signal"
    return None


def check_exits() -> list[PaperTrade]:
    """Auto-close open positions when strategy exit rules fire."""
    closed: list[PaperTrade] = []
    with _lock:
        pf = _load()
        cash_delta = 0.0
        for trade in _open_trades(pf):
            df = get_history(trade.symbol)
            if df is None or df.empty:
                continue
            result = (
                _quick_exit(trade, df)
                if trade.strategy == "quick"
                else _swing_exit(trade, df)
            )
            if result is None:
                continue
            exit_px, reason = result
            _close_trade(trade, exit_px, reason)
            cash_delta += round(exit_px * trade.qty, 2)
            closed.append(trade)
        if closed:
            pf.cash = round(pf.cash + cash_delta, 2)
            _save(pf)
    return closed


def buy(
    symbol: str,
    qty: int,
    strategy: str = "quick",
    price: float | None = None,
    stop_loss: float | None = None,
    target1: float | None = None,
    target2: float | None = None,
    notes: str = "",
    tier: str | None = None,
) -> PaperTrade:
    symbol = symbol.upper()
    if qty <= 0:
        raise ValueError("quantity must be positive")

    if strategy != "quick":
        raise ValueError("only quick strategy is supported")

    notes_lower = notes.lower()
    if tier == "review" or tier == "extended" or "watch" in notes_lower:
        raise ValueError(
            "paper buy only allowed for qualified quick signals — "
            "review/extended/WATCH tiers are not auto-buy eligible"
        )

    entry = price if price is not None else latest_price(symbol)
    if entry is None:
        raise ValueError(f"no price data for {symbol}")

    cost = round(entry * qty, 2)

    with _lock:
        pf = _load()
        if any(t.symbol == symbol and t.status == "open" for t in pf.trades):
            raise ValueError(f"already holding an open paper position in {symbol}")

        if strategy == "quick":
            open_quick = sum(1 for t in _open_trades(pf) if t.strategy == "quick")
            if open_quick >= MAX_OPEN_QUICK:
                raise ValueError(
                    f"max {MAX_OPEN_QUICK} open quick positions — close one before buying"
                )

        last_closed = _last_closed_date(pf, symbol)
        if last_closed is not None:
            cooldown_end = last_closed + timedelta(days=SYMBOL_COOLDOWN_DAYS)
            if date.today() < cooldown_end:
                raise ValueError(
                    f"{symbol} cooldown active until {cooldown_end} "
                    f"({SYMBOL_COOLDOWN_DAYS} days after last exit)"
                )

        if cost > pf.cash + 0.01:
            raise ValueError(
                f"insufficient cash: need ₹{cost:,.0f}, have ₹{pf.cash:,.0f}"
            )

        trade = PaperTrade(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            qty=qty,
            entry_price=round(entry, 2),
            entry_date=str(date.today()),
            strategy=strategy,
            stop_loss=stop_loss,
            target1=target1,
            target2=target2,
            notes=notes,
        )
        pf.cash = round(pf.cash - cost, 2)
        pf.trades.append(trade)
        _save(pf)
        return trade


def sell(trade_id: str, price: float | None = None, reason: str = "manual") -> PaperTrade:
    with _lock:
        pf = _load()
        trade = next((t for t in pf.trades if t.id == trade_id and t.status == "open"), None)
        if trade is None:
            raise ValueError(f"open trade {trade_id!r} not found")

        exit_px = price if price is not None else latest_price(trade.symbol, live=True)
        if exit_px is None:
            raise ValueError(f"no price data for {trade.symbol}")

        _close_trade(trade, exit_px, reason)
        pf.cash = round(pf.cash + round(exit_px * trade.qty, 2), 2)
        _save(pf)
        return trade


def reset_portfolio(starting_capital: float = 20_000) -> PaperPortfolio:
    with _lock:
        pf = PaperPortfolio(starting_capital=starting_capital, cash=starting_capital, trades=[])
        _save(pf)
        return pf


def portfolio_view(*, refresh_prices: bool = False) -> dict:
    if refresh_prices:
        check_exits()

    pf = _load()
    open_t = _open_trades(pf)
    closed_t = _closed_trades(pf)

    open_rows = []
    invested = 0.0
    unrealized = 0.0
    for t in open_t:
        mkt = latest_price(t.symbol, live=refresh_prices) or t.entry_price
        cost = t.entry_price * t.qty
        value = mkt * t.qty
        upnl = value - cost
        invested += cost
        unrealized += upnl
        open_rows.append(
            {
                **asdict(t),
                "current_price": mkt,
                "market_value": round(value, 2),
                "cost_basis": round(cost, 2),
                "unrealized_pnl": round(upnl, 2),
                "unrealized_pnl_pct": round((mkt / t.entry_price - 1) * 100, 2),
            }
        )

    realized = sum(t.pnl or 0 for t in closed_t)
    wins = [t for t in closed_t if (t.pnl or 0) > 0]
    losses = [t for t in closed_t if (t.pnl or 0) <= 0]
    equity = pf.cash + sum(r["market_value"] for r in open_rows)

    return {
        "starting_capital": pf.starting_capital,
        "cash": pf.cash,
        "invested": round(invested, 2),
        "equity": round(equity, 2),
        "unrealized_pnl": round(unrealized, 2),
        "realized_pnl": round(realized, 2),
        "total_pnl": round(realized + unrealized, 2),
        "total_return_pct": round((equity / pf.starting_capital - 1) * 100, 2),
        "open_count": len(open_t),
        "closed_count": len(closed_t),
        "win_rate": round(100 * len(wins) / len(closed_t), 1) if closed_t else None,
        "avg_win": round(sum(t.pnl for t in wins) / len(wins), 2) if wins else None,
        "avg_loss": round(sum(t.pnl for t in losses) / len(losses), 2) if losses else None,
        "open_positions": open_rows,
        "closed_trades": [asdict(t) for t in reversed(closed_t)],
        "ltp_updated_at": datetime.now(timezone.utc).astimezone().strftime("%d %b, %I:%M %p %Z"),
        "ltp_live": refresh_prices,
        "storage": storage_kind(),
        "durable": kv_enabled() or storage_kind() == "disk",
    }
