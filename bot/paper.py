"""Paper trading ledger — pseudo buy/sell with P&L tracking.

State is persisted as JSON under `.cache/paper_portfolio.json` so trades
survive server restarts.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from datetime import datetime, timezone

from .data import get_history, get_ltp
from .storage import CACHE_BASE, kv_enabled, kv_get, kv_set, storage_kind

LEDGER_PATH = CACHE_BASE / "paper_portfolio.json"
KV_KEY = "paper_portfolio"
_lock = threading.Lock()


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
    starting_capital: float = 1_000_000
    cash: float = 1_000_000
    trades: list[PaperTrade] = field(default_factory=list)


def _raw_to_portfolio(raw: dict) -> PaperPortfolio:
    trades = [PaperTrade(**t) for t in raw.get("trades", [])]
    return PaperPortfolio(
        starting_capital=raw.get("starting_capital", 1_000_000),
        cash=raw.get("cash", raw.get("starting_capital", 1_000_000)),
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


def buy(
    symbol: str,
    qty: int,
    strategy: str = "swing",
    price: float | None = None,
    stop_loss: float | None = None,
    target1: float | None = None,
    target2: float | None = None,
    notes: str = "",
) -> PaperTrade:
    symbol = symbol.upper()
    if qty <= 0:
        raise ValueError("quantity must be positive")

    entry = price if price is not None else latest_price(symbol)
    if entry is None:
        raise ValueError(f"no price data for {symbol}")

    cost = round(entry * qty, 2)

    with _lock:
        pf = _load()
        if any(t.symbol == symbol and t.status == "open" for t in pf.trades):
            raise ValueError(f"already holding an open paper position in {symbol}")

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


def sell(trade_id: str, price: float | None = None) -> PaperTrade:
    with _lock:
        pf = _load()
        trade = next((t for t in pf.trades if t.id == trade_id and t.status == "open"), None)
        if trade is None:
            raise ValueError(f"open trade {trade_id!r} not found")

        exit_px = price if price is not None else latest_price(trade.symbol, live=True)
        if exit_px is None:
            raise ValueError(f"no price data for {trade.symbol}")

        exit_px = round(exit_px, 2)
        proceeds = round(exit_px * trade.qty, 2)
        cost_basis = round(trade.entry_price * trade.qty, 2)
        pnl = round(proceeds - cost_basis, 2)
        pnl_pct = round((exit_px / trade.entry_price - 1) * 100, 2)

        trade.status = "closed"
        trade.exit_price = exit_px
        trade.exit_date = str(date.today())
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        pf.cash = round(pf.cash + proceeds, 2)
        _save(pf)
        return trade


def reset_portfolio(starting_capital: float = 1_000_000) -> PaperPortfolio:
    with _lock:
        pf = PaperPortfolio(starting_capital=starting_capital, cash=starting_capital, trades=[])
        _save(pf)
        return pf


def portfolio_view(*, refresh_prices: bool = False) -> dict:
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
