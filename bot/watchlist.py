"""Persistent multibagger research watchlist.

Saved under `.cache/multibagger_watchlist.json` (or Redis KV on Vercel).
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .data import get_history
from .paper import latest_price
from .storage import CACHE_BASE, kv_enabled, kv_get, kv_set, storage_kind

WATCHLIST_PATH = CACHE_BASE / "multibagger_watchlist.json"
KV_KEY = "multibagger_watchlist"
_lock = threading.Lock()


@dataclass
class WatchlistItem:
    symbol: str
    added_at: str
    entry_price: float
    score: float | None = None
    tier: str | None = None
    sector: str = ""
    industry: str = ""
    market_cap_cr: float | None = None
    roe_pct: float | None = None
    revenue_growth_pct: float | None = None
    reasons: list[str] = field(default_factory=list)
    notes: str = ""
    # populated on read
    current_price: float | None = None
    pct_change: float | None = None


@dataclass
class Watchlist:
    items: list[WatchlistItem] = field(default_factory=list)


def _raw_to_watchlist(raw: dict) -> Watchlist:
    items = [WatchlistItem(**i) for i in raw.get("items", [])]
    return Watchlist(items=items)


def _load() -> Watchlist:
    if kv_enabled():
        try:
            raw = kv_get(KV_KEY)
            return _raw_to_watchlist(raw) if raw else Watchlist()
        except Exception:
            return Watchlist()

    if not WATCHLIST_PATH.exists():
        return Watchlist()
    try:
        return _raw_to_watchlist(json.loads(WATCHLIST_PATH.read_text()))
    except (json.JSONDecodeError, TypeError, KeyError):
        return Watchlist()


def _save(wl: Watchlist) -> None:
    payload = {"items": [asdict(i) for i in wl.items]}
    if kv_enabled():
        try:
            kv_set(KV_KEY, payload)
            return
        except Exception:
            pass
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    WATCHLIST_PATH.write_text(json.dumps(payload, indent=2))


def _enrich_item(item: WatchlistItem, *, live: bool = False) -> WatchlistItem:
    px = latest_price(item.symbol, live=live)
    item.current_price = px
    if px is not None and item.entry_price:
        item.pct_change = round((px / item.entry_price - 1) * 100, 2)
    return item


def list_watchlist(*, refresh_prices: bool = False) -> dict:
    with _lock:
        wl = _load()
    items = [_enrich_item(i, live=refresh_prices) for i in wl.items]
    return {
        "count": len(items),
        "items": [asdict(i) for i in items],
        "storage": storage_kind(),
        "durable": kv_enabled() or storage_kind() == "disk",
    }


def add_to_watchlist(
    symbol: str,
    *,
    entry_price: float | None = None,
    score: float | None = None,
    tier: str | None = None,
    sector: str = "",
    industry: str = "",
    market_cap_cr: float | None = None,
    roe_pct: float | None = None,
    revenue_growth_pct: float | None = None,
    reasons: list[str] | None = None,
    notes: str = "",
) -> WatchlistItem:
    symbol = symbol.upper()
    with _lock:
        wl = _load()
        for existing in wl.items:
            if existing.symbol == symbol:
                raise ValueError(f"{symbol} is already on your watchlist")

        price = entry_price or latest_price(symbol)
        if price is None:
            raise ValueError(f"no price data for {symbol}")

        item = WatchlistItem(
            symbol=symbol,
            added_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            entry_price=round(float(price), 2),
            score=score,
            tier=tier,
            sector=sector,
            industry=industry,
            market_cap_cr=market_cap_cr,
            roe_pct=roe_pct,
            revenue_growth_pct=revenue_growth_pct,
            reasons=(reasons or [])[:5],
            notes=notes.strip(),
        )
        wl.items.insert(0, item)
        _save(wl)
    return _enrich_item(item)


def remove_from_watchlist(symbol: str) -> bool:
    symbol = symbol.upper()
    with _lock:
        wl = _load()
        before = len(wl.items)
        wl.items = [i for i in wl.items if i.symbol != symbol]
        if len(wl.items) == before:
            return False
        _save(wl)
    return True


def is_on_watchlist(symbol: str) -> bool:
    symbol = symbol.upper()
    with _lock:
        return any(i.symbol == symbol for i in _load().items)


def refresh_scores() -> dict:
    """Re-evaluate watchlist symbols and update stored snapshots."""
    from .multibagger import evaluate_multibagger

    with _lock:
        wl = _load()
        updated = 0
        for i, item in enumerate(wl.items):
            df = get_history(item.symbol)
            if df is None:
                continue
            sig = evaluate_multibagger(item.symbol, df)
            if sig is None:
                continue
            item.score = sig.score
            item.tier = sig.tier
            item.sector = sig.sector
            item.industry = sig.industry
            item.market_cap_cr = sig.market_cap_cr
            item.roe_pct = sig.roe_pct
            item.revenue_growth_pct = sig.revenue_growth_pct
            item.reasons = sig.reasons[:5]
            wl.items[i] = item
            updated += 1
        _save(wl)
    return {"updated": updated, "watchlist": list_watchlist(refresh_prices=True)}
