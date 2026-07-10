"""FastAPI server: REST API + dashboard.

Run with:  python -m bot.server   (serves http://localhost:8000)
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .data import get_histories, get_history
from .meanrev import DEFAULT_CAPITAL, evaluate_quick, run_quick_scan
from .multibagger import evaluate_multibagger, scan_multibagger
from .paper import buy as paper_buy
from .paper import check_exits, portfolio_view, reset_portfolio, sell as paper_sell
from .universe import INDEX_CSV_URLS, NIFTY_INDEX_TICKER, get_multibagger_universe, get_universe
from .watchlist import add_to_watchlist, list_watchlist, refresh_scores, remove_from_watchlist

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="NSE Trading Bot")

_scan_lock = threading.Lock()
_scan_state: dict = {
    "status": "idle",
    "progress": 0,
    "total": 0,
    "results": [],
    "universe": None,
    "mode": "quick",
    "error": None,
    "scan_diagnostics": None,
}


def _run_scan(universe: str, capital: float, mode: str) -> None:
    global _scan_state
    try:
        if mode == "multibagger":
            symbols = get_multibagger_universe()
            universe_label = "nifty500+microcap250"
        else:
            symbols = get_universe(universe)
            universe_label = universe

        _scan_state.update(total=len(symbols), progress=0, universe=universe_label)

        def cb(done: int, total: int) -> None:
            _scan_state.update(progress=done, total=total)

        data = get_histories(symbols, progress_cb=cb)

        if mode == "multibagger":
            signals, diagnostics = scan_multibagger(data, qualified_only=False)
            results = [asdict(s) for s in signals]
        else:
            nifty = get_history(NIFTY_INDEX_TICKER)
            quick_signals, diagnostics = run_quick_scan(
                data, capital=capital, nifty_df=nifty, qualified_only=True
            )
            results = [asdict(s) for s in quick_signals]

        _scan_state.update(
            status="done",
            results=results,
            progress=len(symbols),
            scan_diagnostics=diagnostics,
            mode=mode,
        )
    except Exception as exc:
        _scan_state.update(status="error", error=str(exc))


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/universes")
def universes():
    return {
        "universes": list(INDEX_CSV_URLS),
        "default_capital": DEFAULT_CAPITAL,
        "modes": ["quick", "multibagger"],
    }


@app.post("/api/scan")
def start_scan(
    universe: str = "nifty500",
    capital: float = DEFAULT_CAPITAL,
    mode: str = "quick",
    sync: bool = False,
):
    if mode not in ("quick", "multibagger"):
        raise HTTPException(400, f"unknown mode {mode!r}")
    if mode == "quick" and universe not in INDEX_CSV_URLS:
        raise HTTPException(400, f"unknown universe {universe!r}")

    if sync:
        _scan_state.update(
            status="running", progress=0, total=0, results=[],
            universe=universe, mode=mode, error=None, scan_diagnostics=None,
        )
        _run_scan(universe, capital, mode)
        return _scan_state

    with _scan_lock:
        if _scan_state["status"] == "running":
            return {"status": "already-running"}
        _scan_state.update(
            status="running", progress=0, total=0, results=[],
            universe=universe, mode=mode, error=None, scan_diagnostics=None,
        )
        threading.Thread(
            target=_run_scan, args=(universe, capital, mode), daemon=True
        ).start()
    return {"status": "started"}


@app.get("/api/scan/status")
def scan_status():
    return _scan_state


@app.get("/api/stock/{symbol}")
def stock_detail(symbol: str, capital: float = DEFAULT_CAPITAL, mode: str = "quick"):
    symbol = symbol.upper()
    df = get_history(symbol)
    if df is None:
        raise HTTPException(404, f"no data for {symbol}")

    from .indicators import enrich

    e = enrich(df).iloc[-120:]
    candles = [
        {
            "time": str(idx.date()),
            "open": round(float(r.Open), 2),
            "high": round(float(r.High), 2),
            "low": round(float(r.Low), 2),
            "close": round(float(r.Close), 2),
            "volume": int(r.Volume),
            "sma50": None if r.sma50 != r.sma50 else round(float(r.sma50), 2),
            "sma200": None if r.sma200 != r.sma200 else round(float(r.sma200), 2),
        }
        for idx, r in e.iterrows()
    ]

    if mode == "multibagger":
        sig = evaluate_multibagger(symbol, df)
        return {
            "symbol": symbol,
            "signal": asdict(sig) if sig else None,
            "candles": candles,
            "mode": "multibagger",
        }

    nifty = get_history(NIFTY_INDEX_TICKER)
    sig = evaluate_quick(symbol, df, capital=capital, nifty_df=nifty)
    return {
        "symbol": symbol,
        "signal": asdict(sig) if sig else None,
        "candles": candles,
        "mode": "quick",
    }


class PaperBuyRequest(BaseModel):
    symbol: str
    qty: int
    strategy: str = "quick"
    price: float | None = None
    stop_loss: float | None = None
    target1: float | None = None
    target2: float | None = None
    notes: str = ""
    tier: str | None = "qualified"


class WatchlistAddRequest(BaseModel):
    symbol: str
    entry_price: float | None = None
    score: float | None = None
    tier: str | None = None
    sector: str = ""
    industry: str = ""
    market_cap_cr: float | None = None
    roe_pct: float | None = None
    revenue_growth_pct: float | None = None
    reasons: list[str] = []
    notes: str = ""


@app.get("/api/paper/portfolio")
def paper_portfolio(refresh: bool = False):
    return portfolio_view(refresh_prices=refresh)


@app.post("/api/paper/buy")
def paper_buy_endpoint(body: PaperBuyRequest):
    try:
        trade = paper_buy(
            symbol=body.symbol,
            qty=body.qty,
            strategy="quick",
            price=body.price,
            stop_loss=body.stop_loss,
            target1=body.target1,
            target2=body.target2,
            notes=body.notes,
            tier=body.tier or "qualified",
        )
        return {"trade": asdict(trade), "portfolio": portfolio_view(refresh_prices=True)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/paper/sell/{trade_id}")
def paper_sell_endpoint(trade_id: str, price: float | None = None):
    try:
        trade = paper_sell(trade_id, price=price)
        return {"trade": asdict(trade), "portfolio": portfolio_view(refresh_prices=True)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/paper/reset")
def paper_reset(starting_capital: float = DEFAULT_CAPITAL):
    pf = reset_portfolio(starting_capital)
    return {"portfolio": portfolio_view(), "message": f"Reset to ₹{starting_capital:,.0f}"}


@app.post("/api/paper/check-exits")
def paper_check_exits():
    closed = check_exits()
    return {
        "closed": [asdict(t) for t in closed],
        "portfolio": portfolio_view(refresh_prices=True),
    }


@app.get("/api/watchlist")
def watchlist_get(refresh: bool = False):
    return list_watchlist(refresh_prices=refresh)


@app.post("/api/watchlist/add")
def watchlist_add(body: WatchlistAddRequest):
    try:
        item = add_to_watchlist(
            symbol=body.symbol,
            entry_price=body.entry_price,
            score=body.score,
            tier=body.tier,
            sector=body.sector,
            industry=body.industry,
            market_cap_cr=body.market_cap_cr,
            roe_pct=body.roe_pct,
            revenue_growth_pct=body.revenue_growth_pct,
            reasons=body.reasons,
            notes=body.notes,
        )
        return {"item": asdict(item), "watchlist": list_watchlist(refresh_prices=True)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/watchlist/{symbol}")
def watchlist_remove(symbol: str):
    if not remove_from_watchlist(symbol):
        raise HTTPException(404, f"{symbol.upper()} not on watchlist")
    return {"watchlist": list_watchlist(refresh_prices=True)}


@app.post("/api/watchlist/refresh")
def watchlist_refresh():
    return refresh_scores()


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
