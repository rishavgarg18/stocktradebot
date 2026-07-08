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
from .paper import buy as paper_buy
from .paper import check_exits, portfolio_view, reset_portfolio, sell as paper_sell
from .universe import INDEX_CSV_URLS, NIFTY_INDEX_TICKER, get_universe

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="NSE Quick Dip Bot")

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


def _run_scan(universe: str, capital: float) -> None:
    global _scan_state
    try:
        symbols = get_universe(universe)
        _scan_state.update(total=len(symbols), progress=0)

        def cb(done: int, total: int) -> None:
            _scan_state.update(progress=done, total=total)

        data = get_histories(symbols, progress_cb=cb)
        nifty = get_history(NIFTY_INDEX_TICKER)
        quick_signals, diagnostics = run_quick_scan(
            data, capital=capital, nifty_df=nifty, qualified_only=True
        )
        _scan_state.update(
            status="done",
            results=[asdict(s) for s in quick_signals],
            progress=len(symbols),
            scan_diagnostics=diagnostics,
            mode="quick",
        )
    except Exception as exc:
        _scan_state.update(status="error", error=str(exc))


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/universes")
def universes():
    return {"universes": list(INDEX_CSV_URLS), "default_capital": DEFAULT_CAPITAL}


@app.post("/api/scan")
def start_scan(
    universe: str = "nifty500",
    capital: float = DEFAULT_CAPITAL,
    sync: bool = False,
):
    if universe not in INDEX_CSV_URLS:
        raise HTTPException(400, f"unknown universe {universe!r}")

    if sync:
        _scan_state.update(
            status="running", progress=0, total=0, results=[],
            universe=universe, mode="quick", error=None, scan_diagnostics=None,
        )
        _run_scan(universe, capital)
        return _scan_state

    with _scan_lock:
        if _scan_state["status"] == "running":
            return {"status": "already-running"}
        _scan_state.update(
            status="running", progress=0, total=0, results=[],
            universe=universe, mode="quick", error=None, scan_diagnostics=None,
        )
        threading.Thread(target=_run_scan, args=(universe, capital), daemon=True).start()
    return {"status": "started"}


@app.get("/api/scan/status")
def scan_status():
    return _scan_state


@app.get("/api/stock/{symbol}")
def stock_detail(symbol: str, capital: float = DEFAULT_CAPITAL):
    symbol = symbol.upper()
    df = get_history(symbol)
    if df is None:
        raise HTTPException(404, f"no data for {symbol}")

    nifty = get_history(NIFTY_INDEX_TICKER)
    sig = evaluate_quick(symbol, df, capital=capital, nifty_df=nifty)

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

    return {
        "symbol": symbol,
        "signal": asdict(sig) if sig else None,
        "candles": candles,
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


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
