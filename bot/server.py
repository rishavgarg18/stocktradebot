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

from .backtest import run_backtest
from .data import get_histories, get_history
from .meanrev import quick_watchlist, scan_quick
from .paper import buy as paper_buy
from .paper import portfolio_view, reset_portfolio, sell as paper_sell
from .strategy import DEFAULT_CAPITAL, evaluate, nifty_roc63_from
from .universe import INDEX_CSV_URLS, NIFTY_INDEX_TICKER, get_universe

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="NSE Swing Signal Bot")

_scan_lock = threading.Lock()
_scan_state: dict = {"status": "idle", "progress": 0, "total": 0, "results": [], "universe": None, "mode": "swing", "error": None}


def _run_scan(universe: str, capital: float, mode: str) -> None:
    global _scan_state
    try:
        symbols = get_universe(universe)
        _scan_state.update(total=len(symbols), progress=0)

        def cb(done: int, total: int) -> None:
            _scan_state.update(progress=done, total=total)

        data = get_histories(symbols, progress_cb=cb)

        if mode == "quick":
            nifty = get_history(NIFTY_INDEX_TICKER)
            signals = scan_quick(data, capital=capital, nifty_df=nifty)
            watch = quick_watchlist(data, capital=capital, nifty_df=nifty)
            results = [asdict(s) for s in signals] + [asdict(s) for s in watch]
        else:
            nifty = get_history(NIFTY_INDEX_TICKER)
            nifty_roc = nifty_roc63_from(nifty)
            results = []
            for sym, df in data.items():
                sig = evaluate(sym, df, nifty_roc63=nifty_roc, capital=capital)
                if sig is not None:
                    results.append(asdict(sig))
            results.sort(key=lambda s: s["score"], reverse=True)
        _scan_state.update(status="done", results=results, progress=len(symbols))
    except Exception as exc:  # surface scan failures to the UI
        _scan_state.update(status="error", error=str(exc))


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/universes")
def universes():
    return {"universes": list(INDEX_CSV_URLS)}


@app.post("/api/scan")
def start_scan(
    universe: str = "nifty500",
    capital: float = DEFAULT_CAPITAL,
    mode: str = "swing",
    sync: bool = False,
):
    if universe not in INDEX_CSV_URLS:
        raise HTTPException(400, f"unknown universe {universe!r}")
    if mode not in ("swing", "quick"):
        raise HTTPException(400, f"unknown mode {mode!r}")

    # Serverless (Vercel) can't share memory between the start request and a
    # separate status poll, so run the whole scan inline and return results.
    if sync:
        _scan_state.update(
            status="running", progress=0, total=0, results=[],
            universe=universe, mode=mode, error=None,
        )
        _run_scan(universe, capital, mode)
        return _scan_state

    with _scan_lock:
        if _scan_state["status"] == "running":
            return {"status": "already-running"}
        _scan_state.update(
            status="running", progress=0, total=0, results=[],
            universe=universe, mode=mode, error=None,
        )
        threading.Thread(target=_run_scan, args=(universe, capital, mode), daemon=True).start()
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
    sig = evaluate(symbol, df, nifty_roc63=nifty_roc63_from(nifty), capital=capital)
    bt = run_backtest(symbol, df, nifty_df=nifty)

    from .indicators import enrich

    e = enrich(df).iloc[-260:]
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
        "backtest": asdict(bt) if bt else None,
        "candles": candles,
    }


class PaperBuyRequest(BaseModel):
    symbol: str
    qty: int
    strategy: str = "swing"
    price: float | None = None
    stop_loss: float | None = None
    target1: float | None = None
    target2: float | None = None
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
            strategy=body.strategy,
            price=body.price,
            stop_loss=body.stop_loss,
            target1=body.target1,
            target2=body.target2,
            notes=body.notes,
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


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
