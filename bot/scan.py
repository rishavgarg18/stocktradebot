"""CLI scanner: python -m bot.scan --universe nifty50 --top 20"""

from __future__ import annotations

import argparse
import sys

from .data import get_histories, get_history
from .meanrev import quick_watchlist, scan_quick
from .strategy import DEFAULT_CAPITAL, evaluate, nifty_roc63_from
from .universe import NIFTY_INDEX_TICKER, get_universe


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan NSE stocks for swing signals")
    ap.add_argument("--universe", default="nifty500",
                    choices=["nifty50", "nifty100", "nifty200", "nifty500"])
    ap.add_argument("--mode", default="swing", choices=["swing", "quick"],
                    help="swing = trend signals; quick = 2-day mean-reversion")
    ap.add_argument("--top", type=int, default=25, help="show top N by score")
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    args = ap.parse_args()

    symbols = get_universe(args.universe)
    print(f"Scanning {len(symbols)} stocks ({args.universe})...", file=sys.stderr)

    def cb(done, total):
        print(f"\r  data: {done}/{total}", end="", file=sys.stderr, flush=True)

    data = get_histories(symbols, progress_cb=cb)
    print(file=sys.stderr)

    if args.mode == "quick":
        nifty = get_history(NIFTY_INDEX_TICKER)
        sigs = scan_quick(data, capital=args.capital, nifty_df=nifty)
        watch = quick_watchlist(data, capital=args.capital, nifty_df=nifty)
        hdr = f"{'SYM':<14}{'CLOSE':>10}{'STOP':>10}{'QTY':>7}{'WIN%':>7}{'TRADES':>8}{'PF':>6}{'1M%':>6}{'AVG%':>7}{'WORST%':>8}"

        def _row(s):
            recent_wr = f"{s.recent_win_rate:.1f}" if s.recent_win_rate is not None else "—"
            print(f"{s.symbol:<14}{s.close:>10.2f}{s.stop_loss:>10.2f}{s.qty_for_1pct_risk:>7}"
                  f"{s.hist_win_rate:>7.1f}{s.hist_trades:>8}{s.hist_profit_factor:>6.1f}{recent_wr:>6}"
                  f"{s.hist_avg_pnl_pct:>7.2f}{s.hist_worst_pct:>8.2f}")

        print(f"\n=== QUICK 2-DAY DIP-BUY SIGNALS ({len(sigs)}) — passed all gates ===")
        print("Stricter v2: costs, stops, Nifty filter, walk-forward. Max 3 picks/scan.\n")
        print(hdr)
        for s in sigs[: args.top]:
            _row(s)
            notes = "; ".join(s.fundamental_notes[:3])
            if notes:
                print(f"{'':<14}{notes}")
        if not sigs:
            print("No qualifying setups today (all gates passed).")

        print(f"\n=== NEAR-MISS WATCHLIST ({len(watch)}) — failed 1-2 gates, review manually ===")
        print(hdr)
        for s in watch[: args.top]:
            _row(s)
            print(f"{'':<14}missed: {'; '.join(s.failed_gates)}")
        if not watch:
            print("No near-misses today.")
        return

    nifty_roc = nifty_roc63_from(get_history(NIFTY_INDEX_TICKER))

    signals = []
    for sym, df in data.items():
        sig = evaluate(sym, df, nifty_roc63=nifty_roc, capital=args.capital)
        if sig:
            signals.append(sig)
    signals.sort(key=lambda s: s.score, reverse=True)

    buys = [s for s in signals if s.action == "BUY"]
    sells = [s for s in signals if s.action == "SELL"]

    hdr = f"{'SYM':<14}{'ACT':<7}{'SCORE':>6}{'CLOSE':>10}{'ENTRY':>10}{'STOP':>10}{'T1':>10}{'T2':>10}{'QTY':>7}"
    print(f"\n=== BUY SIGNALS ({len(buys)}) ===")
    print(hdr)
    for s in buys[: args.top]:
        print(f"{s.symbol:<14}{s.action:<7}{s.score:>6.1f}{s.close:>10.2f}"
              f"{s.entry or 0:>10.2f}{s.stop_loss or 0:>10.2f}"
              f"{s.target1 or 0:>10.2f}{s.target2 or 0:>10.2f}{s.qty_for_1pct_risk or 0:>7}")

    print(f"\n=== SELL SIGNALS ({len(sells)}) ===")
    print(f"{'SYM':<14}{'SCORE':>6}{'CLOSE':>10}  REASONS")
    for s in sells[: args.top]:
        print(f"{s.symbol:<14}{s.score:>6.1f}{s.close:>10.2f}  {'; '.join(s.warnings)}")

    watch = [s for s in signals if s.action == "WATCH"]
    print(f"\n=== WATCHLIST ({len(watch)}) ===")
    for s in watch[: args.top]:
        print(f"{s.symbol:<14}{s.score:>6.1f}{s.close:>10.2f}")


if __name__ == "__main__":
    main()
