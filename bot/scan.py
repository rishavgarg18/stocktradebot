"""CLI scanner: python -m bot.scan [--mode quick|multibagger]"""

from __future__ import annotations

import argparse
import sys

from .data import get_histories, get_history
from .meanrev import DEFAULT_CAPITAL, run_quick_scan
from .multibagger import scan_multibagger
from .universe import NIFTY_INDEX_TICKER, get_multibagger_universe, get_universe


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan NSE for quick dip-buys or multibagger candidates")
    ap.add_argument("--mode", default="quick", choices=["quick", "multibagger"])
    ap.add_argument("--universe", default="nifty500",
                    choices=["nifty50", "nifty100", "nifty200", "nifty500"])
    ap.add_argument("--top", type=int, default=10, help="show top N results")
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    args = ap.parse_args()

    if args.mode == "multibagger":
        symbols = get_multibagger_universe()
        label = "nifty500+microcap250"
    else:
        symbols = get_universe(args.universe)
        label = args.universe
    print(f"Scanning {len(symbols)} stocks ({label}, mode={args.mode})...", file=sys.stderr)

    def cb(done: int, total: int) -> None:
        print(f"\r  data: {done}/{total}", end="", file=sys.stderr, flush=True)

    data = get_histories(symbols, progress_cb=cb)
    print(file=sys.stderr)

    if args.mode == "multibagger":
        results, diag = scan_multibagger(data, qualified_only=False)
        print(f"\n=== MULTIBAGGER SCREENER ===")
        print(f"  Scanned: {diag.get('symbols_scanned', 0)} | sub-₹50: {diag.get('sub50_price_pass', 0)}")
        print(f"  Passed filters: {diag.get('scored', 0)} | qualified: {diag.get('qualified', 0)} | watch: {diag.get('watch', 0)}")
        hdr = f"{'SYM':<14}{'PRICE':>8}{'SCORE':>7}{'TIER':>10}{'ROE%':>7}{'REV%':>7}{'6M%':>7}{'MCAP':>8}"
        print(f"\n=== CANDIDATES ({len(results)}) ===\n{hdr}")
        for s in results[: args.top]:
            print(f"{s.symbol:<14}{s.close:>8.2f}{s.score:>7.1f}{s.tier:>10}"
                  f"{(s.roe_pct or 0):>7.1f}{(s.revenue_growth_pct or 0):>7.1f}"
                  f"{(s.roc_6m_pct or 0):>7.1f}{(s.market_cap_cr or 0):>8.0f}")
            for r in s.reasons[:2]:
                print(f"    · {r}")
        if not results:
            print("No candidates — most sub-₹50 stocks fail quality filters.")
        return

    nifty = get_history(NIFTY_INDEX_TICKER)
    qualified, diag = run_quick_scan(
        data, capital=args.capital, nifty_df=nifty, qualified_only=True
    )

    print(f"\n=== SCAN DIAGNOSTICS ===")
    print(f"  {diag.get('regime_note', '')}")
    print(f"  Dip setups: {diag.get('dip_setups_today', 0)} | passed stats: {diag.get('passed_stats', 0)}")
    print(f"  Qualified: {diag.get('qualified', 0)} (extended/review hidden — wait for qualified only)")
    if diag.get("top_filter_reasons"):
        print(f"  Top blockers: {' · '.join(diag['top_filter_reasons'])}")

    hdr = f"{'SYM':<14}{'CLOSE':>10}{'STOP':>10}{'QTY':>5}{'WIN%':>7}{'TRADES':>7}{'PF':>6}{'AVG%':>7}"
    print(f"\n=== QUALIFIED ({len(qualified)}) ===\n{hdr}")
    for s in qualified[: args.top]:
        print(f"{s.symbol:<14}{s.close:>10.2f}{s.stop_loss:>10.2f}{s.qty_for_1pct_risk:>5}"
              f"{s.hist_win_rate:>7.1f}{s.hist_trades:>7}{s.hist_profit_factor:>6.1f}"
              f"{s.hist_avg_pnl_pct:>7.2f}")
    if not qualified:
        print("No qualified setups — wait.")


if __name__ == "__main__":
    main()
