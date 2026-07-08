"""CLI scanner: python -m bot.scan --universe nifty500 --top 10"""

from __future__ import annotations

import argparse
import sys

from .data import get_histories, get_history
from .meanrev import DEFAULT_CAPITAL, run_quick_scan
from .universe import NIFTY_INDEX_TICKER, get_universe


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan NSE for qualified quick dip-buy signals")
    ap.add_argument("--universe", default="nifty500",
                    choices=["nifty50", "nifty100", "nifty200", "nifty500"])
    ap.add_argument("--top", type=int, default=10, help="show top N qualified")
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    args = ap.parse_args()

    symbols = get_universe(args.universe)
    print(f"Scanning {len(symbols)} stocks ({args.universe})...", file=sys.stderr)

    def cb(done: int, total: int) -> None:
        print(f"\r  data: {done}/{total}", end="", file=sys.stderr, flush=True)

    data = get_histories(symbols, progress_cb=cb)
    print(file=sys.stderr)

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
