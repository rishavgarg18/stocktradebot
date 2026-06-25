# NSE Swing Trading Signal Bot

A swing-trading signal engine for Indian (NSE) stocks. It scans the Nifty 500,
scores every stock against a blend of time-tested strategies, and produces
BUY / SELL signals with entry, stop-loss, targets, and position sizing —
served through a web dashboard with candlestick charts and per-stock backtests.

## Strategy blend

| Pillar | Source of the idea | What it checks |
|---|---|---|
| Trend | Weinstein stage analysis, Minervini trend template | Price above rising 50/200 DMA, MAs stacked bullishly |
| Momentum | O'Neil (CANSLIM), relative strength | Near 52-week high, outperforming Nifty, positive ROC |
| Entry timing | Pullback / mean-reversion within trend | RSI reset, bounce off 20/50 DMA, MACD turn |
| Volume | Wyckoff, O'Neil | Up-moves on above-average volume |
| Risk | Van Tharp position sizing | ATR(14) stop, 2R/3R targets, 1% account risk per trade |

A stock only gets a BUY signal when the trend, momentum, and entry-timing
pillars agree. SELL signals fire on trend breaks, stop violations, and
distribution patterns.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m bot.server          # starts http://localhost:8000
```

Open http://localhost:8000 — hit **Run Scan** to scan the universe
(first scan downloads ~1 year of daily data for every symbol; subsequent
scans use the on-disk cache and are much faster).

## CLI scan (no UI)

```bash
python -m bot.scan --universe nifty50 --top 20
```

## Deploy to Vercel

The app is Vercel-ready (`api/index.py` exposes the FastAPI ASGI app,
`vercel.json` rewrites all routes to it). Because Vercel runs stateless,
short-lived serverless functions, the app adapts automatically:

- **Scans run synchronously** (`?sync=true`) instead of using a background
  thread + status poll, since serverless requests don't share memory.
- **Caches go to `/tmp`** on Vercel (the project filesystem is read-only).
  This is ephemeral — scans just re-download data after a cold start.
- **Use Nifty 50** on Vercel. Nifty 200/500 scans take 60s+ and will hit the
  function timeout (`maxDuration` is set to 60s, the Hobby max).

### Steps

```bash
npm i -g vercel       # CLI (already installed in this workspace)
vercel login          # interactive — log into your account
vercel                # deploy a preview
vercel --prod         # promote to production
```

### Make paper trades persist (important)

`/tmp` is wiped on cold starts, so paper trades won't survive on Vercel
unless you attach a Redis store. Add a **Vercel KV / Upstash Redis**
integration to the project — it sets `KV_REST_API_URL` and
`KV_REST_API_TOKEN` automatically, and the app detects them and stores the
portfolio in Redis. Without it, the dashboard shows a "trades NOT saved"
warning.

> Honest note: this app (long scans, stateful background work, file caches)
> is a better fit for an always-on host like Railway/Render/Fly than for
> serverless. Vercel works for a Nifty-50 demo with KV, but for full Nifty
> 500 scanning and durable state, a small always-on container is simpler.

## Project layout

```
bot/
  universe.py    # Nifty 50/100/200/500 symbol lists (NSE CSV + bundled fallback)
  data.py        # Yahoo Finance OHLCV download with parquet cache
  indicators.py  # SMA, EMA, RSI, MACD, ATR, ADX, Bollinger, ROC, OBV...
  strategy.py    # composite scoring engine -> Signal (entry/stop/targets/size)
  backtest.py    # historical backtest of the signal rules per stock
  server.py      # FastAPI app + REST API
  scan.py        # CLI scanner
static/
  index.html     # dashboard (TradingView lightweight-charts)
```

## Honest disclaimer

No strategy has a guaranteed win rate. This bot enforces discipline —
trend alignment, volume confirmation, predefined stops, small position
sizes — which is what actually keeps risk low. Backtest results are
historical, not predictive. This is a research/educational tool, not
investment advice. Always do your own diligence; SEBI registration is
required to give investment advice in India.
