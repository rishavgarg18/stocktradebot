"""Writable storage that works both locally and on Vercel.

Locally everything lives under the repo's `.cache/` directory. On Vercel the
project filesystem is read-only except `/tmp`, which is also ephemeral (it is
wiped on cold starts and not shared across instances). So:

  - OHLCV / fundamentals / universe caches go to `/tmp` on Vercel. Losing them
    just means a scan re-downloads data; no correctness impact.
  - Paper-trade state is the one thing that MUST persist. If an Upstash/Vercel
    KV (Redis REST) is configured via env vars, we use it; otherwise we fall
    back to a `/tmp` file and flag the portfolio as non-durable so the UI can
    warn the user.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

IS_SERVERLESS = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))


def cache_base() -> Path:
    base = Path("/tmp/stocktradebot_cache") if IS_SERVERLESS else (
        Path(__file__).resolve().parent.parent / ".cache"
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


CACHE_BASE = cache_base()

# Upstash / Vercel KV REST credentials (set automatically when you attach a
# Vercel KV / Upstash Redis integration to the project).
_KV_URL = os.environ.get("KV_REST_API_URL") or os.environ.get("UPSTASH_REDIS_REST_URL")
_KV_TOKEN = os.environ.get("KV_REST_API_TOKEN") or os.environ.get("UPSTASH_REDIS_REST_TOKEN")


def kv_enabled() -> bool:
    return bool(_KV_URL and _KV_TOKEN)


def kv_get(key: str) -> dict | None:
    resp = requests.get(
        f"{_KV_URL}/get/{key}",
        headers={"Authorization": f"Bearer {_KV_TOKEN}"},
        timeout=10,
    )
    resp.raise_for_status()
    result = resp.json().get("result")
    if not result:
        return None
    return json.loads(result)


def kv_set(key: str, value: dict) -> None:
    resp = requests.post(
        f"{_KV_URL}/set/{key}",
        headers={"Authorization": f"Bearer {_KV_TOKEN}"},
        data=json.dumps(value),
        timeout=10,
    )
    resp.raise_for_status()


def storage_kind() -> str:
    if kv_enabled():
        return "redis"
    return "tmp" if IS_SERVERLESS else "disk"
