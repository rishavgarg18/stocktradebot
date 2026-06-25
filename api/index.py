"""Vercel serverless entrypoint.

Vercel's Python runtime serves the ASGI `app` exported here. All routes are
rewritten to this function via vercel.json, so FastAPI handles both the
dashboard HTML and the JSON API.
"""

import sys
from pathlib import Path

# Ensure the repo root is importable so `bot` resolves when Vercel runs this
# file from within the api/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.server import app  # noqa: E402

__all__ = ["app"]
