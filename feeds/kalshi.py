"""
polypredict/feeds/kalshi.py

Kalshi markets via the public trade-api v2 (no key for public market data).
Host: api.elections.kalshi.com (current public endpoint).
"""

import requests

UA = "polypredict/1.0 (research)"
BASE = "https://api.elections.kalshi.com/trade-api/v2/markets"


def fetch_markets(query: str = "", limit: int = 100):
    """Return normalized open Kalshi markets, optionally keyword-filtered on the title."""
    r = requests.get(
        BASE, params={"limit": limit, "status": "open"},
        headers={"User-Agent": UA}, timeout=20,
    )
    r.raise_for_status()
    out = []
    q = query.lower()

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    for m in r.json().get("markets", []):
        title = m.get("title") or m.get("yes_sub_title") or m.get("ticker") or ""
        if q and q not in title.lower():
            continue
        # Kalshi reports prices as dollar-denominated STRINGS already in [0,1].
        yes_price = next(
            (v for v in (_f(m.get("yes_bid_dollars")), _f(m.get("last_price_dollars")), _f(m.get("yes_ask_dollars")))
             if v is not None),
            None,
        )
        out.append({
            "venue": "Kalshi",
            "question": title,
            "yes_price": yes_price,
            "volume": _f(m.get("volume_fp")) or 0.0,
            "end": m.get("close_time"),
        })
    return out
