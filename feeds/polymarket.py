"""
polypredict/feeds/polymarket.py

Polymarket markets via the public Gamma API (no key).
"""

import json
import requests

UA = "polypredict/1.0 (research)"
GAMMA = "https://gamma-api.polymarket.com/markets"


def _yes_price(m):
    """Gamma returns outcomePrices as a JSON string list like '["0.62","0.38"]'."""
    raw = m.get("outcomePrices")
    try:
        prices = json.loads(raw) if isinstance(raw, str) else (raw or [])
        return float(prices[0]) if prices else None
    except (ValueError, TypeError, IndexError):
        return None


def fetch_markets(query: str = "", limit: int = 60):
    """Return normalized active markets, optionally filtered by a keyword in the question."""
    r = requests.get(
        GAMMA,
        params={"active": "true", "closed": "false", "limit": limit, "order": "volume", "ascending": "false"},
        headers={"User-Agent": UA}, timeout=20,
    )
    r.raise_for_status()
    out = []
    q = query.lower()
    for m in r.json():
        question = m.get("question") or ""
        if q and q not in question.lower():
            continue
        out.append({
            "venue": "Polymarket",
            "question": question,
            "yes_price": _yes_price(m),
            "volume": float(m.get("volume") or 0),
            "end": m.get("endDate"),
        })
    return out
