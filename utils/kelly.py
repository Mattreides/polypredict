"""
polypredict/utils/kelly.py

Fractional Kelly position sizing with expected-value calculation for binary
prediction-market contracts priced in [0, 1].
"""


def kelly_fraction(model_prob: float, market_yes_price: float) -> float:
    """Full-Kelly fraction for a YES bet: f* = (p - q) / (1 - q)."""
    p, q = model_prob, market_yes_price
    if q >= 1.0 or q <= 0.0:
        return 0.0
    return (p - q) / (1 - q)


def size_position(model_prob, market_yes_price, bankroll, kelly_frac=0.25, min_edge=0.05):
    """
    Returns a sizing dict: direction (YES/NO/NO_BET), recommended stake, and EV.
    Clamps full Kelly to 25% of bankroll even before applying the fractional multiplier.
    """
    edge = model_prob - market_yes_price

    if abs(edge) < min_edge:
        return {
            "direction": "NO_BET",
            "reason": f"Edge {edge:+.1%} below threshold {min_edge:.1%}",
            "edge": edge, "model_prob": model_prob, "market_prob": market_yes_price,
            "full_kelly_pct": 0.0, "recommended_pct": 0.0, "recommended_usd": 0.0,
            "expected_value_pct": 0.0, "bankroll": bankroll, "kelly_fraction_used": kelly_frac,
        }

    direction = "YES" if edge > 0 else "NO"
    model_p = model_prob if direction == "YES" else (1 - model_prob)
    market_p = market_yes_price if direction == "YES" else (1 - market_yes_price)

    full_k = abs(kelly_fraction(model_prob, market_yes_price))
    full_k_clamped = max(0.0, min(full_k, 0.25))
    recommended_pct = full_k_clamped * kelly_frac

    # EV per $1 staked on `direction`: p*(1/price - 1) - (1-p)
    ev = model_p * ((1 / market_p) - 1) - (1 - model_p)

    return {
        "direction": direction, "edge": edge,
        "model_prob": model_prob, "market_prob": market_yes_price,
        "full_kelly_pct": full_k_clamped, "recommended_pct": recommended_pct,
        "recommended_usd": recommended_pct * bankroll, "expected_value_pct": ev,
        "bankroll": bankroll, "kelly_fraction_used": kelly_frac,
    }


def format_recommendation(s: dict) -> str:
    if s["direction"] == "NO_BET":
        return f"NO BET — {s['reason']}"
    return (
        f"BET {s['direction']} | Edge {s['edge']:+.1%} | "
        f"Model {s['model_prob']:.1%} vs Market {s['market_prob']:.1%} | "
        f"Stake ${s['recommended_usd']:.2f} ({s['recommended_pct']:.1%} of ${s['bankroll']:.0f}) | "
        f"EV {s['expected_value_pct']:+.1%}/$"
    )
