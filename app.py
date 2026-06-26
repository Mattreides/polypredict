"""
polypredict/app.py

Streamlit dashboard for short-timeframe prediction-market edge detection.
Tabs: Crypto Model · Weather Model · Live Markets · AI Analysis.

    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from feeds.crypto import fetch_ohlcv, build_features, DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES
from feeds import weather, polymarket, kalshi
from models.crypto_predictor import train_and_predict
from utils.kelly import size_position, format_recommendation

load_dotenv()
st.set_page_config(page_title="polypredict", layout="wide", initial_sidebar_state="expanded")
st.title("polypredict")
st.caption("Short-timeframe forecasting + calibration for binary prediction markets")

ss = st.session_state
ss.setdefault("crypto_result", None)
ss.setdefault("weather_data", None)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Bankroll & risk")
    bankroll = st.number_input("Bankroll ($)", min_value=10.0, value=1000.0, step=50.0)
    kelly_frac = st.slider("Kelly fraction", 0.05, 1.0, 0.25, 0.05,
                           help="Fraction of full Kelly. 0.25 (quarter) is a sane default.")
    min_edge = st.slider("Min edge to bet", 0.0, 0.25, 0.05, 0.01)
    st.divider()
    st.caption("Crypto: Kraken (US-accessible). Weather: NWS. Markets: Polymarket + Kalshi. All key-less.")

tab_c, tab_w, tab_m, tab_ai = st.tabs(["Crypto Model", "Weather Model", "Live Markets", "AI Analysis"])

# ── Crypto tab ───────────────────────────────────────────────────────────────
with tab_c:
    c1, c2, c3 = st.columns([1, 1, 1])
    symbol = c1.selectbox("Pair", DEFAULT_SYMBOLS, index=0)
    timeframe = c2.selectbox("Timeframe", DEFAULT_TIMEFRAMES, index=DEFAULT_TIMEFRAMES.index("1h"))
    n_train = c3.slider("Train window (candles)", 200, 700, 500, 50)

    if st.button("Train & Predict", type="primary"):
        with st.spinner("Fetching candles and training walk-forward model..."):
            try:
                df = fetch_ohlcv(symbol, timeframe, limit=720)
                X, y, _ = build_features(df)
                res = train_and_predict(X, y, n_train=n_train)
                res.update({"symbol": symbol, "timeframe": timeframe,
                            "last_close": float(df["close"].iloc[-1])})
                ss.crypto_result = res
            except Exception as e:
                st.error(f"Failed: {e}")

    if ss.crypto_result:
        r = ss.crypto_result
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P(up next bar)", f"{r['p_up']:.1%}")
        m2.metric("Walk-forward accuracy", f"{r['cv_accuracy']:.1%}")
        m3.metric("Brier score", f"{r['brier']:.3f}", help="0.25 = coin flip, lower is better")
        m4.metric("Last close", f"${r['last_close']:,.0f}")
        if r["brier"] > 0.245:
            st.info("Brier near 0.25 — treat this signal as weak/noisy.")

        st.subheader("Size it against a market")
        s1, s2 = st.columns(2)
        market_price = s1.number_input("Market YES price (0-1)", 0.01, 0.99, 0.50, 0.01)
        model_p = r["p_up"]
        sizing = size_position(model_p, market_price, bankroll, kelly_frac, min_edge)
        s2.write(""); s2.write("")
        (s2.success if sizing["direction"] != "NO_BET" else s2.warning)(format_recommendation(sizing))

# ── Weather tab ──────────────────────────────────────────────────────────────
with tab_w:
    st.caption("NWS forecasts are well-calibrated; edge comes from market mispricing. "
               "Enter your own probability for the contract's strike.")
    city = st.text_input("US city", "Knoxville")
    if st.button("Fetch NWS forecast"):
        with st.spinner("Geocoding and fetching NWS..."):
            try:
                ss.weather_data = weather.forecast_for_city(city)
            except Exception as e:
                st.error(f"Failed: {e}")

    if ss.weather_data:
        w = ss.weather_data
        st.write(f"**{w['query']}** → NWS grid: {w['location']}")
        for p in w["periods"][:6]:
            pp = f" · precip {p['precip_pct']}%" if p["precip_pct"] is not None else ""
            st.write(f"- **{p['name']}**: {p['temp']}°{p['unit']} — {p['short']}{pp}")

        st.subheader("Edge on a temperature/precip contract")
        w1, w2 = st.columns(2)
        your_p = w1.slider("Your probability for the strike", 0.0, 1.0, 0.5, 0.01)
        mkt_p = w2.slider("Market YES price", 0.01, 0.99, 0.5, 0.01)
        sizing = size_position(your_p, mkt_p, bankroll, kelly_frac, min_edge)
        (st.success if sizing["direction"] != "NO_BET" else st.warning)(format_recommendation(sizing))

# ── Markets tab ──────────────────────────────────────────────────────────────
with tab_m:
    st.caption("Polymarket (Gamma) returns fully-priced public markets. Kalshi's public endpoint "
               "currently returns zeroed prices without API credentials, so priced rows here are "
               "Polymarket-led; add a Kalshi key to enable its pricing.")
    kw = st.text_input("Filter keyword (e.g. bitcoin, temperature, ethereum)", "")
    if st.button("Load markets"):
        with st.spinner("Querying Polymarket + Kalshi..."):
            rows = []
            for src in (polymarket, kalshi):
                try:
                    rows += src.fetch_markets(query=kw, limit=80)
                except Exception as e:
                    st.warning(f"{src.__name__} failed: {e}")
            rows = [r for r in rows if r["yes_price"] not in (None, 0.0)]
            rows.sort(key=lambda r: r["volume"], reverse=True)
            if rows:
                st.dataframe(
                    [{"Venue": r["venue"], "Market": r["question"][:80],
                      "YES": f"{r['yes_price']:.2f}", "Volume": f"{r['volume']:,.0f}"} for r in rows[:60]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No priced markets matched.")

# ── AI Analysis tab ──────────────────────────────────────────────────────────
with tab_ai:
    st.caption("Synthesizes the current crypto/weather signals into a briefing via the Anthropic API.")
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.warning("Set ANTHROPIC_API_KEY in your .env to enable this tab.")
    elif st.button("Generate AI analysis"):
        ctx = []
        if ss.crypto_result:
            r = ss.crypto_result
            ctx.append(f"Crypto {r['symbol']} {r['timeframe']}: P(up)={r['p_up']:.1%}, "
                       f"walk-forward acc={r['cv_accuracy']:.1%}, Brier={r['brier']:.3f}.")
        if ss.weather_data:
            ctx.append(f"Weather: {ss.weather_data['location']} forecast loaded.")
        if not ctx:
            st.info("Train a crypto model or fetch weather first.")
        else:
            with st.spinner("Asking Claude..."):
                try:
                    import anthropic
                    client = anthropic.Anthropic()
                    msg = client.messages.create(
                        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                        max_tokens=700,
                        messages=[{"role": "user", "content":
                            "You are a quantitative trading analyst. Given these model signals, assess "
                            "signal quality (real skill vs noise), the main risks, and one concrete next "
                            "action. Be concise and skeptical.\n\n" + "\n".join(ctx)}],
                    )
                    st.markdown("".join(b.text for b in msg.content if b.type == "text"))
                except Exception as e:
                    st.error(f"API call failed: {e}")
