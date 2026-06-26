# polypredict

A short-timeframe **probabilistic forecasting and calibration** system for binary
markets. It trains a walk-forward gradient-boosting classifier on live market-data
features, calibrates the output into honest probabilities, and sizes positions by
expected value. Built as a Streamlit dashboard; every data source is free and key-less
except the optional AI-analysis tab.

The interesting engineering here is calibration and validation discipline, not the
betting: turning a classifier's raw scores into trustworthy probabilities, measuring
whether the signal is real skill or noise (Brier score, walk-forward CV), and only
acting when the modeled probability diverges from the market by more than a threshold.

> _Add a screenshot:_ `![polypredict](screenshot.png)`

## What it does

- **Crypto direction model** — pulls live OHLCV, computes 19 scale-invariant technical
  features, trains a walk-forward `GradientBoostingClassifier` with Platt (sigmoid)
  calibration, and reports P(up) for the next bar alongside its out-of-sample accuracy
  and Brier score.
- **Weather model** — fetches NWS forecasts and frames temperature/precipitation
  contracts; since NWS is already well-calibrated, edge comes from market mispricing.
- **Live markets browser** — Polymarket (Gamma API) priced markets, keyword-filterable.
- **AI analysis** — optional Anthropic-powered briefing that critiques signal quality
  and flags risks.
- **Kelly sizing** — fractional Kelly with EV, applied across all of the above.

## Architecture

```
feeds/
  crypto.py      Kraken OHLCV (US-accessible) + 19-feature pipeline
  weather.py     NWS forecasts + Open-Meteo geocoding (key-less)
  polymarket.py  Polymarket Gamma public API
  kalshi.py      Kalshi trade-api v2 (see note below)
models/
  crypto_predictor.py   walk-forward GBM + Platt calibration (no stored models)
utils/
  kelly.py       fractional Kelly + expected value
app.py           Streamlit dashboard (4 tabs)
```

## Setup & run

```bash
pip install -r requirements.txt
cp .env.example .env        # only needed for the AI Analysis tab
streamlit run app.py
```

No API keys are required for the crypto, weather, or Polymarket features.

## Two data-source notes (real findings, worth keeping in the README)

**Crypto source is Kraken, not Binance.** The original build used Binance via `ccxt`.
Binance now returns **HTTP 451 ("Unavailable For Legal Reasons")** to US IP addresses,
so that feed is dead from a US location. This version pulls OHLCV directly from Kraken's
public REST API (US-accessible, no key), which also removes the `ccxt` dependency. Set
the symbol map in `feeds/crypto.py` to switch exchanges.

**Kalshi public pricing is gated.** Kalshi's public `/markets` endpoint now returns
zeroed bid/ask/last fields without API credentials (verified across 8,000 markets). The
markets browser therefore leads on Polymarket, which serves full pricing key-less; the
Kalshi feed is wired and ready but will surface prices only once a Kalshi key is added.

## Reading the numbers

- **Edge** = `model_prob − market_price`. Bet only when `|edge|` exceeds the threshold.
- **Brier score** — 0.25 is a coin flip, 0.00 is perfect. On short-timeframe crypto,
  expect **0.22–0.24**; anything under 0.20 on live data deserves a lookahead-bias audit.
- **Walk-forward accuracy** — typically **53–55%** here. That is modest by design; large
  numbers on noisy short-horizon data usually mean a leak, not skill.
- **Platt calibration** — GBM scores cluster near 0/1 and aren't real probabilities; the
  sigmoid wrapper makes them usable in Kelly math.
- **Walk-forward training** — the model retrains fresh on recent candles every run; there
  are no saved model files, because short-horizon regimes drift fast.
- **Kelly** — fractional (default 0.25×) because full Kelly assumes perfect calibration
  and produces brutal variance in practice.

## Configuration

`feeds/crypto.py`: `SYMBOL_MAP`, `TF_MIN` (supported timeframes), exchange endpoint.
`.env`: `ANTHROPIC_API_KEY` (AI tab only), optional `ANTHROPIC_MODEL`.

## License

MIT — see [LICENSE](LICENSE).
