"""
polypredict/feeds/weather.py

National Weather Service forecasts (api.weather.gov) — free, no key, US only.
City names are geocoded via Open-Meteo's free geocoding API.

NWS forecasts are already well-calibrated by meteorologists, so edge on weather
contracts comes from MARKET mispricing, not from beating the forecast. This module
surfaces the forecast; you supply the probability for a given strike (manual
override in the app) and compare it to the market price.
"""

import requests

UA = "polypredict/1.0 (research; contact: you@example.com)"
GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
NWS_POINTS = "https://api.weather.gov/points/{lat},{lon}"


def geocode_city(city: str):
    """Return (lat, lon, label) for a city name, or raise."""
    r = requests.get(GEOCODE, params={"name": city, "count": 1, "country": "US"},
                     headers={"User-Agent": UA}, timeout=15)
    r.raise_for_status()
    results = r.json().get("results") or []
    if not results:
        raise ValueError(f"No geocoding match for '{city}'.")
    g = results[0]
    label = ", ".join(x for x in [g.get("name"), g.get("admin1"), g.get("country_code")] if x)
    return g["latitude"], g["longitude"], label


def fetch_nws_forecast(lat: float, lon: float):
    """Return {location, periods:[{name, temp, unit, short, precip_pct, is_daytime}]}."""
    hdr = {"User-Agent": UA, "Accept": "application/geo+json"}
    pt = requests.get(NWS_POINTS.format(lat=round(lat, 4), lon=round(lon, 4)), headers=hdr, timeout=15)
    pt.raise_for_status()
    props = pt.json()["properties"]
    rel = props.get("relativeLocation", {}).get("properties", {})
    loc = f"{rel.get('city', '')}, {rel.get('state', '')}".strip(", ")

    fc = requests.get(props["forecast"], headers=hdr, timeout=15)
    fc.raise_for_status()
    periods = []
    for p in fc.json()["properties"]["periods"][:8]:
        pop = (p.get("probabilityOfPrecipitation") or {}).get("value")
        periods.append({
            "name": p["name"], "temp": p["temperature"], "unit": p["temperatureUnit"],
            "short": p["shortForecast"], "precip_pct": pop, "is_daytime": p["isDaytime"],
        })
    return {"location": loc or f"{lat:.2f},{lon:.2f}", "periods": periods}


def forecast_for_city(city: str):
    lat, lon, label = geocode_city(city)
    data = fetch_nws_forecast(lat, lon)
    data["query"] = label
    return data
