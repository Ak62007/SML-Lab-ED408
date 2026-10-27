"""Build the World Bank macroeconomic panel used by ED408 Questions 11 and 12.

The script downloads official World Development Indicators data, removes
regional/income aggregates, and saves a local snapshot for offline notebooks.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "world_bank_macro_2010_2023.csv"
METADATA_PATH = ROOT / "data" / "world_bank_macro_metadata.json"
API_ROOT = "https://api.worldbank.org/v2"
START_YEAR = 2010
END_YEAR = 2023

INDICATORS = {
    "gdp_per_capita_constant_2015_usd": "NY.GDP.PCAP.KD",
    "trade_pct_gdp": "NE.TRD.GNFS.ZS",
    "gross_capital_formation_pct_gdp": "NE.GDI.TOTL.ZS",
    "inflation_consumer_pct": "FP.CPI.TOTL.ZG",
    "unemployment_pct": "SL.UEM.TOTL.ZS",
    "fdi_net_inflows_pct_gdp": "BX.KLT.DINV.WD.GD.ZS",
    "government_consumption_pct_gdp": "NE.CON.GOVT.ZS",
    "industry_value_added_pct_gdp": "NV.IND.TOTL.ZS",
    "urban_population_pct": "SP.URB.TOTL.IN.ZS",
    "population_growth_pct": "SP.POP.GROW",
}


def fetch_json(path: str, **params):
    url = f"{API_ROOT}/{path}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "ED408-lab/1.0"})
    last_error = None
    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=180) as response:
                payload = json.load(response)
            break
        except (OSError, TimeoutError) as error:
            last_error = error
            if attempt == 3:
                raise RuntimeError(f"World Bank API failed after 3 attempts: {url}") from error
            print(f"API attempt {attempt} failed ({error}); retrying...", flush=True)
            time.sleep(2 * attempt)
    else:
        raise RuntimeError(f"World Bank API failed: {url}") from last_error
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f"Unexpected response from {url}: {payload!r}")
    return payload[1], url


def fetch_countries():
    records, url = fetch_json("country", format="json", per_page=400)
    countries = {}
    for record in records:
        region = record.get("region", {})
        if not region.get("id") or not record.get("id"):
            continue
        countries[record["id"]] = {
            "country": record["name"],
            "region": region.get("value"),
            "income_level": record.get("incomeLevel", {}).get("value"),
            "income_level_code": record.get("incomeLevel", {}).get("id"),
        }
    return countries, url


def fetch_indicator(code: str):
    records, url = fetch_json(
        f"country/all/indicator/{code}",
        format="json",
        date=f"{START_YEAR}:{END_YEAR}",
        per_page=20000,
        source=2,
    )
    values = {}
    for record in records:
        iso3 = record.get("countryiso3code")
        year = record.get("date")
        if iso3 and year and record.get("value") is not None:
            values[(iso3, int(year))] = float(record["value"])
    return values, url


def main():
    countries, country_url = fetch_countries()
    rows = {
        (iso3, year): {"country_code": iso3, "year": year, **metadata}
        for iso3, metadata in countries.items()
        for year in range(START_YEAR, END_YEAR + 1)
    }

    source_urls = {"country_metadata": country_url}
    for column, code in INDICATORS.items():
        print(f"Downloading {code}: {column}", flush=True)
        values, url = fetch_indicator(code)
        source_urls[code] = url
        for key, value in values.items():
            if key in rows:
                rows[key][column] = value

    frame = pd.DataFrame(rows.values()).sort_values(["country_code", "year"])
    numeric_columns = list(INDICATORS)
    frame = frame.loc[frame[numeric_columns].notna().any(axis=1)].copy()
    if frame.duplicated(["country_code", "year"]).any():
        raise RuntimeError("Duplicate country-year rows were produced")
    if frame["gdp_per_capita_constant_2015_usd"].notna().sum() < 1_000:
        raise RuntimeError("Too few GDP-per-capita observations; check the API response")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False, float_format="%.10g")
    metadata = {
        "title": "World Bank macroeconomic country-year panel",
        "source": "World Development Indicators, World Bank Indicators API v2",
        "license": "CC BY 4.0",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "year_range": [START_YEAR, END_YEAR],
        "country_filter": "Economies with a non-empty World Bank region; aggregates excluded",
        "indicators": INDICATORS,
        "source_urls": source_urls,
        "rows": len(frame),
        "economies": int(frame["country_code"].nunique()),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(frame):,} rows for {frame['country_code'].nunique()} economies")
    print(f"Data: {DATA_PATH.relative_to(ROOT)}")
    print(f"Metadata: {METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
