"""Extract county demographic and economic predictors from ACS."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://api.census.gov/data/{year}/acs/acs5"
VARIABLES = {
    "B19013_001E": "median_household_income",
    "B17001_001E": "poverty_universe",
    "B17001_002E": "population_below_poverty",
    "B23025_003E": "civilian_labor_force",
    "B23025_005E": "unemployed_population",
    "B01002_001E": "median_age",
    "B25002_001E": "housing_units",
    "B25002_003E": "vacant_housing_units",
}


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_demographics_year(year: int, api_key: str | None = None, timeout: int = 60) -> pd.DataFrame:
    params = {"get": "NAME," + ",".join(VARIABLES), "for": "county:*", "in": "state:*"}
    if api_key:
        params["key"] = api_key
    response = _session().get(BASE_URL.format(year=year), params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"Unexpected Census response for {year}")
    frame = pd.DataFrame(payload[1:], columns=payload[0]).rename(columns=VARIABLES)
    frame["year"] = int(year)
    frame["source"] = "Census ACS 5-year"
    return frame


def extract_demographics(years: Iterable[int], output_dir: Path, api_key: str | None = None) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for year in years:
        LOGGER.info("Extracting county demographics for %s", year)
        frame = fetch_demographics_year(year, api_key=api_key)
        frame.to_json(output_dir / f"demographics_{year}.json", orient="records", indent=2)
        frame.to_parquet(output_dir / f"demographics_{year}.parquet", index=False)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
