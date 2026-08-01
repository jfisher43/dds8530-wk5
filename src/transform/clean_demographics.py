"""Demographic cleaning and rate construction."""
from __future__ import annotations
import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    "median_household_income", "poverty_universe", "population_below_poverty",
    "civilian_labor_force", "unemployed_population", "median_age",
    "housing_units", "vacant_housing_units",
]


def clean_demographics(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = df.columns.str.lower()
    df["state"] = df["state"].astype("string").str.zfill(2)
    df["county"] = df["county"].astype("string").str.zfill(3)
    df["full_fips"] = df["state"] + df["county"]
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan
    df["poverty_rate"] = df["population_below_poverty"].div(df["poverty_universe"].replace(0, np.nan))
    df["unemployment_rate"] = df["unemployed_population"].div(df["civilian_labor_force"].replace(0, np.nan))
    df["housing_vacancy_rate"] = df["vacant_housing_units"].div(df["housing_units"].replace(0, np.nan))
    return df.drop_duplicates(subset=["full_fips", "year"]).reset_index(drop=True)
