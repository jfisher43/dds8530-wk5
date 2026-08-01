"""Join sources and create leakage-safe county-year features."""
from __future__ import annotations
import pandas as pd


def build_features(population: pd.DataFrame, demographics: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    demo_keep = [
        "full_fips", "year", "median_household_income", "median_age",
        "poverty_rate", "unemployment_rate", "housing_vacancy_rate",
    ]
    df = population.merge(demographics[demo_keep], on=["full_fips", "year"], how="inner")
    ref = reference.rename(columns={"state_fips": "state"}).copy()
    ref["state"] = ref["state"].astype("string").str.zfill(2)
    df = df.merge(ref[["state", "state_name", "region"]], on="state", how="left")
    df = df.sort_values(["full_fips", "year"]).reset_index(drop=True)
    group = df.groupby("full_fips", group_keys=False)
    df["population_growth_rate"] = group["population"].pct_change(fill_method=None)
    df["population_lag_1"] = group["population"].shift(1)
    df["growth_rate_lag_1"] = group["population_growth_rate"].shift(1)
    next_population = group["population"].shift(-1)
    df["target_growth_next_year"] = (next_population > df["population"]).astype("Int64")
    df.loc[next_population.isna(), "target_growth_next_year"] = pd.NA
    return df
