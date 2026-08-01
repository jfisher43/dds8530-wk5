"""Population cleaning logic."""
from __future__ import annotations
import pandas as pd


def clean_population(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = df.columns.str.lower()
    for col, width in (("state", 2), ("county", 3)):
        df[col] = df[col].astype("string").str.zfill(width)
    df["full_fips"] = df["state"] + df["county"]
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["full_fips", "population", "year"])
    df = df[df["population"] >= 0]
    return df.drop_duplicates(subset=["full_fips", "year"]).reset_index(drop=True)
