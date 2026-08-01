"""Lightweight data-quality gates."""
from __future__ import annotations
import pandas as pd


def validate_population(df: pd.DataFrame) -> None:
    required = {"full_fips", "year", "population"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing population columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Population data is empty")
    if (df["population"] < 0).any():
        raise ValueError("Negative population detected")
    if df.duplicated(["full_fips", "year"]).any():
        raise ValueError("Duplicate county-year population rows detected")
    if not df["full_fips"].astype(str).str.fullmatch(r"\d{5}").all():
        raise ValueError("Invalid county FIPS code")


def validate_features(df: pd.DataFrame) -> None:
    required = {"full_fips", "year", "population", "target_growth_next_year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    valid_targets = set(df["target_growth_next_year"].dropna().astype(int).unique())
    if not valid_targets.issubset({0, 1}):
        raise ValueError(f"Invalid target values: {valid_targets}")
