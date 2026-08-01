import pandas as pd
import pytest

from src.transform.build_features import build_features
from src.transform.clean_demographics import clean_demographics
from src.transform.clean_population import clean_population
from src.transform.validate_data import validate_population


def sample_population():
    return pd.DataFrame({
        "NAME": ["A", "A", "A"], "population": ["100", "110", "105"],
        "state": ["1", "1", "1"], "county": ["1", "1", "1"], "year": [2020, 2021, 2022],
    })


def sample_demographics():
    return pd.DataFrame({
        "NAME": ["A", "A", "A"], "state": ["1"] * 3, "county": ["1"] * 3, "year": [2020, 2021, 2022],
        "median_household_income": [50000, 51000, 52000], "poverty_universe": [100, 110, 105],
        "population_below_poverty": [10, 11, 10], "civilian_labor_force": [60, 65, 63],
        "unemployed_population": [3, 4, 3], "median_age": [40, 40.5, 41],
        "housing_units": [50, 52, 53], "vacant_housing_units": [5, 5, 4],
    })


def test_fips_and_target_are_created_correctly():
    pop = clean_population(sample_population())
    demo = clean_demographics(sample_demographics())
    ref = pd.DataFrame({"state_fips": ["01"], "state_name": ["Alabama"], "region": ["South"]})
    features = build_features(pop, demo, ref)
    assert features["full_fips"].eq("01001").all()
    assert features.loc[features["year"] == 2020, "target_growth_next_year"].iloc[0] == 1
    assert features.loc[features["year"] == 2021, "target_growth_next_year"].iloc[0] == 0
    assert pd.isna(features.loc[features["year"] == 2022, "target_growth_next_year"].iloc[0])


def test_validation_rejects_negative_population():
    frame = pd.DataFrame({"full_fips": ["01001"], "year": [2020], "population": [-1]})
    with pytest.raises(ValueError, match="Negative"):
        validate_population(frame)
