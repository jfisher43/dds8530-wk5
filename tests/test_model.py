from pathlib import Path
import pandas as pd

from src.models.predict import load_model_bundle, predict_one
from src.models.train import train_model


def model_frame():
    rows = []
    for year in range(2018, 2023):
        for i in range(20):
            rows.append({
                "year": year, "population": 1000 + 10 * i + year,
                "population_lag_1": 990 + 10 * i + year,
                "growth_rate_lag_1": (i - 10) / 1000,
                "median_household_income": 40000 + i * 1000,
                "median_age": 30 + i / 2, "poverty_rate": 0.05 + i / 500,
                "unemployment_rate": 0.02 + i / 1000, "housing_vacancy_rate": 0.04 + i / 1000,
                "region": ["South", "West", "Midwest", "Northeast"][i % 4],
                "target_growth_next_year": int(i % 3 != 0),
            })
    return pd.DataFrame(rows)


def test_model_trains_saves_and_predicts(tmp_path: Path):
    path = tmp_path / "model.joblib"
    train_model(model_frame(), path, tracking_uri=None)
    bundle = load_model_bundle(path)
    sample = model_frame().iloc[0][bundle["features"]].to_dict()
    result = predict_one(bundle, sample)
    assert result["prediction"] in {0, 1}
    assert 0 <= result["growth_probability"] <= 1
