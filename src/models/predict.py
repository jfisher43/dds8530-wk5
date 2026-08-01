"""Load the persisted model bundle and make predictions."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import joblib
import pandas as pd


def load_model_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run the training pipeline first.")
    return joblib.load(path)


def predict_one(bundle: dict[str, Any], values: dict[str, Any]) -> dict[str, float | int]:
    frame = pd.DataFrame([values], columns=bundle["features"])
    model = bundle["model"]
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])
    return {"prediction": prediction, "growth_probability": probability}
