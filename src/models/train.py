"""Train, evaluate, persist, and optionally track a county-growth classifier."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
try:
    import mlflow
    import mlflow.sklearn
except ImportError:  # Allows unit tests without the optional tracking package.
    mlflow = None
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)
NUMERIC_FEATURES = [
    "population", "population_lag_1", "growth_rate_lag_1", "median_household_income",
    "median_age", "poverty_rate", "unemployment_rate", "housing_vacancy_rate",
]
CATEGORICAL_FEATURES = ["region"]
TARGET = "target_growth_next_year"


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = df.dropna(subset=[TARGET]).copy()
    years = sorted(int(y) for y in usable["year"].unique())
    if len(years) < 2:
        raise ValueError("At least two target-bearing years are required for chronological evaluation")
    test_year = years[-1]
    return usable[usable["year"] < test_year], usable[usable["year"] == test_year]


def build_model(model_type: str = "logistic_regression", random_state: int = 42) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    if model_type == "random_forest":
        classifier = RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced", random_state=random_state, n_jobs=-1)
    elif model_type == "logistic_regression":
        classifier = LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=random_state)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate_predictions(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if y.nunique() > 1 else float("nan"),
    }


def train_model(
    features: pd.DataFrame,
    model_path: Path,
    model_type: str = "logistic_regression",
    tracking_uri: str | None = None,
    experiment_name: str = "county-population-growth",
) -> dict[str, Any]:
    train_df, test_df = chronological_split(features)
    X_train, y_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_df[TARGET].astype(int)
    X_test, y_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test_df[TARGET].astype(int)
    model = build_model(model_type=model_type)

    if tracking_uri and mlflow is None:
        raise ImportError("MLflow is not installed. Run pip install -r requirements.txt")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    context = mlflow.start_run() if tracking_uri else _NullRun()
    with context:
        model.fit(X_train, y_train)
        metrics = evaluate_predictions(model, X_test, y_test)
        if tracking_uri:
            mlflow.log_params({"model_type": model_type, "train_end_year": int(train_df["year"].max()), "test_year": int(test_df["year"].iloc[0])})
            mlflow.log_metrics({k: v for k, v in metrics.items() if pd.notna(v)})
            mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=X_train.head(3),
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "target": TARGET,
        "model_type": model_type,
        "metrics": metrics,
        "test_year": int(test_df["year"].iloc[0]),
    }
    joblib.dump(bundle, model_path)
    metrics_path = model_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Saved model to %s", model_path)
    return bundle


class _NullRun:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
