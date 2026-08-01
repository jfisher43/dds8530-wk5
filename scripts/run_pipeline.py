"""Run the complete local ETL and model-training pipeline."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import logging
import pandas as pd

from src.config import settings
from src.extract.county_csv import create_state_reference_csv, extract_county_csv
from src.extract.demographics_api import extract_demographics
from src.extract.population_api import extract_population
from src.extract.reference_database import extract_reference_table, initialize_reference_table
from src.load.load_sqlite import load_dataframe
from src.models.train import train_model
from src.transform.build_features import build_features
from src.transform.clean_demographics import clean_demographics
from src.transform.clean_population import clean_population
from src.transform.validate_data import validate_features, validate_population


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def run(model_type: str = "logistic_regression") -> dict:
    settings.ensure_directories()
    years = range(settings.start_year, settings.end_year + 1)

    reference_csv = create_state_reference_csv(settings.reference_dir / "state_regions.csv")
    csv_reference = extract_county_csv(reference_csv)
    initialize_reference_table(csv_reference, settings.database_url)
    db_reference = extract_reference_table(settings.database_url)

    population_raw = extract_population(years, settings.raw_population_dir, settings.census_api_key)
    demographics_raw = extract_demographics(years, settings.raw_demographics_dir, settings.census_api_key)

    population = clean_population(population_raw)
    demographics = clean_demographics(demographics_raw)
    validate_population(population)
    features = build_features(population, demographics, db_reference)
    validate_features(features)

    output_path = settings.processed_dir / "county_features.parquet"
    features.to_parquet(output_path, index=False)
    features.to_csv(settings.processed_dir / "county_features.csv", index=False)
    load_dataframe(features, settings.database_url, "county_features")

    training_frame = features.dropna(subset=["target_growth_next_year"])
    bundle = train_model(training_frame, settings.model_path, model_type=model_type, tracking_uri=settings.mlflow_tracking_uri)
    print(f"Processed {len(features):,} county-year rows")
    print(f"Saved features to {output_path}")
    print(f"Saved model to {settings.model_path}")
    print("Test metrics:", bundle["metrics"])
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["logistic_regression", "random_forest"], default="logistic_regression")
    args = parser.parse_args()
    configure_logging()
    run(model_type=args.model_type)


if __name__ == "__main__":
    main()
