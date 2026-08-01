# County Population ETL and MLOps Pipeline

An end-to-end classroom project that extracts county-level population and demographic data from the U.S. Census ACS 5-year API, joins a CSV reference source and a SQLAlchemy-managed SQLite reference table, builds time-aware features, trains a scikit-learn classifier, tracks the run with MLflow, and serves predictions through FastAPI with Prometheus metrics.

## Architecture

Census population API + Census demographics API + state-region CSV + SQLite reference table → pandas cleaning → county-year feature engineering → Parquet/CSV/SQLite → chronological model training → MLflow + joblib model → FastAPI `/predict` → Prometheus `/metrics`.

## Requirements

- Python 3.11 recommended
- Internet connection for Census extraction
- A Census API key is optional for moderate use

## Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env  # Windows
# cp .env.example .env  # macOS/Linux
```

## Run

```bash
python scripts/initialize_database.py
python scripts/run_pipeline.py
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` and use the interactive `/predict` endpoint.

## Tests

```bash
pytest -q
```

## MLflow

The default tracking URI is the local `mlruns` directory. View it with:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Then open `http://127.0.0.1:5000`.

## Airflow

Airflow is intentionally not in the base requirements because its constraints depend on the operating system, Python version, and Airflow release. Install it separately using the official Airflow constraints for your chosen version, set `AIRFLOW_HOME`, copy or point Airflow to `airflow/dags`, and trigger `county_population_etl_mlops`.

## Main outputs

- `data/raw/population/*.json` and `*.parquet`
- `data/raw/demographics/*.json` and `*.parquet`
- `data/processed/county_features.parquet`
- `data/database/population.db`
- `models/county_population_model.joblib`
- `models/county_population_model.metrics.json`
- `mlruns/`

## API request example

```json
{
  "population": 100000,
  "population_lag_1": 99000,
  "growth_rate_lag_1": 0.01,
  "median_household_income": 65000,
  "median_age": 39.2,
  "poverty_rate": 0.12,
  "unemployment_rate": 0.045,
  "housing_vacancy_rate": 0.08,
  "region": "South"
}
```

## Notes

The chronological split uses the latest target-bearing year as the test set. The target for each county-year is whether population increases in the following year. This avoids using future records during training. ACS estimates are survey estimates and should not be described as exact population counts.
