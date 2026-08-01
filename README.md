# County Population ETL and MLOps Pipeline

An end-to-end data engineering and MLOps project that extracts county-level population and demographic data from the U.S. Census ACS 5-year API. It then joins a CSV reference source and a SQLAlchemy-managed SQLite reference table, builds time-aware features, and trains and evaluates a scikit-learn classifier, after which I track experiments with MLflow, serve predictions with FastAPI, and orchestrates the workflow with Apache Airflow. Finally, the deployed API is monitored with Prometheus and Grafana.

## Project objective

The derived model predicts if a U.S. county's population is expected to increase the following year. Each county-year record contains current and lagged population alongside demographic, socioeconomic, housing, and regional features. The pipeline uses a train-test split so that later observations are evaluated as future data rather than being randomly mixed into training.

## Architecture

```text
Census population API ─┐
Census demographics API ├─> Extract ─> Raw JSON/Parquet
State-region CSV ────────┤
SQLite reference table ──┘
                              │
                              v
                    pandas + Dask transforms
                    cleaning, validation, joins,
                    lags, rates, and target creation
                              │
                              v
                    Processed Parquet + SQLite
                              │
                              v
                    scikit-learn model training
                    chronological evaluation
                              │
                   ┌──────────┴──────────┐
                   v                     v
             MLflow tracking       joblib model artifact
                                         │
                                         v
                                  FastAPI /predict
                                         │
                                         v
                              Prometheus /metrics
                                         │
                                         v
                                      Grafana

Airflow orchestrates the end-to-end pipeline. GitHub Actions runs automated tests and validates the Docker build on each push or pull request.
```

## Technology stack

- Python 3.11
- pandas and Dask
- requests
- SQLAlchemy and SQLite
- scikit-learn
- MLflow
- FastAPI and Uvicorn
- Apache Airflow 2.10.5 in Docker
- Prometheus and Grafana
- Docker and Docker Compose
- pytest
- GitHub Actions

## Repository structure

```text
county-population-mlops/
├── api/                         # FastAPI application and schemas
├── data/
│   └── reference/               # Version-controlled reference CSV
├── models/                      # Metrics JSON; generated model is ignored
├── monitoring/
│   └── prometheus.yml           # Prometheus scrape configuration
├── orchestration/
│   └── dags/                    # Airflow DAG
├── reports/                     # Report and supporting materials
├── scripts/                     # Database initialization and pipeline runner
├── src/
│   ├── extract/                 # API, CSV, and SQL extraction
│   ├── transform/               # Cleaning, validation, and feature engineering
│   ├── load/                    # SQLite loading
│   ├── models/                  # Training, evaluation, and prediction
│   └── monitoring/              # Drift and application metrics
├── tests/                       # Unit and API tests
├── .github/workflows/ci.yml     # GitHub Actions workflow
├── Dockerfile                   # FastAPI image
├── Dockerfile.airflow           # Airflow image
├── docker-compose.yml           # API, Airflow, Prometheus, and Grafana
├── requirements.txt             # Local/API dependencies
└── requirements-airflow.txt     # Airflow-compatible dependencies
```

## Prerequisites

- Python 3.11
- Git
- Docker Desktop with the WSL 2 backend on Windows
- Internet access for Census API calls
- A Census API key

## Local setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Create the environment file

```powershell
Copy-Item .env.example .env
```

Update `.env` with values similar to:

```env
CENSUS_API_KEY=your_key_here
DATABASE_URL=sqlite:///data/database/population.db
MLFLOW_TRACKING_URI=sqlite:///data/database/mlflow.db
MODEL_PATH=models/county_population_model.joblib
START_YEAR=2020
END_YEAR=2023
USE_DASK=true
```

The `.env` file is ignored by Git. Do not commit API keys or passwords.

### 3. Initialize the reference database

```powershell
python scripts/initialize_database.py
```

### 4. Run the complete pipeline

```powershell
python scripts/run_pipeline.py
```

The script extracts data, creates features, loads SQLite tables, trains the model, logs the run to MLflow, and saves the model artifact and metrics.

## Automated tests

```powershell
python -m pytest -v
```

The current suite contains seven tests covering extraction, transformation, validation, model behavior, and API endpoints.

## Model results

The baseline logistic regression model produced the following held-out test results:

| Metric | Value |
|---|---:|
| Accuracy | 0.644 |
| Precision | 0.633 |
| Recall | 0.745 |
| F1 score | 0.684 |
| ROC AUC | 0.712 |

The test year was 2022, and the training data ended in 2021.

## MLflow experiment tracking

The local MLflow backend uses SQLite:

```powershell
mlflow ui --backend-store-uri sqlite:///data/database/mlflow.db --port 5000
```

Open `http://127.0.0.1:5000`, select `county-population-growth`, and open the newest finished run. The run includes model parameters, evaluation metrics, and a logged model artifact.

The Airflow container uses a separate compatible tracking database:

```text
data/database/mlflow_airflow.db
```

This prevents schema conflicts between MLflow versions in the Windows environment and the Airflow image.

## Run the FastAPI service locally

```powershell
uvicorn api.main:app
```

Open `http://127.0.0.1:8000/docs`.

Available endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `GET /metrics`

Example request:

```json
{
  "population": 125000,
  "population_lag_1": 123500,
  "growth_rate_lag_1": 0.0121,
  "median_household_income": 72000,
  "median_age": 39.4,
  "poverty_rate": 0.11,
  "unemployment_rate": 0.045,
  "housing_vacancy_rate": 0.08,
  "region": "South"
}
```

Example response:

```json
{
  "prediction": 1,
  "growth_probability": 0.7163,
  "model_type": "logistic_regression"
}
```

## Docker deployment

Build and run the API image:

```powershell
docker build -t county-population-api .
docker run --rm -p 8000:8000 county-population-api
```

Or start the API and monitoring services with Docker Compose:

```powershell
docker compose up -d api prometheus grafana
```

Service URLs:

| Service | URL |
|---|---|
| FastAPI | `http://127.0.0.1:8000/docs` |
| Prometheus | `http://127.0.0.1:9090` |
| Grafana | `http://127.0.0.1:3000` |

## Airflow orchestration

Airflow runs in Docker because native Apache Airflow installation is not supported on Windows.

Build the image:

```powershell
docker compose build airflow
```

Start Airflow:

```powershell
docker compose up airflow
```

Open `http://127.0.0.1:8080`, log in with the standalone credentials, unpause `county_population_etl_mlops`, and trigger the DAG. The DAG runs `scripts/run_pipeline.py` inside the mounted project directory.

To display the generated standalone password:

```powershell
docker compose exec airflow cat /opt/airflow/standalone_admin_password.txt
```

## Monitoring

Prometheus scrapes the FastAPI `/metrics` endpoint every five seconds using `monitoring/prometheus.yml`.

Verify the target at:

```text
http://127.0.0.1:9090/targets
```

The `county_population_api` target should be `UP`.

In Grafana, add Prometheus with this internal Docker URL:

```text
http://prometheus:9090
```

Recommended dashboard queries:

```promql
model_predictions_total
```

```promql
model_prediction_errors_total or vector(0)
```

## CI/CD

The GitHub Actions workflow runs on pushes and pull requests to `main`. It installs dependencies, runs the test suite, checks Python compilation, and builds the Docker image. A green workflow run confirms that the repository can be tested and packaged in a clean hosted environment.

## Main generated outputs

- `data/raw/population/*.json` and `*.parquet`
- `data/raw/demographics/*.json` and `*.parquet`
- `data/processed/county_features.parquet`
- `data/database/population.db`
- `data/database/mlflow.db`
- `data/database/mlflow_airflow.db`
- `models/county_population_model.joblib`
- `models/county_population_model.metrics.json`

Generated databases, raw files, processed files, model binaries, credentials, and local environment files are excluded from Git.

## Known limitations

- ACS values are survey estimates and should not be described as exact counts.
- The model is a baseline classifier and is not intended for policy or operational decision-making without further validation.
- Some lagged features may be unavailable when the selected year range is too short. The warning involving `growth_rate_lag_1` should be addressed by increasing the historical window or removing the feature.
- SQLite and Airflow's SequentialExecutor are appropriate for development and coursework, not production deployment.
- The Grafana dashboard is configured manually unless dashboard provisioning is added.

## Reproduce the complete demonstration

```powershell
python -m pytest -v
python scripts/initialize_database.py
python scripts/run_pipeline.py
mlflow ui --backend-store-uri sqlite:///data/database/mlflow.db --port 5000
uvicorn api.main:app

docker compose up -d api prometheus grafana
docker compose up airflow
```

## Repository

GitHub: `https://github.com/jfisher43/dds8530-wk5`
