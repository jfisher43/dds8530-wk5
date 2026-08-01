"""Airflow DAG that invokes the tested project pipeline."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import sys

try:
    from airflow.decorators import dag, task
except ImportError:  # Keeps normal pytest/import checks usable without Airflow installed.
    dag = task = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if dag is not None:
    @dag(
        dag_id="county_population_etl_mlops",
        schedule="@monthly",
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["etl", "mlops", "census"],
    )
    def population_pipeline():
        @task
        def run_end_to_end() -> str:
            completed = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "run_pipeline.py")],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout[-4_000:]

        run_end_to_end()

    population_pipeline()
