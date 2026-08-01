"""Central configuration for the county population MLOps project."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    census_api_key: str | None = os.getenv("CENSUS_API_KEY") or None
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///data/database/population.db")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    model_path: Path = _resolve_path(os.getenv("MODEL_PATH", "models/county_population_model.joblib"))
    start_year: int = int(os.getenv("START_YEAR", "2018"))
    end_year: int = int(os.getenv("END_YEAR", "2023"))
    use_dask: bool = os.getenv("USE_DASK", "true").lower() in {"1", "true", "yes"}

    @property
    def raw_population_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "raw" / "population"

    @property
    def raw_demographics_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "raw" / "demographics"

    @property
    def reference_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "reference"

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "processed"

    @property
    def database_path(self) -> Path:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise ValueError("This classroom implementation currently expects a SQLite DATABASE_URL")
        return _resolve_path(self.database_url.removeprefix(prefix))

    def ensure_directories(self) -> None:
        for path in (
            self.raw_population_dir,
            self.raw_demographics_dir,
            self.reference_dir,
            self.processed_dir,
            self.database_path.parent,
            self.model_path.parent,
            PROJECT_ROOT / "artifacts",
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
