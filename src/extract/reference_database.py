"""Read and write the reference table through SQLAlchemy."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_engine(database_url: str) -> Engine:
    return create_engine(database_url, future=True)


def initialize_reference_table(reference: pd.DataFrame, database_url: str) -> None:
    reference.to_sql("state_reference", get_engine(database_url), if_exists="replace", index=False)


def extract_reference_table(database_url: str) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM state_reference", get_engine(database_url))
