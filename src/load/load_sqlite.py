"""Load transformed frames into SQLite with SQLAlchemy."""
from __future__ import annotations
import pandas as pd
from sqlalchemy import create_engine


def load_dataframe(df: pd.DataFrame, database_url: str, table_name: str, if_exists: str = "replace") -> int:
    engine = create_engine(database_url, future=True)
    df.to_sql(table_name, engine, if_exists=if_exists, index=False, chunksize=1_000, method="multi")
    return len(df)
