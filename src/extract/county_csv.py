"""CSV extraction utilities for county/state reference data."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

STATE_REGIONS = {
    "01": ("Alabama", "South"), "02": ("Alaska", "West"), "04": ("Arizona", "West"),
    "05": ("Arkansas", "South"), "06": ("California", "West"), "08": ("Colorado", "West"),
    "09": ("Connecticut", "Northeast"), "10": ("Delaware", "South"), "11": ("District of Columbia", "South"),
    "12": ("Florida", "South"), "13": ("Georgia", "South"), "15": ("Hawaii", "West"),
    "16": ("Idaho", "West"), "17": ("Illinois", "Midwest"), "18": ("Indiana", "Midwest"),
    "19": ("Iowa", "Midwest"), "20": ("Kansas", "Midwest"), "21": ("Kentucky", "South"),
    "22": ("Louisiana", "South"), "23": ("Maine", "Northeast"), "24": ("Maryland", "South"),
    "25": ("Massachusetts", "Northeast"), "26": ("Michigan", "Midwest"), "27": ("Minnesota", "Midwest"),
    "28": ("Mississippi", "South"), "29": ("Missouri", "Midwest"), "30": ("Montana", "West"),
    "31": ("Nebraska", "Midwest"), "32": ("Nevada", "West"), "33": ("New Hampshire", "Northeast"),
    "34": ("New Jersey", "Northeast"), "35": ("New Mexico", "West"), "36": ("New York", "Northeast"),
    "37": ("North Carolina", "South"), "38": ("North Dakota", "Midwest"), "39": ("Ohio", "Midwest"),
    "40": ("Oklahoma", "South"), "41": ("Oregon", "West"), "42": ("Pennsylvania", "Northeast"),
    "44": ("Rhode Island", "Northeast"), "45": ("South Carolina", "South"), "46": ("South Dakota", "Midwest"),
    "47": ("Tennessee", "South"), "48": ("Texas", "South"), "49": ("Utah", "West"),
    "50": ("Vermont", "Northeast"), "51": ("Virginia", "South"), "53": ("Washington", "West"),
    "54": ("West Virginia", "South"), "55": ("Wisconsin", "Midwest"), "56": ("Wyoming", "West"),
    "72": ("Puerto Rico", "Other"),
}


def create_state_reference_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"state_fips": fips, "state_name": values[0], "region": values[1]} for fips, values in STATE_REGIONS.items()]
    ).to_csv(path, index=False)
    return path


def extract_county_csv(path: Path, chunksize: int = 10_000) -> pd.DataFrame:
    if not path.exists():
        create_state_reference_csv(path)
    chunks = [chunk for chunk in pd.read_csv(path, dtype={"state_fips": "string"}, chunksize=chunksize)]
    frame = pd.concat(chunks, ignore_index=True)
    frame.columns = frame.columns.str.strip().str.lower().str.replace(" ", "_")
    frame["state_fips"] = frame["state_fips"].astype("string").str.zfill(2)
    return frame
