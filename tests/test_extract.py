from pathlib import Path
from unittest.mock import Mock, patch

from src.extract.county_csv import create_state_reference_csv, extract_county_csv
from src.extract.population_api import fetch_population_year


def test_population_api_payload_converts_to_dataframe():
    response = Mock()
    response.json.return_value = [["NAME", "B01003_001E", "state", "county"], ["Example County", "1000", "01", "001"]]
    response.raise_for_status.return_value = None
    with patch("src.extract.population_api._session") as session_factory:
        session_factory.return_value.get.return_value = response
        frame = fetch_population_year(2023)
    assert frame.loc[0, "population"] == "1000"
    assert frame.loc[0, "year"] == 2023


def test_reference_csv_round_trip(tmp_path: Path):
    path = create_state_reference_csv(tmp_path / "state_regions.csv")
    frame = extract_county_csv(path)
    assert {"state_fips", "state_name", "region"}.issubset(frame.columns)
    assert frame["state_fips"].str.len().eq(2).all()
