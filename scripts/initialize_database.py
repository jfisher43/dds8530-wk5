from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.extract.county_csv import create_state_reference_csv, extract_county_csv
from src.extract.reference_database import initialize_reference_table


def main() -> None:
    settings.ensure_directories()
    csv_path = create_state_reference_csv(settings.reference_dir / "state_regions.csv")
    reference = extract_county_csv(csv_path)
    initialize_reference_table(reference, settings.database_url)
    print(f"Initialized {len(reference)} state reference rows in {settings.database_path}")


if __name__ == "__main__":
    main()
