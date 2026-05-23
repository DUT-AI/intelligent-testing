from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "XES3G5M"
    / "XES3G5M"
    / "kc_level"
)

CSV_FILES = [
    ("train", DATA_DIR / "final_train.csv"),
    ("valid", DATA_DIR / "final_val.csv"),
    ("test", DATA_DIR / "final_test.csv"),
]

CHUNK_SIZE = 500


sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.infrastructure.database.connection import SessionLocal, engine  # noqa: E402
from app.infrastructure.database.models import Base, KcLevelRecord  # noqa: E402


def clean_value(value):
    if value is None:
        return None
    if pd.isna(value):
        return None
    if value == "":
        return None
    return value


def build_record(row: dict, fallback_dataset_type: str) -> dict:
    record = {
        "fold": int(row["fold"]),
        "uid": int(row["uid"]),
        "questions": str(row["questions"]),
        "concepts": str(row["concepts"]),
        "responses": str(row["responses"]),
        "timestamps": str(row["timestamps"]),
        "selectmasks": str(row["selectmasks"]),
        "is_repeat": str(row["is_repeat"]),
        "dataset_type": str(clean_value(row.get("dataset_type")) or fallback_dataset_type),
    }

    for optional_field in ("qidxs", "rest", "orirow"):
        if optional_field in row:
            record[optional_field] = clean_value(row.get(optional_field))

    return record


def load_csv_file(session, csv_path: Path, fallback_dataset_type: str) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    loaded_rows = 0
    reader = pd.read_csv(
        csv_path,
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
    )

    for chunk in reader:
        records = [build_record(row, fallback_dataset_type) for row in chunk.to_dict(orient="records")]
        session.bulk_insert_mappings(KcLevelRecord, records)
        session.commit()
        loaded_rows += len(records)

    return loaded_rows


def main() -> None:
    print(f"Using database: {engine.url}")

    Base.metadata.create_all(bind=engine)

    session = SessionLocal()
    try:
        session.execute(text("TRUNCATE TABLE kc_level_records RESTART IDENTITY"))
        session.commit()

        total_rows = 0
        for dataset_type, csv_path in CSV_FILES:
            print(f"Loading {csv_path.name}...")
            loaded_rows = load_csv_file(session, csv_path, dataset_type)
            total_rows += loaded_rows
            print(f"Loaded {loaded_rows} rows from {csv_path.name}")

        print(f"Done. Total rows inserted: {total_rows}")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()