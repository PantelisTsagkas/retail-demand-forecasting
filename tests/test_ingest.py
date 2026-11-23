from pathlib import Path

import duckdb
import pytest

from retail_demand.ingest import ingest


def test_ingest_loads_every_table_with_row_counts(raw_dir: Path, tmp_path: Path) -> None:
    counts = ingest(raw_dir, tmp_path / "test.duckdb")

    assert counts == {
        "sales_raw": 7,
        "items": 3,
        "stores": 2,
        "transactions": 1,
        "oil": 2,
        "holidays_events": 3,
    }


def test_ingest_applies_explicit_types(raw_dir: Path, tmp_path: Path) -> None:
    db_path = tmp_path / "test.duckdb"
    ingest(raw_dir, db_path)

    with duckdb.connect(db_path) as con:
        types = dict(
            con.execute("SELECT column_name, column_type FROM (DESCRIBE sales_raw)").fetchall()
        )
        promo_nulls = con.execute(
            "SELECT count(*) FROM sales_raw WHERE onpromotion IS NULL"
        ).fetchone()
        missing_oil = con.execute("SELECT count(*) FROM oil WHERE dcoilwtico IS NULL").fetchone()

    assert types["date"] == "DATE"
    assert types["unit_sales"] == "DOUBLE"
    assert types["onpromotion"] == "BOOLEAN"
    assert promo_nulls == (3,)
    assert missing_oil == (1,)


def test_ingest_is_rerunnable(raw_dir: Path, tmp_path: Path) -> None:
    db_path = tmp_path / "test.duckdb"
    ingest(raw_dir, db_path)

    counts = ingest(raw_dir, db_path)

    assert counts["sales_raw"] == 7


def test_ingest_names_every_missing_file(raw_dir: Path, tmp_path: Path) -> None:
    (raw_dir / "train.csv").unlink()
    (raw_dir / "oil.csv").unlink()

    with pytest.raises(FileNotFoundError, match="train.csv, oil.csv"):
        ingest(raw_dir, tmp_path / "test.duckdb")
