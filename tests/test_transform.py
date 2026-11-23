from collections.abc import Iterator
from datetime import date
from pathlib import Path

import duckdb
import pytest

from retail_demand.ingest import ingest
from retail_demand.transform import build_modelling_tables


@pytest.fixture
def con(raw_dir: Path, tmp_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    db_path = tmp_path / "test.duckdb"
    ingest(raw_dir, db_path)
    with duckdb.connect(db_path) as connection:
        build_modelling_tables(connection)
        yield connection


def sales(con: duckdb.DuckDBPyConnection, store: int, family: str) -> list[tuple[date, float]]:
    return con.execute(
        "SELECT date, unit_sales FROM family_daily "
        "WHERE store_nbr = ? AND family = ? ORDER BY date",
        [store, family],
    ).fetchall()


def test_family_daily_sums_items_and_fills_zero_days(con: duckdb.DuckDBPyConnection) -> None:
    assert sales(con, 1, "GROCERY I") == [
        (date(2013, 1, 1), 7.0),
        (date(2013, 1, 2), 4.0),
        (date(2013, 1, 3), 0.0),  # only a return that day: not demand
    ]
    assert sales(con, 1, "PRODUCE") == [
        (date(2013, 1, 1), 3.5),
        (date(2013, 1, 2), 0.0),  # missing in raw data, filled with zero
        (date(2013, 1, 3), 6.0),
    ]


def test_family_daily_starts_each_store_at_its_first_sale(con: duckdb.DuckDBPyConnection) -> None:
    first_day = con.execute("SELECT min(date) FROM family_daily WHERE store_nbr = 2").fetchone()

    assert first_day == (date(2013, 1, 2),)


def test_family_info_flags_perishable_families(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute("SELECT family, perishable FROM family_info ORDER BY family").fetchall()

    assert rows == [("GROCERY I", False), ("PRODUCE", True)]


def test_store_calendar_holiday_flags(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        "SELECT date, store_nbr, is_national_holiday, is_local_holiday "
        "FROM store_calendar ORDER BY date, store_nbr"
    ).fetchall()

    assert rows == [
        (date(2013, 1, 1), 1, True, False),
        (date(2013, 1, 2), 1, False, False),
        (date(2013, 1, 2), 2, False, True),  # Manta's local holiday, store 2 is in Manta
        (date(2013, 1, 3), 1, False, False),  # transferred away, so a normal day
        (date(2013, 1, 3), 2, False, False),
    ]
