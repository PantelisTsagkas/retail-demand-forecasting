from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from retail_demand.checks import run_checks
from retail_demand.ingest import ingest


@pytest.fixture
def con(raw_dir: Path, tmp_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    db_path = tmp_path / "test.duckdb"
    ingest(raw_dir, db_path)
    with duckdb.connect(db_path) as connection:
        yield connection


def failing(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    return {r.name: r.failing_rows for r in run_checks(con) if not r.passed}


def test_clean_data_passes_every_check(con: duckdb.DuckDBPyConnection) -> None:
    assert failing(con) == {}


@pytest.mark.parametrize(
    ("sql", "expected_check"),
    [
        (
            "INSERT INTO sales_raw VALUES (9, '2013-01-01', 1, 100, 1.0, NULL)",
            "sales_raw: one row per date, store and item",
        ),
        (
            "INSERT INTO sales_raw VALUES (9, NULL, 1, 100, 1.0, NULL)",
            "sales_raw: no null keys",
        ),
        (
            "INSERT INTO sales_raw VALUES (9, '2013-01-05', 99, 100, 1.0, NULL)",
            "sales_raw: every store exists in stores",
        ),
        (
            "INSERT INTO sales_raw VALUES (9, '2013-01-05', 1, 999, 1.0, NULL)",
            "sales_raw: every item exists in items",
        ),
        (
            "INSERT INTO items VALUES (300, 'DAIRY', 3, 2)",
            "items: perishable flag is 0 or 1",
        ),
    ],
)
def test_each_check_catches_its_problem(
    con: duckdb.DuckDBPyConnection, sql: str, expected_check: str
) -> None:
    con.execute(sql)

    assert failing(con) == {expected_check: 1}
