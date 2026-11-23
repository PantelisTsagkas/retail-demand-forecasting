"""Load the raw Kaggle CSVs into DuckDB, check them, and build the modelling tables.

Run with: uv run ingest --raw-dir data/raw --db data/favorita.duckdb
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb

from retail_demand.checks import run_checks
from retail_demand.transform import build_modelling_tables

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawTable:
    """One raw CSV and the DuckDB table it lands in."""

    table: str
    filename: str
    columns: dict[str, str]


# Explicit schemas: a typo or format change in the source fails loudly here
# instead of silently becoming a VARCHAR column three steps later.
RAW_TABLES: tuple[RawTable, ...] = (
    RawTable(
        table="sales_raw",
        filename="train.csv",
        columns={
            "id": "BIGINT",
            "date": "DATE",
            "store_nbr": "INTEGER",
            "item_nbr": "INTEGER",
            "unit_sales": "DOUBLE",
            "onpromotion": "BOOLEAN",
        },
    ),
    RawTable(
        table="items",
        filename="items.csv",
        columns={
            "item_nbr": "INTEGER",
            "family": "VARCHAR",
            "class": "INTEGER",
            "perishable": "INTEGER",
        },
    ),
    RawTable(
        table="stores",
        filename="stores.csv",
        columns={
            "store_nbr": "INTEGER",
            "city": "VARCHAR",
            "state": "VARCHAR",
            "type": "VARCHAR",
            "cluster": "INTEGER",
        },
    ),
    RawTable(
        table="transactions",
        filename="transactions.csv",
        columns={"date": "DATE", "store_nbr": "INTEGER", "transactions": "INTEGER"},
    ),
    RawTable(
        table="oil",
        filename="oil.csv",
        columns={"date": "DATE", "dcoilwtico": "DOUBLE"},
    ),
    RawTable(
        table="holidays_events",
        filename="holidays_events.csv",
        columns={
            "date": "DATE",
            "type": "VARCHAR",
            "locale": "VARCHAR",
            "locale_name": "VARCHAR",
            "description": "VARCHAR",
            "transferred": "BOOLEAN",
        },
    ),
)


def check_raw_files(raw_dir: Path, tables: tuple[RawTable, ...]) -> None:
    """Fail early with one clear message if any expected CSV is missing.

    Args:
        raw_dir: Directory holding the extracted Kaggle files.
        tables: The tables whose source files must exist.

    Raises:
        FileNotFoundError: If one or more source files are missing.
    """
    missing: list[str] = []
    for raw in tables:
        if not (raw_dir / raw.filename).is_file():
            missing.append(raw.filename)
    if missing:
        raise FileNotFoundError(
            f"Missing in {raw_dir}: {', '.join(missing)}. "
            "Download the Kaggle 'favorita-grocery-sales-forecasting' data first (see README)."
        )


def load_table(con: duckdb.DuckDBPyConnection, raw_dir: Path, raw: RawTable) -> int:
    """Replace one DuckDB table from its CSV and return the row count.

    Args:
        con: Open DuckDB connection.
        raw_dir: Directory holding the CSV.
        raw: The table definition to load.

    Returns:
        Number of rows loaded.
    """
    csv_path = (raw_dir / raw.filename).as_posix()
    con.execute(
        f"CREATE OR REPLACE TABLE {raw.table} AS "
        "SELECT * FROM read_csv(?, header = true, columns = ?)",
        [csv_path, raw.columns],
    )
    row = con.execute(f"SELECT count(*) FROM {raw.table}").fetchone()
    if row is None:
        raise RuntimeError(f"Could not count rows in {raw.table}")
    return int(row[0])


def ingest(
    raw_dir: Path, db_path: Path, tables: tuple[RawTable, ...] = RAW_TABLES
) -> dict[str, int]:
    """Load every raw CSV into DuckDB.

    Args:
        raw_dir: Directory holding the extracted Kaggle files.
        db_path: DuckDB file to create or update.
        tables: Table definitions to load. Defaults to the full Favorita set.

    Returns:
        Row count per table name.
    """
    check_raw_files(raw_dir, tables)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with duckdb.connect(db_path) as con:
        for raw in tables:
            counts[raw.table] = load_table(con, raw_dir, raw)
            logger.info("Loaded %s: %d rows", raw.table, counts[raw.table])
    return counts


def main() -> None:
    """Command-line entry point: load, check, then build modelling tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--db", type=Path, default=Path("data/favorita.duckdb"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingest(args.raw_dir, args.db)
    with duckdb.connect(args.db) as con:
        results = run_checks(con)
        failed = [r.name for r in results if not r.passed]
        if failed:
            raise SystemExit(f"{len(failed)} data quality check(s) failed: {'; '.join(failed)}")
        build_modelling_tables(con)


if __name__ == "__main__":
    main()
