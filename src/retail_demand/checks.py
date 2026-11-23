"""Data quality checks on the raw DuckDB tables.

Each check is a SQL query that returns the number of offending rows.
Zero means the check passes.
"""

import logging
from dataclasses import dataclass

import duckdb

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Check:
    """A named SQL check that counts bad rows."""

    name: str
    sql: str


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check."""

    name: str
    failing_rows: int

    @property
    def passed(self) -> bool:
        """True when no rows broke the rule."""
        return self.failing_rows == 0


# Negative unit_sales are product returns and are allowed in the raw data.
# The modelling step counts only positive sales as demand.
CHECKS: tuple[Check, ...] = (
    Check(
        name="sales_raw: one row per date, store and item",
        sql="""
            SELECT count(*) FROM (
                SELECT date, store_nbr, item_nbr FROM sales_raw
                GROUP BY ALL HAVING count(*) > 1
            )
        """,
    ),
    Check(
        name="sales_raw: no null keys",
        sql="""
            SELECT count(*) FROM sales_raw
            WHERE date IS NULL OR store_nbr IS NULL OR item_nbr IS NULL OR unit_sales IS NULL
        """,
    ),
    Check(
        name="sales_raw: every store exists in stores",
        sql="SELECT count(*) FROM sales_raw ANTI JOIN stores USING (store_nbr)",
    ),
    Check(
        name="sales_raw: every item exists in items",
        sql="SELECT count(*) FROM sales_raw ANTI JOIN items USING (item_nbr)",
    ),
    Check(
        name="items: perishable flag is 0 or 1",
        sql="SELECT count(*) FROM items WHERE perishable NOT IN (0, 1)",
    ),
)


def run_checks(
    con: duckdb.DuckDBPyConnection, checks: tuple[Check, ...] = CHECKS
) -> list[CheckResult]:
    """Run every check and log the outcome.

    Args:
        con: Open DuckDB connection with the tables loaded.
        checks: Checks to run. Defaults to the full set.

    Returns:
        One result per check, in order.
    """
    results: list[CheckResult] = []
    for check in checks:
        row = con.execute(check.sql).fetchone()
        if row is None:
            raise RuntimeError(f"Check returned no row: {check.name}")
        result = CheckResult(name=check.name, failing_rows=int(row[0]))
        status = "PASS" if result.passed else "FAIL"
        level = logging.INFO if result.passed else logging.WARNING
        logger.log(level, "%s %s (%d rows)", status, check.name, result.failing_rows)
        results.append(result)
    return results
