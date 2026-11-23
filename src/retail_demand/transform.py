"""SQL transforms from raw item-level rows to the store x family modelling tables."""

import logging

import duckdb

logger = logging.getLogger(__name__)

# The raw file only holds rows where something sold, so zero-sales days are missing.
# We rebuild the full grid: every day from each store's first sale, for every family.
# Returns (negative unit_sales) are not demand, so they are excluded.
FAMILY_DAILY_SQL = """
CREATE OR REPLACE TABLE family_daily AS
WITH sold AS (
    SELECT s.date, s.store_nbr, i.family, sum(s.unit_sales) AS unit_sales
    FROM sales_raw AS s
    JOIN items AS i USING (item_nbr)
    WHERE s.unit_sales > 0
    GROUP BY ALL
),
calendar AS (
    SELECT unnest(generate_series(min(date), max(date), INTERVAL 1 DAY))::DATE AS date
    FROM sales_raw
),
store_open AS (
    SELECT store_nbr, min(date) AS first_date FROM sales_raw GROUP BY store_nbr
),
families AS (
    SELECT DISTINCT family FROM items
),
grid AS (
    SELECT c.date, so.store_nbr, f.family
    FROM calendar AS c
    JOIN store_open AS so ON c.date >= so.first_date
    CROSS JOIN families AS f
)
SELECT g.date, g.store_nbr, g.family, coalesce(sold.unit_sales, 0.0) AS unit_sales
FROM grid AS g
LEFT JOIN sold USING (date, store_nbr, family)
ORDER BY g.store_nbr, g.family, g.date
"""

# A family counts as perishable when at least half of its items are.
FAMILY_INFO_SQL = """
CREATE OR REPLACE TABLE family_info AS
SELECT family, count(*) AS n_items, avg(perishable) >= 0.5 AS perishable
FROM items
GROUP BY family
"""

# Holiday flags per store and day. Transferred holidays are moved to another date,
# which appears as its own 'Transfer' row, so the original date is a normal day.
# 'Work Day' rows are make-up working days, so they are not holidays.
STORE_CALENDAR_SQL = """
CREATE OR REPLACE TABLE store_calendar AS
WITH days AS (
    SELECT DISTINCT date, store_nbr FROM family_daily
),
h AS (
    SELECT * FROM holidays_events
    WHERE NOT transferred AND type IN ('Holiday', 'Transfer', 'Additional', 'Bridge', 'Event')
)
SELECT
    d.date,
    d.store_nbr,
    bool_or(h.locale = 'National' AND h.type <> 'Event') IS TRUE AS is_national_holiday,
    bool_or(
        (h.locale = 'Regional' AND h.locale_name = st.state)
        OR (h.locale = 'Local' AND h.locale_name = st.city)
    ) IS TRUE AS is_local_holiday,
    bool_or(h.type = 'Event') IS TRUE AS is_event
FROM days AS d
JOIN stores AS st USING (store_nbr)
LEFT JOIN h ON h.date = d.date
GROUP BY d.date, d.store_nbr
"""


def build_modelling_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create family_daily, family_info and store_calendar from the raw tables.

    Args:
        con: Open DuckDB connection with the raw tables loaded.
    """
    con.execute(FAMILY_DAILY_SQL)
    con.execute(FAMILY_INFO_SQL)
    con.execute(STORE_CALENDAR_SQL)
    row = con.execute(
        "SELECT count(*), count(DISTINCT (store_nbr, family)) FROM family_daily"
    ).fetchone()
    if row is None:
        raise RuntimeError("family_daily is empty")
    logger.info("Built family_daily: %d rows, %d series", row[0], row[1])
