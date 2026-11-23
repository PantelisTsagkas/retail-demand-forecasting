"""Load the modelling tables into a dense series x day panel."""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SalesPanel:
    """Daily sales for every store x family series, plus per-day calendar flags.

    Attributes:
        values: Units sold, shape (n_series, n_days). NaN before the store opened.
        dates: The calendar day for each column.
        series: One row per series: store_nbr, family, perishable, store_type, cluster.
        national_holiday: Bool flags, same shape as values.
        local_holiday: Bool flags, same shape as values.
        event: Bool flags, same shape as values.
    """

    values: np.ndarray
    dates: pd.DatetimeIndex
    series: pd.DataFrame
    national_holiday: np.ndarray
    local_holiday: np.ndarray
    event: np.ndarray


def _pivot(
    df: pd.DataFrame, column: str, index: pd.MultiIndex, dates: pd.DatetimeIndex
) -> np.ndarray:
    wide = df.pivot(index=["store_nbr", "family"], columns="date", values=column)
    return wide.reindex(index=index, columns=dates).to_numpy(dtype=float)


def load_panel(db_path: Path) -> SalesPanel:
    """Read family_daily, family_info, stores and store_calendar into a SalesPanel.

    Args:
        db_path: The DuckDB file built by `uv run ingest`.

    Returns:
        The panel, with series sorted by store then family.
    """
    with duckdb.connect(db_path, read_only=True) as con:
        daily = con.execute(
            """
            SELECT d.date, d.store_nbr, d.family, d.unit_sales,
                   c.is_national_holiday::INT AS national_holiday,
                   c.is_local_holiday::INT AS local_holiday,
                   c.is_event::INT AS event
            FROM family_daily AS d
            JOIN store_calendar AS c USING (date, store_nbr)
            """
        ).df()
        series = con.execute(
            """
            SELECT s.store_nbr, f.family, f.perishable, st.type AS store_type, st.cluster
            FROM (SELECT DISTINCT store_nbr, family FROM family_daily) AS s
            JOIN family_info AS f USING (family)
            JOIN stores AS st USING (store_nbr)
            ORDER BY s.store_nbr, f.family
            """
        ).df()

    daily["date"] = pd.to_datetime(daily["date"])
    dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    index = pd.MultiIndex.from_frame(series[["store_nbr", "family"]])
    flags = {
        name: np.nan_to_num(_pivot(daily, name, index, dates)).astype(bool)
        for name in ("national_holiday", "local_holiday", "event")
    }
    return SalesPanel(
        values=_pivot(daily, "unit_sales", index, dates),
        dates=dates,
        series=series.reset_index(drop=True),
        national_holiday=flags["national_holiday"],
        local_holiday=flags["local_holiday"],
        event=flags["event"],
    )
