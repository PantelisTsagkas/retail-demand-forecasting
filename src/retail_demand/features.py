"""Turn the panel into one training row per (series, forecast origin, horizon day).

A forecast is made at the end of an origin day `o` for days o+1 ... o+HORIZON.
Every feature is computed from columns <= o only. That rule is what prevents
leakage, and tests/test_features.py checks it directly.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retail_demand.panel import SalesPanel

HORIZON = 7
HISTORY = 56  # days of history a series needs at the origin to be included

CATEGORICAL = ("store_nbr", "family", "store_type", "cluster")
FEATURES = (
    "store_nbr",
    "family",
    "store_type",
    "cluster",
    "perishable",
    "horizon",
    "dayofweek",
    "dayofmonth",
    "is_payday",
    "national_holiday",
    "local_holiday",
    "event",
    "last_1",
    "mean_7",
    "mean_14",
    "mean_28",
    "mean_56",
    "std_28",
    "zero_share_28",
    "same_weekday_last_week",
    "same_weekday_mean_4",
)


@dataclass(frozen=True)
class Splits:
    """Forecast origin day indices for each time period. No target day is shared."""

    train: list[int]
    validation: list[int]
    test: list[int]


def make_splits(
    n_days: int,
    n_test: int = 8,
    n_validation: int = 8,
    n_train: int = 104,
    min_origin: int = HISTORY - 1,
) -> Splits:
    """Weekly forecast origins, ordered in time: train, then validation, then test.

    The last origin leaves exactly HORIZON days of actuals after it. Origins are
    7 days apart, so each period's targets end on the next period's first origin.

    Args:
        n_days: Number of days in the panel.
        n_test: Weekly origins in the final, untouched test period.
        n_validation: Weekly origins used for early stopping and model choice.
        n_train: Maximum weekly origins for training (104 = two years).
        min_origin: Earliest allowed origin index, e.g. to skip a period of bad data.

    Returns:
        The origin indices for each period, each list in ascending order.
    """
    last = n_days - 1 - HORIZON
    test = [last - 7 * k for k in range(n_test)][::-1]
    validation = [test[0] - 7 * k for k in range(1, n_validation + 1)][::-1]
    train: list[int] = []
    origin = validation[0] - 7
    while origin >= max(min_origin, HISTORY - 1) and len(train) < n_train:
        train.append(origin)
        origin -= 7
    return Splits(train=train[::-1], validation=validation, test=test)


def _is_payday(dates: pd.DatetimeIndex) -> np.ndarray:
    # Public sector wages in Ecuador are paid on the 15th and the last day of the month.
    return np.asarray((dates.day == 15) | dates.is_month_end)


def origin_rows(panel: SalesPanel, origin: int) -> pd.DataFrame:
    """Rows for one forecast origin: every eligible series x every horizon day.

    Args:
        panel: The sales panel.
        origin: Day index of the last observed day.

    Returns:
        One row per (series, horizon) with features and the `target` column.
        Series without HISTORY days of history at the origin are skipped.
    """
    values = panel.values
    window = values[:, origin - HISTORY + 1 : origin + 1]
    eligible = ~np.isnan(window).any(axis=1)
    idx = np.flatnonzero(eligible)
    w = window[idx]

    per_series = {
        "last_1": w[:, -1],
        "mean_7": w[:, -7:].mean(axis=1),
        "mean_14": w[:, -14:].mean(axis=1),
        "mean_28": w[:, -28:].mean(axis=1),
        "mean_56": w.mean(axis=1),
        "std_28": w[:, -28:].std(axis=1),
        "zero_share_28": (w[:, -28:] == 0).mean(axis=1),
    }

    frames: list[pd.DataFrame] = []
    for h in range(1, HORIZON + 1):
        t = origin + h
        target_date = panel.dates[t]
        # t-7, t-14, t-21, t-28 are all <= origin because h <= 7.
        same_weekday = values[idx][:, [t - 7, t - 14, t - 21, t - 28]]
        frame = panel.series.iloc[idx].reset_index(drop=True).copy()
        frame["origin"] = panel.dates[origin]
        frame["target_date"] = target_date
        frame["horizon"] = h
        frame["dayofweek"] = target_date.day_of_week
        frame["dayofmonth"] = target_date.day
        frame["is_payday"] = bool(_is_payday(pd.DatetimeIndex([target_date]))[0])
        frame["national_holiday"] = panel.national_holiday[idx, t]
        frame["local_holiday"] = panel.local_holiday[idx, t]
        frame["event"] = panel.event[idx, t]
        for name, column in per_series.items():
            frame[name] = column
        frame["same_weekday_last_week"] = same_weekday[:, 0]
        frame["same_weekday_mean_4"] = same_weekday.mean(axis=1)
        frame["target"] = values[idx, t] if t < values.shape[1] else np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_samples(panel: SalesPanel, origins: list[int]) -> pd.DataFrame:
    """Stack origin_rows for many origins and set categorical dtypes.

    Args:
        panel: The sales panel.
        origins: Day indices of forecast origins.

    Returns:
        The stacked rows.
    """
    rows = pd.concat([origin_rows(panel, o) for o in origins], ignore_index=True)
    for name in CATEGORICAL:
        categories = sorted(panel.series[name].unique())
        rows[name] = pd.Categorical(rows[name], categories=categories)
    rows["perishable"] = rows["perishable"].astype(int)
    for name in ("national_holiday", "local_holiday", "event", "is_payday"):
        rows[name] = rows[name].astype(int)
    return rows
