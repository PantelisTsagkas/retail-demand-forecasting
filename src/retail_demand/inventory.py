"""A simple weekly replenishment simulation driven by each model's forecasts.

Assumptions (kept deliberately simple, and stated in the README):
- One order per store x family per week, placed at the forecast origin and delivered
  before the week starts (no lead time beyond that).
- Order-up-to policy: stock up to the 7-day forecast plus safety stock.
- Safety stock = z x the standard deviation of that series' weekly forecast error on the
  validation period, so each model gets a buffer sized from its own past mistakes.
- Models are compared at the same fill rate: we sweep z, then read each model's waste
  at a matched fill rate. Comparing at one fixed z would favour whichever model happens
  to hold more stock.
- Demand that cannot be served is lost (no backorders).
- Perishable families: anything left at the end of the week is thrown away.
  Non-perishable families: leftovers carry over to the next week.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

Z_GRID: tuple[float, ...] = tuple(float(z) for z in np.arange(0.0, 4.01, 0.25))
FILL_RATE_TARGETS: tuple[float, ...] = (0.97, 0.98, 0.99)
KEYS = ["store_nbr", "family"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeekOutcome:
    """Totals from simulating one series over several weeks."""

    demand: float
    served: float
    lost: float
    wasted: float
    ending_stock: float


def simulate_series(
    weekly_demand: np.ndarray, weekly_forecast: np.ndarray, safety_stock: float, perishable: bool
) -> WeekOutcome:
    """Run the order-up-to policy week by week for one series.

    Args:
        weekly_demand: Actual units per week, in time order.
        weekly_forecast: Forecast units per week, same order.
        safety_stock: Extra units held above the forecast.
        perishable: If True, unsold stock is wasted at the end of each week.

    Returns:
        Totals over all weeks. ending_stock sums the stock left after each week.
    """
    on_hand = 0.0
    served_total = lost_total = wasted_total = ending_total = 0.0
    for demand, forecast in zip(weekly_demand, weekly_forecast, strict=True):
        target_level = forecast + safety_stock
        on_hand += max(0.0, target_level - on_hand)
        served = min(on_hand, demand)
        on_hand -= served
        served_total += served
        lost_total += demand - served
        if perishable:
            wasted_total += on_hand
            on_hand = 0.0
        ending_total += on_hand
    return WeekOutcome(
        demand=float(np.sum(weekly_demand)),
        served=served_total,
        lost=lost_total,
        wasted=wasted_total,
        ending_stock=ending_total,
    )


def weekly_totals(predictions: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Sum daily rows into one row per series per forecast origin (a week)."""
    columns = ["target", *models]
    grouped = predictions.groupby([*KEYS, "perishable", "origin"], observed=True)[columns].sum()
    return grouped.reset_index().sort_values([*KEYS, "origin"])


def safety_stocks(validation_weekly: pd.DataFrame, model: str, z: float) -> pd.Series:
    """z x std of weekly forecast error per series, from the validation weeks."""
    errors = validation_weekly[model] - validation_weekly["target"]
    spread = errors.groupby([validation_weekly[k] for k in KEYS], observed=True).std()
    return (z * spread).fillna(0.0)


def simulate_model(
    validation_weekly: pd.DataFrame, test_weekly: pd.DataFrame, model: str, z: float
) -> dict[str, float | str]:
    """Simulate one model at one safety stock level across every series.

    Args:
        validation_weekly: Weekly totals for the validation period (sizes safety stock).
        test_weekly: Weekly totals for the test period (the simulation itself).
        model: Prediction column name.
        z: Safety stock multiplier.

    Returns:
        Fill rate, lost units, wasted perishable units and average leftover stock.
    """
    buffers = safety_stocks(validation_weekly, model, z)
    n_weeks = test_weekly["origin"].nunique()
    demand = served = lost = wasted = ending = 0.0
    for _, weeks in test_weekly.groupby(KEYS, observed=True):
        first = weeks.iloc[0]
        outcome = simulate_series(
            weeks["target"].to_numpy(),
            weeks[model].to_numpy(),
            float(buffers.get((first["store_nbr"], first["family"]), 0.0)),
            bool(first["perishable"]),
        )
        demand += outcome.demand
        served += outcome.served
        lost += outcome.lost
        wasted += outcome.wasted
        ending += outcome.ending_stock
    return {
        "model": model,
        "z": z,
        "fill_rate": served / demand,
        "lost_units": lost,
        "wasted_units_perishable": wasted,
        "avg_leftover_stock_non_perishable": ending / n_weeks,
    }


def service_frontier(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    models: list[str],
    z_grid: tuple[float, ...] = Z_GRID,
) -> pd.DataFrame:
    """Simulate every model at every safety stock level: the fill rate vs waste trade-off.

    Args:
        validation_predictions: Daily rows for the validation period.
        test_predictions: Daily rows for the test period.
        models: Prediction column names.
        z_grid: Safety stock multipliers to try.

    Returns:
        One row per (model, z).
    """
    validation_weekly = weekly_totals(validation_predictions, models)
    test_weekly = weekly_totals(test_predictions, models)
    records: list[dict[str, float | str]] = []
    for model in models:
        for z in z_grid:
            records.append(simulate_model(validation_weekly, test_weekly, model, z))
    return pd.DataFrame(records)


def at_matched_fill_rate(
    frontier: pd.DataFrame, targets: tuple[float, ...] = FILL_RATE_TARGETS
) -> pd.DataFrame:
    """Interpolate each model's waste and leftover stock at the same fill rates.

    Args:
        frontier: Output of service_frontier.
        targets: Fill rates to compare at. Each must lie inside every model's swept range.

    Returns:
        One row per (model, target fill rate).

        Targets a model never reaches in the sweep get NaN values.
    """
    records: list[dict[str, float | str]] = []
    for model, rows in frontier.groupby("model", sort=False):
        rows = rows.sort_values("fill_rate")
        fill = rows["fill_rate"].to_numpy()
        for target in targets:
            if not fill.min() <= target <= fill.max():
                # Unreachable in the swept range: record it as missing, never extrapolate.
                logger.warning("%s never reaches a %.0f%% fill rate", model, target * 100)
                records.append(
                    {
                        "model": str(model),
                        "fill_rate": target,
                        "wasted_units_perishable": float("nan"),
                        "avg_leftover_stock_non_perishable": float("nan"),
                    }
                )
                continue
            records.append(
                {
                    "model": str(model),
                    "fill_rate": target,
                    "wasted_units_perishable": float(
                        np.interp(target, fill, rows["wasted_units_perishable"].to_numpy())
                    ),
                    "avg_leftover_stock_non_perishable": float(
                        np.interp(
                            target, fill, rows["avg_leftover_stock_non_perishable"].to_numpy()
                        )
                    ),
                }
            )
    return pd.DataFrame(records)
