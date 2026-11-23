import numpy as np
import pandas as pd
import pytest

from retail_demand.inventory import at_matched_fill_rate, simulate_model, simulate_series


def test_perfect_forecast_without_buffer_serves_everything_with_no_waste() -> None:
    demand = np.array([10.0, 12.0, 8.0])

    outcome = simulate_series(demand, demand, safety_stock=0.0, perishable=True)

    assert outcome.served == 30.0
    assert outcome.lost == 0.0
    assert outcome.wasted == 0.0


def test_under_forecast_loses_sales() -> None:
    outcome = simulate_series(np.array([10.0]), np.array([6.0]), safety_stock=0.0, perishable=False)

    assert outcome.lost == 4.0
    assert outcome.served == 6.0


def test_perishable_leftovers_are_wasted_each_week() -> None:
    outcome = simulate_series(
        np.array([5.0, 5.0]), np.array([5.0, 5.0]), safety_stock=2.0, perishable=True
    )

    assert outcome.wasted == 4.0  # 2 units of buffer thrown away each week
    assert outcome.lost == 0.0


def test_non_perishable_leftovers_carry_over_and_reduce_the_next_order() -> None:
    outcome = simulate_series(
        np.array([5.0, 5.0]), np.array([5.0, 5.0]), safety_stock=2.0, perishable=False
    )

    assert outcome.wasted == 0.0
    assert outcome.ending_stock == pytest.approx(4.0)  # 2 left after each week
    assert outcome.served == 10.0


def test_matched_fill_rate_interpolates_between_sweep_points() -> None:
    frontier = pd.DataFrame(
        {
            "model": ["A", "A"],
            "z": [0.0, 1.0],
            "fill_rate_perishable": [0.96, 1.00],
            "wasted_units_perishable": [100.0, 300.0],
        }
    )

    result = at_matched_fill_rate(
        frontier, "fill_rate_perishable", "wasted_units_perishable", targets=(0.98,)
    )

    assert result.loc[0, "wasted_units_perishable"] == pytest.approx(200.0)


def test_matched_fill_rate_marks_unreachable_targets_as_missing() -> None:
    frontier = pd.DataFrame(
        {
            "model": ["A", "A"],
            "z": [0.0, 1.0],
            "fill_rate_perishable": [0.96, 0.97],
            "wasted_units_perishable": [100.0, 300.0],
        }
    )

    result = at_matched_fill_rate(
        frontier, "fill_rate_perishable", "wasted_units_perishable", targets=(0.99,)
    )

    assert np.isnan(result.loc[0, "wasted_units_perishable"])


def _weekly(perishable: list[int], target: list[float], forecast: list[float]) -> pd.DataFrame:
    """One week per series: series i is store i, family 'F'."""
    n = len(target)
    return pd.DataFrame(
        {
            "store_nbr": list(range(n)),
            "family": ["F"] * n,
            "perishable": perishable,
            "origin": [0] * n,
            "target": target,
            "m": forecast,
        }
    )


def test_fill_rate_is_reported_per_segment_not_pooled() -> None:
    # Perishable series is fully served, non-perishable series is half served.
    # A pooled fill rate would read 150 / 200 = 75% and hide the split.
    weekly = _weekly(perishable=[1, 0], target=[100.0, 100.0], forecast=[100.0, 50.0])

    result = simulate_model(weekly, weekly, "m", z=0.0)

    assert result["fill_rate_perishable"] == pytest.approx(1.0)
    assert result["fill_rate_non_perishable"] == pytest.approx(0.5)


def test_leftover_stock_only_counts_non_perishables() -> None:
    weekly = _weekly(perishable=[1, 0], target=[10.0, 10.0], forecast=[15.0, 12.0])

    result = simulate_model(weekly, weekly, "m", z=0.0)

    assert result["wasted_units_perishable"] == pytest.approx(5.0)
    assert result["avg_leftover_stock_non_perishable"] == pytest.approx(2.0)
