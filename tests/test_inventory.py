import numpy as np
import pandas as pd
import pytest

from retail_demand.inventory import at_matched_fill_rate, simulate_series


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
            "fill_rate": [0.96, 1.00],
            "wasted_units_perishable": [100.0, 300.0],
            "avg_leftover_stock_non_perishable": [10.0, 30.0],
        }
    )

    result = at_matched_fill_rate(frontier, targets=(0.98,))

    assert result.loc[0, "wasted_units_perishable"] == pytest.approx(200.0)
    assert result.loc[0, "avg_leftover_stock_non_perishable"] == pytest.approx(20.0)


def test_matched_fill_rate_marks_unreachable_targets_as_missing() -> None:
    frontier = pd.DataFrame(
        {
            "model": ["A", "A"],
            "z": [0.0, 1.0],
            "fill_rate": [0.96, 0.97],
            "wasted_units_perishable": [100.0, 300.0],
            "avg_leftover_stock_non_perishable": [10.0, 30.0],
        }
    )

    result = at_matched_fill_rate(frontier, targets=(0.99,))

    assert np.isnan(result.loc[0, "wasted_units_perishable"])
