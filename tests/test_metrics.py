import numpy as np
import pytest

from retail_demand.metrics import bias, mae, rmse, rmsle, wape

ACTUAL = np.array([10.0, 20.0, 30.0, 40.0])
PREDICTED = np.array([12.0, 18.0, 30.0, 50.0])  # errors: +2, -2, 0, +10


def test_mae() -> None:
    assert mae(ACTUAL, PREDICTED) == pytest.approx(14 / 4)


def test_rmse_punishes_the_big_miss_more() -> None:
    assert rmse(ACTUAL, PREDICTED) == pytest.approx(np.sqrt((4 + 4 + 0 + 100) / 4))
    assert rmse(ACTUAL, PREDICTED) > mae(ACTUAL, PREDICTED)


def test_wape_is_error_over_total_demand() -> None:
    assert wape(ACTUAL, PREDICTED) == pytest.approx(14 / 100)


def test_bias_is_signed() -> None:
    assert bias(ACTUAL, PREDICTED) == pytest.approx(10 / 100)
    assert bias(ACTUAL, ACTUAL * 0.5) == pytest.approx(-0.5)


def test_rmsle_is_zero_for_perfect_forecast_and_clips_negatives() -> None:
    assert rmsle(ACTUAL, ACTUAL) == 0.0
    assert rmsle(np.array([0.0]), np.array([-5.0])) == 0.0
