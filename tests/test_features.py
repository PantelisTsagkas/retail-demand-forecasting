import numpy as np
import pandas as pd
import pytest

from retail_demand.features import FEATURES, HISTORY, HORIZON, build_samples, make_splits
from retail_demand.panel import SalesPanel


def make_panel(values: np.ndarray) -> SalesPanel:
    n_series, n_days = values.shape
    shape = (n_series, n_days)
    return SalesPanel(
        values=values,
        dates=pd.date_range("2016-01-01", periods=n_days, freq="D"),
        series=pd.DataFrame(
            {
                "store_nbr": list(range(1, n_series + 1)),
                "family": ["PRODUCE"] * n_series,
                "perishable": [True] * n_series,
                "store_type": ["A"] * n_series,
                "cluster": [1] * n_series,
            }
        ),
        national_holiday=np.zeros(shape, dtype=bool),
        local_holiday=np.zeros(shape, dtype=bool),
        event=np.zeros(shape, dtype=bool),
    )


@pytest.fixture
def panel() -> SalesPanel:
    rng = np.random.default_rng(0)
    return make_panel(rng.poisson(20, size=(3, 120)).astype(float))


def test_features_ignore_everything_after_the_origin(panel: SalesPanel) -> None:
    origin = 80
    changed = panel.values.copy()
    changed[:, origin + 1 :] = 9999.0
    future_changed = make_panel(changed)

    before = build_samples(panel, [origin])
    after = build_samples(future_changed, [origin])

    pd.testing.assert_frame_equal(before[list(FEATURES)], after[list(FEATURES)])
    assert (after["target"] == 9999.0).all()  # only the target sees the future


def test_one_row_per_series_and_horizon_day(panel: SalesPanel) -> None:
    rows = build_samples(panel, [80])

    assert len(rows) == 3 * HORIZON
    assert sorted(rows["horizon"].unique()) == list(range(1, HORIZON + 1))


def test_same_weekday_features_use_the_right_days() -> None:
    values = np.arange(120, dtype=float).reshape(1, 120)  # value equals day index
    rows = build_samples(make_panel(values), [80])
    day_3 = rows[rows["horizon"] == 3].iloc[0]  # target day 83

    assert day_3["target"] == 83
    assert day_3["last_1"] == 80
    assert day_3["same_weekday_last_week"] == 76
    assert day_3["same_weekday_mean_4"] == np.mean([76, 69, 62, 55])
    assert day_3["mean_7"] == np.mean(range(74, 81))


def test_series_without_enough_history_are_skipped() -> None:
    values = np.full((2, 120), 5.0)
    values[1, :40] = np.nan  # store 2 opened on day 40
    origin = 40 + HISTORY - 2  # one day short of HISTORY days for store 2

    rows = build_samples(make_panel(values), [origin])

    assert set(rows["store_nbr"].astype(int)) == {1}


def test_splits_are_in_time_order_and_never_share_target_days() -> None:
    splits = make_splits(n_days=400, n_test=4, n_validation=4, n_train=10)

    assert splits.train[-1] + HORIZON <= splits.validation[0]
    assert splits.validation[-1] + HORIZON <= splits.test[0]
    assert splits.test[-1] + HORIZON == 399
    assert len(splits.train) == 10


def test_splits_respect_the_earliest_allowed_origin() -> None:
    splits = make_splits(n_days=400, n_test=4, n_validation=4, n_train=100, min_origin=200)

    assert min(splits.train) >= 200
