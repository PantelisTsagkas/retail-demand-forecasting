import pandas as pd
import pytest

from retail_demand.evaluate import series_wape, series_win_summary


def _predictions() -> pd.DataFrame:
    # Store 1: model wins (error 2 vs 4). Store 2: baseline wins (error 3 vs 1).
    # Store 3: no sales, so it is dropped.
    return pd.DataFrame(
        {
            "store_nbr": [1, 1, 2, 2, 3],
            "family": ["F"] * 5,
            "target": [10.0, 10.0, 5.0, 5.0, 0.0],
            "model": [11.0, 9.0, 8.0, 5.0, 1.0],
            "baseline": [12.0, 8.0, 6.0, 5.0, 0.0],
        }
    )


def test_series_wape_scores_each_series_and_drops_series_without_sales() -> None:
    result = series_wape(_predictions(), "model", "baseline")

    assert list(result["store_nbr"]) == [1, 2]
    assert result.loc[0, "model_wape"] == pytest.approx(2 / 20)
    assert result.loc[0, "baseline_wape"] == pytest.approx(4 / 20)
    assert list(result["model_wins"]) == [True, False]


def test_series_win_summary_reports_share_and_size_of_wins() -> None:
    per_series = series_wape(_predictions(), "model", "baseline")

    summary = series_win_summary(per_series, "model", "baseline").iloc[0]

    assert summary["n_series"] == 2
    assert summary["share_won"] == pytest.approx(0.5)
    assert summary["share_won_by_over_5pct"] == pytest.approx(0.5)
    # Relative improvements are +50% (store 1) and -200% (store 2).
    assert summary["median_relative_improvement"] == pytest.approx((0.5 - 2.0) / 2)
