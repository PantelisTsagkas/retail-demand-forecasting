"""Run baselines and LightGBM on the same forecast origins and write every result to reports/.

Run with: uv run forecast
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from retail_demand.features import FEATURES, HISTORY, Splits, build_samples, make_splits
from retail_demand.inventory import (
    NON_PERISHABLE_TARGETS,
    PERISHABLE_TARGETS,
    at_matched_fill_rate,
    service_frontier,
)
from retail_demand.metrics import score
from retail_demand.model import BASELINES, fit_with_early_stopping, predict, refit
from retail_demand.panel import SalesPanel, load_panel

logger = logging.getLogger(__name__)

MODEL_NAME = "LightGBM"

# Several families (PRODUCE, BEVERAGES and others) are missing from the raw data for long
# stretches of 2014 and early 2015, then return at normal levels from this date. Every
# training window must sit fully after it, so the model never learns from the gap.
DATA_COMPLETE_FROM = pd.Timestamp("2015-06-01")


def add_predictions(rows: pd.DataFrame, model_predictions: np.ndarray) -> pd.DataFrame:
    """Return a copy of rows with one prediction column per model."""
    out = rows.copy()
    for name, column in BASELINES.items():
        out[name] = out[column]
    out[MODEL_NAME] = model_predictions
    return out


def model_names() -> list[str]:
    """Every model in report order: baselines first."""
    return [*BASELINES, MODEL_NAME]


def score_table(predictions: pd.DataFrame, period: str) -> pd.DataFrame:
    """Every metric for every model on one period."""
    records: list[dict[str, float | str]] = []
    actual = predictions["target"].to_numpy()
    for name in model_names():
        records.append(
            {"period": period, "model": name, **score(actual, predictions[name].to_numpy())}
        )
    return pd.DataFrame(records)


def volume_band(panel: SalesPanel, splits: Splits) -> pd.Series:
    """Label each series high, medium or low volume, using only training-period sales.

    Args:
        panel: The sales panel.
        splits: Forecast origins; volume uses the 365 days before the first validation origin.

    Returns:
        A band per (store_nbr, family).
    """
    end = splits.validation[0] + 1
    history = panel.values[:, max(0, end - 365) : end]
    mean_sales = pd.Series(np.nanmean(history, axis=1))
    bands = pd.qcut(mean_sales.rank(method="first"), 3, labels=["low", "medium", "high"])
    index = pd.MultiIndex.from_frame(panel.series[["store_nbr", "family"]])
    return pd.Series(bands.astype(str).to_numpy(), index=index, name="volume_band")


def best_baseline(overall: pd.DataFrame) -> str:
    """The baseline with the lowest validation WAPE (chosen on validation, not test)."""
    validation = overall[
        (overall["period"] == "validation") & (overall["model"].isin(list(BASELINES)))
    ]
    return str(validation.sort_values("WAPE").iloc[0]["model"])


def series_wape(predictions: pd.DataFrame, model: str, baseline: str) -> pd.DataFrame:
    """WAPE per store x family series for a model and a baseline.

    Series with no sales in the period are dropped, as WAPE is undefined for them.

    Args:
        predictions: Daily rows with `target` and one column per model.
        model: Prediction column to judge.
        baseline: Prediction column to compare against.

    Returns:
        One row per series: store_nbr, family, actual units, both WAPEs and whether the
        model's WAPE is lower.
    """
    errors = pd.DataFrame(
        {
            "store_nbr": predictions["store_nbr"],
            "family": predictions["family"],
            "actual": predictions["target"],
            "model_abs_error": (predictions[model] - predictions["target"]).abs(),
            "baseline_abs_error": (predictions[baseline] - predictions["target"]).abs(),
        }
    )
    totals = errors.groupby(["store_nbr", "family"], observed=True).sum().reset_index()
    totals = totals[totals["actual"] > 0].copy()
    totals["model_wape"] = totals["model_abs_error"] / totals["actual"]
    totals["baseline_wape"] = totals["baseline_abs_error"] / totals["actual"]
    totals["model_wins"] = totals["model_wape"] < totals["baseline_wape"]
    keep = ["store_nbr", "family", "actual", "model_wape", "baseline_wape", "model_wins"]
    return totals[keep].reset_index(drop=True)


def series_win_summary(per_series: pd.DataFrame, model: str, baseline: str) -> pd.DataFrame:
    """How often, and by how much, the model beats the baseline across series.

    Args:
        per_series: Output of series_wape.
        model: Name of the model, for the report.
        baseline: Name of the baseline, for the report.

    Returns:
        A single row: series count, share won, share won by more than 5% relative WAPE,
        and the median relative WAPE improvement.
    """
    relative = (per_series["baseline_wape"] - per_series["model_wape"]) / per_series[
        "baseline_wape"
    ]
    return pd.DataFrame(
        [
            {
                "model": model,
                "baseline": baseline,
                "n_series": len(per_series),
                "share_won": float(per_series["model_wins"].mean()),
                "share_won_by_over_5pct": float((relative > 0.05).mean()),
                "median_relative_improvement": float(relative.median()),
            }
        ]
    )


def segment_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Test metrics per segment: perishable vs not, and volume band."""
    records: list[dict[str, float | str]] = []
    segments = {
        "perishable": predictions["perishable"] == 1,
        "non-perishable": predictions["perishable"] == 0,
        "high volume": predictions["volume_band"] == "high",
        "medium volume": predictions["volume_band"] == "medium",
        "low volume": predictions["volume_band"] == "low",
    }
    for segment, mask in segments.items():
        part = predictions[mask]
        actual = part["target"].to_numpy()
        for name in model_names():
            metrics = score(actual, part[name].to_numpy())
            records.append({"segment": segment, "model": name, **metrics})
    return pd.DataFrame(records)


def run(db_path: Path, reports: Path) -> None:
    """Full experiment: features, baselines, model, metrics, inventory simulation."""
    reports.mkdir(parents=True, exist_ok=True)
    panel = load_panel(db_path)
    complete_from = int(panel.dates.get_loc(DATA_COMPLETE_FROM))
    splits = make_splits(len(panel.dates), min_origin=complete_from + HISTORY - 1)
    logger.info(
        "Origins: %d train (%s to %s), %d validation, %d test (%s to %s)",
        len(splits.train),
        panel.dates[splits.train[0]].date(),
        panel.dates[splits.train[-1]].date(),
        len(splits.validation),
        len(splits.test),
        panel.dates[splits.test[0]].date(),
        panel.dates[splits.test[-1]].date(),
    )

    train = build_samples(panel, splits.train)
    validation = build_samples(panel, splits.validation)
    test = build_samples(panel, splits.test)
    logger.info("Rows: train %d, validation %d, test %d", len(train), len(validation), len(test))

    # 1. Choose the number of trees on validation. 2. Refit on train + validation.
    # 3. Score once on test, which nothing above has seen.
    selector = fit_with_early_stopping(train, validation)
    validation_predictions = add_predictions(validation, predict(selector, validation))
    final = refit(pd.concat([train, validation], ignore_index=True), selector.best_iteration_)
    test_predictions = add_predictions(test, predict(final, test))

    bands = volume_band(panel, splits)
    for frame in (validation_predictions, test_predictions):
        keys = pd.MultiIndex.from_arrays(
            [frame["store_nbr"].astype(int), frame["family"].astype(str)]
        )
        frame["volume_band"] = bands.reindex(keys).to_numpy()

    overall = pd.concat(
        [score_table(validation_predictions, "validation"), score_table(test_predictions, "test")]
    )
    overall.to_csv(reports / "metrics_overall.csv", index=False)
    segment_table(test_predictions).to_csv(reports / "metrics_segments.csv", index=False)

    importance = pd.DataFrame(
        {"feature": list(FEATURES), "gain": final.booster_.feature_importance("gain")}
    ).sort_values("gain", ascending=False)
    importance.to_csv(reports / "feature_importance.csv", index=False)

    keep = [
        "store_nbr",
        "family",
        "perishable",
        "volume_band",
        "origin",
        "target_date",
        "horizon",
        "target",
    ]
    validation_predictions[keep + model_names()].to_parquet(
        reports / "predictions_validation.parquet"
    )
    test_predictions[keep + model_names()].to_parquet(reports / "predictions_test.parquet")

    baseline = best_baseline(overall)
    per_series = series_wape(test_predictions, MODEL_NAME, baseline)
    per_series.to_csv(reports / "series_wape.csv", index=False)
    wins = series_win_summary(per_series, MODEL_NAME, baseline)
    wins.to_csv(reports / "series_wins.csv", index=False)

    frontier = service_frontier(validation_predictions, test_predictions, model_names())
    frontier.to_csv(reports / "inventory_frontier.csv", index=False)
    waste = at_matched_fill_rate(
        frontier, "fill_rate_perishable", "wasted_units_perishable", PERISHABLE_TARGETS
    )
    waste.to_csv(reports / "inventory_perishable_waste.csv", index=False)
    stock = at_matched_fill_rate(
        frontier,
        "fill_rate_non_perishable",
        "avg_leftover_stock_non_perishable",
        NON_PERISHABLE_TARGETS,
    )
    stock.to_csv(reports / "inventory_non_perishable_stock.csv", index=False)

    logger.info("\n%s", overall.round(3).to_string(index=False))
    logger.info("\n%s", wins.round(3).to_string(index=False))
    logger.info("\n%s", waste.round(0).to_string(index=False))
    logger.info("\n%s", stock.round(0).to_string(index=False))


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/favorita.duckdb"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.db, args.reports)


if __name__ == "__main__":
    main()
