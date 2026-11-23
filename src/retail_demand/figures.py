"""Draw the README figures from the DuckDB tables and the files in reports/.

Run with: uv run figures
"""

import argparse
import logging
from pathlib import Path
from typing import Literal

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b9b8b3"
MODEL_COLOURS = {"LightGBM": "#2a78d6", "baseline": "#eb6834"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "font.size": 10,
        "legend.frameon": False,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def _grid(ax: Axes, axis: Literal["x", "y", "both"]) -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, out: Path) -> None:
    fig.savefig(out)
    plt.close(fig)
    logger.info("Wrote %s", out)


def best_baseline(overall: pd.DataFrame) -> str:
    """The baseline with the lowest validation WAPE (chosen on validation, not test)."""
    validation = overall[(overall["period"] == "validation") & (overall["model"] != "LightGBM")]
    return str(validation.sort_values("WAPE").iloc[0]["model"])


def weekly_sales(con: duckdb.DuckDBPyConnection, out: Path) -> None:
    """Total units sold per week across all stores."""
    df = con.execute(
        """
        SELECT date_trunc('week', date) AS week, sum(unit_sales) / 1e6 AS units_m
        FROM family_daily GROUP BY 1 HAVING count(DISTINCT date) = 7 ORDER BY 1
        """
    ).df()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(df["week"], df["units_m"], color=MODEL_COLOURS["LightGBM"], linewidth=2)
    ax.set_title("Weekly units sold, all 54 stores")
    ax.set_ylabel("Million units")
    _grid(ax, "y")
    _save(fig, out)


def weekday_pattern(con: duckdb.DuckDBPyConnection, out: Path) -> None:
    """Average units per store-family series by weekday, last full year."""
    df = con.execute(
        """
        SELECT isodow(date) AS dow, strftime(date, '%a') AS day, avg(unit_sales) AS units
        FROM family_daily WHERE date >= DATE '2016-08-15'
        GROUP BY ALL ORDER BY dow
        """
    ).df()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.bar(df["day"], df["units"], color=MODEL_COLOURS["LightGBM"], width=0.6)
    ax.bar_label(bars, fmt="%.0f", color=INK_SECONDARY, padding=3)
    ax.set_title("Average daily units per store and family, by weekday")
    ax.set_ylabel("Units")
    _grid(ax, "y")
    _save(fig, out)


def wape_by_model(overall: pd.DataFrame, out: Path) -> None:
    """Test WAPE per model, best model highlighted."""
    test = overall[overall["period"] == "test"].sort_values("WAPE", ascending=False)
    colours = [MODEL_COLOURS["LightGBM"] if m == "LightGBM" else NEUTRAL for m in test["model"]]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.barh(test["model"], test["WAPE"] * 100, color=colours, height=0.55)
    ax.bar_label(bars, fmt="%.1f%%", color=INK, padding=4)
    ax.set_title("Forecast error on the 8 test weeks (WAPE, lower is better)")
    ax.set_xlabel("WAPE (%)")
    _grid(ax, "x")
    _save(fig, out)


def wape_by_segment(segments: pd.DataFrame, baseline: str, out: Path) -> None:
    """LightGBM vs the best baseline in each segment."""
    order = ["perishable", "non-perishable", "high volume", "medium volume", "low volume"]
    pivot = segments.pivot(index="segment", columns="model", values="WAPE").loc[order[::-1]]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = range(len(pivot))
    height = 0.36
    for offset, model, colour in (
        (height / 2, "LightGBM", MODEL_COLOURS["LightGBM"]),
        (-height / 2, baseline, MODEL_COLOURS["baseline"]),
    ):
        values = pivot[model] * 100
        bars = ax.barh(
            [i + offset for i in y], values, height=height - 0.04, color=colour, label=model
        )
        ax.bar_label(bars, fmt="%.1f%%", color=INK_SECONDARY, padding=3, fontsize=9)
    ax.set_yticks(list(y), pivot.index)
    ax.set_title("Test WAPE by segment")
    ax.set_xlabel("WAPE (%)")
    ax.legend(loc="upper right")
    _grid(ax, "x")
    _save(fig, out)


def example_series(predictions: pd.DataFrame, baseline: str, out: Path) -> None:
    """Actual vs forecasts for one high-volume perishable series over the test weeks."""
    store, family = 44, "PRODUCE"
    part = predictions[
        (predictions["store_nbr"].astype(int) == store)
        & (predictions["family"].astype(str) == family)
    ].sort_values("target_date")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(part["target_date"], part["target"], color=INK, linewidth=2, label="Actual")
    ax.plot(
        part["target_date"],
        part["LightGBM"],
        color=MODEL_COLOURS["LightGBM"],
        linewidth=2,
        label="LightGBM",
    )
    ax.plot(
        part["target_date"],
        part[baseline],
        color=MODEL_COLOURS["baseline"],
        linewidth=2,
        linestyle="--",
        label=baseline,
    )
    ax.set_title(f"Store {store}, {family.title()}: 8 test weeks, forecast each Tuesday for 7 days")
    ax.set_ylabel("Units")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + (top - bottom) * 0.15)  # headroom so the legend sits clear
    ax.legend(loc="upper left", ncols=3)
    _grid(ax, "y")
    _save(fig, out)


def feature_importance(importance: pd.DataFrame, out: Path) -> None:
    """Top features by total gain."""
    top = importance.head(12).iloc[::-1]
    share = top["gain"] / importance["gain"].sum() * 100
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(top["feature"], share, color=MODEL_COLOURS["LightGBM"], height=0.6)
    ax.bar_label(bars, fmt="%.1f%%", color=INK_SECONDARY, padding=3, fontsize=9)
    ax.set_title("What the model relies on (share of total gain)")
    ax.set_xlabel("% of gain")
    _grid(ax, "x")
    _save(fig, out)


def inventory_frontier(frontier: pd.DataFrame, baseline: str, out: Path) -> None:
    """Fill rate vs perishable waste as safety stock grows, best baseline vs LightGBM."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for model, colour in (
        ("LightGBM", MODEL_COLOURS["LightGBM"]),
        (baseline, MODEL_COLOURS["baseline"]),
    ):
        rows = frontier[frontier["model"] == model].sort_values("fill_rate")
        ax.plot(
            rows["fill_rate"] * 100,
            rows["wasted_units_perishable"] / 1e6,
            color=colour,
            linewidth=2,
            marker="o",
            markersize=5,
            label=model,
        )
    ax.set_xlim(95, 100)
    ax.set_title("More safety stock: fewer lost sales, more waste")
    ax.set_xlabel("Fill rate: share of demand served (%)")
    ax.set_ylabel("Perishable units wasted (M)")
    ax.legend(loc="upper left")
    _grid(ax, "both")
    _save(fig, out)


def run(db_path: Path, reports: Path) -> None:
    """Draw every figure into reports/figures/."""
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    overall = pd.read_csv(reports / "metrics_overall.csv")
    segments = pd.read_csv(reports / "metrics_segments.csv")
    predictions = pd.read_parquet(reports / "predictions_test.parquet")
    importance = pd.read_csv(reports / "feature_importance.csv")
    frontier = pd.read_csv(reports / "inventory_frontier.csv")
    baseline = best_baseline(overall)

    with duckdb.connect(db_path, read_only=True) as con:
        weekly_sales(con, figures / "eda_weekly_sales.png")
        weekday_pattern(con, figures / "eda_weekday.png")
    wape_by_model(overall, figures / "wape_by_model.png")
    wape_by_segment(segments, baseline, figures / "wape_by_segment.png")
    example_series(predictions, baseline, figures / "example_forecast.png")
    feature_importance(importance, figures / "feature_importance.png")
    inventory_frontier(frontier, baseline, figures / "inventory_frontier.png")


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
