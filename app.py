"""Streamlit dashboard: explore test-period forecasts, errors and the inventory simulation.

Run with: uv run streamlit run app.py
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPORTS = Path(__file__).parent / "reports"
# Same colours as the README figures: actual is ink, LightGBM blue, best baseline orange.
COLOURS = {"Actual": "#0b0b0b", "LightGBM": "#2a78d6", "baseline": "#eb6834"}
MODEL_ORDER = ["Last day", "Same weekday last week", "Mean of last 4 same weekdays", "LightGBM"]
SEGMENT_ORDER = ["perishable", "non-perishable", "high volume", "medium volume", "low volume"]
REPO = "https://github.com/PantelisTsagkas/retail-demand-forecasting"


@st.cache_data
def load() -> dict[str, pd.DataFrame]:
    """Read every report file once."""
    predictions = pd.read_parquet(REPORTS / "predictions_test.parquet")
    predictions["store_nbr"] = predictions["store_nbr"].astype(int)
    predictions["family"] = predictions["family"].astype(str)
    return {
        "predictions": predictions,
        "overall": pd.read_csv(REPORTS / "metrics_overall.csv"),
        "segments": pd.read_csv(REPORTS / "metrics_segments.csv"),
        "frontier": pd.read_csv(REPORTS / "inventory_frontier.csv"),
        "matched": pd.read_csv(REPORTS / "inventory_matched_fill_rate.csv"),
        "importance": pd.read_csv(REPORTS / "feature_importance.csv"),
    }


def best_baseline(overall: pd.DataFrame) -> str:
    """Baseline with the lowest validation WAPE."""
    validation = overall[(overall["period"] == "validation") & (overall["model"] != "LightGBM")]
    return str(validation.sort_values("WAPE").iloc[0]["model"])


def percent(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Format share columns as percentages for display."""
    out = table.copy()
    for column in columns:
        out[column] = (out[column] * 100).map("{:.1f}%".format)
    return out


st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
data = load()
baseline = best_baseline(data["overall"])
test = data["overall"][data["overall"]["period"] == "test"].set_index("model")

st.title("Retail demand forecasting")
st.caption(
    "Daily units per store and product family, forecast every Tuesday for the next 7 days. "
    "Real data from Corporación Favorita (Ecuador, public Kaggle dataset). "
    "Test period: 8 weeks, 21 Jun to 15 Aug 2017, never seen during training or model choice. "
    f"[Code and write-up on GitHub]({REPO})."
)

col1, col2, col3 = st.columns(3)
model_wape = test.loc["LightGBM", "WAPE"]
baseline_wape = test.loc[baseline, "WAPE"]
col1.metric("LightGBM test WAPE", f"{model_wape:.1%}")
col2.metric("Best baseline test WAPE", f"{baseline_wape:.1%}", help=baseline)
col3.metric(
    "Relative error reduction",
    f"{(baseline_wape - model_wape) / baseline_wape:.0%}",
)

st.subheader("Forecast vs actual")
predictions = data["predictions"]
left, right = st.columns(2)
store = left.selectbox("Store", sorted(predictions["store_nbr"].unique()), index=43)
families = sorted(predictions["family"].unique())
family = right.selectbox("Product family", families, index=families.index("PRODUCE"))

series = predictions[(predictions["store_nbr"] == store) & (predictions["family"] == family)]
if series.empty:
    st.info("This store and family has no test rows (not enough sales history).")
else:
    long = series.melt(
        id_vars="target_date",
        value_vars=["target", "LightGBM", baseline],
        var_name="series",
        value_name="units",
    ).replace({"series": {"target": "Actual"}})
    domain = ["Actual", "LightGBM", baseline]
    chart = (
        alt.Chart(long, mark=alt.MarkDef(type="line", strokeWidth=2))
        .encode(
            x=alt.X("target_date:T", title=None),
            y=alt.Y("units:Q", title="Units"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=domain,
                    range=[COLOURS["Actual"], COLOURS["LightGBM"], COLOURS["baseline"]],
                ),
                legend=alt.Legend(title=None, orient="top", labelLimit=400),
            ),
            strokeDash=alt.condition(
                alt.datum.series == baseline, alt.value([6, 4]), alt.value([1, 0])
            ),
            tooltip=[
                alt.Tooltip("target_date:T", title="Date"),
                "series:N",
                alt.Tooltip("units:Q", format=",.0f"),
            ],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, width="stretch")
    actual = series["target"].sum()
    errors = {
        name: (series[name] - series["target"]).abs().sum() / actual
        for name in ["LightGBM", baseline]
    }
    st.caption(
        f"WAPE for this series: LightGBM {errors['LightGBM']:.1%}, "
        f"{baseline} {errors[baseline]:.1%}"
    )

st.subheader("Error by model (test period)")
st.dataframe(
    test.reset_index()
    .drop(columns="period")
    .style.format(
        {"MAE": "{:.1f}", "RMSE": "{:.1f}", "WAPE": "{:.1%}", "RMSLE": "{:.3f}", "Bias": "{:+.1%}"}
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("Error by segment (test WAPE)")
segment_wape = data["segments"].pivot(index="segment", columns="model", values="WAPE")
st.dataframe(
    segment_wape.loc[SEGMENT_ORDER, MODEL_ORDER].style.format("{:.1%}"),
    width="stretch",
)

st.subheader("Inventory simulation")
st.caption(
    "Weekly order-up-to policy: stock = 7-day forecast + safety stock, sized from each model's "
    "own validation errors. Unsold perishables are thrown away at week end; lost sales are not "
    "backordered. Models are compared at the same fill rate. Simplified, at family level."
)
matched = data["matched"].copy()
matched["fill_rate"] = (matched["fill_rate"] * 100).map("{:.0f}%".format)
waste = matched.pivot(index="model", columns="fill_rate", values="wasted_units_perishable")
waste = waste.loc[MODEL_ORDER]
st.markdown("**Perishable units wasted over 8 weeks, at the same fill rate**")
st.dataframe(
    waste.map(lambda v: "not reached" if pd.isna(v) else f"{v:,.0f}"),
    width="stretch",
)
frontier = data["frontier"][data["frontier"]["model"].isin(["LightGBM", baseline])]
curve = (
    alt.Chart(frontier, mark=alt.MarkDef(type="line", point=True, strokeWidth=2))
    .encode(
        x=alt.X(
            "fill_rate:Q",
            title="Fill rate",
            axis=alt.Axis(format="%"),
            scale=alt.Scale(domain=[0.95, 1.0]),
        ),
        y=alt.Y(
            "wasted_units_perishable:Q",
            title="Perishable units wasted",
            axis=alt.Axis(format="~s"),
        ),
        color=alt.Color(
            "model:N",
            scale=alt.Scale(
                domain=["LightGBM", baseline], range=[COLOURS["LightGBM"], COLOURS["baseline"]]
            ),
            legend=alt.Legend(title=None, orient="top", labelLimit=400),
        ),
        tooltip=[
            "model:N",
            alt.Tooltip("z:Q", title="Safety stock z"),
            alt.Tooltip("fill_rate:Q", format=".1%"),
            alt.Tooltip("wasted_units_perishable:Q", format=",.0f", title="Wasted"),
        ],
    )
    .transform_filter(alt.datum.fill_rate >= 0.95)
    .properties(height=320)
)
st.altair_chart(curve, width="stretch")

st.subheader("What the model relies on")
importance = data["importance"].head(12).copy()
importance["share"] = importance["gain"] / data["importance"]["gain"].sum()
bars = (
    alt.Chart(importance, mark=alt.MarkDef(type="bar", color=COLOURS["LightGBM"]))
    .encode(
        x=alt.X("share:Q", title="Share of total gain", axis=alt.Axis(format="%")),
        y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)),
        tooltip=["feature:N", alt.Tooltip("share:Q", format=".1%")],
    )
    .properties(height=320)
)
st.altair_chart(bars, width="stretch")
