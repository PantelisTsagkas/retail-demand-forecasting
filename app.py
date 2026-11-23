"""Streamlit dashboard: explore test-period forecasts, errors and the inventory simulation.

Run with: uv run streamlit run app.py
"""

import base64
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
ASSETS = ROOT / "assets"
# Optional photos. Only commit images you hold a licence for (for example Unsplash or Pexels).
# Without them the hero uses a solid colour and the summary runs full width.
HERO_PHOTO = ASSETS / "hero.jpg"
SUMMARY_PHOTO = ASSETS / "summary.jpg"
REPO = "https://github.com/PantelisTsagkas/retail-demand-forecasting"
DATASET = "https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting"

# Same colours as the README figures: actual is ink, LightGBM blue, best baseline orange.
INK = "#0b0b0b"
INK_SECONDARY = "#55534e"
BORDER = "#e4e2dc"
GRID = "#ecebe6"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1a8a5a"
COLOURS = {"Actual": INK, "LightGBM": BLUE, "baseline": ORANGE}

MODEL_ORDER = ["Last day", "Same weekday last week", "Mean of last 4 same weekdays", "LightGBM"]
SEGMENT_ORDER = ["perishable", "non-perishable", "high volume", "medium volume", "low volume"]

CSS = f"""
<style>
[data-testid="stMainBlockContainer"] {{ max-width: 1180px; padding-top: 2rem; }}
.hero {{ background-color: #17352b; background-size: cover; background-position: center 45%;
  border-radius: 4px; overflow: hidden; margin-bottom: 1.5rem; }}
.hero.has-photo .hero-shade {{ background: rgba(11, 11, 11, 0.58); min-height: 360px; }}
.hero-shade {{ padding: 56px 48px 44px; min-height: 300px;
  display: flex; flex-direction: column; justify-content: flex-end; }}
.hero .eyebrow {{ color: #ffd9c7; }}
.hero-title {{ color: #ffffff; font-size: 2.7rem; font-weight: 700; line-height: 1.1;
  max-width: 760px; margin: 0.4rem 0 0.8rem; }}
.hero-lede {{ color: #f2f1ed; font-size: 1.1rem; max-width: 680px; line-height: 1.5;
  margin: 0 0 1.4rem; }}
.hero-links a {{ display: inline-block; padding: 9px 16px; margin-right: 10px; border-radius: 4px;
  font-weight: 600; font-size: 0.92rem; text-decoration: none; }}
.hero-links a.primary {{ background: #ffffff; color: {INK}; }}
.hero-links a.secondary {{ border: 1px solid rgba(255, 255, 255, 0.7); color: #ffffff; }}
.eyebrow {{ font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase; color: {INK_SECONDARY}; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px;
  margin: 0 0 2rem; }}
.kpi {{ background: #ffffff; border: 1px solid {BORDER}; border-top: 4px solid var(--accent);
  border-radius: 4px; padding: 18px 20px; }}
.kpi-value {{ font-family: 'JetBrains Mono', monospace; font-size: 2.1rem; font-weight: 600;
  color: var(--accent); line-height: 1.1; }}
.kpi-label {{ font-weight: 600; font-size: 0.98rem; margin-top: 6px; color: {INK}; }}
.kpi-note {{ font-size: 0.86rem; color: {INK_SECONDARY}; margin-top: 4px; line-height: 1.4; }}
.section {{ margin: 2.6rem 0 0.8rem; padding-top: 1.4rem; border-top: 1px solid {BORDER}; }}
.section-title {{ font-size: 1.65rem; font-weight: 700; margin: 0.3rem 0 0.4rem; color: {INK}; }}
.section-lede {{ font-size: 1.02rem; color: {INK_SECONDARY}; max-width: 820px; line-height: 1.55; }}
.note {{ background: #ffffff; border: 1px solid {BORDER}; border-left: 4px solid var(--accent);
  border-radius: 4px; padding: 12px 16px; font-size: 0.93rem; line-height: 1.5;
  margin: 0.6rem 0 1rem; }}
.legend-swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px;
  margin: 0 6px -1px 0; }}
.footer {{ margin-top: 3rem; padding: 1.4rem 0 2rem; border-top: 1px solid {BORDER};
  font-size: 0.9rem; color: {INK_SECONDARY}; line-height: 1.6; }}
.footer a {{ color: {BLUE}; }}
@media (max-width: 640px) {{
  .hero-shade {{ padding: 32px 20px 26px; min-height: 300px; }}
  .hero-title {{ font-size: 1.85rem; }}
  .hero-lede {{ font-size: 1rem; }}
  .hero-links a {{ margin-bottom: 8px; }}
}}
</style>
"""


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
        "waste": pd.read_csv(REPORTS / "inventory_perishable_waste.csv"),
        "stock": pd.read_csv(REPORTS / "inventory_non_perishable_stock.csv"),
        "wins": pd.read_csv(REPORTS / "series_wins.csv"),
        "importance": pd.read_csv(REPORTS / "feature_importance.csv"),
    }


@st.cache_data
def image_data_uri(path: Path) -> str:
    """A JPEG as a data URI, so CSS can use it as a background."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def best_baseline(overall: pd.DataFrame) -> str:
    """Baseline with the lowest validation WAPE."""
    validation = overall[(overall["period"] == "validation") & (overall["model"] != "LightGBM")]
    return str(validation.sort_values("WAPE").iloc[0]["model"])


def change_at(table: pd.DataFrame, value: str, baseline: str, fill_rate: float) -> float:
    """LightGBM's relative change against the baseline at one matched fill rate."""
    rows = table[table["fill_rate"].round(4) == round(fill_rate, 4)].set_index("model")[value]
    return float(rows["LightGBM"] / rows[baseline] - 1)


def html(markup: str) -> None:
    """Render raw HTML. Lines are kept unindented so markdown never reads them as code."""
    st.markdown(markup, unsafe_allow_html=True)


def kpi(value: str, label: str, note: str, accent: str) -> str:
    """One headline number card."""
    return (
        f'<div class="kpi" style="--accent:{accent}">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-note">{note}</div></div>'
    )


def section(eyebrow: str, title: str, lede: str) -> None:
    """Numbered section header with a one-paragraph intro."""
    html(
        f'<div class="section"><div class="eyebrow">{eyebrow}</div>'
        f'<div class="section-title">{title}</div>'
        f'<div class="section-lede">{lede}</div></div>'
    )


def note(text: str, accent: str = BLUE) -> None:
    """A callout box with a coloured left edge."""
    html(f'<div class="note" style="--accent:{accent}">{text}</div>')


@alt.theme.register("dashboard", enable=True)
def dashboard_theme() -> alt.theme.ThemeConfig:
    """Shared chart look: DM Sans, quiet grid, no frame."""
    return {
        "config": {
            "font": "DM Sans",
            "background": "transparent",
            "view": {"stroke": None},
            "axis": {
                "labelColor": INK_SECONDARY,
                "titleColor": INK_SECONDARY,
                "gridColor": GRID,
                "domainColor": BORDER,
                "tickColor": BORDER,
                "labelFontSize": 12,
                "titleFontSize": 12,
                "titleFontWeight": 500,
            },
            "legend": {"labelFontSize": 13, "labelColor": INK, "symbolStrokeWidth": 3},
        }
    }


def show(chart: alt.Chart | alt.LayerChart) -> None:
    """Render with the dashboard theme instead of Streamlit's default chart theme."""
    st.altair_chart(chart, width="stretch", theme=None)


def highlight_lightgbm(row: pd.Series) -> list[str]:
    """Tint the LightGBM row in a results table."""
    tint = "background-color: #eaf2fc; font-weight: 600" if row.name == "LightGBM" else ""
    return [tint] * len(row)


def matched_table(table: pd.DataFrame, value: str) -> pd.DataFrame:
    """Models as rows, target fill rates as columns, formatted for display."""
    out = table.copy()
    out["fill_rate"] = (out["fill_rate"] * 100).map("{:.1f}%".format)
    wide = out.pivot(index="model", columns="fill_rate", values=value).loc[MODEL_ORDER]
    return wide.map(lambda v: "not reached" if pd.isna(v) else f"{v:,.0f}")


st.set_page_config(page_title="Retail Demand Forecasting", page_icon="🛒", layout="wide")
html(CSS)
data = load()
baseline = best_baseline(data["overall"])
test = data["overall"][data["overall"]["period"] == "test"].set_index("model")
model_wape = float(test.loc["LightGBM", "WAPE"])
baseline_wape = float(test.loc[baseline, "WAPE"])
reduction = (baseline_wape - model_wape) / baseline_wape
wins = data["wins"].iloc[0]
stock_change = change_at(data["stock"], "avg_leftover_stock_non_perishable", baseline, 0.98)
waste_change = change_at(data["waste"], "wasted_units_perishable", baseline, 0.99)

# Hero
hero_attrs = 'class="hero"'
if HERO_PHOTO.exists():
    photo_uri = image_data_uri(HERO_PHOTO)
    hero_attrs = f'class="hero has-photo" style="background-image:url({photo_uri})"'
html(
    f"<div {hero_attrs}>"
    '<div class="hero-shade">'
    '<div class="eyebrow">Retail demand forecasting &middot; 125M rows of real sales</div>'
    '<div class="hero-title">Forecasting grocery demand for 54 supermarkets</div>'
    '<div class="hero-lede">Every Tuesday, a 7-day forecast for every store and product family. '
    "Scored on 8 weeks the model never saw, then tested on the question that matters to a store: "
    "does a better forecast mean less stock left on the shelf?</div>"
    '<div class="hero-links">'
    f'<a class="primary" href="{REPO}" target="_blank">View the code on GitHub</a>'
    f'<a class="secondary" href="{DATASET}" target="_blank">About the data</a>'
    "</div></div></div>"
)

# Headline numbers
html(
    '<div class="kpis">'
    + kpi(
        f"{model_wape:.1%}",
        "Forecast error (WAPE)",
        f"Against {baseline_wape:.1%} for the best simple rule a planner could use.",
        BLUE,
    )
    + kpi(
        f"{reduction:.0%}",
        "Less error than the best rule",
        f"Best rule: {baseline.lower()}.",
        BLUE,
    )
    + kpi(
        f"{wins['share_won']:.0%}",
        "Of store-family series improved",
        f"{int(wins['n_series']):,} series with sales in the test weeks.",
        GREEN,
    )
    + kpi(
        f"{-stock_change:.0%}",
        "Less unsold dry stock",
        "At the same 98% non-perishable fill rate, in a simplified simulation.",
        ORANGE,
    )
    + "</div>"
)

# Plain-English summary
if SUMMARY_PHOTO.exists():
    photo, text = st.columns([2, 3], gap="large", vertical_alignment="center")
    photo.image(str(SUMMARY_PHOTO), width="stretch")
else:
    text = st.container()
with text:
    html('<div class="eyebrow">In plain English</div>')
    st.markdown(
        f"""
A supermarket orders stock every week. Order too little and shelves go empty. Order too much
and fresh food is thrown away. This project measures how much a machine learning model
(LightGBM) improves on the simple rules a planner could run in a spreadsheet, and what that
improvement means for stock.

**WAPE** is the total forecast miss as a share of total sales. A WAPE of {model_wape:.1%}
means that for every 100 units sold, the forecasts were off by about
{model_wape * 100:.0f} units in total.

**The honest version:** the model clearly beats the simple rules on accuracy. In the stock
simulation the gain is large for dry goods and small for fresh food, which the simple rule
already forecasts well. The stock figures are a simplified look-back, not a measured saving.
"""
    )

# 01 Forecasts
section(
    "01 &middot; Forecasts",
    "Forecast against actual sales",
    "Pick any store and product family to see the 8 test weeks. Each line is a separate "
    "7-day forecast made on a Tuesday, using only data up to that day.",
)
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
        alt.Chart(long, mark=alt.MarkDef(type="line", strokeWidth=2.5))
        .encode(
            x=alt.X("target_date:T", title=None),
            y=alt.Y("units:Q", title="Units sold per day"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=domain, range=[INK, BLUE, ORANGE]),
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
        .properties(height=360)
    )
    show(chart)
    actual = series["target"].sum()
    errors = {
        name: (series[name] - series["target"]).abs().sum() / actual
        for name in ["LightGBM", baseline]
    }
    model_won = errors["LightGBM"] < errors[baseline]
    verdict = (
        "LightGBM wins on this series."
        if model_won
        else f"The simple rule wins on this series. The model wins on {wins['share_won']:.0%} "
        "of series, not all of them, and this example is left as the default on purpose."
    )
    note(
        f"WAPE for this series: <b>LightGBM {errors['LightGBM']:.1%}</b>, "
        f"<b>{baseline.lower()} {errors[baseline]:.1%}</b>. {verdict}",
        BLUE if model_won else ORANGE,
    )

# 02 Accuracy
section(
    "02 &middot; Accuracy",
    "How wrong is each model?",
    "All four models are scored on exactly the same rows. Lower is better for every error "
    "metric. Bias shows whether a model over-forecasts (+) or under-forecasts (-) in total.",
)
st.dataframe(
    test.drop(columns="period")
    .loc[MODEL_ORDER]
    .rename_axis("Model")
    .style.apply(highlight_lightgbm, axis=1)
    .format(
        {"MAE": "{:.1f}", "RMSE": "{:.1f}", "WAPE": "{:.1%}", "RMSLE": "{:.3f}", "Bias": "{:+.1%}"}
    ),
    width="stretch",
)

segments = data["segments"][data["segments"]["model"].isin([baseline, "LightGBM"])].copy()
segment_bars = (
    alt.Chart(segments, mark=alt.MarkDef(type="bar", cornerRadiusEnd=2, height=14))
    .encode(
        y=alt.Y("segment:N", sort=SEGMENT_ORDER, title=None, axis=alt.Axis(labelFontSize=13)),
        yOffset=alt.YOffset("model:N", sort=[baseline, "LightGBM"]),
        x=alt.X(
            "WAPE:Q",
            title="Test WAPE (lower is better)",
            axis=alt.Axis(format="%", tickCount=7),
            scale=alt.Scale(domain=[0, float(segments["WAPE"].max()) * 1.08]),
        ),
        color=alt.Color(
            "model:N",
            scale=alt.Scale(domain=[baseline, "LightGBM"], range=[ORANGE, BLUE]),
            legend=alt.Legend(title=None, orient="top", labelLimit=400),
        ),
        tooltip=["segment:N", "model:N", alt.Tooltip("WAPE:Q", format=".1%")],
    )
    .properties(height=340)
)
show(segment_bars)
lgbm_bias = data["segments"][data["segments"]["model"] == "LightGBM"].set_index("segment")["Bias"]
note(
    "Low-volume products, like books or baby care, stay hard for every model. The total bias "
    f"is close to zero, but LightGBM under-forecasts medium-volume series by "
    f"{-lgbm_bias['medium volume']:.0%} and low-volume series by {-lgbm_bias['low volume']:.0%}: "
    "a side effect of training on log(1 + units).",
    ORANGE,
)

# 03 Inventory
section(
    "03 &middot; Stock decisions",
    "Does a better forecast mean less waste?",
    "Each week the simulation stocks up to the forecast plus a safety buffer. Unsold fresh food "
    "is thrown away, unsold dry goods carry over, and empty shelves lose the sale. Models are "
    "compared at the same fill rate (the share of demand actually served), separately for fresh "
    "and dry goods.",
)
html(
    '<div class="kpis">'
    + kpi(
        f"{-stock_change:.0%}",
        "Less unsold dry stock",
        "Non-perishables, at the same 98% fill rate.",
        GREEN,
    )
    + kpi(
        f"{-waste_change:.0%}",
        "Less fresh food thrown away",
        "Perishables, at the same 99% fill rate. Small, because the simple rule already "
        "forecasts fresh food well.",
        BLUE,
    )
    + "</div>"
)
frontier = data["frontier"][data["frontier"]["model"].isin(["LightGBM", baseline])]
curve = (
    alt.Chart(frontier, mark=alt.MarkDef(type="line", point=True, strokeWidth=2.5))
    .encode(
        x=alt.X(
            "fill_rate_perishable:Q",
            title="Perishable fill rate (share of fresh-food demand served)",
            axis=alt.Axis(format=".1%"),
            scale=alt.Scale(domain=[0.975, 1.0]),
        ),
        y=alt.Y(
            "wasted_units_perishable:Q",
            title="Perishable units thrown away, 8 weeks",
            axis=alt.Axis(format="~s"),
        ),
        color=alt.Color(
            "model:N",
            scale=alt.Scale(domain=["LightGBM", baseline], range=[BLUE, ORANGE]),
            legend=alt.Legend(title=None, orient="top", labelLimit=400),
        ),
        tooltip=[
            "model:N",
            alt.Tooltip("z:Q", title="Safety stock z"),
            alt.Tooltip("fill_rate_perishable:Q", format=".1%", title="Perishable fill rate"),
            alt.Tooltip("wasted_units_perishable:Q", format=",.0f", title="Wasted"),
        ],
    )
    .transform_filter(alt.datum.fill_rate_perishable >= 0.975)
    .properties(height=360)
)
show(curve)
note(
    "More safety stock serves more customers but throws away more food. A curve further down "
    "and to the right is better. The curves cross near a 99.5% fill rate, and above it the "
    "simple rule wastes slightly less. The sweep runs on the test weeks, so this is a look-back "
    "comparison, not a measured saving.",
    INK_SECONDARY,
)
with st.expander("Full tables at matched fill rates"):
    left, right = st.columns(2)
    left.markdown("**Perishable units wasted over 8 weeks**")
    left.dataframe(matched_table(data["waste"], "wasted_units_perishable"), width="stretch")
    right.markdown("**Average non-perishable stock left per week**")
    right.dataframe(
        matched_table(data["stock"], "avg_leftover_stock_non_perishable"), width="stretch"
    )

# 04 Drivers
section(
    "04 &middot; What drives the model",
    "Last week's pattern does most of the work",
    "Share of the model's training gain by feature. The same weekday last week and the average "
    "of the last four same weekdays carry most of it. The model's edge comes from combining "
    "them with recent sales levels and the calendar.",
)
importance = data["importance"].head(8).copy()
importance["share"] = importance["gain"] / data["importance"]["gain"].sum()
importance["weekly"] = importance["feature"].str.startswith("same_weekday")
share_x = alt.X(
    "share:Q",
    title="Share of total training gain",
    axis=alt.Axis(format="%"),
    scale=alt.Scale(domain=[0, float(importance["share"].max()) * 1.12]),
)
feature_y = alt.Y(
    "feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300, labelFontSize=13)
)
bars = alt.Chart(importance, mark=alt.MarkDef(type="bar", cornerRadiusEnd=2, height=22)).encode(
    x=share_x,
    y=feature_y,
    color=alt.condition(alt.datum.weekly, alt.value(BLUE), alt.value("#9fc2ec")),
    tooltip=["feature:N", alt.Tooltip("share:Q", format=".1%")],
)
labels = alt.Chart(
    importance,
    mark=alt.MarkDef(type="text", align="left", dx=6, fontSize=12, color=INK_SECONDARY),
).encode(x=share_x, y=feature_y, text=alt.Text("share:Q", format=".1%"))
show(alt.LayerChart(layer=[bars, labels], height=300))

html(
    '<div class="footer">'
    "<b>Data:</b> Corporación Favorita, Ecuador (public Kaggle dataset), January 2013 to "
    "August 2017. Test period: 21 June to 15 August 2017, never used for training or model "
    "choice.<br>"
    "<b>Limits:</b> forecasts are per product family, not per item. The stock simulation has "
    "no lead times or costs and is a look-back comparison. One summer test window.<br>"
    f'<a href="{REPO}" target="_blank">Code, method and full write-up on GitHub</a>'
    "</div>"
)
