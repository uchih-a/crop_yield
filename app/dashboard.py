"""
dashboard.py — Dashboard analytics charts and KPI summaries.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Refined dark-green luxury palette ────────────────────────
P = {
    "bg":        "#0d1f14",
    "surface":   "#122a1a",
    "border":    "#1e4029",
    "primary":   "#2ecc71",
    "secondary": "#f0a500",
    "accent":    "#e74c3c",
    "text":      "#e8f5e9",
    "muted":     "#7fad8c",
    "grid":      "#1a3525",
    "card":      "#0f2318",
}

CROP_COLORS = {
    "Coffee":    "#6f4e37",
    "Maize":     "#f5c518",
    "Sugarcane": "#2ecc71",
    "Tea":       "#1a9f5a",
    "Wheat":     "#e67e22",
}

LAYOUT_BASE = dict(
    plot_bgcolor=P["surface"],
    paper_bgcolor=P["card"],
    font=dict(family="'DM Sans', 'Segoe UI', sans-serif", color=P["text"], size=12),
    margin=dict(l=60, r=30, t=60, b=60),
    hovermode="closest",
)


def _get_df() -> pd.DataFrame:
    try:
        from app.database import get_crop_records_df
        df = get_crop_records_df()
        if not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _empty_fig(message: str = "No data loaded yet. Upload your CSV dataset first.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=15, color=P["muted"]),
    )
    fig.update_layout(**LAYOUT_BASE, height=420)
    return fig


# ──────────────────────────────────────────────────────────────
# Chart 1: Rainfall vs Yield
# ──────────────────────────────────────────────────────────────

def chart_rainfall_vs_yield(region_filter="All", crop_filter="All") -> go.Figure:
    df = _get_df()
    if df.empty:
        return _empty_fig()

    if region_filter != "All":
        df = df[df["Region"] == region_filter]
    if crop_filter != "All":
        df = df[df["Crop"] == crop_filter]
    if df.empty:
        return _empty_fig("No data for this selection.")

    z = np.polyfit(df["Rainfall_mm"], df["Past_Yield_tons_acre"], 1)
    p_fn = np.poly1d(z)
    x_line = np.linspace(df["Rainfall_mm"].min(), df["Rainfall_mm"].max(), 300)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Rainfall_mm"], y=df["Past_Yield_tons_acre"],
        mode="markers",
        marker=dict(
            color=df["Temperature_C"], colorscale="Viridis",
            size=5, opacity=0.6,
            colorbar=dict(title="Temp (°C)", thickness=12, titleside="right",
                          tickfont=dict(size=9), bgcolor=P["surface"]),
            line=dict(width=0),
        ),
        text=df.apply(lambda r: (
            f"<b>{r['Region']} · {r['Crop']}</b><br>"
            f"Rainfall: {r['Rainfall_mm']:.0f} mm<br>"
            f"Yield: {r['Past_Yield_tons_acre']:.3f} t/acre<br>"
            f"Temp: {r['Temperature_C']:.1f}°C"
        ), axis=1),
        hoverinfo="text", name="Observations",
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=p_fn(x_line),
        mode="lines",
        line=dict(color=P["secondary"], width=2, dash="dash"),
        name=f"Trend (slope={z[0]:.4f})",
    ))

    corr = df["Rainfall_mm"].corr(df["Past_Yield_tons_acre"])
    fig.add_annotation(
        text=f"<b>r = {corr:.3f}</b>",
        xref="paper", yref="paper", x=0.02, y=0.96,
        showarrow=False, font=dict(size=13, color=P["secondary"]),
        bgcolor="rgba(0,0,0,0.5)", bordercolor=P["secondary"],
        borderwidth=1, borderpad=5,
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Rainfall vs Yield — coloured by Temperature",
                   font=dict(size=15, color=P["primary"]), x=0.02),
        xaxis=dict(title="Rainfall (mm)", gridcolor=P["grid"], gridwidth=1),
        yaxis=dict(title="Yield (tons/acre)", gridcolor=P["grid"], gridwidth=1),
        legend=dict(orientation="h", y=-0.18, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        height=460,
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Chart 2: Average Yield by Region & Crop
# ──────────────────────────────────────────────────────────────

def chart_yield_by_region_crop() -> go.Figure:
    df = _get_df()
    if df.empty:
        return _empty_fig()

    pivot = (
        df.groupby(["Region", "Crop"])["Past_Yield_tons_acre"]
        .mean().reset_index()
        .rename(columns={"Past_Yield_tons_acre": "Avg_Yield"})
    )
    fig = px.bar(
        pivot, x="Region", y="Avg_Yield", color="Crop",
        barmode="group",
        color_discrete_map=CROP_COLORS,
        labels={"Avg_Yield": "Avg Yield (t/acre)"},
        text_auto=".2f",
    )
    fig.update_traces(
        textposition="outside", textfont_size=9,
        marker_line_color=P["bg"], marker_line_width=0.8,
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Average Yield by Region and Crop Type",
                   font=dict(size=15, color=P["primary"]), x=0.02),
        xaxis=dict(gridcolor=P["grid"], tickangle=-30),
        yaxis=dict(gridcolor=P["grid"]),
        legend=dict(title="Crop", x=1.01, y=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=60, r=130, t=60, b=80),
        height=460,
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Chart 3: Soil pH Distribution & Yield
# ──────────────────────────────────────────────────────────────

def chart_soil_ph_yield() -> go.Figure:
    df = _get_df()
    if df.empty:
        return _empty_fig()

    bins = np.arange(4.0, 9.25, 0.25)
    df["pH_bin"] = pd.cut(df["Soil_pH"], bins=bins)
    grouped = df.groupby("pH_bin", observed=False).agg(
        Freq=("Soil_pH", "count"),
        Avg_Yield=("Past_Yield_tons_acre", "mean"),
    ).reset_index()
    grouped["pH_mid"] = grouped["pH_bin"].apply(
        lambda x: (x.left + x.right) / 2 if pd.notna(x) else np.nan)
    grouped = grouped.dropna()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=grouped["pH_mid"], y=grouped["Freq"],
        name="Frequency",
        marker_color="rgba(46, 204, 113, 0.5)",
        marker_line_color=P["primary"], marker_line_width=0.8,
        width=0.2,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=grouped["pH_mid"], y=grouped["Avg_Yield"],
        name="Avg Yield", mode="lines+markers",
        line=dict(color=P["secondary"], width=2.5),
        marker=dict(size=8, color=P["secondary"],
                    line=dict(color=P["bg"], width=1.5)),
    ), secondary_y=True)
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Soil pH Distribution & Yield Relationship",
                   font=dict(size=15, color=P["primary"]), x=0.02),
        legend=dict(orientation="h", y=-0.2, x=0, bgcolor="rgba(0,0,0,0)"),
        height=440,
    )
    fig.update_xaxes(title_text="Soil pH", gridcolor=P["grid"])
    fig.update_yaxes(title_text="Frequency", gridcolor=P["grid"],
                     secondary_y=False, title_font_color=P["primary"])
    fig.update_yaxes(title_text="Avg Yield (t/acre)", secondary_y=True,
                     title_font_color=P["secondary"], showgrid=False)
    return fig


# ──────────────────────────────────────────────────────────────
# Chart 4: Temperature vs Yield by Crop
# ──────────────────────────────────────────────────────────────

def chart_temperature_vs_yield() -> go.Figure:
    df = _get_df()
    if df.empty:
        return _empty_fig()

    fig = go.Figure()
    for crop, color in CROP_COLORS.items():
        sub = df[df["Crop"] == crop]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["Temperature_C"], y=sub["Past_Yield_tons_acre"],
            mode="markers",
            marker=dict(size=5, color=color, opacity=0.55),
            name=crop,
            text=sub.apply(lambda r: (
                f"<b>{crop}</b><br>Temp: {r['Temperature_C']:.1f}°C<br>"
                f"Yield: {r['Past_Yield_tons_acre']:.3f} t/acre"
            ), axis=1),
            hoverinfo="text",
        ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Temperature vs Yield by Crop Type",
                   font=dict(size=15, color=P["primary"]), x=0.02),
        xaxis=dict(title="Temperature (°C)", gridcolor=P["grid"]),
        yaxis=dict(title="Yield (t/acre)", gridcolor=P["grid"]),
        legend=dict(x=1.01, y=1, bgcolor="rgba(0,0,0,0)"),
        height=440,
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Chart 5: Yield distribution violin by crop
# ──────────────────────────────────────────────────────────────

def chart_yield_distribution() -> go.Figure:
    df = _get_df()
    if df.empty:
        return _empty_fig()

    fig = go.Figure()
    for crop, color in CROP_COLORS.items():
        sub = df[df["Crop"] == crop]["Past_Yield_tons_acre"]
        if sub.empty:
            continue
        fig.add_trace(go.Violin(
            y=sub, name=crop,
            box_visible=True, meanline_visible=True,
            fillcolor=color, opacity=0.65,
            line_color=P["bg"], marker_color=color,
        ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Yield Distribution by Crop Type",
                   font=dict(size=15, color=P["primary"]), x=0.02),
        yaxis=dict(title="Yield (t/acre)", gridcolor=P["grid"]),
        xaxis=dict(gridcolor=P["grid"]),
        violingap=0.15, violinmode="overlay",
        height=440,
    )
    return fig


# ──────────────────────────────────────────────────────────────
# KPI Summary
# ──────────────────────────────────────────────────────────────

def get_kpi_summary() -> dict:
    df = _get_df()
    if df.empty:
        return {
            "total_records": "—", "avg_yield": "—",
            "top_region": "—", "top_crop": "—",
            "avg_rainfall": "—", "avg_temp": "—",
            "data_loaded": False,
        }
    return {
        "total_records": f"{len(df):,}",
        "avg_yield":     f"{df['Past_Yield_tons_acre'].mean():.3f} t/acre",
        "top_region":    df.groupby("Region")["Past_Yield_tons_acre"].mean().idxmax(),
        "top_crop":      df.groupby("Crop")["Past_Yield_tons_acre"].mean().idxmax(),
        "avg_rainfall":  f"{df['Rainfall_mm'].mean():.1f} mm",
        "avg_temp":      f"{df['Temperature_C'].mean():.1f}°C",
        "data_loaded":   True,
    }


def get_filter_options() -> tuple:
    df = _get_df()
    if df.empty:
        return ["All"], ["All"]
    regions = ["All"] + sorted(df["Region"].dropna().unique().tolist())
    crops   = ["All"] + sorted(df["Crop"].dropna().unique().tolist())
    return regions, crops
