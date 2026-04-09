"""
predictions.py — LSTM prediction tab with 12-month forecasting.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helpers import yield_category, yield_color, month_name

P = {
    "bg":       "#0d1f14",
    "surface":  "#122a1a",
    "border":   "#1e4029",
    "primary":  "#2ecc71",
    "secondary":"#f0a500",
    "accent":   "#e74c3c",
    "text":     "#e8f5e9",
    "muted":    "#7fad8c",
    "grid":     "#1a3525",
    "card":     "#0f2318",
    "forecast": "#3498db",
    "ci_fill":  "rgba(52,152,219,0.1)",
}


def run_prediction(
    region, crop, soil_texture, month,
    rainfall, temperature, humidity,
    soil_ph, soil_sat, land_size,
    user_id=None,
):
    inputs = {
        "region":          region,
        "crop":            crop,
        "soil_texture":    soil_texture,
        "month":           int(month),
        "rainfall_mm":     float(rainfall),
        "temperature_c":   float(temperature),
        "humidity_pct":    float(humidity),
        "soil_ph":         float(soil_ph),
        "soil_sat_pct":    float(soil_sat),
        "land_size_acres": float(land_size),
    }

    try:
        from models.lstm_model import predict_yield, predict_forecast
        predicted = predict_yield(inputs)
        forecast  = predict_forecast(inputs, horizon=12)
    except Exception as e:
        return (
            f'<div style="color:#e74c3c;padding:20px;">Model not loaded: {e}<br>'
            f'Place your trained model files in <code>models/saved/</code></div>',
            go.Figure(),
        )

    cat   = yield_category(predicted)
    color = yield_color(predicted)

    if user_id:
        try:
            from app.database import save_prediction
            save_prediction(user_id, inputs, predicted, cat)
        except Exception:
            pass

    result_html  = _build_result_html(predicted, cat, color, inputs)
    forecast_fig = _build_forecast_chart(forecast, predicted, inputs)
    return result_html, forecast_fig


def _build_result_html(predicted: float, cat: str, color: str, inputs: dict) -> str:
    pct_bar = min(100, int(predicted / 2.5 * 100))
    stars   = "★" * min(5, max(1, round(predicted / 0.5)))
    empty   = "☆" * (5 - len(stars))
    m_name  = month_name(inputs["month"])

    return f"""
<div style="font-family:'DM Sans','Segoe UI',sans-serif; background:#0f2318;
            border-radius:16px; padding:32px 36px;
            border:1px solid #1e4029; max-width:680px;
            box-shadow:0 8px 32px rgba(0,0,0,0.4);">

  <div style="display:flex; align-items:flex-start; gap:20px; margin-bottom:24px;">
    <div style="font-size:52px; line-height:1;">🌾</div>
    <div style="flex:1;">
      <div style="font-size:11px; color:#7fad8c; letter-spacing:2px;
                  text-transform:uppercase; margin-bottom:6px;">
        LSTM PREDICTION &nbsp;·&nbsp; {inputs['region']} &nbsp;·&nbsp; {inputs['crop']} &nbsp;·&nbsp; {m_name}
      </div>
      <div style="font-size:48px; font-weight:700; color:{color}; line-height:1.05;
                  font-family:'Georgia',serif;">
        {predicted:.3f}
        <span style="font-size:20px; font-weight:400; color:#7fad8c;"> tons / acre</span>
      </div>
      <div style="margin-top:8px; font-size:20px; color:{color}; letter-spacing:4px;">
        {stars}<span style="color:#2a4a35;">{empty}</span>
      </div>
    </div>
  </div>

  <div style="background:rgba(255,255,255,0.03); border-radius:12px; padding:18px 22px;
              border-left:4px solid {color}; margin-bottom:20px;">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <div style="font-size:14px; color:#b2d8bc;">
        Yield Category: <strong style="color:{color}; font-size:16px;">{cat}</strong>
      </div>
      <div style="font-size:12px; color:#7fad8c;">{pct_bar}% of peak (2.5 t/acre)</div>
    </div>
    <div style="margin-top:12px; background:#0d1f14; border-radius:20px;
                height:8px; overflow:hidden;">
      <div style="width:{pct_bar}%; height:100%; background:linear-gradient(90deg,{color}88,{color});
                  border-radius:20px;"></div>
    </div>
  </div>

  <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:10px;
              font-size:12px;">
    {"".join([
      f'<div style="background:#122a1a; padding:12px 14px; border-radius:10px; '
      f'border:1px solid #1e4029; text-align:center;">'
      f'<div style="color:#7fad8c; margin-bottom:4px;">{label}</div>'
      f'<strong style="color:#e8f5e9; font-size:14px;">{value}</strong></div>'
      for label, value in [
          ("Rainfall", f"{inputs['rainfall_mm']:.0f} mm"),
          ("Temp", f"{inputs['temperature_c']:.1f}°C"),
          ("Soil pH", f"{inputs['soil_ph']:.2f}"),
          ("Humidity", f"{inputs['humidity_pct']:.0f}%"),
      ]
    ])}
  </div>

  <div style="margin-top:20px; font-size:10px; color:#3a5a44; text-align:right;
              letter-spacing:1px; text-transform:uppercase;">
    Bidirectional LSTM · Kenya Crop Yield Intelligence System
  </div>
</div>
"""


def _build_forecast_chart(forecast: list, current_yield: float, inputs: dict) -> go.Figure:
    if not forecast:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor=P["surface"], paper_bgcolor=P["card"],
            font=dict(color=P["text"]), height=500,
        )
        fig.add_annotation(text="Forecast unavailable", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=P["muted"]))
        return fig

    labels = [f[0] for f in forecast]
    values = [f[1] for f in forecast]
    upper  = [v * 1.10 for v in values]
    lower  = [v * 0.90 for v in values]

    mom = [0.0] + [
        (values[i] - values[i-1]) / max(values[i-1], 0.001) * 100
        for i in range(1, len(values))
    ]

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.72, 0.28],
        shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=(
            f"12-Month Yield Forecast — {inputs['region']} · {inputs['crop']}",
            "Month-over-Month Change (%)",
        ),
    )

    # CI band
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1],
        y=upper + lower[::-1],
        fill="toself", fillcolor=P["ci_fill"],
        line=dict(color="rgba(0,0,0,0)"),
        name="±10% CI", showlegend=True,
    ), row=1, col=1)

    # Forecast line
    fig.add_trace(go.Scatter(
        x=labels, y=values, mode="lines+markers",
        line=dict(color=P["forecast"], width=2.5),
        marker=dict(
            size=9, color=values,
            colorscale=[[0,"#e74c3c"],[0.4,"#f39c12"],[0.7,"#2ecc71"],[1,"#1a9f5a"]],
            cmin=0, cmax=2.5,
            line=dict(color=P["bg"], width=1.5),
        ),
        name="Predicted Yield",
        hovertemplate="<b>%{x}</b><br>Yield: %{y:.3f} t/acre<extra></extra>",
    ), row=1, col=1)

    # Current month
    fig.add_trace(go.Scatter(
        x=[labels[0]], y=[current_yield],
        mode="markers",
        marker=dict(size=14, color=P["secondary"], symbol="star",
                    line=dict(color=P["bg"], width=2)),
        name="Current",
        hovertemplate=f"Current: {current_yield:.3f} t/acre<extra></extra>",
    ), row=1, col=1)

    # MoM bars
    fig.add_trace(go.Bar(
        x=labels, y=mom,
        marker_color=["#27ae60" if v >= 0 else "#e74c3c" for v in mom],
        marker_line_width=0,
        name="MoM %", showlegend=False,
        hovertemplate="<b>%{x}</b><br>Change: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    # Peak/trough annotations
    peak_i   = int(np.argmax(values))
    trough_i = int(np.argmin(values))
    for idx, txt, col in [
        (peak_i,   f"▲ Peak {values[peak_i]:.3f}",   "#27ae60"),
        (trough_i, f"▼ Trough {values[trough_i]:.3f}", "#e74c3c"),
    ]:
        fig.add_annotation(
            x=labels[idx], y=upper[idx] + 0.06,
            text=txt, showarrow=False,
            font=dict(color=col, size=11),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor=col, borderwidth=1, borderpad=4,
        )

    fig.update_layout(
        height=560,
        plot_bgcolor=P["surface"], paper_bgcolor=P["card"],
        font=dict(family="'DM Sans','Segoe UI',sans-serif", color=P["text"], size=11),
        legend=dict(orientation="h", y=-0.05, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=60, r=40, t=70, b=60),
        hovermode="x unified",
    )
    for row in [1, 2]:
        fig.update_xaxes(gridcolor=P["grid"], row=row, col=1)
        fig.update_yaxes(gridcolor=P["grid"], row=row, col=1)
    fig.update_yaxes(title_text="Yield (t/acre)", row=1, col=1)
    fig.update_yaxes(title_text="Δ %", row=2, col=1)

    return fig
