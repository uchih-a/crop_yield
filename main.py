"""
main.py — Kenya Crop Yield Intelligence System
Luxury Gradio UI with Login/Signup, Dashboard, Predictions, Data Upload, Reports.
"""

import os
import logging

import gradio as gr

from utils.helpers import setup_logging, load_env, yield_category, yield_color, month_name

setup_logging(os.getenv("LOG_LEVEL", "INFO"))
load_env()

logger = logging.getLogger(__name__)

# ── Initialise DB ─────────────────────────────────────────────
logger.info("Initialising database…")
from app.database import init_db
db_ok = init_db()
logger.info(f"DB ready — MySQL={db_ok}")

# ── Load Model ────────────────────────────────────────────────
logger.info("Loading LSTM model…")
from models.lstm_model import load_model
model_ok = load_model()
logger.info(f"Model ready — loaded={model_ok}")

# ── Imports ───────────────────────────────────────────────────
from app.auth       import login_handler, signup_handler, logout_handler
from app.dashboard  import (
    chart_rainfall_vs_yield, chart_yield_by_region_crop,
    chart_soil_ph_yield, chart_temperature_vs_yield,
    chart_yield_distribution, get_kpi_summary, get_filter_options,
)
from app.predictions import run_prediction
from app.reports     import generate_report
from app.data_upload import process_csv_upload, ingest_to_db


# ─────────────────────────────────────────────────────────────
# Theme & CSS
# ─────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
  --bg:       #0a1a10;
  --surface:  #0f2318;
  --surface2: #122a1a;
  --border:   #1e4029;
  --primary:  #2ecc71;
  --secondary:#f0a500;
  --accent:   #e74c3c;
  --text:     #e8f5e9;
  --muted:    #7fad8c;
  --radius:   14px;
}

body, .gradio-container {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
}

/* ── Auth wrapper ── */
.auth-wrap {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: radial-gradient(ellipse at 30% 20%, #0d3320 0%, #0a1a10 60%),
              radial-gradient(ellipse at 70% 80%, #1a2e0d 0%, transparent 50%);
}

.auth-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 48px 52px;
  width: 100%;
  max-width: 480px;
  box-shadow: 0 24px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(46,204,113,0.08);
}

.auth-logo {
  text-align: center;
  font-family: 'DM Serif Display', serif;
  font-size: 32px;
  color: var(--primary);
  margin-bottom: 4px;
  letter-spacing: -0.5px;
}

.auth-sub {
  text-align: center;
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 36px;
  letter-spacing: 0.5px;
}

/* ── Tabs ── */
.tab-nav {
  background: var(--surface) !important;
  border-bottom: 1px solid var(--border) !important;
}

.tab-nav button {
  color: var(--muted) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: 0.3px !important;
  padding: 14px 24px !important;
  border-radius: 0 !important;
  transition: all 0.2s !important;
}

.tab-nav button.selected {
  color: var(--primary) !important;
  border-bottom: 2px solid var(--primary) !important;
  background: transparent !important;
}

/* ── Inputs ── */
input[type=text], input[type=email], input[type=password],
textarea, select, .gr-input {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
  padding: 12px 16px !important;
  transition: border-color 0.2s !important;
}

input:focus, textarea:focus {
  border-color: var(--primary) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(46,204,113,0.12) !important;
}

label, .gr-form label {
  color: var(--muted) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  letter-spacing: 0.5px !important;
  text-transform: uppercase !important;
  margin-bottom: 6px !important;
}

/* ── Buttons ── */
.btn-primary {
  background: linear-gradient(135deg, var(--primary), #27ae60) !important;
  color: #0a1a10 !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  padding: 14px 28px !important;
  cursor: pointer !important;
  transition: all 0.2s !important;
  letter-spacing: 0.3px !important;
  width: 100% !important;
}

.btn-primary:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 24px rgba(46,204,113,0.3) !important;
}

.btn-secondary {
  background: transparent !important;
  color: var(--muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  font-size: 13px !important;
  padding: 10px 20px !important;
  cursor: pointer !important;
  transition: all 0.2s !important;
}

.btn-secondary:hover {
  border-color: var(--primary) !important;
  color: var(--primary) !important;
}

.btn-danger {
  background: rgba(231,76,60,0.15) !important;
  color: #e74c3c !important;
  border: 1px solid rgba(231,76,60,0.3) !important;
  border-radius: 10px !important;
  font-size: 13px !important;
  padding: 10px 20px !important;
  cursor: pointer !important;
}

/* ── KPI cards ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

.kpi-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 24px;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s, border-color 0.2s;
}

.kpi-card:hover {
  transform: translateY(-2px);
  border-color: var(--primary);
}

.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--primary), transparent);
}

.kpi-label {
  font-size: 11px;
  color: var(--muted);
  letter-spacing: 1px;
  text-transform: uppercase;
  margin-bottom: 8px;
}

.kpi-value {
  font-family: 'DM Serif Display', serif;
  font-size: 26px;
  color: var(--text);
  line-height: 1.1;
}

.kpi-sub {
  font-size: 11px;
  color: var(--muted);
  margin-top: 4px;
}

/* ── Navbar ── */
.navbar {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 16px 32px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0;
}

.navbar-brand {
  font-family: 'DM Serif Display', serif;
  font-size: 22px;
  color: var(--primary);
  letter-spacing: -0.3px;
}

.navbar-user {
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 13px;
  color: var(--muted);
}

/* ── Section headers ── */
.section-header {
  font-family: 'DM Serif Display', serif;
  font-size: 22px;
  color: var(--text);
  margin-bottom: 6px;
}

.section-sub {
  font-size: 13px;
  color: var(--muted);
  margin-bottom: 24px;
}

/* ── Upload area ── */
.upload-zone {
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--surface2) !important;
  transition: border-color 0.2s !important;
}

.upload-zone:hover {
  border-color: var(--primary) !important;
}

/* ── Status messages ── */
.status-success {
  background: rgba(46,204,113,0.1);
  border: 1px solid rgba(46,204,113,0.3);
  border-radius: 10px;
  padding: 14px 18px;
  color: var(--primary);
  font-size: 13px;
}

.status-error {
  background: rgba(231,76,60,0.1);
  border: 1px solid rgba(231,76,60,0.3);
  border-radius: 10px;
  padding: 14px 18px;
  color: #e74c3c;
  font-size: 13px;
}

/* ── Slider overrides ── */
.gr-slider input[type=range] {
  accent-color: var(--primary) !important;
}

/* ── Plotly charts ── */
.plot-container {
  border-radius: var(--radius) !important;
  overflow: hidden !important;
  border: 1px solid var(--border) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }

/* ── Mobile responsive ── */
@media (max-width: 768px) {
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .auth-card { padding: 32px 24px; margin: 16px; }
}
"""


def _kpi_html(kpis: dict) -> str:
    if not kpis.get("data_loaded"):
        return """
<div style="background:#0f2318; border:1px dashed #1e4029; border-radius:14px;
            padding:48px; text-align:center;">
  <div style="font-size:40px; margin-bottom:12px;">📊</div>
  <div style="font-family:'DM Serif Display',serif; font-size:20px; color:#e8f5e9;
              margin-bottom:8px;">No Dataset Loaded</div>
  <div style="color:#7fad8c; font-size:14px;">
    Go to <strong style="color:#2ecc71;">Data Upload</strong> tab to load your CSV dataset.
  </div>
</div>"""

    cards = [
        ("Total Records",    kpis["total_records"], "📋", "#2ecc71"),
        ("Average Yield",    kpis["avg_yield"],     "🌾", "#f0a500"),
        ("Top Region",       kpis["top_region"],    "📍", "#3498db"),
        ("Top Crop",         kpis["top_crop"],      "🌱", "#27ae60"),
        ("Avg Rainfall",     kpis["avg_rainfall"],  "🌧️", "#2980b9"),
        ("Avg Temperature",  kpis["avg_temp"],      "🌡️", "#e67e22"),
    ]
    items = ""
    for label, value, icon, color in cards:
        items += f"""
<div style="background:#0f2318; border:1px solid #1e4029; border-radius:14px;
            padding:20px 22px; position:relative; overflow:hidden;
            transition:transform 0.2s;">
  <div style="position:absolute; top:0; left:0; right:0; height:3px;
              background:linear-gradient(90deg,{color},transparent);"></div>
  <div style="font-size:24px; margin-bottom:8px;">{icon}</div>
  <div style="font-size:11px; color:#7fad8c; letter-spacing:1px;
              text-transform:uppercase; margin-bottom:6px;">{label}</div>
  <div style="font-family:'DM Serif Display',serif; font-size:24px;
              color:#e8f5e9; line-height:1.1;">{value}</div>
</div>"""

    return f"""
<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:8px;">
  {items}
</div>"""


# ─────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="Kenya Crop Yield Intelligence",
        theme=gr.themes.Base(
            primary_hue="green",
            neutral_hue="gray",
        ),
    ) as demo:

        state = gr.State({})

        # ── Auth Section ─────────────────────────────────────
        with gr.Column(visible=True, elem_id="auth_section") as auth_col:

            gr.HTML("""
<div class="auth-wrap">
  <div style="width:100%; max-width:500px; padding:24px;">
    <div style="text-align:center; margin-bottom:40px;">
      <div style="font-family:'DM Serif Display',serif; font-size:38px;
                  color:#2ecc71; letter-spacing:-1px; margin-bottom:4px;">
        🌾 CropYield AI
      </div>
      <div style="color:#7fad8c; font-size:14px; letter-spacing:0.5px;">
        Kenya Crop Yield Intelligence System
      </div>
    </div>
  </div>
</div>
""")

            with gr.Tabs(elem_classes="auth-tabs") as auth_tabs:

                # ── Login Tab ──────────────────────────────
                with gr.TabItem("Sign In", id="login_tab"):
                    with gr.Column(elem_classes="auth-card",
                                   scale=1, min_width=400):
                        gr.HTML('<div class="auth-logo">Welcome back</div>'
                                '<div class="auth-sub">Sign in to your account</div>')

                        login_username = gr.Textbox(
                            label="Username or Email",
                            placeholder="Enter your username or email",
                        )
                        login_password = gr.Textbox(
                            label="Password",
                            type="password",
                            placeholder="Enter your password",
                        )
                        login_error = gr.HTML(visible=False)
                        login_btn   = gr.Button("Sign In →",
                                                elem_classes="btn-primary",
                                                variant="primary")

                # ── Signup Tab ─────────────────────────────
                with gr.TabItem("Create Account", id="signup_tab"):
                    with gr.Column(elem_classes="auth-card",
                                   scale=1, min_width=400):
                        gr.HTML('<div class="auth-logo">Get started</div>'
                                '<div class="auth-sub">Create your free account</div>')

                        signup_fname    = gr.Textbox(label="Full Name",
                                                     placeholder="Your full name")
                        signup_username = gr.Textbox(label="Username",
                                                     placeholder="Choose a username (min 3 chars)")
                        signup_email    = gr.Textbox(label="Email Address",
                                                     placeholder="your@email.com")
                        signup_pw       = gr.Textbox(label="Password",
                                                     type="password",
                                                     placeholder="Min 8 characters")
                        signup_pw2      = gr.Textbox(label="Confirm Password",
                                                     type="password",
                                                     placeholder="Re-enter password")
                        signup_error    = gr.HTML(visible=False)
                        signup_btn      = gr.Button("Create Account →",
                                                    elem_classes="btn-primary",
                                                    variant="primary")

        # ── Main App ─────────────────────────────────────────
        with gr.Column(visible=False) as main_col:

            # Navbar
            navbar_html = gr.HTML()

            def _navbar(state):
                uname = state.get("username", "")
                fname = state.get("full_name", uname)
                admin = "· Admin" if state.get("is_admin") else ""
                return f"""
<div class="navbar">
  <div class="navbar-brand">🌾 CropYield <span style="color:#7fad8c;
       font-family:'DM Sans',sans-serif; font-size:16px; font-weight:400;">AI</span></div>
  <div class="navbar-user">
    <span>👤 {fname} {admin}</span>
  </div>
</div>"""

            with gr.Tabs(elem_classes="main-tabs") as main_tabs:

                # ══════════════════════════════════════════════
                # TAB 1 — DASHBOARD
                # ══════════════════════════════════════════════
                with gr.TabItem("📊  Dashboard"):
                    gr.HTML("""
<div style="padding:28px 32px 8px;">
  <div class="section-header">Analytics Dashboard</div>
  <div class="section-sub">Explore historical yield trends, climate relationships, and regional performance.</div>
</div>""")

                    dash_kpi = gr.HTML()

                    with gr.Row():
                        dash_refresh = gr.Button("↻  Refresh Data",
                                                  elem_classes="btn-secondary",
                                                  scale=1)
                        dash_region  = gr.Dropdown(choices=["All"], value="All",
                                                    label="Region Filter", scale=2)
                        dash_crop    = gr.Dropdown(choices=["All"], value="All",
                                                    label="Crop Filter", scale=2)

                    with gr.Row():
                        chart_rf  = gr.Plot(label="Rainfall vs Yield")
                        chart_reg = gr.Plot(label="Yield by Region & Crop")

                    with gr.Row():
                        chart_ph   = gr.Plot(label="Soil pH & Yield")
                        chart_temp = gr.Plot(label="Temperature vs Yield")

                    chart_violin = gr.Plot(label="Yield Distribution by Crop")

                # ══════════════════════════════════════════════
                # TAB 2 — PREDICTIONS
                # ══════════════════════════════════════════════
                with gr.TabItem("🔮  Predict Yield"):
                    gr.HTML("""
<div style="padding:28px 32px 8px;">
  <div class="section-header">Yield Prediction</div>
  <div class="section-sub">Enter field conditions to get an LSTM-powered yield prediction and 12-month forecast.</div>
</div>""")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML('<div style="color:#7fad8c; font-size:12px; letter-spacing:1px; text-transform:uppercase; margin-bottom:16px; padding:0 4px;">Field Information</div>')
                            p_region   = gr.Dropdown(
                                choices=["Central","Coast","Eastern","Nairobi",
                                         "North Eastern","Nyanza","Rift Valley","Western"],
                                value="Central", label="Region")
                            p_crop     = gr.Dropdown(
                                choices=["Coffee","Maize","Sugarcane","Tea","Wheat"],
                                value="Maize", label="Crop Type")
                            p_soil_tex = gr.Dropdown(
                                choices=["Clay","Loam","Sandy","Silt"],
                                value="Loam", label="Soil Texture")
                            p_month    = gr.Slider(1, 12, value=6, step=1,
                                                   label="Month")

                            gr.HTML('<div style="color:#7fad8c; font-size:12px; letter-spacing:1px; text-transform:uppercase; margin:20px 0 16px; padding:0 4px;">Climate & Soil</div>')
                            p_rain  = gr.Slider(0, 400, value=120, step=1,
                                                label="Rainfall (mm)")
                            p_temp  = gr.Slider(10, 40, value=22, step=0.1,
                                                label="Temperature (°C)")
                            p_humid = gr.Slider(20, 100, value=65, step=1,
                                                label="Humidity (%)")
                            p_ph    = gr.Slider(4.0, 9.0, value=6.5, step=0.1,
                                                label="Soil pH")
                            p_sat   = gr.Slider(10, 100, value=55, step=1,
                                                label="Soil Saturation (%)")
                            p_land  = gr.Slider(0.5, 50, value=5, step=0.5,
                                                label="Land Size (acres)")

                            predict_btn = gr.Button("🔮  Run Prediction",
                                                     elem_classes="btn-primary",
                                                     variant="primary")

                        with gr.Column(scale=2):
                            pred_result = gr.HTML(
                                '<div style="color:#7fad8c; text-align:center; '
                                'padding:60px 20px; font-size:14px;">'
                                '← Set conditions and click Run Prediction</div>'
                            )
                            pred_chart = gr.Plot(label="12-Month Forecast")

                # ══════════════════════════════════════════════
                # TAB 3 — DATA UPLOAD
                # ══════════════════════════════════════════════
                with gr.TabItem("📁  Data Upload"):
                    gr.HTML("""
<div style="padding:28px 32px 8px;">
  <div class="section-header">Dataset Management</div>
  <div class="section-sub">Upload your CSV dataset to power the dashboard and improve predictions.</div>
</div>""")

                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.HTML("""
<div style="background:#0f2318; border:1px solid #1e4029; border-radius:14px;
            padding:24px; margin-bottom:16px;">
  <div style="font-size:14px; font-weight:600; color:#e8f5e9; margin-bottom:12px;">
    📋 Expected CSV Columns
  </div>
  <div style="font-family:'Courier New',monospace; font-size:12px; color:#7fad8c;
              line-height:2; column-count:2;">
    Month_Year · Region · Crop<br>
    Soil_Texture · Rainfall_mm<br>
    Temperature_C · Humidity_pct<br>
    Soil_pH · Soil_Saturation_pct<br>
    Land_Size_acres · Past_Yield_tons_acre
  </div>
</div>""")

                            upload_file = gr.File(
                                label="Drop CSV file here or click to browse",
                                file_types=[".csv"],
                                elem_classes="upload-zone",
                            )
                            upload_replace = gr.Checkbox(
                                label="Replace existing dataset (uncheck to append)",
                                value=True,
                            )

                            with gr.Row():
                                validate_btn = gr.Button("✔  Validate CSV",
                                                          elem_classes="btn-secondary")
                                upload_btn   = gr.Button("⬆  Load to Database",
                                                          elem_classes="btn-primary",
                                                          variant="primary")

                            upload_status = gr.Markdown()

                        with gr.Column(scale=1):
                            gr.HTML("""
<div style="background:#0f2318; border:1px solid #1e4029; border-radius:14px; padding:24px;">
  <div style="font-size:14px; font-weight:600; color:#e8f5e9; margin-bottom:16px;">
    💡 Tips
  </div>
  <div style="font-size:13px; color:#7fad8c; line-height:1.8;">
    • Column names are case-insensitive<br>
    • Common aliases are auto-mapped<br>
    • Rows with missing values are dropped<br>
    • Replace mode clears old data first<br>
    • Append mode adds to existing rows<br>
    • Refresh Dashboard after upload<br>
    • Supports files up to ~100k rows
  </div>
</div>""")
                            db_status = gr.HTML()

                # ══════════════════════════════════════════════
                # TAB 4 — PREDICTION HISTORY
                # ══════════════════════════════════════════════
                with gr.TabItem("📜  History"):
                    gr.HTML("""
<div style="padding:28px 32px 8px;">
  <div class="section-header">Prediction History</div>
  <div class="section-sub">All saved predictions for your account.</div>
</div>""")

                    history_refresh = gr.Button("↻  Load History",
                                                 elem_classes="btn-secondary")
                    history_table   = gr.Dataframe(
                        headers=["Date","Region","Crop","Rainfall","Temp",
                                  "Soil pH","Humidity","Predicted Yield","Category"],
                        datatype=["str"]*9,
                        interactive=False,
                        wrap=True,
                    )
                    history_html = gr.HTML()

                # ══════════════════════════════════════════════
                # TAB 5 — REPORTS
                # ══════════════════════════════════════════════
                with gr.TabItem("📄  Reports"):
                    gr.HTML("""
<div style="padding:28px 32px 8px;">
  <div class="section-header">Report Generator</div>
  <div class="section-sub">Generate a comprehensive PDF report with KPIs, charts summary, and prediction history.</div>
</div>""")

                    with gr.Row():
                        rpt_region   = gr.Dropdown(choices=["All"], value="All",
                                                    label="Region Filter")
                        rpt_crop     = gr.Dropdown(choices=["All"], value="All",
                                                    label="Crop Filter")
                        rpt_inc_pred = gr.Checkbox(
                            label="Include prediction history", value=True)

                    rpt_btn    = gr.Button("📄  Generate PDF Report",
                                           elem_classes="btn-primary",
                                           variant="primary")
                    rpt_status = gr.HTML()
                    rpt_file   = gr.File(label="Download Report", visible=False)

                # ══════════════════════════════════════════════
                # TAB 6 — ACCOUNT
                # ══════════════════════════════════════════════
                with gr.TabItem("👤  Account"):
                    gr.HTML("""
<div style="padding:28px 32px 8px;">
  <div class="section-header">Account</div>
</div>""")
                    acct_info = gr.HTML()
                    logout_btn = gr.Button("Sign Out",
                                           elem_classes="btn-danger")

        # ─────────────────────────────────────────────────────
        # EVENT HANDLERS
        # ─────────────────────────────────────────────────────

        def _error_html(msg):
            return f'<div style="background:rgba(231,76,60,0.1); border:1px solid rgba(231,76,60,0.3); border-radius:10px; padding:12px 16px; color:#e74c3c; font-size:13px; margin-top:8px;">{msg}</div>'

        # Login
        login_btn.click(
            fn=login_handler,
            inputs=[login_username, login_password, state],
            outputs=[state, auth_col, main_col, login_error, login_password],
        ).then(
            fn=lambda s: _navbar(s),
            inputs=[state], outputs=[navbar_html],
        ).then(
            fn=lambda s: _acct_html(s),
            inputs=[state], outputs=[acct_info],
        ).then(
            fn=_load_dashboard,
            inputs=[state],
            outputs=[dash_kpi, chart_rf, chart_reg, chart_ph,
                     chart_temp, chart_violin, dash_region, dash_crop,
                     rpt_region, rpt_crop],
        )

        login_username.submit(
            fn=login_handler,
            inputs=[login_username, login_password, state],
            outputs=[state, auth_col, main_col, login_error, login_password],
        )

        # Signup
        signup_btn.click(
            fn=signup_handler,
            inputs=[signup_fname, signup_username, signup_email,
                    signup_pw, signup_pw2, state],
            outputs=[state, auth_col, main_col, signup_error],
        ).then(
            fn=lambda s: _navbar(s),
            inputs=[state], outputs=[navbar_html],
        ).then(
            fn=lambda s: _acct_html(s),
            inputs=[state], outputs=[acct_info],
        ).then(
            fn=_load_dashboard,
            inputs=[state],
            outputs=[dash_kpi, chart_rf, chart_reg, chart_ph,
                     chart_temp, chart_violin, dash_region, dash_crop,
                     rpt_region, rpt_crop],
        )

        # Dashboard refresh
        def _load_dashboard(s):
            kpis = get_kpi_summary()
            regions, crops = get_filter_options()
            return (
                _kpi_html(kpis),
                chart_rainfall_vs_yield(),
                chart_yield_by_region_crop(),
                chart_soil_ph_yield(),
                chart_temperature_vs_yield(),
                chart_yield_distribution(),
                gr.update(choices=regions, value="All"),
                gr.update(choices=crops,   value="All"),
                gr.update(choices=regions, value="All"),
                gr.update(choices=crops,   value="All"),
            )

        dash_refresh.click(
            fn=_load_dashboard,
            inputs=[state],
            outputs=[dash_kpi, chart_rf, chart_reg, chart_ph,
                     chart_temp, chart_violin, dash_region, dash_crop,
                     rpt_region, rpt_crop],
        )

        # Dashboard filters
        def _filter_charts(region, crop):
            return chart_rainfall_vs_yield(region, crop), chart_yield_by_region_crop()

        dash_region.change(
            fn=_filter_charts,
            inputs=[dash_region, dash_crop],
            outputs=[chart_rf, chart_reg],
        )
        dash_crop.change(
            fn=_filter_charts,
            inputs=[dash_region, dash_crop],
            outputs=[chart_rf, chart_reg],
        )

        # Prediction
        def _predict(region, crop, soil_tex, month, rain, temp, humid,
                     ph, sat, land, s):
            uid = s.get("user_id")
            return run_prediction(region, crop, soil_tex, month, rain, temp,
                                  humid, ph, sat, land, user_id=uid)

        predict_btn.click(
            fn=_predict,
            inputs=[p_region, p_crop, p_soil_tex, p_month, p_rain,
                    p_temp, p_humid, p_ph, p_sat, p_land, state],
            outputs=[pred_result, pred_chart],
        )

        # CSV validate
        _validated_df = gr.State(None)

        def _validate(file):
            if file is None:
                return "⚠️ Please select a file first.", None
            status, df = process_csv_upload(file.name)
            return status, df

        validate_btn.click(
            fn=_validate,
            inputs=[upload_file],
            outputs=[upload_status, _validated_df],
        )

        # CSV upload to DB
        def _upload_to_db(file, replace, df_state):
            if file is None:
                return "⚠️ Please select a file first.", _db_status_html()
            if df_state is None:
                status, df = process_csv_upload(file.name)
                if df is None:
                    return status, _db_status_html()
            else:
                df = df_state
            ok, msg = ingest_to_db(df, replace=replace)
            return msg, _db_status_html()

        upload_btn.click(
            fn=_upload_to_db,
            inputs=[upload_file, upload_replace, _validated_df],
            outputs=[upload_status, db_status],
        )

        # History
        def _load_history(s):
            uid      = s.get("user_id")
            is_admin = s.get("is_admin", False)
            if not uid:
                return [], '<div style="color:#7fad8c;padding:20px;">Not logged in.</div>'
            try:
                from app.database import get_predictions_for_user, get_all_predictions
                preds = get_all_predictions(100) if is_admin else get_predictions_for_user(uid, 100)
            except Exception as e:
                return [], f'<div style="color:#e74c3c;">Error: {e}</div>'

            if not preds:
                return [], '<div style="color:#7fad8c; padding:20px; text-align:center;">No predictions yet. Run a prediction first.</div>'

            rows = [[
                p.created_at.strftime("%d/%m/%Y %H:%M"),
                p.region or "—",
                p.crop   or "—",
                f"{p.rainfall_mm:.0f} mm"  if p.rainfall_mm    else "—",
                f"{p.temperature_c:.1f}°C" if p.temperature_c  else "—",
                f"{p.soil_ph:.2f}"         if p.soil_ph        else "—",
                f"{p.humidity_pct:.0f}%"   if p.humidity_pct   else "—",
                f"{p.predicted_yield:.3f}" if p.predicted_yield else "—",
                p.yield_category           or "—",
            ] for p in preds]

            summary = f'<div style="color:#7fad8c; font-size:13px; margin-bottom:8px;">{len(rows)} prediction(s) found</div>'
            return rows, summary

        history_refresh.click(
            fn=_load_history,
            inputs=[state],
            outputs=[history_table, history_html],
        )

        # Reports
        def _gen_report(region, crop, inc_pred, s):
            uid     = s.get("user_id")
            uname   = s.get("username", "unknown")
            fname   = s.get("full_name", uname)
            is_admin= s.get("is_admin", False)
            filepath = generate_report(
                username=uname, full_name=fname,
                region=region, crop=crop,
                include_predictions=inc_pred,
                user_id=uid, is_admin=is_admin,
            )
            if filepath:
                html = '<div style="background:rgba(46,204,113,0.1); border:1px solid rgba(46,204,113,0.3); border-radius:10px; padding:12px 18px; color:#2ecc71; font-size:13px;">✅ Report generated successfully. Click below to download.</div>'
                return html, gr.update(value=filepath, visible=True)
            return '<div style="color:#e74c3c;">❌ Report generation failed.</div>', gr.update(visible=False)

        rpt_btn.click(
            fn=_gen_report,
            inputs=[rpt_region, rpt_crop, rpt_inc_pred, state],
            outputs=[rpt_status, rpt_file],
        )

        # Account info
        def _acct_html(s):
            if not s.get("logged_in"):
                return ""
            admin_badge = ""
            if s.get("is_admin"):
                admin_badge = '<span style="background:rgba(240,165,0,0.2); color:#f0a500; font-size:11px; padding:3px 10px; border-radius:20px; border:1px solid rgba(240,165,0,0.3); margin-left:10px;">Admin</span>'
            return f"""
<div style="background:#0f2318; border:1px solid #1e4029; border-radius:16px;
            padding:32px; max-width:480px; margin:0 auto;">
  <div style="font-size:48px; text-align:center; margin-bottom:16px;">👤</div>
  <div style="text-align:center; font-family:'DM Serif Display',serif; font-size:24px;
              color:#e8f5e9; margin-bottom:4px;">
    {s.get('full_name', s.get('username',''))} {admin_badge}
  </div>
  <div style="text-align:center; color:#7fad8c; font-size:14px; margin-bottom:28px;">
    @{s.get('username','')}
  </div>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; font-size:13px;">
    <div style="background:#122a1a; padding:14px; border-radius:10px; border:1px solid #1e4029;">
      <div style="color:#7fad8c; margin-bottom:4px;">Role</div>
      <strong style="color:#e8f5e9;">{'Administrator' if s.get('is_admin') else 'Analyst'}</strong>
    </div>
    <div style="background:#122a1a; padding:14px; border-radius:10px; border:1px solid #1e4029;">
      <div style="color:#7fad8c; margin-bottom:4px;">Status</div>
      <strong style="color:#2ecc71;">● Active</strong>
    </div>
  </div>
</div>"""

        # Logout
        logout_btn.click(
            fn=logout_handler,
            inputs=[state],
            outputs=[state, auth_col, main_col, login_error],
        )

        def _db_status_html():
            try:
                from app.database import get_db_record_count
                count = get_db_record_count()
                return f'<div style="background:rgba(46,204,113,0.1); border:1px solid rgba(46,204,113,0.3); border-radius:10px; padding:14px 18px; color:#2ecc71; font-size:13px; margin-top:12px;">📊 Database: <strong>{count:,} crop records</strong></div>'
            except Exception:
                return ""

    return demo


if __name__ == "__main__":
    app  = build_ui()
    port = int(os.environ.get("PORT", os.environ.get("APP_PORT", 7860)))
    logger.info(f"Starting app on 0.0.0.0:{port}")
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        favicon_path=None,
    )
