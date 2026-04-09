# 🌾 Kenya Crop Yield Intelligence System

A production-grade Gradio application for crop yield prediction and analytics, powered by a Bidirectional LSTM model.

---

## Features

- **Authentication** — Login & Signup with bcrypt password hashing
- **Dashboard** — 5 interactive Plotly charts, 6 KPI cards, region/crop filters
- **Yield Prediction** — LSTM-powered single prediction + 12-month forecast
- **Data Upload** — Upload your CSV dataset with validation and column auto-mapping
- **Prediction History** — Tabular view of all saved predictions
- **Reports** — Downloadable PDF with KPIs, breakdown tables, and prediction history
- **MySQL + SQLite fallback** — Production MySQL on Railway; SQLite for local dev

---

## Project Structure

```
crop_yield_app/
├── main.py                   # Gradio UI entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── app/
│   ├── database.py           # SQLAlchemy models & CRUD
│   ├── auth.py               # Login / signup handlers
│   ├── dashboard.py          # Chart functions & KPIs
│   ├── predictions.py        # Prediction & forecast logic
│   ├── data_upload.py        # CSV validation & DB ingestion
│   └── reports.py            # PDF report generation
├── models/
│   ├── lstm_model.py         # Model loader & inference
│   └── saved/                # ← Place your model files here
│       ├── model.keras        (trained model)
│       ├── scalers.pkl        (group scalers dict)
│       ├── label_encoders.pkl (optional)
│       └── config.json        (feature config)
└── utils/
    └── helpers.py            # Shared utilities
```

---

## Model Files (Your Colab-Trained Model)

Place these files in `models/saved/`:

| File | Description |
|------|-------------|
| `model.keras` | Full saved Keras model (`model.save('model.keras')`) |
| `scalers.pkl` | Dict of scalers: `{"Region_Crop": scaler, "global": scaler}` |
| `label_encoders.pkl` | Optional LabelEncoders for categorical features |
| `config.json` | Feature names and lookback window |

### Example `config.json`
```json
{
  "lookback": 12,
  "temporal_features": [
    "Rainfall_mm", "Temperature_C", "Humidity_pct",
    "Soil_pH", "Soil_Saturation_pct", "Land_Size_acres"
  ],
  "categorical_features": ["Region", "Crop", "Soil_Texture"]
}
```

### Saving your model in Colab
```python
# Save full model (recommended)
model.save('model.keras')

# Save scalers
import joblib
joblib.dump(scalers_dict, 'scalers.pkl')

# Save config
import json
config = {
    "lookback": 12,
    "temporal_features": ["Rainfall_mm", "Temperature_C", ...]
}
with open('config.json', 'w') as f:
    json.dump(config, f)
```

---

## CSV Dataset Format

Your CSV must include these columns (case-insensitive, common aliases accepted):

```
Month_Year, Region, Crop, Soil_Texture, Rainfall_mm,
Temperature_C, Humidity_pct, Soil_pH, Soil_Saturation_pct,
Land_Size_acres, Past_Yield_tons_acre
```

---

## Local Development

```bash
cp .env.example .env
# Edit .env with your values

docker-compose up --build
# App: http://localhost:7860
```

---

## Railway Deployment

1. Push code to GitHub
2. Create Railway project → connect repo
3. Add MySQL service
4. In your app service → Variables, add all values from `.env.example`
5. Railway auto-injects `PORT` — the app reads it automatically
6. Set `TF_CPP_MIN_LOG_LEVEL=3` to suppress TF noise in logs

---

## Default Admin Account

Created automatically on first run using env vars:
- Username: `ADMIN_USERNAME` (default: `admin`)
- Password: `ADMIN_PASSWORD` (default: `admin123`)

**Change these in your Railway environment variables before deploying.**
