"""
data_upload.py — CSV dataset upload, validation, and DB ingestion.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "Month_Year", "Region", "Crop", "Soil_Texture",
    "Rainfall_mm", "Temperature_C", "Humidity_pct",
    "Soil_pH", "Soil_Saturation_pct", "Land_Size_acres",
    "Past_Yield_tons_acre",
}

COLUMN_ALIASES = {
    # lowercase / common variations → canonical
    "month_year":           "Month_Year",
    "monthyear":            "Month_Year",
    "region":               "Region",
    "crop":                 "Crop",
    "soil_texture":         "Soil_Texture",
    "soiltexture":          "Soil_Texture",
    "rainfall_mm":          "Rainfall_mm",
    "rainfall":             "Rainfall_mm",
    "temperature_c":        "Temperature_C",
    "temperature":          "Temperature_C",
    "temp_c":               "Temperature_C",
    "humidity_pct":         "Humidity_pct",
    "humidity":             "Humidity_pct",
    "soil_ph":              "Soil_pH",
    "soilph":               "Soil_pH",
    "ph":                   "Soil_pH",
    "soil_saturation_pct":  "Soil_Saturation_pct",
    "soil_saturation":      "Soil_Saturation_pct",
    "land_size_acres":      "Land_Size_acres",
    "land_size":            "Land_Size_acres",
    "past_yield_tons_acre": "Past_Yield_tons_acre",
    "past_yield":           "Past_Yield_tons_acre",
    "yield":                "Past_Yield_tons_acre",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    return df.rename(columns=rename_map)


def process_csv_upload(file_path: str) -> tuple[str, pd.DataFrame | None]:
    """
    Read, validate, and return a cleaned DataFrame.
    Returns (status_message, df_or_None).
    """
    if file_path is None:
        return "⚠️ No file selected.", None

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"❌ Could not read CSV: {e}", None

    df = _normalise_columns(df)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return (
            f"❌ Missing columns: {', '.join(sorted(missing))}\n\n"
            f"Expected columns: {', '.join(sorted(REQUIRED_COLUMNS))}",
            None,
        )

    original_len = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    dropped = original_len - len(df)

    # Type coercions
    numeric_cols = [
        "Rainfall_mm", "Temperature_C", "Humidity_pct",
        "Soil_pH", "Soil_Saturation_pct", "Land_Size_acres",
        "Past_Yield_tons_acre",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)

    if df.empty:
        return "❌ No valid rows after cleaning.", None

    n = len(df)
    note = f" ({dropped} rows dropped with missing values)" if dropped else ""
    status = (
        f"✅ Validated: **{n:,} rows**{note}\n\n"
        f"Regions: {', '.join(sorted(df['Region'].unique()))}\n"
        f"Crops: {', '.join(sorted(df['Crop'].unique()))}"
    )
    return status, df


def ingest_to_db(df: pd.DataFrame, replace: bool = True) -> tuple[bool, str]:
    """Write validated DataFrame to crop_records table."""
    from app.database import bulk_insert_records, clear_crop_records

    if replace:
        clear_crop_records()

    records = []
    for _, row in df.iterrows():
        records.append({
            "month_year":           str(row["Month_Year"]),
            "region":               str(row["Region"]),
            "crop":                 str(row["Crop"]),
            "soil_texture":         str(row["Soil_Texture"]),
            "rainfall_mm":          float(row["Rainfall_mm"]),
            "temperature_c":        float(row["Temperature_C"]),
            "humidity_pct":         float(row["Humidity_pct"]),
            "soil_ph":              float(row["Soil_pH"]),
            "soil_saturation_pct":  float(row["Soil_Saturation_pct"]),
            "land_size_acres":      float(row["Land_Size_acres"]),
            "past_yield_tons_acre": float(row["Past_Yield_tons_acre"]),
        })

    success, count = bulk_insert_records(records)
    if success:
        return True, f"✅ **{count:,} records** successfully loaded into the database."
    return False, "❌ Database insertion failed. Check logs for details."
