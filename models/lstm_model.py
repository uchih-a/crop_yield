"""
lstm_model.py — LSTM model loader and inference wrapper.
Drop your trained model files into models/saved/:
  - model_weights.h5  (or model.keras)
  - scalers.pkl        (dict of group scalers)
  - label_encoders.pkl (dict of LabelEncoders)
  - config.json        (feature lists, lookback, etc.)
"""

import os
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

_model        = None
_scalers      = None
_encoders     = None
_config       = None
_model_loaded = False

SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")

MONTHS_SHORT = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]


def load_model():
    global _model, _scalers, _encoders, _config, _model_loaded

    weights_path  = os.path.join(SAVED_DIR, "model_weights.h5")
    keras_path    = os.path.join(SAVED_DIR, "model.keras")
    scalers_path  = os.path.join(SAVED_DIR, "scalers.pkl")
    encoders_path = os.path.join(SAVED_DIR, "label_encoders.pkl")
    config_path   = os.path.join(SAVED_DIR, "config.json")

    # Load config
    if os.path.exists(config_path):
        with open(config_path) as f:
            _config = json.load(f)
        logger.info("Model config loaded.")
    else:
        logger.warning("config.json not found — using defaults.")
        _config = {
            "lookback": 12,
            "temporal_features": [
                "Rainfall_mm", "Temperature_C", "Humidity_pct",
                "Soil_pH", "Soil_Saturation_pct", "Land_Size_acres",
            ],
            "categorical_features": ["Region", "Crop", "Soil_Texture"],
        }

    # Load scalers
    if os.path.exists(scalers_path):
        import joblib
        _scalers = joblib.load(scalers_path)
        logger.info(f"Loaded {len(_scalers)} scalers.")

    # Load encoders
    if os.path.exists(encoders_path):
        import joblib
        _encoders = joblib.load(encoders_path)
        logger.info("Label encoders loaded.")

    # Load model
    try:
        import tensorflow as tf
        if os.path.exists(keras_path):
            _model = tf.keras.models.load_model(keras_path)
            logger.info("Keras model loaded from model.keras")
            _model_loaded = True
        elif os.path.exists(weights_path):
            # Architecture must be reconstructed — users should prefer .keras
            logger.warning(
                "model_weights.h5 found but no architecture. "
                "Save your model with model.save('model.keras') for full loading."
            )
        else:
            logger.warning("No model file found in models/saved/. Predictions will fail.")
    except Exception as e:
        logger.error(f"Model load error: {e}")

    return _model_loaded


def _build_input(inputs: dict) -> np.ndarray:
    """Build model input array from the inputs dict."""
    temporal = _config.get("temporal_features", [
        "Rainfall_mm", "Temperature_C", "Humidity_pct",
        "Soil_pH", "Soil_Saturation_pct", "Land_Size_acres",
    ])

    feat_vec = np.array([
        inputs.get(k.lower().replace(" ", "_"), 0.0)
        for k in temporal
    ], dtype=np.float32)

    # Apply scaler if available
    group_key = f"{inputs.get('region','All')}_{inputs.get('crop','All')}"
    if _scalers and group_key in _scalers:
        feat_vec = _scalers[group_key].transform(feat_vec.reshape(1, -1))[0]
    elif _scalers and "global" in _scalers:
        feat_vec = _scalers["global"].transform(feat_vec.reshape(1, -1))[0]

    lookback = _config.get("lookback", 12)
    # Repeat single timestep to fill lookback window
    seq = np.tile(feat_vec, (lookback, 1))[np.newaxis, :, :]  # (1, lookback, features)
    return seq


def predict_yield(inputs: dict) -> float:
    if not _model_loaded or _model is None:
        raise RuntimeError(
            "Model not loaded. Place model.keras in models/saved/ and restart."
        )
    seq = _build_input(inputs)
    pred = float(_model.predict(seq, verbose=0)[0][0])
    return max(0.0, round(pred, 4))


def predict_forecast(inputs: dict, horizon: int = 12) -> list:
    """Return list of (month_label, yield) tuples for the next `horizon` months."""
    if not _model_loaded or _model is None:
        return []

    results = []
    current_month = inputs.get("month", 1)

    for i in range(horizon):
        m = ((current_month - 1 + i) % 12) + 1
        inp = {**inputs, "month": m}
        try:
            y = predict_yield(inp)
        except Exception:
            y = 0.0
        label = f"{MONTHS_SHORT[m-1]}"
        results.append((label, y))

    return results
