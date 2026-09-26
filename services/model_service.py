"""
Model service — loads the LightGBM model and the feature/mapping config,
and exposes a single predict() function.
"""

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve paths relative to this file so they work on any OS / Render
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent.parent

MODEL_PATH   = _BASE / "mandi_lightgbm_model.pkl"
FEATURES_PATH = _BASE / "feature_columns.json"
DISTRICT_MAP_PATH  = _BASE / "district_mapping.json"
COMMODITY_MAP_PATH = _BASE / "commodity_mapping.json"

# ---------------------------------------------------------------------------
# Load model & config at import time — fail fast if anything is missing
# ---------------------------------------------------------------------------
try:
    _model = joblib.load(MODEL_PATH)
    logger.info("LightGBM model loaded from %s", MODEL_PATH)
    MODEL_LOADED = True
except Exception as exc:
    _model = None
    MODEL_LOADED = False
    logger.error("Failed to load model: %s", exc)

with open(FEATURES_PATH, "r") as f:
    FEATURE_COLUMNS: list[str] = json.load(f)

with open(DISTRICT_MAP_PATH, "r") as f:
    DISTRICT_MAPPING: dict[str, int] = json.load(f)

with open(COMMODITY_MAP_PATH, "r") as f:
    COMMODITY_MAPPING: dict[str, int] = json.load(f)


def get_district_code(district: str) -> int:
    """Return the numeric District_Code for a district name, or raise KeyError."""
    if district not in DISTRICT_MAPPING:
        raise KeyError(
            f"District '{district}' not found in district_mapping.json. "
            f"Available: {sorted(DISTRICT_MAPPING.keys())}"
        )
    return DISTRICT_MAPPING[district]


def get_commodity_code(commodity: str) -> int:
    """Return the numeric Commodity_Code for a commodity name, or raise KeyError."""
    if commodity not in COMMODITY_MAPPING:
        raise KeyError(
            f"Commodity '{commodity}' not found in commodity_mapping.json. "
            f"Available: {sorted(COMMODITY_MAPPING.keys())}"
        )
    return COMMODITY_MAPPING[commodity]


def predict(features: dict) -> float:
    """
    Given a dict with keys matching FEATURE_COLUMNS, run model inference.

    Parameters
    ----------
    features : dict
        Must contain exactly the keys in FEATURE_COLUMNS with numeric values.

    Returns
    -------
    float
        Predicted Modal_Price.
    """
    if not MODEL_LOADED or _model is None:
        raise RuntimeError("Model is not loaded. Check server logs.")

    # Build DataFrame with exact column order required by the model
    try:
        df = pd.DataFrame([features])[FEATURE_COLUMNS]
    except KeyError as exc:
        raise ValueError(f"Missing required feature: {exc}") from exc

    # Ensure numeric types
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    prediction = _model.predict(df)
    return float(prediction[0])
