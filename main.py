"""
AgroAid Mandi Price Prediction API
===================================
Production-ready FastAPI backend using a trained LightGBM model.

Endpoints
---------
GET  /          — API status
GET  /health    — health check (model loaded flag)
POST /predict   — predict modal price for a district + commodity
POST /forecast  — rolling 7-day price forecast
POST /trend     — price trend (UP / STABLE / DOWN)
POST /confidence — confidence score
POST /recommendation — sell/hold recommendation
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Load .env before any service imports so env vars are present
load_dotenv()

from schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
)
from services import model_service, supabase_service

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CORS origins
# ---------------------------------------------------------------------------
_FRONTEND_URL = os.environ.get("FRONTEND_URL", "")
_origins = [o.strip() for o in _FRONTEND_URL.split(",") if o.strip()] if _FRONTEND_URL else ["*"]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AgroAid API starting up …")
    logger.info("Model loaded: %s", model_service.MODEL_LOADED)
    logger.info("CORS origins: %s", _origins)
    yield
    logger.info("AgroAid API shutting down.")


app = FastAPI(
    title="AgroAid Mandi Price Prediction API",
    description=(
        "Predicts agricultural commodity prices using a trained LightGBM model. "
        "Fetches lag/rolling features automatically from Supabase."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/", tags=["Status"])
def root():
    """API liveness probe."""
    return {"message": "AgroAid Mandi Price Prediction API is running"}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health():
    """Returns API health and model status."""
    return HealthResponse(
        status="healthy" if model_service.MODEL_LOADED else "degraded",
        model_loaded=model_service.MODEL_LOADED,
    )


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def _build_and_predict(
    state: str,
    district: str,
    commodity: str,
) -> tuple[float, dict]:
    """
    Shared helper:
    1. Validate district & commodity against mappings.
    2. Fetch features from Supabase.
    3. Assemble full feature dict.
    4. Run model.

    Returns (predicted_price, feature_dict).
    """
    # --- Validate mappings ---
    try:
        district_code = model_service.get_district_code(district)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    try:
        commodity_code = model_service.get_commodity_code(commodity)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # --- Fetch Supabase features ---
    try:
        supabase_features = supabase_service.get_prediction_features(
            state, district, commodity
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Supabase error: {exc}",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    # --- Assemble feature dict (exact order matches model training) ---
    feature_dict = {
        "District_Code":  district_code,
        "Commodity_Code": commodity_code,
        "Min_Price":      supabase_features["Min_Price"],
        "Max_Price":      supabase_features["Max_Price"],
        "lag_1":          supabase_features["lag_1"],
        "lag_7":          supabase_features["lag_7"],
        "lag_30":         supabase_features["lag_30"],
        "rolling_7":      supabase_features["rolling_7"],
        "rolling_30":     supabase_features["rolling_30"],
    }

    # --- Predict ---
    try:
        predicted_price = model_service.predict(feature_dict)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {exc}",
        )
    except (ValueError, Exception) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        )

    return predicted_price, feature_dict


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict today's modal price",
)
def predict(request: PredictionRequest):
    """
    Accepts state, district, and commodity.
    Fetches historical features from Supabase automatically.
    Returns the predicted modal price.
    """
    predicted_price, _ = _build_and_predict(
        request.state, request.district, request.commodity
    )

    return PredictionResponse(
        state=request.state,
        district=request.district,
        commodity=request.commodity,
        predicted_price=round(predicted_price, 2),
    )


# ---------------------------------------------------------------------------
# POST /forecast  (rolling 7-day)
# ---------------------------------------------------------------------------
@app.post("/forecast", tags=["Prediction"], summary="7-day price forecast")
def forecast(request: PredictionRequest):
    """
    Generates a rolling 7-day price forecast starting from today's prediction.
    """
    _, feature_dict = _build_and_predict(
        request.state, request.district, request.commodity
    )

    forecasts = []
    current = feature_dict.copy()

    for day in range(1, 8):
        try:
            price = model_service.predict(current)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Forecast error on day {day}: {exc}",
            )

        forecasts.append({"day": day, "predicted_price": round(price, 2)})

        # Roll lags forward
        current["lag_30"] = current["lag_7"]
        current["lag_7"]  = current["lag_1"]
        current["lag_1"]  = price
        current["rolling_7"]  = (current["rolling_7"]  * 6  + price) / 7
        current["rolling_30"] = (current["rolling_30"] * 29 + price) / 30

    return {
        "state": request.state,
        "district": request.district,
        "commodity": request.commodity,
        "forecast": forecasts,
    }


# ---------------------------------------------------------------------------
# POST /trend
# ---------------------------------------------------------------------------
@app.post("/trend", tags=["Prediction"], summary="Price trend UP/STABLE/DOWN")
def trend(request: PredictionRequest):
    """Returns whether the predicted price is trending UP, STABLE, or DOWN."""
    predicted_price, feature_dict = _build_and_predict(
        request.state, request.district, request.commodity
    )

    current_price = feature_dict["lag_1"]

    if predicted_price > current_price * 1.02:
        trend_label = "UP"
    elif predicted_price < current_price * 0.98:
        trend_label = "DOWN"
    else:
        trend_label = "STABLE"

    return {
        "state": request.state,
        "district": request.district,
        "commodity": request.commodity,
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "trend": trend_label,
    }


# ---------------------------------------------------------------------------
# POST /confidence
# ---------------------------------------------------------------------------
@app.post("/confidence", tags=["Prediction"], summary="Prediction confidence score")
def confidence(request: PredictionRequest):
    """Returns a confidence score (50–95) based on price volatility."""
    predicted_price, feature_dict = _build_and_predict(
        request.state, request.district, request.commodity
    )

    rolling_30 = feature_dict["rolling_30"]
    lag_1      = feature_dict["lag_1"]

    volatility = abs(lag_1 - rolling_30)
    confidence_score = max(
        50,
        min(95, 95 - (volatility / max(rolling_30, 1)) * 100),
    )

    return {
        "state": request.state,
        "district": request.district,
        "commodity": request.commodity,
        "predicted_price": round(predicted_price, 2),
        "confidence": round(confidence_score, 2),
    }


# ---------------------------------------------------------------------------
# POST /recommendation
# ---------------------------------------------------------------------------
@app.post("/recommendation", tags=["Prediction"], summary="Sell/Hold recommendation")
def recommendation(request: PredictionRequest):
    """Returns HOLD, SELL, or SELL_NOW based on expected price change."""
    predicted_price, feature_dict = _build_and_predict(
        request.state, request.district, request.commodity
    )

    current_price = feature_dict["lag_1"]
    change_pct = ((predicted_price - current_price) / max(current_price, 1)) * 100

    if change_pct > 5:
        rec = "HOLD"
    elif change_pct < -5:
        rec = "SELL_NOW"
    else:
        rec = "SELL"

    return {
        "state": request.state,
        "district": request.district,
        "commodity": request.commodity,
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "expected_change_percent": round(change_pct, 2),
        "recommendation": rec,
    }


# ---------------------------------------------------------------------------
# Local dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
