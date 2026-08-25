from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(
    title="🌾 AgroAid Mandi Price Prediction API",
    description="ML-powered API to predict agricultural commodity prices, 7-day forecasts, market trends, confidence scores, and sell/hold recommendations.",
    version="1.0.0"
)

model = joblib.load("apmc_price_model.pkl")

FEATURE_COLUMNS = [
    "District",
    "Commodity",
    "Min_Price",
    "Max_Price",
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_7",
    "rolling_30"
]


class MandiPriceFeatures(BaseModel):
    District: int = Field(
        default=47,
        description="Encoded District numerical ID (e.g., 47 for Rewa, 12 for Bhopal)",
        examples=[47]
    )
    Commodity: int = Field(
        default=58,
        description="Encoded Commodity numerical ID (e.g., 58 for Wheat, 125 for Soybean)",
        examples=[58]
    )
    Min_Price: float = Field(
        default=5000.0,
        description="Current minimum market/mandi price (in Rs./Quintal)",
        examples=[5000.0]
    )
    Max_Price: float = Field(
        default=5500.0,
        description="Current maximum market/mandi price (in Rs./Quintal)",
        examples=[5500.0]
    )
    lag_1: float = Field(
        default=5100.0,
        description="Previous day's (1 day ago) modal price (in Rs./Quintal)",
        examples=[5100.0]
    )
    lag_7: float = Field(
        default=4950.0,
        description="Modal price 7 days ago (in Rs./Quintal)",
        examples=[4950.0]
    )
    lag_30: float = Field(
        default=4800.0,
        description="Modal price 30 days ago (in Rs./Quintal)",
        examples=[4800.0]
    )
    rolling_7: float = Field(
        default=5000.0,
        description="7-day moving average modal price (in Rs./Quintal)",
        examples=[5000.0]
    )
    rolling_30: float = Field(
        default=4900.0,
        description="30-day moving average modal price (in Rs./Quintal)",
        examples=[4900.0]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "District": 47,
                    "Commodity": 58,
                    "Min_Price": 5000.0,
                    "Max_Price": 5500.0,
                    "lag_1": 5100.0,
                    "lag_7": 4950.0,
                    "lag_30": 4800.0,
                    "rolling_7": 5000.0,
                    "rolling_30": 4900.0
                }
            ]
        }
    }


@app.get("/")
def home():
    return {"message": "Mandi Price Prediction API Running"}


@app.post("/predict", summary="Predict Single Mandi Price", description="Predicts commodity modal price based on APMC market features.")
def predict(data: MandiPriceFeatures):
    payload = data.model_dump()
    df = pd.DataFrame([payload])[FEATURE_COLUMNS]
    prediction = float(model.predict(df)[0])
    return {
        "predicted_price": round(prediction, 2)
    }


@app.post("/forecast", summary="7-Day Rolling Price Forecast", description="Generates a 7-day rolling daily modal price forecast.")
def forecast(data: MandiPriceFeatures):
    forecasts = []
    current_data = data.model_dump()

    for day in range(1, 8):
        df = pd.DataFrame([current_data])[FEATURE_COLUMNS]
        prediction = float(model.predict(df)[0])
        forecasts.append({
            "day": day,
            "predicted_price": round(prediction, 2)
        })

        current_data["lag_30"] = current_data["lag_7"]
        current_data["lag_7"] = current_data["lag_1"]
        current_data["lag_1"] = prediction

        current_data["rolling_7"] = (
            current_data["rolling_7"] * 6 + prediction
        ) / 7

        current_data["rolling_30"] = (
            current_data["rolling_30"] * 29 + prediction
        ) / 30

    return {
        "forecast": forecasts
    }


@app.post("/trend", summary="Price Trend Direction", description="Analyzes if commodity price trend is UP, DOWN, or STABLE.")
def trend(data: MandiPriceFeatures):
    payload = data.model_dump()
    df = pd.DataFrame([payload])[FEATURE_COLUMNS]
    predicted_price = float(model.predict(df)[0])
    current_price = float(payload["lag_1"])

    if predicted_price > current_price * 1.02:
        trend_status = "UP"
    elif predicted_price < current_price * 0.98:
        trend_status = "DOWN"
    else:
        trend_status = "STABLE"

    return {
        "current_price": current_price,
        "predicted_price": round(predicted_price, 2),
        "trend": trend_status
    }


@app.post("/confidence", summary="Prediction Confidence Score", description="Calculates prediction confidence score (50-95%) based on market volatility.")
def confidence(data: MandiPriceFeatures):
    payload = data.model_dump()
    df = pd.DataFrame([payload])[FEATURE_COLUMNS]
    predicted_price = float(model.predict(df)[0])

    volatility = abs(
        float(payload["lag_1"]) -
        float(payload["rolling_30"])
    )

    confidence_score = max(
        50.0,
        min(
            95.0,
            95.0 - (
                volatility /
                max(float(payload["rolling_30"]), 1.0)
            ) * 100.0
        )
    )

    return {
        "predicted_price": round(predicted_price, 2),
        "confidence": round(confidence_score, 2)
    }


@app.post("/recommendation", summary="Sell / Hold Recommendation", description="Provides actionable selling advice (HOLD, SELL, SELL_NOW) for farmers.")
def recommendation(data: MandiPriceFeatures):
    payload = data.model_dump()
    df = pd.DataFrame([payload])[FEATURE_COLUMNS]
    predicted_price = float(model.predict(df)[0])
    current_price = float(payload["lag_1"])

    change_pct = (
        (predicted_price - current_price)
        / current_price
    ) * 100.0

    if change_pct > 5.0:
        advice = "HOLD"
    elif change_pct < -5.0:
        advice = "SELL_NOW"
    else:
        advice = "SELL"

    return {
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "expected_change_percent": round(change_pct, 2),
        "recommendation": advice
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
