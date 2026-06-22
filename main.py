from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

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


@app.get("/")
def home():
    return {"message": "Mandi Price Prediction API Running"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])[FEATURE_COLUMNS]

    prediction = float(model.predict(df)[0])

    return {
        "predicted_price": prediction
    }


@app.post("/forecast")
def forecast(data: dict):

    forecasts = []

    current_data = data.copy()

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


@app.post("/trend")
def trend(data: dict):

    df = pd.DataFrame([data])[FEATURE_COLUMNS]

    predicted_price = float(model.predict(df)[0])

    current_price = float(data["lag_1"])

    if predicted_price > current_price * 1.02:
        trend = "UP"
    elif predicted_price < current_price * 0.98:
        trend = "DOWN"
    else:
        trend = "STABLE"

    return {
        "current_price": current_price,
        "predicted_price": round(predicted_price, 2),
        "trend": trend
    }


@app.post("/confidence")
def confidence(data: dict):

    df = pd.DataFrame([data])[FEATURE_COLUMNS]

    predicted_price = float(model.predict(df)[0])

    volatility = abs(
        float(data["lag_1"]) -
        float(data["rolling_30"])
    )

    confidence_score = max(
        50,
        min(
            95,
            95 - (
                volatility /
                max(float(data["rolling_30"]), 1)
            ) * 100
        )
    )

    return {
        "predicted_price": round(predicted_price, 2),
        "confidence": round(confidence_score, 2)
    }


@app.post("/recommendation")
def recommendation(data: dict):

    df = pd.DataFrame([data])[FEATURE_COLUMNS]

    predicted_price = float(model.predict(df)[0])

    current_price = float(data["lag_1"])

    change_pct = (
        (predicted_price - current_price)
        / current_price
    ) * 100

    if change_pct > 5:
        recommendation = "HOLD"
    elif change_pct < -5:
        recommendation = "SELL_NOW"
    else:
        recommendation = "SELL"

    return {
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "expected_change_percent": round(change_pct, 2),
        "recommendation": recommendation
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
