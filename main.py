from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib

app = FastAPI()

# Load model
model = joblib.load("apmc_price_model.pkl")


@app.get("/")
def home():
    return {"message": "Mandi Price Prediction API Running"}


@app.post("/predict")
def predict(data: dict):
    try:
        columns = [
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

        # Create dataframe from request
        df = pd.DataFrame([data])

        # Validate required columns
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing_cols}"
            )

        # Keep only training columns
        df = df[columns]

        # Predict
        prediction = model.predict(df)[0]

        return {
            "predicted_price": float(prediction)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))