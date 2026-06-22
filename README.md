# 🌾 AgroAid Mandi Price Prediction API

## Overview

The AgroAid Mandi Price Prediction API is a Machine Learning-powered service that predicts agricultural commodity prices using historical mandi market data.

The API is designed to integrate with the AgroAid platform and provide:

* Real-time price prediction
* 7-day price forecasting
* Market trend analysis
* Confidence scoring
* Sell/Hold recommendations

The model is trained on historical APMC mandi data and deployed using FastAPI and Render.

---

# Project Architecture

```
AgroAid App
      │
      ▼
Supabase Database
      │
      ▼
Feature Generation
      │
      ▼
Mandi Prediction API
      │
      ▼
LightGBM Model
      │
      ▼
Prediction Results
```

---

# Technology Stack

### Backend

* Python
* FastAPI
* Pandas
* NumPy

### Machine Learning

* LightGBM
* Scikit-Learn
* Joblib

### Database

* Supabase PostgreSQL

### Deployment

* Render

---

# Project Structure

```
mandi-price-api/
│
├── main.py
├── requirements.txt
├── district_mapping.json
├── commodity_mapping.json
├── README.md
│
├── mandi_price_model.pkl
│
└── training/
    ├── train_model.ipynb
    ├── clean_apmc_data.csv
    └── feature_engineering.py
```

---

# File Descriptions

## main.py

Main FastAPI application.

Responsibilities:

* Load trained model
* Accept API requests
* Generate predictions
* Generate forecasts
* Calculate trends
* Calculate confidence scores
* Generate recommendations

Available endpoints:

* GET /
* POST /predict
* POST /forecast
* POST /trend
* POST /confidence
* POST /recommendation

---

## mandi_price_model.pkl

Serialized LightGBM model.

Generated after training.

Used for:

* Single day prediction
* Forecast generation

---

## district_mapping.json

District encoder mapping.

Example:

```json
{
  "Rewa": 47,
  "Bhopal": 12
}
```

Purpose:

Convert district names into numerical values used by the model.

---

## commodity_mapping.json

Commodity encoder mapping.

Example:

```json
{
  "Wheat": 58,
  "Soybean": 125
}
```

Purpose:

Convert commodity names into numerical values used by the model.

---

## requirements.txt

Contains all Python dependencies required for deployment.

Example:

```txt
fastapi
uvicorn
pandas
numpy
lightgbm
scikit-learn
joblib
```

---

# Machine Learning Features

The model uses the following engineered features:

| Feature    | Description                 |
| ---------- | --------------------------- |
| District   | Encoded district            |
| Commodity  | Encoded commodity           |
| Min_Price  | Current minimum mandi price |
| Max_Price  | Current maximum mandi price |
| lag_1      | Previous day's modal price  |
| lag_7      | Modal price 7 days ago      |
| lag_30     | Modal price 30 days ago     |
| rolling_7  | 7-day average modal price   |
| rolling_30 | 30-day average modal price  |

---

# Prediction Workflow

## Step 1

User selects:

* State
* District
* Commodity

---

## Step 2

AgroAid fetches feature values from Supabase.

RPC Function:

```sql
get_prediction_features()
```

---

## Step 3

Frontend sends request:

```json
{
  "District": 47,
  "Commodity": 58,
  "Min_Price": 5000,
  "Max_Price": 5500,
  "lag_1": 5100,
  "lag_7": 4950,
  "lag_30": 4800,
  "rolling_7": 5000,
  "rolling_30": 4900
}
```

---

## Step 4

FastAPI loads the model.

```python
model.predict()
```

---

## Step 5

Predicted mandi price is returned.

Example:

```json
{
  "predicted_price": 5247.99
}
```

---

# Forecast API

Endpoint:

```
POST /forecast
```

Generates a rolling 7-day prediction.

Returns:

```json
{
  "forecast": [
    {
      "day": 1,
      "predicted_price": 5247
    }
  ]
}
```

---

# Trend API

Endpoint:

```
POST /trend
```

Possible outputs:

* UP
* DOWN
* STABLE

Example:

```json
{
  "trend": "UP"
}
```

---

# Confidence API

Endpoint:

```
POST /confidence
```

Returns:

```json
{
  "confidence": 87.4
}
```

Range:

```
50 - 95
```

---

# Recommendation API

Endpoint:

```
POST /recommendation
```

Possible outputs:

* HOLD
* SELL
* SELL_NOW

Example:

```json
{
  "recommendation": "HOLD"
}
```

---

# Supabase Integration

Data source:

```sql
mandi_prices
```

Prediction features generated through:

```sql
get_prediction_features()
```

Used to calculate:

* lag_1
* lag_7
* lag_30
* rolling_7
* rolling_30

---

# Local Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
uvicorn main:app --reload
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

# Deployment

Platform:

Render

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

# Future Enhancements

* Multi-market comparison
* Seasonal forecasting
* Weather impact analysis
* AI crop advisory
* Best selling day prediction
* Price alert notifications
* WhatsApp integration
* News-based market intelligence

---

# Author

Shivam Pandey

AgroAid AI Platform

AI-Powered Agricultural Intelligence & Mandi Analytics
