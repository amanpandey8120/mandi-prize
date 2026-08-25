# 🌾 AgroAid Mandi Price Prediction API

An ML-powered service that predicts agricultural commodity prices (mandi prices) using historical APMC market data, built with **FastAPI**, **LightGBM**, and **Scikit-Learn**.

---

## ⚡ Quick Start (Run locally)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python main.py
```
*(The server will start on `http://localhost:8000`)*

### 3. Open Interactive Web Documentation (Swagger UI)
Open your browser and visit:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 📋 API Data Request Format

All prediction and analytics endpoints (`/predict`, `/forecast`, `/trend`, `/confidence`, `/recommendation`) expect a `POST` request with a **JSON body** containing the following 9 numerical fields:

| Field Name | Type | Description | Example Value |
| :--- | :--- | :--- | :--- |
| `District` | Integer | Encoded District numerical ID | `47` |
| `Commodity` | Integer | Encoded Commodity numerical ID | `58` |
| `Min_Price` | Float / Int | Current minimum mandi price (in ₹) | `5000` |
| `Max_Price` | Float / Int | Current maximum mandi price (in ₹) | `5500` |
| `lag_1` | Float / Int | Yesterday's modal price (1 day ago in ₹) | `5100` |
| `lag_7` | Float / Int | Modal price 7 days ago (in ₹) | `4950` |
| `lag_30` | Float / Int | Modal price 30 days ago (in ₹) | `4800` |
| `rolling_7` | Float / Int | 7-day moving average modal price (in ₹) | `5000` |
| `rolling_30` | Float / Int | 30-day moving average modal price (in ₹) | `4900` |

---

## 📦 Copy-Paste Ready Sample JSON Payload

Use this exact JSON body when testing any endpoint:

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

## 🚀 How to Test the API

### Method 1: Using Interactive Browser UI (Recommended)
1. Open `http://localhost:8000/docs` in your browser.
2. Select an endpoint (e.g., `POST /predict`).
3. Click **Try it out**.
4. Paste the sample JSON above into the **Request body** box.
5. Click **Execute**.

### Method 2: Using cURL (Linux / macOS / Git Bash)
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "District": 47,
  "Commodity": 58,
  "Min_Price": 5000,
  "Max_Price": 5500,
  "lag_1": 5100,
  "lag_7": 4950,
  "lag_30": 4800,
  "rolling_7": 5000,
  "rolling_30": 4900
}'
```

### Method 3: Using PowerShell (Windows)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body '{"District": 47, "Commodity": 58, "Min_Price": 5000, "Max_Price": 5500, "lag_1": 5100, "lag_7": 4950, "lag_30": 4800, "rolling_7": 5000, "rolling_30": 4900}'
```

---

## 📡 API Endpoints & Expected Outputs

### 1. `POST /predict`
Predicts the mandi price for a commodity.

* **Sample Response:**
  ```json
  {
    "predicted_price": 5247.99
  }
  ```

---

### 2. `POST /forecast`
Generates a 7-day rolling daily price forecast.

* **Sample Response:**
  ```json
  {
    "forecast": [
      { "day": 1, "predicted_price": 5247.99 },
      { "day": 2, "predicted_price": 5253.12 },
      { "day": 3, "predicted_price": 5260.45 },
      { "day": 4, "predicted_price": 5265.80 },
      { "day": 5, "predicted_price": 5270.15 },
      { "day": 6, "predicted_price": 5275.30 },
      { "day": 7, "predicted_price": 5280.90 }
    ]
  }
  ```

---

### 3. `POST /trend`
Calculates market price direction (`UP`, `DOWN`, or `STABLE`).

* **Sample Response:**
  ```json
  {
    "current_price": 5100,
    "predicted_price": 5247.99,
    "trend": "UP"
  }
  ```

---

### 4. `POST /confidence`
Returns a confidence percentage score (between 50% and 95%) based on market volatility.

* **Sample Response:**
  ```json
  {
    "predicted_price": 5247.99,
    "confidence": 90.92
  }
  ```

---

### 5. `POST /recommendation`
Provides an actionable advice for farmers: `HOLD`, `SELL`, or `SELL_NOW`.

* **Sample Response:**
  ```json
  {
    "current_price": 5100,
    "predicted_price": 5247.99,
    "expected_change_percent": 2.9,
    "recommendation": "SELL"
  }
  ```

---

## 📂 Project Files

```
mandi-prize/
├── main.py                   # FastAPI backend server & route handlers
├── requirements.txt           # Required Python packages
├── apmc_price_model.pkl       # Trained LightGBM model
├── district_encoder.pkl       # LabelEncoder for district names
├── commodity_encoder.pkl      # LabelEncoder for commodity names
├── commodity_mapping.json     # Encoded mapping dictionary for commodities
├── district_mapping.json      # Encoded mapping dictionary for districts
└── README.md                  # Project documentation & API guide
```
