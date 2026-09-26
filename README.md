# 🌾 AgroAid Mandi Price Prediction API — v2.0

Production-ready FastAPI backend powered by a trained **LightGBM** model.  
Predicts agricultural commodity modal prices using historical mandi data from Supabase.

---

## Model

| Property | Value |
|----------|-------|
| Algorithm | LightGBM (LGBMRegressor) |
| File | `mandi_lightgbm_model.pkl` |
| Test MAE | 164.17 |
| Test RMSE | 445.81 |
| Test R² | 0.9675 |
| Target | `Modal_Price` |

---

## Features (exact order required by model)

| # | Feature | Source |
|---|---------|--------|
| 1 | `District_Code` | `district_mapping.json` |
| 2 | `Commodity_Code` | `commodity_mapping.json` |
| 3 | `Min_Price` | Supabase RPC |
| 4 | `Max_Price` | Supabase RPC |
| 5 | `lag_1` | Supabase RPC |
| 6 | `lag_7` | Supabase RPC |
| 7 | `lag_30` | Supabase RPC |
| 8 | `rolling_7` | Supabase RPC |
| 9 | `rolling_30` | Supabase RPC |

---

## Project Structure

```
mandi-prize/
├── main.py                    # FastAPI application
├── mandi_lightgbm_model.pkl   # Trained LightGBM model
├── feature_columns.json       # Feature column names & order
├── district_mapping.json      # District name → District_Code
├── commodity_mapping.json     # Commodity name → Commodity_Code
│
├── services/
│   ├── __init__.py
│   ├── model_service.py       # Model loading & predict()
│   └── supabase_service.py    # Supabase RPC feature fetching
│
├── schemas/
│   ├── __init__.py
│   └── prediction.py          # Pydantic request/response models
│
├── requirements.txt
├── render.yaml                # Render deployment config
├── .env.example               # Environment variable template
├── .gitignore
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status |
| GET | `/health` | Health check + model status |
| POST | `/predict` | Single-day modal price prediction |
| POST | `/forecast` | Rolling 7-day forecast |
| POST | `/trend` | Price trend (UP/STABLE/DOWN) |
| POST | `/confidence` | Confidence score (50–95) |
| POST | `/recommendation` | Sell/Hold recommendation |

### POST /predict — Request

```json
{
  "state": "Madhya Pradesh",
  "district": "Bhopal",
  "commodity": "Wheat"
}
```

### POST /predict — Response

```json
{
  "state": "Madhya Pradesh",
  "district": "Bhopal",
  "commodity": "Wheat",
  "predicted_price": 2347.85
}
```

---

## Environment Variables

Copy `.env.example` → `.env` and fill in your values:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-or-service-role-key
FRONTEND_URL=https://your-frontend-domain.com
```

> ⚠️ Never commit `.env` to Git.

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env with your Supabase credentials
cp .env.example .env

# Run the API
uvicorn main:app --reload

# Open Swagger UI
http://localhost:8000/docs
```

---

## Supabase RPC

The API expects a Supabase RPC function named `get_prediction_features` that accepts:

| Parameter | Type |
|-----------|------|
| `p_state` | text |
| `p_district` | text |
| `p_commodity` | text |

And returns a row containing: `Min_Price`, `Max_Price`, `lag_1`, `lag_7`, `lag_30`, `rolling_7`, `rolling_30`.

---

## Deployment on Render

**Build Command:**
```
pip install -r requirements.txt
```

**Start Command:**
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Environment Variables** (set in Render dashboard):
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `FRONTEND_URL`

---

## Files to Push to GitHub

✅ Include:
- `main.py`
- `mandi_lightgbm_model.pkl`
- `feature_columns.json`
- `district_mapping.json`
- `commodity_mapping.json`
- `services/`
- `schemas/`
- `requirements.txt`
- `render.yaml`
- `.env.example`
- `.gitignore`
- `README.md`

❌ Do NOT push:
- `.env`
- `apmc_price_model.pkl`
- `commodity_encoder.pkl`
- `district_encoder.pkl`
- `__pycache__/`
- `.venv/`

---

## Author

Shivam Pandey — AgroAid AI Platform  
*AI-Powered Agricultural Intelligence & Mandi Analytics*
