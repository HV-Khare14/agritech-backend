# AgriTech Predictive Platform — Backend

A production-ready FastAPI backend that recommends the mathematically **least-risk
mixed cropping strategy** for Indian farmers using historical environmental data,
XGBoost yield modelling, and Mean-Variance Portfolio Optimisation.

---

## Quick Start

```bash
# 1. Clone / extract the project
cd agritech_backend

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place all CSV data files in ./data/
#    Required:
#      APY__1_.csv
#      District_Rainfall_Normal_0.csv
#      Sub_Division_IMD_2017.csv
#      district_to_subdivision.csv
#      usingnew.csv

# 5. Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The first startup takes **10–30 seconds** while the pipeline loads CSVs,
runs KNN imputation, and trains the XGBoost model. Subsequent requests are
sub-second.

---

## API Endpoints

### `GET /health`
Returns model readiness status and training metrics.

### `POST /api/ml-crop-recommendation`
Main recommendation endpoint.

**Request body:**
```json
{
  "state": "Punjab",
  "district": "Ludhiana",
  "land_area_hectares": 5.0,
  "soil_type": "Sandy loam",
  "current_climate_conditions": "Normal monsoon expected",
  "risk_preference": "balanced"
}
```

**`risk_preference` options:**
| Value | λ (risk aversion) | Effect |
|-------|-------------------|--------|
| `conservative` | 0.75 | Minimise variance; safer but lower yield |
| `balanced` | 0.50 | Equal weight to yield and risk (default) |
| `aggressive` | 0.25 | Maximise yield; higher variance accepted |

**Example response:**
```json
{
  "state": "Punjab",
  "district": "Ludhiana",
  "land_area_hectares": 5.0,
  "soil_type_used": "Sandy loam",
  "optimised_crop_mix": [
    {
      "crop": "Wheat",
      "season": "RABI",
      "percentage": 45.2,
      "area_hectares": 2.26,
      "expected_yield_tonnes": 8.14,
      "yield_ci_lower_tonnes": 6.90,
      "yield_ci_upper_tonnes": 9.38,
      "risk_contribution_pct": 38.5
    },
    {
      "crop": "Rice",
      "season": "KHARIF",
      "percentage": 32.1,
      "area_hectares": 1.61,
      "expected_yield_tonnes": 5.62,
      "yield_ci_lower_tonnes": 4.80,
      "yield_ci_upper_tonnes": 6.44,
      "risk_contribution_pct": 41.2
    },
    {
      "crop": "Maize",
      "season": "KHARIF",
      "percentage": 22.7,
      "area_hectares": 1.14,
      "expected_yield_tonnes": 3.98,
      "yield_ci_lower_tonnes": 3.20,
      "yield_ci_upper_tonnes": 4.76,
      "risk_contribution_pct": 20.3
    }
  ],
  "portfolio_risk_score": 34.8,
  "portfolio_risk_label": "Moderate",
  "total_expected_yield_tonnes": 17.74,
  "data_integrity_score": 78.5,
  "data_integrity_label": "Moderate – some values were imputed; treat with care",
  "model_metadata": {
    "yield_model_r2": 0.823,
    "yield_model_mae": 0.412,
    "training_sample_size": 293535,
    "optimiser_converged": true,
    "optimiser_message": "Optimization terminated successfully"
  },
  "warnings": []
}
```

---

## Project Structure

```
agritech_backend/
├── app/
│   ├── __init__.py
│   ├── main.py            ← FastAPI app, lifespan, endpoints
│   ├── preprocessing.py   ← Data loading, KNN imputation, feature engineering
│   ├── model_pipeline.py  ← XGBoost yield predictor + Climate Risk Scorer
│   ├── optimizer.py       ← Mean-Variance Portfolio Optimisation (SLSQP)
│   └── schemas.py         ← Pydantic request/response models
├── data/
│   ├── APY__1_.csv
│   ├── District_Rainfall_Normal_0.csv
│   ├── Sub_Division_IMD_2017.csv
│   ├── district_to_subdivision.csv
│   └── usingnew.csv
├── requirements.txt
└── README.md
```

---

## ML Architecture

### Model A — XGBoost Yield Predictor
- **Features:** annual/kharif/rabi rainfall, rainfall CV, crop, season, soil type
- **Target:** log₁⁺¹(Yield) — log-transform stabilises the right-skewed distribution
- **Training:** 85/15 train-validation split; XGBoost with histogram tree method
- **Output:** point estimate + 95% confidence interval per crop

### Model B — Climate Risk Scorer
A composite rule-based scorer combining:
- `cv_annual` (40%): Inter-annual rainfall volatility from IMD 1901–2017 data
- `rain_cv` (30%): Within-year rainfall unevenness from district normals
- `raw_score` (30%): Pre-normalised CV-based score
- Soil type drought premium (±3–5 points)

### Portfolio Optimiser — Mean-Variance Optimisation
Solves: **min** λ·wᵀΣw − (1−λ)·wᵀμ subject to Σwᵢ=1, 0.05≤wᵢ≤0.60

- **μ**: predicted yields from Model A
- **Σ**: diagonal covariance from historical yield variance × climate risk multiplier
- **λ**: risk aversion controlled by `risk_preference` parameter
- Solved via `scipy.optimize.minimize(method="SLSQP")`

### Data Integrity Score
Each row tracks whether its values were directly observed or KNN-imputed.
The score propagates to the API response so farmers and agronomists know
how much to trust the recommendation.

---

## Interactive Docs

Once the server is running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
