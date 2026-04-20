"""
main.py
=======
AgriTech Predictive Platform – FastAPI Application Entry Point

Architecture
------------
┌────────────────────────────────────────────────────┐
│  POST /api/ml-crop-recommendation                  │
│                                                    │
│  1. Parse + validate request (Pydantic)            │
│  2. preprocessing.get_district_profile()           │
│     ├─ Lookup weather features for district        │
│     ├─ Filter APY crop history                     │
│     └─ Compute per-row data integrity score        │
│  3. optimizer.build_candidates()                   │
│     └─ model_pipeline.YieldPredictor.predict()     │
│        per crop → (yield_hat, ci_lo, ci_hi)        │
│  4. optimizer.CropPortfolioOptimiser.optimise()    │
│     └─ Mean-Variance Optimisation (SLSQP)          │
│  5. Assemble + return CropRecommendationResponse   │
└────────────────────────────────────────────────────┘

Startup / shutdown lifecycle
-----------------------------
On startup the app loads all CSVs, runs imputation, trains the
XGBoost model, and caches everything in module-level state.
This takes ~10-30 s the first time but every subsequent request
is sub-second.

Run
---
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import sys
import time
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Internal modules
from app import preprocessing as prep
from app import model_pipeline as mp
from app import optimizer as opt
from app.schemas import (
    CropRecommendationRequest,
    CropRecommendationResponse,
    CropAllocation,
    ModelMetadata,
    HealthResponse,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application state (populated during lifespan startup)
# ---------------------------------------------------------------------------
class AppState:
    apy              = None   # pd.DataFrame
    district_weather = None   # pd.DataFrame
    imd_volatility   = None   # pd.DataFrame
    soil             = None   # pd.DataFrame
    ready            = False
    error            = None   # str – last training error if any


state = AppState()


# ---------------------------------------------------------------------------
# Background model training – runs in a separate thread so the HTTP
# port opens immediately (required by Render's port scanner)
# ---------------------------------------------------------------------------

def _train_models_background():
    """Load data and train models in a background thread."""
    t0 = time.time()
    logger.info("=== AgriTech platform: background model training starting ===")

    try:
        # Step 1 – Build master dataset (load + impute + feature-engineer)
        logger.info("Step 1/2: Building master dataset …")
        (state.apy,
         state.district_weather,
         state.imd_volatility,
         state.soil) = prep.build_master_dataset()

        # Step 2 – Train ML models
        logger.info("Step 2/2: Training ML models …")
        mp.train_all_models(state.apy, state.district_weather)

        state.ready = True
        elapsed = round(time.time() - t0, 1)
        logger.info("=== Platform ready in %.1f s ===", elapsed)

    except Exception as exc:
        import traceback
        state.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.exception("FATAL: model training failed – %s", exc)
        state.ready = False


# ---------------------------------------------------------------------------
# Lifespan handler – kicks off background training, port opens immediately
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Starts model training in a background thread so the HTTP server
    begins accepting requests (and opens the port) immediately.
    The /health endpoint reports readiness status.
    """
    logger.info("=== AgriTech platform starting up ===")
    training_thread = threading.Thread(target=_train_models_background, daemon=True)
    training_thread.start()

    yield  # Application runs here

    logger.info("=== AgriTech platform shutting down ===")


# ---------------------------------------------------------------------------
# FastAPI app instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "AgriTech Predictive Platform API",
    description = (
        "Least-risk mixed cropping strategy engine for Indian farmers. "
        "Uses XGBoost yield prediction + Mean-Variance Portfolio Optimisation."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS – allow any origin during development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ---------------------------------------------------------------------------
# Middleware – request timing logger
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed_ms = round((time.time() - t0) * 1000)
    logger.info("%s %s → %d  (%d ms)",
                request.method, request.url.path,
                response.status_code, elapsed_ms)
    return response


# ---------------------------------------------------------------------------
# Helper: translate risk_preference to λ (risk aversion parameter)
# ---------------------------------------------------------------------------

RISK_AVERSION_MAP = {
    "conservative": 0.75,  # Heavily penalise variance → safer mix
    "balanced"    : 0.50,  # Equal weight to yield and variance
    "aggressive"  : 0.25,  # Maximise yield; accept higher variance
}

CLIMATE_KEYWORD_MULTIPLIERS = {
    "drought": 1.20,
    "flood"  : 1.15,
    "storm"  : 1.10,
    "normal" : 1.00,
    "good"   : 0.95,
}


def _parse_climate_risk_adjustment(climate_text: Optional[str]) -> float:
    """
    Scan the free-text current_climate_conditions field for known risk
    keywords and return a multiplier to apply to the base climate risk score.
    """
    if not climate_text:
        return 1.0
    text_lower = climate_text.lower()
    for keyword, mult in CLIMATE_KEYWORD_MULTIPLIERS.items():
        if keyword in text_lower:
            logger.info("Climate keyword '%s' detected – risk multiplier %.2f", keyword, mult)
            return mult
    return 1.0


def _integrity_label(score: float) -> str:
    """Convert a data integrity score (0-100) to a human-readable label."""
    if score >= 85:
        return "High – recommendation strongly grounded in real data"
    elif score >= 60:
        return "Moderate – some values were imputed; treat with care"
    else:
        return "Low – significant imputation; recommendation is indicative only"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint.
    Returns whether the ML models are trained and ready to serve requests.
    """
    status = "ok" if state.ready else ("error" if state.error else "initialising")
    metrics = mp.yield_predictor.metrics if state.ready else {}
    if state.error:
        metrics["error"] = state.error
    return HealthResponse(
        status          = status,
        models_ready    = state.ready,
        training_metrics= metrics,
    )


@app.post(
    "/api/ml-crop-recommendation",
    response_model = CropRecommendationResponse,
    summary        = "Optimised Crop Portfolio Recommendation",
    tags           = ["Crop Recommendation"],
)
async def crop_recommendation(request: CropRecommendationRequest):
    """
    **Core recommendation endpoint.**

    Given a farmer's location, land size, and optional soil / climate
    information, returns the mathematically least-risk mixed cropping
    strategy using:

    - **Model A** (XGBoost Yield Predictor) – predicts expected yield/ha
    - **Model B** (Climate Risk Scorer) – quantifies district-level climate risk
    - **MVO Optimiser** (SLSQP) – allocates land across crops to minimise
      variance while maximising yield

    ### Response
    - `optimised_crop_mix` : array of crops with %, hectares, yield, and CI
    - `portfolio_risk_score` : 0–100 composite risk index
    - `data_integrity_score` : how much real (vs imputed) data was used
    - `total_expected_yield_tonnes` : aggregate farm output estimate
    """
    if not state.ready:
        raise HTTPException(
            status_code = 503,
            detail      = "Models are still initialising. Please retry in 30 seconds.",
        )

    warnings: list[str] = []
    soil_type = request.soil_type or "Loam"

    # -----------------------------------------------------------------------
    # Step 1 – Retrieve district profile
    # -----------------------------------------------------------------------
    logger.info("Recommendation request: state=%s district=%s area=%.2fha soil=%s",
                request.state, request.district,
                request.land_area_hectares, soil_type)

    profile = prep.get_district_profile(
        state            = request.state,
        district         = request.district,
        apy              = state.apy,
        district_weather = state.district_weather,
        soil             = state.soil,
        soil_type        = soil_type,
    )

    weather_features  = profile["weather_features"]
    climate_risk_raw  = profile["climate_risk_raw"]
    data_integrity    = profile["data_integrity"]
    crops_available   = profile["crops_available"]
    apy_district      = profile["apy_district"]

    if not crops_available:
        warnings.append(
            f"No crop history found for district '{request.district}'. "
            "Using state-level data as fallback."
        )

    # -----------------------------------------------------------------------
    # Step 2 – Adjust climate risk for current conditions
    # -----------------------------------------------------------------------
    climate_multiplier = _parse_climate_risk_adjustment(
        request.current_climate_conditions
    )
    adjusted_climate_risk = min(100.0, climate_risk_raw * climate_multiplier)

    if climate_multiplier != 1.0:
        warnings.append(
            f"Climate condition keywords detected in input; risk score adjusted "
            f"by ×{climate_multiplier:.2f} (raw: {climate_risk_raw:.1f} → "
            f"adjusted: {adjusted_climate_risk:.1f})."
        )

    # -----------------------------------------------------------------------
    # Step 3 – Score district climate risk (Model B)
    # -----------------------------------------------------------------------
    final_climate_risk = mp.climate_risk_scorer.score(
        cv_annual  = weather_features.get("cv_annual",  0.2),
        rain_cv    = weather_features.get("rain_cv",    0.5),
        raw_score  = adjusted_climate_risk,
        soil_type  = soil_type,
    )

    # -----------------------------------------------------------------------
    # Step 4 – Build crop candidates using Model A predictions
    # -----------------------------------------------------------------------
    candidates = opt.build_candidates(
        crops              = crops_available,
        apy_district       = apy_district,
        weather_features   = weather_features,
        climate_risk_score = final_climate_risk,
        soil_type          = soil_type,
        target_season      = request.target_season,
        yield_predictor    = mp.yield_predictor,
        climate_risk_scorer= mp.climate_risk_scorer,
        max_candidates     = 25,
    )

    if len(candidates) == 0:
        season_msg = f" for season '{request.target_season}'" if request.target_season and request.target_season != "ALL" else ""
        raise HTTPException(
            status_code = 422,
            detail      = (
                f"Could not generate yield predictions for any crops in "
                f"'{request.district}', '{request.state}'{season_msg}. "
                "Please verify the state and district names."
            ),
        )

    if len(candidates) < 3:
        warnings.append(
            f"Only {len(candidates)} crop(s) had sufficient data for modelling. "
            "Portfolio diversification is limited."
        )

    # -----------------------------------------------------------------------
    # Step 5 – Mean-Variance Optimisation
    # -----------------------------------------------------------------------
    risk_aversion = RISK_AVERSION_MAP.get(request.risk_preference, 0.5)

    optimiser = opt.CropPortfolioOptimiser(
        risk_aversion = risk_aversion,
        min_weight    = 0.05,
        max_weight    = 0.60,
    )

    portfolio = optimiser.optimise(
        candidates         = candidates,
        land_area_hectares = request.land_area_hectares,
        climate_risk_score = final_climate_risk,
    )

    if not portfolio.optimiser_converged:
        warnings.append(
            "Portfolio optimiser did not fully converge – using equal-weight "
            "fallback. Results are approximate."
        )

    # -----------------------------------------------------------------------
    # Step 6 – Assemble response
    # -----------------------------------------------------------------------
    crop_allocations = [
        CropAllocation(
            crop                   = a.crop,
            season                 = a.season,
            percentage             = a.percentage,
            area_hectares          = a.area_hectares,
            expected_yield_tonnes  = a.expected_yield_tonnes,
            yield_ci_lower_tonnes  = a.yield_ci_lower,
            yield_ci_upper_tonnes  = a.yield_ci_upper,
            risk_contribution_pct  = a.risk_contribution,
        )
        for a in portfolio.allocations
    ]

    integrity_0_100 = round(data_integrity * 100.0, 1)

    response = CropRecommendationResponse(
        state                       = request.state,
        district                    = request.district,
        land_area_hectares          = request.land_area_hectares,
        soil_type_used              = soil_type,
        optimised_crop_mix          = crop_allocations,
        portfolio_risk_score        = portfolio.portfolio_risk_score,
        portfolio_risk_label        = portfolio.risk_label,
        total_expected_yield_tonnes = portfolio.total_expected_yield,
        data_integrity_score        = integrity_0_100,
        data_integrity_label        = _integrity_label(integrity_0_100),
        model_metadata              = ModelMetadata(
            yield_model_r2       = mp.yield_predictor.metrics.get("r2",  0.0),
            yield_model_mae      = mp.yield_predictor.metrics.get("mae", 0.0),
            training_sample_size = mp.yield_predictor.metrics.get("n_train", 0),
            optimiser_converged  = portfolio.optimiser_converged,
            optimiser_message    = portfolio.optimiser_message,
        ),
        warnings = warnings,
    )

    logger.info(
        "Recommendation served: %d crops | risk=%.1f | integrity=%.1f%%",
        len(crop_allocations),
        portfolio.portfolio_risk_score,
        integrity_0_100,
    )
    return response


# ---------------------------------------------------------------------------
# Global exception handler – returns JSON for any unhandled exception
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code = 500,
        content     = {
            "detail": "An internal server error occurred. "
                      "Please check server logs for details.",
            "type"  : type(exc).__name__,
        },
    )


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        workers = 1,  # Use 1 worker with reload=True; scale up in production
    )
