"""
model_pipeline.py
=================
Two-model ML pipeline for the AgriTech platform.

Model A – Yield Predictor
--------------------------
An XGBoost regression ensemble that predicts the expected yield (tonnes/ha)
for a given (crop, district, soil_type, weather_conditions) combination.

Training strategy
  • Features : annual_rain, kharif_rain, rabi_rain, rain_cv,
               encoded soil_type, encoded season, encoded crop.
  • Target   : log1p(Yield) – log-transform stabilises the heavily
               right-skewed yield distribution.
  • We train one global model across all crops but include crop as a
    feature so the model learns crop-specific yield patterns.

Model B – Climate Risk Scorer
------------------------------
Not a trainable model – instead it synthesises several risk signals into
a scalar 0-100 "Portfolio Climate Risk Score":
  1. cv_annual       (inter-annual coefficient of variation from IMD data)
  2. rain_cv         (within-year rainfall unevenness from district normals)
  3. A domain-knowledge seasonal drought multiplier

All models are cached as module-level singletons so they are trained
once on startup and reused for every API request.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


# ===========================================================================
#  Label encoders – kept as module-level state for consistency between
#  training and inference.
# ===========================================================================

class EncoderStore:
    """Holds fitted LabelEncoders for categorical features."""
    def __init__(self):
        self.crop_enc   = LabelEncoder()
        self.season_enc = LabelEncoder()
        self.soil_enc   = LabelEncoder()
        self._fitted    = False

    def fit(self, crops: List[str], seasons: List[str], soils: List[str]):
        # Add an UNKNOWN sentinel so unseen values can be handled gracefully
        # Filter out any NaN / non-string values from the data
        clean_crops   = [c for c in crops   if isinstance(c, str)]
        clean_seasons = [s for s in seasons if isinstance(s, str)]
        clean_soils   = [s for s in soils   if isinstance(s, str)]
        self.crop_enc.fit(sorted(set(clean_crops))   + ["UNKNOWN"])
        self.season_enc.fit(sorted(set(clean_seasons)) + ["UNKNOWN"])
        self.soil_enc.fit(sorted(set(clean_soils))   + ["UNKNOWN"])
        self._fitted = True

    def _safe_transform(self, enc: LabelEncoder, value: str) -> int:
        """Transform a single value; return the UNKNOWN class on unseen input."""
        try:
            return int(enc.transform([value])[0])
        except ValueError:
            return int(enc.transform(["UNKNOWN"])[0])

    def transform_crop(self, crop: str)   -> int:
        return self._safe_transform(self.crop_enc,   crop.strip())

    def transform_season(self, season: str) -> int:
        return self._safe_transform(self.season_enc, season.strip().upper())

    def transform_soil(self, soil: str)  -> int:
        return self._safe_transform(self.soil_enc,   soil.strip())


# Module-level singleton
encoder_store = EncoderStore()


# ===========================================================================
#  Feature builder
# ===========================================================================

FEATURE_COLS = [
    "annual_rain", "kharif_rain", "rabi_rain", "rain_cv",
    "crop_enc", "season_enc", "soil_enc",
]


def _build_feature_row(crop: str,
                       season: str,
                       soil_type: str,
                       weather: dict) -> np.ndarray:
    """
    Build a single feature vector (1-D numpy array) for prediction.

    Parameters
    ----------
    crop        : Crop name string.
    season      : Season string (e.g., 'KHARIF', 'RABI').
    soil_type   : Soil type string (e.g., 'Sandy loam').
    weather     : Dict with keys matching those in FEATURE_COLS.

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    row = [
        weather.get("annual_rain",  800.0),
        weather.get("kharif_rain",  500.0),
        weather.get("rabi_rain",    200.0),
        weather.get("rain_cv",        0.5),
        encoder_store.transform_crop(crop),
        encoder_store.transform_season(season),
        encoder_store.transform_soil(soil_type),
    ]
    return np.array(row, dtype=float).reshape(1, -1)


def _build_feature_matrix(df: pd.DataFrame,
                           district_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Join APY data with district weather to produce a flat feature matrix
    for model training.
    """
    # Merge APY with district weather on State + District
    merged = df.merge(
        district_weather[["State", "District",
                          "annual_rain", "kharif_rain", "rabi_rain", "rain_cv"]],
        on=["State", "District"],
        how="left",
    )

    # Fill remaining weather NaNs with national medians
    for col in ["annual_rain", "kharif_rain", "rabi_rain", "rain_cv"]:
        merged[col].fillna(merged[col].median(), inplace=True)

    # Encode categoricals
    merged["crop_enc"]   = merged["Crop"].apply(encoder_store.transform_crop)
    merged["season_enc"] = merged["Season"].apply(
        lambda s: encoder_store.transform_season(s.strip().upper()))
    # soil_type is not in APY; use a neutral "LOAM" default during training
    merged["soil_enc"]   = encoder_store.transform_soil("Loam")

    return merged


# ===========================================================================
#  Model A – XGBoost Yield Predictor
# ===========================================================================

class YieldPredictor:
    """
    Ensemble XGBoost regression model for yield prediction.

    Attributes
    ----------
    model     : Fitted XGBRegressor.
    is_fitted : bool – True after .train() has been called.
    metrics   : dict with training performance metrics.
    """

    def __init__(self):
        # Hyperparameters tuned for typical agro-tabular datasets.
        # n_estimators=400 + early stopping gives robust performance without
        # over-fitting to any single region's idiosyncrasies.
        self.model = XGBRegressor(
            n_estimators      = 400,
            learning_rate     = 0.05,
            max_depth         = 6,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            reg_alpha         = 0.1,    # L1 regularisation
            reg_lambda        = 1.0,    # L2 regularisation
            random_state      = 42,
            tree_method       = "hist", # Memory-efficient histogram method
            n_jobs            = -1,
            verbosity         = 0,
        )
        self.is_fitted = False
        self.metrics: Dict = {}
        # Per-crop mean and std of log-yield for CI computation later
        self._crop_yield_stats: Dict[str, Dict] = {}

    def train(self, apy: pd.DataFrame, district_weather: pd.DataFrame) -> None:
        """
        Fit the XGBoost model on the full APY + weather feature matrix.

        Steps
        -----
        1. Fit encoders (must happen before _build_feature_matrix).
        2. Build feature matrix.
        3. Log-transform target.
        4. Train/val split → fit with early stopping.
        5. Record per-crop yield stats for CI estimation.
        """
        logger.info("Fitting label encoders …")
        encoder_store.fit(
            crops   = apy["Crop"].unique().tolist(),
            seasons = apy["Season"].unique().tolist(),
            soils   = ["Sandy loam", "Loam", "Black", "Clay loam", "UNKNOWN"],
        )

        logger.info("Building feature matrix …")
        feat_df = _build_feature_matrix(apy, district_weather)

        X = feat_df[FEATURE_COLS].values
        # Log-transform to reduce skew; expm1 on the way out
        y = np.log1p(feat_df["Yield"].values)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

        logger.info("Training XGBoost yield predictor (%d samples) …", len(X_train))
        self.model.fit(
            X_train, y_train,
            eval_set          = [(X_val, y_val)],
            verbose           = False,
        )

        y_pred_val = self.model.predict(X_val)
        self.metrics = {
            "r2"  : round(r2_score(y_val, y_pred_val),  4),
            "mae" : round(mean_absolute_error(y_val, y_pred_val), 4),
            "n_train": len(X_train),
        }
        logger.info("YieldPredictor trained | R²=%.3f  MAE=%.3f (log scale)",
                    self.metrics["r2"], self.metrics["mae"])

        # Store per-crop yield statistics (natural scale) for CI computation
        for crop, grp in apy.groupby("Crop"):
            vals = grp["Yield"].dropna()
            if len(vals) < 2:
                continue
            self._crop_yield_stats[crop] = {
                "mean": float(vals.mean()),
                "std" : float(vals.std()),
                "n"   : int(len(vals)),
            }

        self.is_fitted = True

    def predict(self,
                crop: str,
                season: str,
                soil_type: str,
                weather: dict) -> Tuple[float, float, float]:
        """
        Predict yield for one (crop, conditions) combination.

        Returns
        -------
        (predicted_yield, ci_lower, ci_upper)
        All in tonnes/ha (natural scale).
        95% confidence interval is computed from the historical per-crop
        standard deviation when available, falling back to ±30% of the
        point estimate.
        """
        if not self.is_fitted:
            raise RuntimeError("YieldPredictor has not been trained yet.")

        X = _build_feature_row(crop, season, soil_type, weather)
        y_log = float(self.model.predict(X)[0])
        y_hat = float(np.expm1(y_log))  # Back to natural scale

        # 95% CI using historical variance for this crop
        stats = self._crop_yield_stats.get(crop)
        if stats and stats["n"] >= 5:
            # t-critical ≈ 1.96 for large n; use 2.0 as conservative approximation
            se = stats["std"] / np.sqrt(stats["n"])
            ci_lower = max(0.0, y_hat - 2.0 * se)
            ci_upper = y_hat + 2.0 * se
        else:
            ci_lower = max(0.0, y_hat * 0.70)
            ci_upper = y_hat * 1.30

        return round(y_hat, 4), round(ci_lower, 4), round(ci_upper, 4)


# ===========================================================================
#  Model B – Climate Risk Scorer
# ===========================================================================

class ClimateRiskScorer:
    """
    Composite rule-based + data-driven climate risk scorer.

    The final score (0–100) is a weighted combination of:
      - cv_annual    (40%) : inter-annual volatility from IMD time-series
      - rain_cv      (30%) : within-year unevenness of rainfall distribution
      - raw_score    (30%) : the pre-normalised CV-based score from the
                             preprocessing module

    Score interpretation:
      0–25   : Low risk   (stable rainfall, predictable seasons)
      25–50  : Moderate risk
      50–75  : High risk  (erratic rainfall, drought/flood prone)
      75–100 : Very high risk
    """

    # Weights must sum to 1.0
    WEIGHTS = {
        "cv_annual"  : 0.40,
        "rain_cv"    : 0.30,
        "raw_score"  : 0.30,
    }

    def score(self,
              cv_annual     : float,
              rain_cv       : float,
              raw_score     : float,
              soil_type     : str = "Loam") -> float:
        """
        Compute a 0-100 risk score.

        Parameters
        ----------
        cv_annual  : Coefficient of variation of annual rainfall (IMD).
        rain_cv    : Coefficient of variation across monthly normals.
        raw_score  : Pre-computed normalised score from preprocessing.
        soil_type  : Soil type string – sandy soils add a small drought
                     risk premium.
        """
        # --- Normalise each component to [0, 100] ---
        # cv_annual: typical range 0.05–0.60 in India
        cv_annual_score = min(100.0, (cv_annual / 0.60) * 100.0)

        # rain_cv: typical range 0.3–2.0
        rain_cv_score = min(100.0, (rain_cv / 2.0) * 100.0)

        # raw_score: already 0-100
        raw_clamped = min(100.0, max(0.0, raw_score))

        # --- Weighted sum ---
        base_score = (
            self.WEIGHTS["cv_annual"] * cv_annual_score +
            self.WEIGHTS["rain_cv"]   * rain_cv_score   +
            self.WEIGHTS["raw_score"] * raw_clamped
        )

        # --- Soil-type drought premium ---
        soil_premium = 0.0
        if soil_type.lower() in ("sandy", "sandy loam"):
            # Sandy soils drain faster → higher drought sensitivity
            soil_premium = 5.0
        elif soil_type.lower() in ("black", "clay", "clay loam"):
            # Heavy soils retain water → slightly lower drought risk
            soil_premium = -3.0

        final_score = min(100.0, max(0.0, base_score + soil_premium))
        return round(final_score, 2)

    @staticmethod
    def label(score: float) -> str:
        """Convert numeric score to a human-readable risk label."""
        if score < 25:
            return "Low"
        elif score < 50:
            return "Moderate"
        elif score < 75:
            return "High"
        else:
            return "Very High"


# ===========================================================================
#  Module-level singletons – initialised by the app lifespan handler
# ===========================================================================

yield_predictor    = YieldPredictor()
climate_risk_scorer = ClimateRiskScorer()


def train_all_models(apy: pd.DataFrame, district_weather: pd.DataFrame) -> None:
    """
    Entry point called once during FastAPI startup to train / warm-up
    all ML models.
    """
    logger.info("=== Starting model training pipeline ===")
    yield_predictor.train(apy, district_weather)
    logger.info("=== Model training complete ===")
    logger.info("Training metrics: %s", yield_predictor.metrics)
