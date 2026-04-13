"""
preprocessing.py
================
Robust preprocessing pipeline for the AgriTech ML platform.

Responsibilities
----------------
1. Load all raw CSVs and normalise column names.
2. Join datasets on State / District / Subdivision keys.
3. Engineer features (seasonal rainfall aggregates, rainfall CV, etc.).
4. Run KNNImputer-based missing-value filling on numeric columns.
5. Encode categoricals.
6. Track a per-row "data_integrity_score" that records what fraction of
   each row's features came from real observations vs. imputed values.
   This score is propagated all the way to the final API response so
   farmers know how trustworthy the recommendation is.

All public functions return plain DataFrames; no side-effects.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File paths  (resolved relative to this module's parent directory)
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

APY_FILE        = DATA_DIR / "APY__1_.csv"           # Agricultural yield data
RAINFALL_FILE   = DATA_DIR / "District_Rainfall_Normal_0.csv"
IMD_FILE        = DATA_DIR / "Sub_Division_IMD_2017.csv"
DIST_MAP_FILE   = DATA_DIR / "district_to_subdivision.csv"
SOIL_FILE       = DATA_DIR / "usingnew.csv"          # Soil + fertiliser data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MONTHLY_COLS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# KNN neighbours used for imputation – small datasets get fewer neighbours
KNN_NEIGHBORS = 5


# ===========================================================================
#  Internal helpers
# ===========================================================================

def _normalise_str(s: pd.Series) -> pd.Series:
    """Strip, upper-case, and collapse whitespace in a string Series."""
    return s.astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)


def _strip_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading/trailing whitespace from every column name."""
    df.columns = [c.strip() for c in df.columns]
    return df


def _missing_mask(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Return a boolean DataFrame (True = value was MISSING before imputation)
    restricted to *cols* that actually exist in *df*.
    """
    existing = [c for c in cols if c in df.columns]
    return df[existing].isnull()


def _data_integrity_score(missing_before: pd.DataFrame, total_numeric_cols: int) -> pd.Series:
    """
    For each row compute the fraction of numeric features that were
    actually observed (not imputed).  Returns a Series in [0, 1]:
      1.0 = all values came from real data
      0.0 = all values were imputed
    """
    if missing_before.empty or total_numeric_cols == 0:
        return pd.Series(1.0, index=missing_before.index)
    imputed_count = missing_before.sum(axis=1)
    return 1.0 - (imputed_count / total_numeric_cols)


# ===========================================================================
#  Loaders – each returns a tidy DataFrame
# ===========================================================================

def load_apy() -> pd.DataFrame:
    """
    Load the Agriculture Production & Yield CSV.
    Cleans whitespace from State, District, Season.
    Drops rows with non-positive yield (data errors).
    Memory-optimised for constrained environments (e.g. Render free tier).
    """
    logger.info("Loading APY yield data …")
    # Load all columns first, then strip names (CSV has trailing spaces like "District ")
    df = pd.read_csv(APY_FILE, low_memory=True)
    df = _strip_col_names(df)

    # Now select only columns we need to save memory
    needed_cols = ["State", "District", "Crop", "Season", "Area", "Production", "Yield", "Crop_Year"]
    available = [c for c in needed_cols if c in df.columns]
    df = df[available].copy()

    # Normalise key join columns
    df["State"]    = _normalise_str(df["State"])
    df["District"] = _normalise_str(df["District"])
    df["Season"]   = _normalise_str(df["Season"])
    df["Crop"]     = df["Crop"].astype(str).str.strip()

    # Drop obvious data-entry errors
    df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
    df = df[df["Yield"] > 0].copy()
    df.dropna(subset=["Yield"], inplace=True)

    # Use category dtype for string columns to save memory
    for col in ["State", "District", "Crop", "Season"]:
        df[col] = df[col].astype("category")

    # Downsample if dataset is too large (keep most recent years)
    if "Crop_Year" in df.columns and len(df) > 100000:
        cutoff = df["Crop_Year"].max() - 15  # Keep last 15 years
        df = df[df["Crop_Year"] >= cutoff].copy()
        logger.info("Downsampled APY to years >= %d", cutoff)

    logger.info("APY loaded: %d rows, %d crops, %d districts",
                len(df), df["Crop"].nunique(), df["District"].nunique())
    return df


def load_rainfall_normal() -> pd.DataFrame:
    """
    Load district-level *normal* (long-run average) monthly rainfall.
    Returns one row per district with mm values for each month.
    """
    logger.info("Loading district normal rainfall …")
    df = pd.read_csv(RAINFALL_FILE)
    df = _strip_col_names(df)
    df.rename(columns={"STATE/UT": "State", "DISTRICT": "District"}, inplace=True)
    df["State"]    = _normalise_str(df["State"])
    df["District"] = _normalise_str(df["District"])

    # Coerce all monthly columns to numeric
    for col in MONTHLY_COLS + ["ANNUAL"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Normal rainfall loaded: %d districts", len(df))
    return df


def load_imd_timeseries() -> pd.DataFrame:
    """
    Load the IMD subdivision-level historical rainfall time-series (1901–2017).
    Used to compute inter-annual rainfall volatility (risk score).
    """
    logger.info("Loading IMD historical rainfall time-series …")
    df = pd.read_csv(IMD_FILE)
    df = _strip_col_names(df)
    df["SUBDIVISION"] = _normalise_str(df["SUBDIVISION"])
    for col in MONTHLY_COLS + ["ANNUAL"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info("IMD series loaded: %d rows, %d subdivisions, years %d–%d",
                len(df), df["SUBDIVISION"].nunique(),
                df["YEAR"].min(), df["YEAR"].max())
    return df


def load_district_subdivision_map() -> pd.DataFrame:
    """Return the district→IMD-subdivision mapping table."""
    df = pd.read_csv(DIST_MAP_FILE)
    df = _strip_col_names(df)
    df.rename(columns={"STATE/UT": "State", "DISTRICT": "District"}, inplace=True)
    df["State"]       = _normalise_str(df["State"])
    df["District"]    = _normalise_str(df["District"])
    df["SUBDIVISION"] = _normalise_str(df["SUBDIVISION"])
    return df


def load_soil_data() -> pd.DataFrame:
    """
    Load per-state, per-soil-type fertiliser recommendation data (usingnew.csv).
    Will be used to enrich district records with soil-type information.
    """
    df = pd.read_csv(SOIL_FILE)
    df = _strip_col_names(df)
    df["state"]     = _normalise_str(df["state"])
    df["soil_type"] = df["soil_type"].astype(str).str.strip()
    df["crop"]      = df["crop"].astype(str).str.strip()
    df["season"]    = df["season"].astype(str).str.strip()
    return df


# ===========================================================================
#  Feature Engineering
# ===========================================================================

def engineer_rainfall_features(rain_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the normal rainfall DataFrame add derived agronomic features:
      - kharif_rain  : Jun–Sep (monsoon season)
      - rabi_rain    : Oct–Mar (winter season)
      - annual_rain  : sum of all months
      - rain_cv      : coefficient of variation across months (proxy for
                       within-year variability)
    """
    df = rain_df.copy()
    kharif_cols = ["JUN", "JUL", "AUG", "SEP"]
    rabi_cols   = ["OCT", "NOV", "DEC", "JAN", "FEB", "MAR"]

    available = [c for c in kharif_cols if c in df.columns]
    df["kharif_rain"] = df[available].sum(axis=1, min_count=1)

    available = [c for c in rabi_cols if c in df.columns]
    df["rabi_rain"] = df[available].sum(axis=1, min_count=1)

    month_available = [c for c in MONTHLY_COLS if c in df.columns]
    df["annual_rain"] = df[month_available].sum(axis=1, min_count=1)

    # Coefficient of variation – higher value = more seasonal unevenness
    month_data = df[month_available]
    df["rain_cv"] = month_data.std(axis=1) / (month_data.mean(axis=1) + 1e-6)

    return df


def compute_imd_volatility(imd_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each IMD subdivision compute the inter-annual rainfall volatility:
      - mean_annual   : long-run average annual rainfall
      - std_annual    : standard deviation across years
      - cv_annual     : coefficient of variation (std / mean) – this is
                        the primary component of the Climate Risk Score.

    Returns one row per SUBDIVISION.
    """
    grp = imd_df.groupby("SUBDIVISION")["ANNUAL"].agg(
        mean_annual="mean",
        std_annual="std",
    ).reset_index()
    grp["cv_annual"] = grp["std_annual"] / (grp["mean_annual"] + 1e-6)
    # Normalise cv_annual to a 0-100 climate risk score
    cv_min, cv_max = grp["cv_annual"].min(), grp["cv_annual"].max()
    grp["climate_risk_score_raw"] = (
        (grp["cv_annual"] - cv_min) / (cv_max - cv_min + 1e-9) * 100
    )
    return grp


# ===========================================================================
#  KNN Imputation
# ===========================================================================

def knn_impute(df: pd.DataFrame,
               numeric_cols: list[str],
               n_neighbors: int = KNN_NEIGHBORS) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply KNNImputer to *numeric_cols* in *df*.

    Returns
    -------
    df_imputed : DataFrame with missing values filled.
    integrity  : Series[float] in [0,1]; fraction of real (non-imputed)
                 values per row ACROSS the numeric_cols subset.
    """
    cols_present = [c for c in numeric_cols if c in df.columns]
    if not cols_present:
        logger.warning("knn_impute called but no numeric cols found – skipping.")
        return df.copy(), pd.Series(1.0, index=df.index)

    # Record which cells were missing BEFORE imputation
    missing_before = _missing_mask(df, cols_present)

    imputer = KNNImputer(n_neighbors=min(n_neighbors, max(1, len(df) - 1)),
                         weights="distance")
    arr_imputed = imputer.fit_transform(df[cols_present])

    df_out = df.copy()
    df_out[cols_present] = arr_imputed

    integrity = _data_integrity_score(missing_before, len(cols_present))
    return df_out, integrity


# ===========================================================================
#  Master pipeline
# ===========================================================================

def build_master_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate all data loading, joining, feature engineering, and imputation.

    Returns
    -------
    apy_clean        : Cleaned yield DataFrame (per crop/district/year).
    district_weather : One row per district with rainfall + risk features.
    imd_volatility   : One row per IMD subdivision with climate volatility stats.
    soil_clean       : Cleaned soil / fertiliser recommendation DataFrame.
    """
    # ---- Load raw data ----
    apy        = load_apy()
    rain_norm  = load_rainfall_normal()
    imd        = load_imd_timeseries()
    dist_map   = load_district_subdivision_map()
    soil       = load_soil_data()

    # ---- Engineer rainfall features ----
    rain_feat = engineer_rainfall_features(rain_norm)

    # ---- Compute IMD volatility per subdivision ----
    imd_vol = compute_imd_volatility(imd)

    # ---- Join: district → subdivision → volatility ----
    district_weather = rain_feat.merge(dist_map, on=["State", "District"], how="left")
    district_weather = district_weather.merge(imd_vol, on="SUBDIVISION", how="left")

    # ---- KNN Impute rainfall features ----
    rain_numeric_cols = (MONTHLY_COLS +
                         ["kharif_rain", "rabi_rain", "annual_rain", "rain_cv",
                          "mean_annual", "std_annual", "cv_annual",
                          "climate_risk_score_raw"])
    district_weather, dw_integrity = knn_impute(district_weather, rain_numeric_cols)
    district_weather["data_integrity"] = dw_integrity

    # ---- KNN Impute APY ----
    apy_numeric = ["Area", "Production", "Yield"]
    apy, apy_integrity = knn_impute(apy, apy_numeric)
    apy["data_integrity"] = apy_integrity

    logger.info("Master dataset built: %d APY rows, %d district-weather rows",
                len(apy), len(district_weather))
    return apy, district_weather, imd_vol, soil


def get_district_profile(state: str,
                          district: str,
                          apy: pd.DataFrame,
                          district_weather: pd.DataFrame,
                          soil: pd.DataFrame,
                          soil_type: Optional[str] = None) -> dict:
    """
    Extract all available features for a requested (state, district) pair.

    Returns a dictionary with:
      - weather_features : dict of rainfall metrics
      - climate_risk_raw : float
      - data_integrity   : float in [0, 1]
      - crops_available  : list of crop names found in APY for this district
      - soil_info        : DataFrame slice for this state (and optional soil type)
    """
    state_upper    = state.strip().upper()
    district_upper = district.strip().upper()

    # --- Weather profile ---
    dw = district_weather[
        (district_weather["State"].str.upper() == state_upper) &
        (district_weather["District"].str.upper() == district_upper)
    ]

    if dw.empty:
        # Fuzzy fallback: match by state only and take state median
        logger.warning("District '%s' not found in weather data; using state median.", district)
        dw = district_weather[
            district_weather["State"].str.upper() == state_upper
        ]
        if dw.empty:
            logger.warning("State '%s' not found; using national median.", state)
            dw = district_weather

    weather_row       = dw.iloc[0]
    climate_risk_raw  = float(weather_row.get("climate_risk_score_raw", 50.0))
    data_integrity_dw = float(weather_row.get("data_integrity", 0.5))

    weather_features = {
        "annual_rain"  : float(weather_row.get("annual_rain",  800.0)),
        "kharif_rain"  : float(weather_row.get("kharif_rain",  500.0)),
        "rabi_rain"    : float(weather_row.get("rabi_rain",    200.0)),
        "rain_cv"      : float(weather_row.get("rain_cv",      0.5)),
        "mean_annual"  : float(weather_row.get("mean_annual",  800.0)),
        "cv_annual"    : float(weather_row.get("cv_annual",    0.2)),
    }

    # --- Crop history for this district ---
    apy_district = apy[
        (apy["State"].str.upper()    == state_upper) &
        (apy["District"].str.upper() == district_upper)
    ]
    if apy_district.empty:
        apy_district = apy[apy["State"].str.upper() == state_upper]

    crops_available = sorted(apy_district["Crop"].unique().tolist())
    data_integrity_apy = float(apy_district["data_integrity"].mean()) if not apy_district.empty else 0.5

    # --- Soil info ---
    soil_slice = soil[soil["state"].str.upper() == state_upper]
    if soil_type:
        st = soil_type.strip()
        filtered = soil_slice[soil_slice["soil_type"].str.lower() == st.lower()]
        if not filtered.empty:
            soil_slice = filtered

    # Combined data integrity score (average of weather + yield sources)
    data_integrity = round((data_integrity_dw + data_integrity_apy) / 2.0, 4)

    return {
        "weather_features"  : weather_features,
        "climate_risk_raw"  : climate_risk_raw,
        "data_integrity"    : data_integrity,
        "crops_available"   : crops_available,
        "apy_district"      : apy_district,
        "soil_info"         : soil_slice,
    }
