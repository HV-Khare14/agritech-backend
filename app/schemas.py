"""
schemas.py
==========
Pydantic v2 request and response models for the AgriTech API.

All monetary / area / yield fields use float for flexibility.
Strict validation is applied where agronomically appropriate
(e.g., land_area must be positive).
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ===========================================================================
#  Request model
# ===========================================================================

class CropRecommendationRequest(BaseModel):
    """
    Input payload for the /api/ml-crop-recommendation endpoint.

    Fields
    ------
    state                    : Indian state name (e.g., 'Punjab', 'Maharashtra').
    district                 : District within the state (e.g., 'Ludhiana').
    land_area_hectares       : Total cultivable land in hectares (must be > 0).
    soil_type                : Optional soil type string. Supported values:
                               'Sandy loam', 'Loam', 'Black', 'Clay loam'.
                               If omitted, 'Loam' is used as the neutral default.
    current_climate_conditions: Optional free-text description of current
                               season or weather observations.  Currently used
                               to adjust the risk-aversion parameter (e.g.,
                               'drought', 'flood', 'normal').
    risk_preference          : 'conservative' | 'balanced' | 'aggressive'.
                               Controls the λ parameter in MVO.
                               Default 'balanced'.
    """

    state                     : str   = Field(..., min_length=2, max_length=100,
                                              example="Punjab")
    district                  : str   = Field(..., min_length=2, max_length=100,
                                              example="Ludhiana")
    land_area_hectares        : float = Field(..., gt=0, le=100_000,
                                              example=5.0)
    soil_type                 : Optional[str] = Field(
                                    default=None,
                                    example="Sandy loam",
                                    description="One of: Sandy loam, Loam, Black, Clay loam")
    current_climate_conditions: Optional[str] = Field(
                                    default=None,
                                    example="Normal monsoon expected",
                                    max_length=500)
    risk_preference           : str   = Field(
                                    default="balanced",
                                    pattern="^(conservative|balanced|aggressive)$",
                                    example="balanced")

    @field_validator("state", "district", mode="before")
    @classmethod
    def strip_whitespace(cls, v):
        return v.strip() if isinstance(v, str) else v


# ===========================================================================
#  Response sub-models
# ===========================================================================

class CropAllocation(BaseModel):
    """Per-crop result within the recommended portfolio."""
    crop                   : str
    season                 : str
    percentage             : float = Field(description="Portfolio weight as percentage (0–100)")
    area_hectares          : float = Field(description="Allocated land area in hectares")
    expected_yield_tonnes  : float = Field(description="Expected total yield for allocated area")
    yield_ci_lower_tonnes  : float = Field(description="95% CI lower bound – total yield")
    yield_ci_upper_tonnes  : float = Field(description="95% CI upper bound – total yield")
    risk_contribution_pct  : float = Field(description="This crop's % contribution to portfolio variance")


class ModelMetadata(BaseModel):
    """Information about the ML models used for this prediction."""
    yield_model_r2         : float
    yield_model_mae        : float
    training_sample_size   : int
    optimiser_converged    : bool
    optimiser_message      : str


class CropRecommendationResponse(BaseModel):
    """
    Full API response for a crop portfolio recommendation.

    Fields
    ------
    state / district         : Echo of the request inputs.
    land_area_hectares       : Echo of the request inputs.
    optimised_crop_mix       : Ordered list of CropAllocation objects
                               (descending by percentage).
    portfolio_risk_score     : Composite risk score 0-100.
    portfolio_risk_label     : Human-readable label (Low / Moderate / High / Very High).
    total_expected_yield_tonnes : Sum of yield across all allocated crops.
    data_integrity_score     : 0-100 score indicating how much real (non-imputed)
                               data underpinned this recommendation.
    data_integrity_label     : Human-readable label.
    model_metadata           : Technical details about the ML pipeline used.
    warnings                 : Any non-fatal issues (e.g., district data unavailable,
                               using state-level fallback).
    """
    state                       : str
    district                    : str
    land_area_hectares          : float
    soil_type_used              : str
    optimised_crop_mix          : List[CropAllocation]
    portfolio_risk_score        : float
    portfolio_risk_label        : str
    total_expected_yield_tonnes : float
    data_integrity_score        : float = Field(description="0–100; higher = more real data used")
    data_integrity_label        : str
    model_metadata              : ModelMetadata
    warnings                    : List[str] = Field(default_factory=list)


# ===========================================================================
#  Health check response
# ===========================================================================

class HealthResponse(BaseModel):
    status         : str
    models_ready   : bool
    training_metrics: dict
