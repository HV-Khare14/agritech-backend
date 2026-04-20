"""
optimizer.py
============
Crop Portfolio Optimiser – Mean-Variance Optimisation (MVO)

Concept (adapted from Markowitz Modern Portfolio Theory)
---------------------------------------------------------
In financial portfolio theory, an investor selects asset weights w that
maximise expected return μ = wᵀr subject to minimising variance σ² = wᵀΣw.

We apply the same framework to crop selection:

    • "Return"   → predicted yield (tonnes/ha) per crop
    • "Risk"     → historical yield variance for that crop in the district,
                   scaled by the Climate Risk Score so that districts with
                   erratic rainfall carry a heavier penalty.
    • "Portfolio variance" → weighted sum of individual crop variances
                   (we use a simplified diagonal covariance matrix because
                   cross-crop yield correlations are sparse in the data;
                   however the structure supports a full Σ when available).

Objective (minimise)
--------------------
    f(w) = λ · wᵀΣw  -  (1-λ) · wᵀμ

    λ ∈ [0,1] controls the risk-aversion level.
    λ = 0.5 by default → equal weight to yield and risk.

Constraints
-----------
    Σ wᵢ = 1          (weights sum to 100%)
    wᵢ ≥ w_min        (no crop can take less than `min_weight` of the farm)
    wᵢ ≤ w_max        (diversification cap – no single crop over `max_weight`)

The optimisation is solved via scipy.optimize.minimize with SLSQP, which
handles linear equality constraints natively.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

logger = logging.getLogger(__name__)


# ===========================================================================
#  Data structures
# ===========================================================================

@dataclass
class CropCandidate:
    """
    Encapsulates all ML-derived metrics for one candidate crop.

    Attributes
    ----------
    name         : Crop name string.
    season       : Best-fit growing season.
    predicted_yield : Expected yield in tonnes/ha (from Model A).
    yield_ci_lower  : 95% CI lower bound.
    yield_ci_upper  : 95% CI upper bound.
    yield_variance  : Historical yield variance for this crop (district-level
                      when available, state-level otherwise).
    climate_risk    : Climate Risk Score (0-100) for the target district.
    """
    name            : str
    season          : str
    predicted_yield : float
    yield_ci_lower  : float
    yield_ci_upper  : float
    yield_variance  : float
    climate_risk    : float


@dataclass
class PortfolioAllocation:
    """One crop's share in the optimised portfolio."""
    crop            : str
    season          : str
    percentage      : float         # 0–100
    area_hectares   : float
    expected_yield_tonnes : float
    yield_ci_lower  : float
    yield_ci_upper  : float
    risk_contribution : float       # This crop's share of total portfolio variance


@dataclass
class OptimisedPortfolio:
    """Full output of the MVO optimisation."""
    allocations          : List[PortfolioAllocation]
    portfolio_risk_score : float    # 0-100 scale
    risk_label           : str
    total_expected_yield : float    # tonnes across entire farm
    optimiser_converged  : bool
    optimiser_message    : str


# ===========================================================================
#  Helper functions
# ===========================================================================

def _build_covariance_matrix(candidates: List[CropCandidate],
                              climate_risk_score: float) -> np.ndarray:
    """
    Construct a diagonal covariance matrix Σ from individual yield variances,
    scaled by the district-level Climate Risk Score.

    The risk score acts as a global multiplier: in a high-risk district every
    crop's variance is inflated proportionally – reflecting the fact that
    climate shocks tend to affect multiple crops simultaneously.

    Parameters
    ----------
    candidates         : List of CropCandidate objects.
    climate_risk_score : 0–100 scalar from the ClimateRiskScorer.

    Returns
    -------
    Σ : np.ndarray of shape (n_crops, n_crops)
    """
    n = len(candidates)
    # Climate multiplier: 1.0 at risk=0, 2.0 at risk=100
    climate_multiplier = 1.0 + (climate_risk_score / 100.0)

    variances = np.array([c.yield_variance for c in candidates], dtype=float)
    variances = np.maximum(variances, 1e-6)   # Avoid zero-variance edge case

    # Diagonal matrix (no cross-crop correlations assumed by default)
    Sigma = np.diag(variances * climate_multiplier)
    return Sigma


def _portfolio_objective(weights: np.ndarray,
                          mu: np.ndarray,
                          Sigma: np.ndarray,
                          risk_aversion: float) -> float:
    """
    Objective function for SLSQP minimisation.

    f(w) = λ·wᵀΣw  -  (1-λ)·wᵀμ

    • Minimising this function maximises yield while controlling variance.
    • risk_aversion = λ; higher → more weight on minimising variance.
    """
    portfolio_variance = weights @ Sigma @ weights
    portfolio_return   = weights @ mu
    return risk_aversion * portfolio_variance - (1.0 - risk_aversion) * portfolio_return


def _portfolio_objective_grad(weights: np.ndarray,
                               mu: np.ndarray,
                               Sigma: np.ndarray,
                               risk_aversion: float) -> np.ndarray:
    """Analytical gradient of the objective (speeds up convergence)."""
    return 2.0 * risk_aversion * (Sigma @ weights) - (1.0 - risk_aversion) * mu


# ===========================================================================
#  Main optimiser
# ===========================================================================

class CropPortfolioOptimiser:
    """
    Implements Mean-Variance Optimisation for crop portfolio selection.

    Usage
    -----
    >>> opt = CropPortfolioOptimiser()
    >>> portfolio = opt.optimise(candidates, land_area_hectares=5.0,
    ...                          climate_risk_score=42.0)
    """

    def __init__(self,
                 risk_aversion : float = 0.5,
                 min_weight    : float = 0.05,
                 max_weight    : float = 0.60):
        """
        Parameters
        ----------
        risk_aversion : λ ∈ [0,1]. 0 = pure yield maximiser.
                        1 = pure risk minimiser. Default 0.5 (balanced).
        min_weight    : Minimum portfolio weight for any selected crop.
        max_weight    : Maximum portfolio weight for any single crop.
        """
        self.risk_aversion = risk_aversion
        self.min_weight    = min_weight
        self.max_weight    = max_weight

    def _select_top_crops(self,
                           candidates: List[CropCandidate],
                           max_crops: int = 6) -> List[CropCandidate]:
        """
        Pre-filter the candidate list to at most *max_crops* crops.
        Selection criterion: highest Sharpe-like ratio = yield / sqrt(variance).
        This avoids a degenerate many-asset problem when dozens of crops are
        available for a state.
        """
        if len(candidates) <= max_crops:
            return candidates

        scored = sorted(
            candidates,
            key=lambda c: c.predicted_yield / (np.sqrt(c.yield_variance + 1e-6)),
            reverse=True,
        )
        logger.info("Pre-filtered %d → %d candidates by Sharpe-like score.",
                    len(candidates), max_crops)
        return scored[:max_crops]

    def optimise(self,
                 candidates         : List[CropCandidate],
                 land_area_hectares : float,
                 climate_risk_score : float) -> OptimisedPortfolio:
        """
        Run MVO and return a fully-populated OptimisedPortfolio.

        Parameters
        ----------
        candidates          : List of CropCandidate objects (≥ 1).
        land_area_hectares  : Total farm size in hectares.
        climate_risk_score  : 0-100 district climate risk.

        Returns
        -------
        OptimisedPortfolio
        """
        # --- Guard: need at least 1 candidate ---
        if not candidates:
            raise ValueError("No crop candidates supplied to optimiser.")

        # --- Pre-filter to manageable set ---
        candidates = self._select_top_crops(candidates, max_crops=6)
        n = len(candidates)

        # --- Build expected-return vector and covariance matrix ---
        mu    = np.array([c.predicted_yield for c in candidates], dtype=float)
        Sigma = _build_covariance_matrix(candidates, climate_risk_score)

        # --- SLSQP optimisation ---
        # Initial guess: equal weights
        w0 = np.full(n, 1.0 / n)

        # Constraints: weights sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Bounds: each weight ∈ [min_weight, max_weight]
        bounds = [(self.min_weight, self.max_weight)] * n

        result: OptimizeResult = minimize(
            fun         = _portfolio_objective,
            x0          = w0,
            args        = (mu, Sigma, self.risk_aversion),
            jac         = _portfolio_objective_grad,
            method      = "SLSQP",
            bounds      = bounds,
            constraints = constraints,
            options     = {"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            logger.warning("SLSQP did not converge: %s. Falling back to equal weights.",
                           result.message)
            weights = w0.copy()
        else:
            weights = result.x
            # Clip & re-normalise to fix floating-point drift
            weights = np.clip(weights, self.min_weight, self.max_weight)
            weights /= weights.sum()

        # --- Portfolio variance (for risk score) ---
        port_variance = float(weights @ Sigma @ weights)

        # Scale portfolio risk to 0-100
        # We use the worst-case single-crop variance as the normaliser
        worst_case_variance = float(np.max(np.diag(Sigma)))
        port_risk_score = min(
            100.0,
            (port_variance / (worst_case_variance + 1e-9)) * climate_risk_score
        )

        # --- Per-crop risk contribution ---
        # Marginal contribution = wᵢ * (Σw)ᵢ / portfolio_variance
        sigma_w = Sigma @ weights
        risk_contributions = (weights * sigma_w) / (port_variance + 1e-9)

        # --- Assemble allocations ---
        allocations: List[PortfolioAllocation] = []
        for i, cand in enumerate(candidates):
            pct   = round(float(weights[i]) * 100.0, 2)
            area  = round(float(weights[i]) * land_area_hectares, 4)
            exp_y = round(cand.predicted_yield * area, 4)
            ci_lo = round(cand.yield_ci_lower  * area, 4)
            ci_hi = round(cand.yield_ci_upper  * area, 4)
            rc    = round(float(risk_contributions[i]) * 100.0, 2)

            allocations.append(PortfolioAllocation(
                crop                   = cand.name,
                season                 = cand.season,
                percentage             = pct,
                area_hectares          = area,
                expected_yield_tonnes  = exp_y,
                yield_ci_lower         = ci_lo,
                yield_ci_upper         = ci_hi,
                risk_contribution      = rc,
            ))

        # Sort descending by allocation percentage
        allocations.sort(key=lambda a: a.percentage, reverse=True)

        total_yield = round(sum(a.expected_yield_tonnes for a in allocations), 4)

        from app.model_pipeline import ClimateRiskScorer
        risk_label = ClimateRiskScorer.label(port_risk_score)

        return OptimisedPortfolio(
            allocations          = allocations,
            portfolio_risk_score = round(port_risk_score, 2),
            risk_label           = risk_label,
            total_expected_yield = total_yield,
            optimiser_converged  = bool(result.success),
            optimiser_message    = result.message,
        )


# ===========================================================================
#  Candidate builder
# ===========================================================================

def build_candidates(crops               : List[str],
                     apy_district        : pd.DataFrame,
                     weather_features    : dict,
                     climate_risk_score  : float,
                     soil_type           : str,
                     target_season       : Optional[str],
                     yield_predictor,
                     climate_risk_scorer,
                     max_candidates      : int = 15) -> List[CropCandidate]:
    """
    For each crop in *crops*, query Model A and compute historical variance
    to assemble a list of CropCandidate objects ready for MVO.

    Parameters
    ----------
    crops              : Candidate crop names.
    apy_district       : APY slice for the target district.
    weather_features   : Dict of weather feature values.
    climate_risk_score : District climate risk (0-100).
    soil_type          : Soil type string.
    target_season      : Season string to isolate (e.g. KHARIF, RABI, ALL)
    yield_predictor    : Fitted YieldPredictor instance.
    climate_risk_scorer: ClimateRiskScorer instance.
    max_candidates     : Limit total candidates fed to optimiser.

    Returns
    -------
    List[CropCandidate]
    """
    candidates: List[CropCandidate] = []

    # Season lookup: map crop → most common season in historical data
    if not apy_district.empty:
        season_map = (
            apy_district.groupby("Crop")["Season"]
            .agg(lambda s: s.mode()[0] if not s.empty else "KHARIF")
            .to_dict()
        )
        # Historical variance per crop in this district
        variance_map = (
            apy_district.groupby("Crop")["Yield"]
            .var()
            .fillna(1.0)
            .to_dict()
        )
    else:
        season_map   = {}
        variance_map = {}

    for crop in crops[:max_candidates]:
        season        = season_map.get(crop, "KHARIF").strip().upper()
        
        if target_season and target_season != "ALL" and season != target_season.upper():
            continue

        var_yield     = float(variance_map.get(crop, 1.0))
        var_yield     = max(var_yield, 0.01)   # Floor to prevent zero-variance

        try:
            y_hat, ci_lo, ci_hi = yield_predictor.predict(
                crop=crop,
                season=season,
                soil_type=soil_type,
                weather=weather_features,
            )
        except Exception as exc:
            logger.warning("Yield prediction failed for crop '%s': %s", crop, exc)
            continue

        # Skip crops with negligible predicted yield (data artefacts)
        if y_hat < 0.01:
            continue

        candidates.append(CropCandidate(
            name             = crop,
            season           = season,
            predicted_yield  = y_hat,
            yield_ci_lower   = ci_lo,
            yield_ci_upper   = ci_hi,
            yield_variance   = var_yield,
            climate_risk     = climate_risk_score,
        ))

    logger.info("Built %d valid crop candidates for optimisation.", len(candidates))
    return candidates
