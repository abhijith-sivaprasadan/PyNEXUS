"""Data reconciliation and gross-error detection for the hydrogen mass balance.

Mirrors the method already used in this portfolio's ThermoTwin repository
(single-constraint weighted least squares + a chi-square(1) global test) —
same statistical method, applied here to redundant "sensor" measurements of
electrolyser production, storage charge/discharge, and pipeline delivery
instead of a thermal energy balance.

This operates on SYNTHETIC noisy measurements (see tests). It demonstrates
the reconciliation method; it is not validated against plant instrumentation
and must never be described as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Chi-square(1) critical value at 99.9% confidence — same threshold used by
# ThermoTwin's data_reconciliation.f90, for consistency across the portfolio.
CHI_SQUARE_1_CRIT_999 = 10.83


@dataclass
class ReconciliationResult:
    measured: dict[str, float]
    adjusted: dict[str, float]
    adjustment: dict[str, float]
    raw_imbalance: float
    test_statistic: float
    gross_error: bool


def reconcile_hydrogen_balance(
    production_kg_h: float,
    to_store_kg_h: float,
    from_store_kg_h: float,
    to_pipeline_kg_h: float,
    variance_production: float,
    variance_to_store: float,
    variance_from_store: float,
    variance_to_pipeline: float,
    crit: float = CHI_SQUARE_1_CRIT_999,
) -> ReconciliationResult:
    """Reconcile four redundant hydrogen-flow measurements against one balance.

    True balance (mass conservation at the storage/pipeline node):
        production == to_store - from_store + to_pipeline

    i.e. everything the electrolyser produces either goes to storage, comes
    back out of storage, or heads to the pipeline toward demand. Given noisy
    measurements of all four quantities with stated variances, this finds
    the minimum-adjustment correction (weighted by each sensor's stated
    variance) that closes the balance exactly, and a chi-square(1) test on
    the raw (pre-adjustment) imbalance to flag a gross error — a residual
    too large to be plausible measurement noise given the stated variances.

    Sign convention matches REVAMP_PLAN.md Phase A4 exactly: coefficients
    are (+1, -1, +1, -1) on (production, to_store, from_store, to_pipeline)
    respectively, i.e. production - to_store + from_store - to_pipeline = 0.

    Deliberately does NOT report a "likely culprit" sensor. A per-sensor
    fault-localization heuristic (rank sensors by their WLS adjustment,
    raw or standardized) was tried and is wrong: with a single constraint
    across four sensors (one redundant degree of freedom), the WLS
    adjustment vector's *shape* across sensors is determined entirely by
    the configured variances — proof: adjustment_i = lam * var_i * a_i
    where lam = imbalance / sum_j(var_j) is a single scalar shared by every
    sensor, so |adjustment_i| : |adjustment_j| = var_i : var_j regardless
    of which sensor's measurement actually contains the error. Verified
    numerically: biasing each of the four sensors in turn, in a fixture with
    distinct variances, always "identified" the same highest-variance
    sensor as the culprit — the ranking never once depended on the data.
    Localizing the actual faulty sensor needs either repeated/historical
    readings per sensor or additional redundant constraints (more sensors
    or equations than this single balance provides); see docs/reconciliation.md.
    """
    names = ["production", "to_store", "from_store", "to_pipeline"]
    y = np.array([production_kg_h, to_store_kg_h, from_store_kg_h, to_pipeline_kg_h], dtype=float)
    var = np.array(
        [variance_production, variance_to_store, variance_from_store, variance_to_pipeline],
        dtype=float,
    )
    if np.any(var <= 0):
        raise ValueError("All variances must be strictly positive")
    a = np.array([1.0, -1.0, 1.0, -1.0])  # production - to_store + from_store - to_pipeline

    imbalance = float(np.dot(a, y))
    denom = float(np.dot(a * var, a))  # a^T * diag(var) * a
    lam = imbalance / denom
    adjustment = lam * var * a  # WLS closed-form for one linear equality constraint
    x = y - adjustment

    test_stat = imbalance**2 / denom
    gross_error = test_stat > crit

    return ReconciliationResult(
        measured=dict(zip(names, y.tolist())),
        adjusted=dict(zip(names, x.tolist())),
        adjustment=dict(zip(names, adjustment.tolist())),
        raw_imbalance=imbalance,
        test_statistic=test_stat,
        gross_error=bool(gross_error),
    )
