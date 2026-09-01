import numpy as np
import pytest

from optimization.reconciliation import CHI_SQUARE_1_CRIT_999, reconcile_hydrogen_balance

# True, exactly-balanced flows: production - to_store + from_store - to_pipeline = 0
TRUE_PRODUCTION = 1000.0
TRUE_TO_STORE = 200.0
TRUE_FROM_STORE = 50.0
TRUE_TO_PIPELINE = TRUE_PRODUCTION - TRUE_TO_STORE + TRUE_FROM_STORE  # 850.0


def test_true_balanced_flows_have_zero_imbalance_and_no_gross_error() -> None:
    result = reconcile_hydrogen_balance(
        TRUE_PRODUCTION,
        TRUE_TO_STORE,
        TRUE_FROM_STORE,
        TRUE_TO_PIPELINE,
        variance_production=4.0,
        variance_to_store=1.0,
        variance_from_store=1.0,
        variance_to_pipeline=4.0,
    )
    assert result.raw_imbalance == pytest.approx(0.0, abs=1e-9)
    assert result.test_statistic == pytest.approx(0.0, abs=1e-9)
    assert not result.gross_error


def test_small_measurement_noise_reconciles_cleanly() -> None:
    """Small noise (well within stated sensor variance) must not trigger a
    gross-error flag, and adjustment should be modest relative to the noise."""
    rng = np.random.default_rng(42)
    var = {"production": 4.0, "to_store": 1.0, "from_store": 1.0, "to_pipeline": 4.0}
    noisy = {
        "production": TRUE_PRODUCTION + rng.normal(0, np.sqrt(var["production"])),
        "to_store": TRUE_TO_STORE + rng.normal(0, np.sqrt(var["to_store"])),
        "from_store": TRUE_FROM_STORE + rng.normal(0, np.sqrt(var["from_store"])),
        "to_pipeline": TRUE_TO_PIPELINE + rng.normal(0, np.sqrt(var["to_pipeline"])),
    }

    result = reconcile_hydrogen_balance(
        noisy["production"],
        noisy["to_store"],
        noisy["from_store"],
        noisy["to_pipeline"],
        variance_production=var["production"],
        variance_to_store=var["to_store"],
        variance_from_store=var["from_store"],
        variance_to_pipeline=var["to_pipeline"],
    )

    assert not result.gross_error
    # Reconciled flows must satisfy the balance exactly.
    a = np.array([1.0, -1.0, 1.0, -1.0])
    adjusted = np.array(
        [
            result.adjusted["production"],
            result.adjusted["to_store"],
            result.adjusted["from_store"],
            result.adjusted["to_pipeline"],
        ]
    )
    assert np.dot(a, adjusted) == pytest.approx(0.0, abs=1e-6)
    # Each adjustment should be small relative to that sensor's noise scale.
    for name in var:
        assert abs(result.adjustment[name]) < 5 * np.sqrt(var[name])


def test_deliberately_biased_sensor_triggers_gross_error() -> None:
    """A 50 kg/h offset on one sensor, far outside plausible noise, must trip
    the chi-square(1) global test — this is the actual, correctly-identifiable
    result of reconciliation with a single constraint: THAT something is
    wrong, not WHICH sensor (see test below)."""
    biased_to_pipeline = TRUE_TO_PIPELINE - 50.0

    result = reconcile_hydrogen_balance(
        TRUE_PRODUCTION,
        TRUE_TO_STORE,
        TRUE_FROM_STORE,
        biased_to_pipeline,
        variance_production=4.0,
        variance_to_store=1.0,
        variance_from_store=2.0,
        variance_to_pipeline=0.25,
    )

    assert result.gross_error
    assert result.test_statistic > CHI_SQUARE_1_CRIT_999


def test_per_sensor_fault_is_not_identifiable_from_one_constraint() -> None:
    """A single balance equation across four sensors has exactly one redundant
    degree of freedom, which is enough to detect that a measurement set is
    inconsistent (the test above) but not enough to say which sensor caused
    it. This test locks in that real limitation: the WLS adjustment ratio
    between any two sensors is exactly var_i/var_j, provably independent of
    which sensor actually carries the bias (adjustment_i = lam*var_i*a_i with
    a single shared scalar lam, so the adjustment *shape* is fixed by the
    variances alone). Concretely: biasing any one of the four sensors in turn
    (holding variances fixed) must leave the *ratio* between two sensors'
    adjustments unchanged, even though which one is actually wrong changes
    every time — proof that ranking sensors by adjustment size would silently
    always point at the same (highest-variance) sensor regardless of the
    data, which is why this module does not offer that ranking as a feature.
    """
    variances = dict(
        variance_production=1.0,
        variance_to_store=0.25,
        variance_from_store=1.0,
        variance_to_pipeline=9.0,
    )
    offset = 80.0
    scenarios = [
        (TRUE_PRODUCTION - offset, TRUE_TO_STORE, TRUE_FROM_STORE, TRUE_TO_PIPELINE),
        (TRUE_PRODUCTION, TRUE_TO_STORE - offset, TRUE_FROM_STORE, TRUE_TO_PIPELINE),
        (TRUE_PRODUCTION, TRUE_TO_STORE, TRUE_FROM_STORE - offset, TRUE_TO_PIPELINE),
        (TRUE_PRODUCTION, TRUE_TO_STORE, TRUE_FROM_STORE, TRUE_TO_PIPELINE - offset),
    ]

    ratios = []
    for production, to_store, from_store, to_pipeline in scenarios:
        result = reconcile_hydrogen_balance(
            production, to_store, from_store, to_pipeline, **variances
        )
        assert result.gross_error  # each scenario is a real, detectable inconsistency
        ratios.append(result.adjustment["to_pipeline"] / result.adjustment["production"])

    # Same ratio (= -var_to_pipeline / var_production, sign from a_i = -1/+1)
    # no matter which sensor was actually the one that got biased.
    assert ratios == pytest.approx([-9.0] * 4, rel=1e-9)


def test_biased_sensor_with_small_variance_is_more_readily_flagged() -> None:
    """A sensor stated to be precise (small variance) that disagrees with the
    others is a stronger signal of a gross error than the same disagreement
    from a sensor stated to be noisy — the WLS weighting should reflect that."""
    offset = 10.0
    precise = reconcile_hydrogen_balance(
        TRUE_PRODUCTION,
        TRUE_TO_STORE,
        TRUE_FROM_STORE,
        TRUE_TO_PIPELINE - offset,
        variance_production=4.0,
        variance_to_store=1.0,
        variance_from_store=1.0,
        variance_to_pipeline=0.01,  # sensor claims very high precision
    )
    imprecise = reconcile_hydrogen_balance(
        TRUE_PRODUCTION,
        TRUE_TO_STORE,
        TRUE_FROM_STORE,
        TRUE_TO_PIPELINE - offset,
        variance_production=4.0,
        variance_to_store=1.0,
        variance_from_store=1.0,
        variance_to_pipeline=100.0,  # sensor claims low precision
    )
    assert precise.test_statistic > imprecise.test_statistic
    assert precise.gross_error
    assert not imprecise.gross_error


def test_nonpositive_variance_is_rejected() -> None:
    with pytest.raises(ValueError):
        reconcile_hydrogen_balance(
            TRUE_PRODUCTION,
            TRUE_TO_STORE,
            TRUE_FROM_STORE,
            TRUE_TO_PIPELINE,
            variance_production=0.0,
            variance_to_store=1.0,
            variance_from_store=1.0,
            variance_to_pipeline=1.0,
        )
