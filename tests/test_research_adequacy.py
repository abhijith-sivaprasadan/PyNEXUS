"""WP1: hand-checkable adequacy-first and grid-settlement regression tests."""

import numpy as np
import pytest

from optimization.dispatch import ElectrolyzerDispatchOptimizer
from optimization.verification import verify_dispatch
from research.adequacy import solve_adequacy_first


def opt():
    return ElectrolyzerDispatchOptimizer("configs/tiny_test.yaml")


def test_stage_a_shortfall_is_independent_of_price_and_stage_b_respects_cap():
    o = opt()
    inputs = np.array([0.0, 0.0])
    expensive = solve_adequacy_first(o, inputs, np.array([80.0, 100.0]), enable_storage=False)
    cheap = solve_adequacy_first(o, inputs, np.array([5.0, 10.0]), enable_storage=False)
    expected = 2 * (o.hourly_demand_kg - o.grid_connection_mw * o.h2_coeff)
    assert expensive["stage_a_unserved_kg"] == pytest.approx(expected, rel=1e-7)
    assert cheap["stage_a_unserved_kg"] == pytest.approx(expected, rel=1e-7)
    assert expensive["stage_b_unserved_kg"] <= expensive["shortfall_cap_kg"] + 1e-5
    assert expensive["verification_a"]["passed"]
    assert expensive["verification_b"]["passed"]


def test_stage_b_grid_cashflow_does_not_charge_imports_twice():
    o = opt()
    h2_price = np.array([12.0])
    a = o.optimize(
        [0.0], h2_price, objective="minimize_unserved", demand_mode="hourly", enable_grid=True
    )
    assert a["status"] == "optimal"
    b = o.optimize(
        [0.0],
        h2_price,
        objective="minimize_cash_cost",
        demand_mode="hourly",
        enable_grid=True,
        shortfall_cap_kg=a["objective_value"] + 1e-4,
    )
    df = b["results_df"]
    expected = float(((df["grid_import_mw"] - df["grid_export_mw"]) * h2_price[0] * o.dt).sum())
    assert b["objective_value"] == pytest.approx(expected, abs=1e-5)
    assert b["objective_value"] == pytest.approx(12.0 * o.grid_connection_mw)
    assert verify_dispatch(
        o,
        df,
        b["objective_value"],
        "minimize_cash_cost",
        "hourly",
        enable_grid=True,
        shortfall_cap_kg=a["objective_value"] + 1e-4,
    )["passed"]


def test_zero_shortfall_with_enough_wind_and_grid():
    o = opt()
    result = solve_adequacy_first(o, [100.0, 100.0], [5.0, 20.0], enable_storage=False)
    assert result["stage_a_unserved_kg"] == pytest.approx(0.0, abs=1e-5)
    assert result["stage_b_unserved_kg"] <= result["numerical_allowance_kg"] + 1e-5


@pytest.mark.parametrize("cap", [None, -1.0, float("nan")])
def test_cashflow_requires_explicit_valid_shortfall_cap(cap):
    with pytest.raises(ValueError, match="shortfall_cap_kg"):
        opt().optimize(
            [100.0],
            [10.0],
            objective="minimize_cash_cost",
            demand_mode="hourly",
            enable_grid=True,
            shortfall_cap_kg=cap,
        )


def test_research_and_legacy_objectives_have_distinct_units():
    o = opt()
    legacy = o.optimize([100.0, 100.0], [1.0, 10.0])
    a = o.optimize(
        [100.0, 100.0],
        [1.0, 10.0],
        objective="minimize_unserved",
        demand_mode="hourly",
        enable_grid=True,
    )
    assert legacy["objective_definition"] == "legacy_electricity_plus_grid_and_shortfall_penalty"
    assert a["objective_definition"] == "total_unserved_hydrogen_kg"
    expected_legacy = float(
        (legacy["results_df"]["power_optimized_mw"]
         * legacy["results_df"]["electricity_price"] * o.dt).sum()
    )
    assert legacy["status"] == "optimal"
    assert legacy["objective_value"] == pytest.approx(expected_legacy, rel=1e-8)
    assert a["objective_value"] == pytest.approx(0.0, abs=1e-5)


def test_research_objectives_reject_incompatible_scope():
    o = opt()
    with pytest.raises(ValueError, match="hourly"):
        o.optimize([100.0], [10.0], objective="minimize_unserved")
    with pytest.raises(ValueError, match="enable_grid"):
        o.optimize(
            [100.0],
            [10.0],
            objective="minimize_cash_cost",
            demand_mode="hourly",
            shortfall_cap_kg=0.0,
        )
    with pytest.raises(ValueError, match="heat coupling"):
        o.optimize(
            [100.0], [10.0], objective="minimize_unserved", demand_mode="hourly", enable_heat=True
        )


def test_adequacy_wrapper_fails_on_unacceptable_gap():
    o = opt()
    o.mip_gap = 0.01
    with pytest.raises(ValueError, match="mip_rel_gap"):
        solve_adequacy_first(o, [100.0], [10.0], enable_storage=False)
