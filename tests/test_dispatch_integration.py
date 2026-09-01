"""All three Phase A/B opt-in features together: storage + grid + heat.

Each feature has its own dedicated test file (test_dispatch_storage_grid.py,
test_dispatch_heat.py) verifying it in isolation. This file exists because
composing all three was never actually exercised until asked to check it
manually — every constraint block references model variables/expressions
defined in a different `if enable_*` block, so a naming or ordering mistake
in any one of them could easily only show up when all three are active
together. Verifies every balance/recursion/terminal-condition holds
simultaneously, not just that the solver returns "optimal".
"""

import numpy as np
import pytest

from optimization.dispatch import ElectrolyzerDispatchOptimizer
from optimization.verification import verify_dispatch

CONFIG = "configs/tiny_test.yaml"


@pytest.fixture
def combined_result():
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    result = opt.optimize(
        [10.0, 200.0, 60.0, 90.0],
        [5.0, 50.0, -5.0, 20.0],
        demand_mode="hourly",
        enable_storage=True,
        enable_grid=True,
        enable_heat=True,
        heat_demand_mw=[3.0, 8.0, 4.0, 6.0],
        carbon_intensity=[100.0, 200.0, 150.0, 180.0],
    )
    return opt, result


def test_combined_run_is_optimal(combined_result) -> None:
    _, result = combined_result
    assert result["status"] == "optimal"


def test_combined_grid_balance_holds(combined_result) -> None:
    opt, result = combined_result
    df = result["results_df"]
    wind = np.array([10.0, 200.0, 60.0, 90.0])
    lhs = wind + df["grid_import_mw"]
    rhs = df["power_optimized_mw"] + df["grid_export_mw"] + df["curtailment_mw"]
    assert lhs.to_numpy() == pytest.approx(rhs.to_numpy(), abs=1e-6)


def test_combined_hydrogen_balance_holds(combined_result) -> None:
    opt, result = combined_result
    df = result["results_df"]
    produced = (
        df["power_optimized_mw"] * opt.h2_coeff
        - df["storage_charge_kg_h"]
        + df["storage_discharge_kg_h"]
    )
    assert (
        (produced + df["demand_slack_kg_h"]).to_numpy() >= opt.hourly_demand_kg - 1e-4
    ).all()


def test_combined_hydrogen_storage_recursion_and_terminal_condition(combined_result) -> None:
    opt, result = combined_result
    df = result["results_df"]
    s_prev = np.concatenate(([opt.storage_initial_kg], df["storage_level_kg"].to_numpy()[:-1]))
    loss = opt.storage_loss_fraction_per_hour * s_prev * opt.dt
    expected = s_prev + (df["storage_charge_kg_h"] - df["storage_discharge_kg_h"]) * opt.dt - loss
    assert df["storage_level_kg"].to_numpy() == pytest.approx(expected, abs=1e-4)
    assert df["storage_level_kg"].iloc[-1] >= opt.storage_final_min_kg - 1e-6


def test_combined_heat_balance_holds(combined_result) -> None:
    opt, result = combined_result
    df = result["results_df"]
    delivered = (
        df["waste_heat_recovered_mw"]
        - df["heat_storage_charge_mw"]
        - df["heat_dumped_mw"]
        + df["heat_storage_discharge_mw"]
        + df["boiler_output_mw"]
        + df["heat_slack_mw"]
    )
    assert (delivered.to_numpy() >= df["heat_demand_mw"].to_numpy() - 1e-4).all()


def test_combined_heat_storage_recursion_and_terminal_condition(combined_result) -> None:
    opt, result = combined_result
    df = result["results_df"]
    e_prev = np.concatenate(
        ([opt.heat_storage_initial_mwh], df["heat_storage_level_mwh"].to_numpy()[:-1])
    )
    loss = opt.heat_storage_loss_fraction_per_hour * e_prev * opt.dt
    expected = (
        e_prev + (df["heat_storage_charge_mw"] - df["heat_storage_discharge_mw"]) * opt.dt - loss
    )
    assert df["heat_storage_level_mwh"].to_numpy() == pytest.approx(expected, abs=1e-4)
    assert df["heat_storage_level_mwh"].iloc[-1] >= opt.heat_storage_final_min_mwh - 1e-6


def test_combined_waste_heat_never_exceeds_thermal_rejection(combined_result) -> None:
    opt, result = combined_result
    df = result["results_df"]
    total_rejected = df["power_optimized_mw"] * (1 - opt.eta_linearized)
    assert (df["waste_heat_recovered_mw"] <= total_rejected + 1e-6).all()


def test_verify_dispatch_passes_the_combined_run(combined_result) -> None:
    """verify_dispatch's storage/grid/heat-aware recomputation must agree with
    the solver's own objective for a run using all three features at once —
    not just report `passed` for whichever checks happen to be relevant."""
    opt, result = combined_result
    evidence = verify_dispatch(
        opt,
        result["results_df"],
        result["objective_value"],
        "minimize_cost",
        "hourly",
        carbon=None,
        enable_storage=True,
        enable_grid=True,
        enable_heat=True,
    )
    assert evidence["passed"], evidence["checks"]
    assert evidence["cost_objective"] == pytest.approx(result["objective_value"], abs=1e-4)


def test_verify_dispatch_catches_tampering_in_a_combined_run(combined_result) -> None:
    """A tampered dispatch.csv (post-hoc edited boiler output) must fail
    verification once storage/grid/heat checks are active, not just the
    baseline checks that existed before Phase A/B."""
    opt, result = combined_result
    tampered = result["results_df"].copy()
    tampered.loc[0, "boiler_output_mw"] += 5.0  # inconsistent with the heat balance now

    evidence = verify_dispatch(
        opt,
        tampered,
        result["objective_value"],
        "minimize_cost",
        "hourly",
        carbon=None,
        enable_storage=True,
        enable_grid=True,
        enable_heat=True,
    )
    assert not evidence["passed"]
