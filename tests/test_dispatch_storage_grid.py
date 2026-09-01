"""Phase A1/A2: hydrogen storage and grid import/export (opt-in, off by default)."""

import numpy as np
import pytest

from optimization.dispatch import ElectrolyzerDispatchOptimizer
from optimization.verification import verify_dispatch

CONFIG = "configs/tiny_test.yaml"


def test_storage_disabled_by_default_matches_pre_phase_a_behaviour() -> None:
    """enable_storage/enable_grid default False must be a true no-op versus the
    pre-Phase-A model — the actual locked reference value (825902.3904382455 EUR
    for the 168h synthetic case) is checked directly in docs/reproducibility.md's
    own workflow and by test_dispatch_contract.py; this checks the *mechanism*
    (explicit False args change nothing) rather than re-asserting that number."""
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    implicit_default = opt.optimize([100.0, 100.0], [1.0, 10.0])
    explicit_false = opt.optimize(
        [100.0, 100.0], [1.0, 10.0], enable_storage=False, enable_grid=False
    )
    assert implicit_default["objective_value"] == pytest.approx(explicit_false["objective_value"])
    assert implicit_default["power_schedule"] == pytest.approx(explicit_false["power_schedule"])
    assert "storage_level_kg" not in implicit_default["results_df"].columns


def test_storage_balance_closes_over_the_full_horizon() -> None:
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    result = opt.optimize(
        [100.0, 100.0, 100.0, 100.0],
        [10.0, 10.0, 10.0, 10.0],
        demand_mode="hourly",
        enable_storage=True,
    )
    assert result["status"] == "optimal"
    df = result["results_df"]
    s_prev = np.concatenate(([opt.storage_initial_kg], df["storage_level_kg"].to_numpy()[:-1]))
    loss = opt.storage_loss_fraction_per_hour * s_prev * opt.dt
    expected_s = s_prev + (df["storage_charge_kg_h"] - df["storage_discharge_kg_h"]) * opt.dt - loss
    assert df["storage_level_kg"].to_numpy() == pytest.approx(expected_s.to_numpy(), abs=1e-6)


def test_storage_terminal_condition_prevents_end_of_horizon_dumping() -> None:
    """Without the terminal condition the optimiser would drain the store for free profit."""
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    # Cheap late price would otherwise tempt the optimiser to discharge everything
    # on the last hour and never restock, since discharging has no direct cost.
    result = opt.optimize(
        [0.0, 0.0, 0.0, 100.0],
        [10.0, 10.0, 10.0, 10.0],
        demand_mode="hourly",
        enable_storage=True,
    )
    assert result["status"] == "optimal"
    final_level = result["results_df"]["storage_level_kg"].iloc[-1]
    assert final_level >= opt.storage_final_min_kg - 1e-6


def test_storage_charge_cannot_exceed_production() -> None:
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    result = opt.optimize(
        [100.0, 100.0, 100.0, 100.0],
        [10.0, 1.0, 10.0, 1.0],
        demand_mode="hourly",
        enable_storage=True,
    )
    assert result["status"] == "optimal"
    df = result["results_df"]
    assert (df["storage_charge_kg_h"] <= df["power_optimized_mw"] * opt.h2_coeff + 1e-6).all()


def test_zero_production_with_lossy_storage_and_a_floor_is_correctly_infeasible() -> None:
    """With no wind at all, storage can only leak (h_in is bounded by production,
    which is zero) — requiring it to still meet final_level_min_kg after a
    nonzero boil-off loss is a genuine infeasibility, not a solver or model bug.
    This is the failure mode a fail-loud model should surface, not paper over."""
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    result = opt.optimize([0.0, 0.0], [10.0, 10.0], demand_mode="hourly", enable_storage=True)
    assert result["status"] == "infeasible"
    assert result["results_df"] is None


def test_enable_storage_without_config_capacity_is_rejected(tmp_path) -> None:
    import yaml

    with open(CONFIG, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    cfg["hydrogen_storage"]["capacity_kg"] = 0
    path = tmp_path / "no_storage.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    opt = ElectrolyzerDispatchOptimizer(str(path))

    with pytest.raises(ValueError, match="capacity_kg"):
        opt.optimize([100.0], [10.0], demand_mode="hourly", enable_storage=True)


# --- Grid import/export -----------------------------------------------------


def test_grid_disabled_by_default_matches_pre_phase_a_behaviour() -> None:
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    result = opt.optimize([80.0, 150.0], [-10.0, -10.0], demand_mode="hourly")
    assert np.all(result["power_schedule"] <= [80.0, 150.0])
    assert (result["results_df"]["grid_import_mw"] == 0).all()
    assert (result["results_df"]["grid_export_mw"] == 0).all()


def test_grid_import_respects_connection_limit() -> None:
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    # Wind is far below what's needed for the electrolyzer to run at all;
    # only grid import can supply it.
    result = opt.optimize([0.0], [10.0], demand_mode="hourly", enable_grid=True)
    assert result["status"] == "optimal"
    assert result["results_df"]["grid_import_mw"].iloc[0] <= opt.grid_connection_mw + 1e-6


def test_grid_export_respects_connection_limit() -> None:
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    # Huge wind surplus with electrolyzer maxed out; excess must be export-or-curtail,
    # and export is capped at the connection limit.
    result = opt.optimize([500.0], [10.0], demand_mode="hourly", enable_grid=True)
    assert result["status"] == "optimal"
    assert result["results_df"]["grid_export_mw"].iloc[0] <= opt.grid_connection_mw + 1e-6


def test_grid_balance_holds_every_hour() -> None:
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    result = opt.optimize(
        [10.0, 200.0, 60.0], [5.0, 50.0, -5.0], demand_mode="hourly", enable_grid=True
    )
    assert result["status"] == "optimal"
    df = result["results_df"]
    lhs = df["wind_available_mw"] + df["grid_import_mw"]
    rhs = df["power_optimized_mw"] + df["grid_export_mw"] + df["curtailment_mw"]
    assert lhs.to_numpy() == pytest.approx(rhs.to_numpy(), abs=1e-6)


def test_enable_grid_without_config_connection_is_rejected(tmp_path) -> None:
    import yaml

    with open(CONFIG, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    cfg["grid"]["connection_capacity_mw"] = 0
    path = tmp_path / "no_grid.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    opt = ElectrolyzerDispatchOptimizer(str(path))

    with pytest.raises(ValueError, match="connection_capacity_mw"):
        opt.optimize([100.0], [10.0], demand_mode="hourly", enable_grid=True)


# --- Objective independently reconstructed (enlarged model) ----------------


def test_objective_reconstructed_for_storage_and_grid_enabled_run() -> None:
    """verify_dispatch's cost recomputation must account for grid import/export,
    not just electrolyzer cost, once grid is enabled — otherwise a real dispatch
    cost would silently fail verification."""
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    result = opt.optimize(
        [10.0, 200.0, 60.0, 90.0],
        [5.0, 50.0, -5.0, 20.0],
        demand_mode="hourly",
        enable_storage=True,
        enable_grid=True,
    )
    assert result["status"] == "optimal"
    df = result["results_df"]

    recomputed_cost = (
        (df["power_optimized_mw"] * df["electricity_price"] * opt.dt).sum()
        + (df["grid_import_mw"] * df["electricity_price"] * opt.dt).sum()
        - (df["grid_export_mw"] * df["electricity_price"] * opt.dt).sum()
        + opt.DEMAND_PENALTY * df["demand_slack_kg_h"].sum() * opt.dt
    )
    assert result["objective_value"] == pytest.approx(recomputed_cost, abs=1e-6)


def test_model_size_grows_with_storage_and_grid_enabled() -> None:
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    baseline = opt.optimize([100.0, 100.0], [1.0, 10.0])
    enlarged = opt.optimize(
        [100.0, 100.0], [1.0, 10.0], demand_mode="hourly", enable_storage=True, enable_grid=True
    )
    assert enlarged["variables"] > baseline["variables"]
    assert enlarged["constraints"] > baseline["constraints"]


def test_verify_dispatch_still_passes_baseline_run_unaffected() -> None:
    """verify_dispatch itself is unchanged by Phase A; confirm the pre-existing
    (no storage, no grid) verification path still passes end to end."""
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    result = opt.optimize([100.0, 100.0], [1.0, 10.0])
    evidence = verify_dispatch(
        opt, result["results_df"], result["objective_value"], "minimize_cost", "cumulative"
    )
    assert evidence["passed"]
