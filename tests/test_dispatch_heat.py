"""Phase B1-B3: electrolyser waste heat, heat storage, backup boiler,
coupled objective (opt-in, off by default)."""

import numpy as np
import pytest

from optimization.dispatch import ElectrolyzerDispatchOptimizer

CONFIG = "configs/tiny_test.yaml"


def test_heat_disabled_by_default_matches_pre_phase_b_behaviour() -> None:
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    implicit_default = opt.optimize([100.0, 100.0], [1.0, 10.0])
    explicit_false = opt.optimize([100.0, 100.0], [1.0, 10.0], enable_heat=False)
    assert implicit_default["objective_value"] == pytest.approx(explicit_false["objective_value"])
    assert "waste_heat_recovered_mw" not in implicit_default["results_df"].columns


def test_enable_heat_requires_heat_demand() -> None:
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    with pytest.raises(ValueError, match="heat_demand_mw"):
        opt.optimize([100.0], [10.0], demand_mode="hourly", enable_heat=True)


def test_enable_heat_without_boiler_config_is_rejected(tmp_path) -> None:
    import yaml

    with open(CONFIG, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    cfg["boiler"]["max_output_mw_th"] = 0
    path = tmp_path / "no_boiler.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    opt = ElectrolyzerDispatchOptimizer(str(path))

    with pytest.raises(ValueError, match="max_output_mw_th"):
        opt.optimize([100.0], [10.0], demand_mode="hourly", enable_heat=True, heat_demand_mw=[5.0])


def test_waste_heat_never_exceeds_electrolyser_thermal_rejection() -> None:
    """q_wh[t] = p[t] * (1 - eta) * f_recoverable must never exceed the total
    energy the electrolyser actually rejected, p[t] * (1 - eta) — i.e.
    f_recoverable must genuinely act as a <=1 fraction, not manufacture heat."""
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    result = opt.optimize(
        [100.0, 60.0, 30.0, 0.0],
        [10.0, 1.0, 10.0, 1.0],
        demand_mode="hourly",
        enable_heat=True,
        heat_demand_mw=[2.0, 2.0, 2.0, 2.0],
    )
    assert result["status"] == "optimal"
    df = result["results_df"]
    total_rejected = df["power_optimized_mw"] * (1 - opt.eta_linearized)
    assert (df["waste_heat_recovered_mw"] <= total_rejected + 1e-6).all()
    assert df["waste_heat_recovered_mw"].to_numpy() == pytest.approx(
        (total_rejected * opt.recoverable_heat_fraction).to_numpy(), abs=1e-6
    )


def test_heat_balance_closes_every_timestep() -> None:
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    result = opt.optimize(
        [100.0, 20.0, 80.0, 40.0],
        [10.0, 1.0, 5.0, 1.0],
        demand_mode="hourly",
        enable_heat=True,
        heat_demand_mw=[8.0, 8.0, 8.0, 8.0],
    )
    assert result["status"] == "optimal"
    df = result["results_df"]
    delivered = (
        df["waste_heat_recovered_mw"]
        - df["heat_storage_charge_mw"]
        - df["heat_dumped_mw"]
        + df["heat_storage_discharge_mw"]
        + df["boiler_output_mw"]
        + df["heat_slack_mw"]
    )
    assert (delivered.to_numpy() >= df["heat_demand_mw"].to_numpy() - 1e-6).all()


def test_heat_storage_soc_closes_over_the_horizon() -> None:
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    result = opt.optimize(
        [100.0, 100.0, 100.0, 100.0],
        [10.0, 1.0, 10.0, 1.0],
        demand_mode="hourly",
        enable_heat=True,
        heat_demand_mw=[5.0, 5.0, 5.0, 5.0],
    )
    assert result["status"] == "optimal"
    df = result["results_df"]
    e_prev = np.concatenate(
        ([opt.heat_storage_initial_mwh], df["heat_storage_level_mwh"].to_numpy()[:-1])
    )
    loss = opt.heat_storage_loss_fraction_per_hour * e_prev * opt.dt
    expected = (
        e_prev + (df["heat_storage_charge_mw"] - df["heat_storage_discharge_mw"]) * opt.dt - loss
    )
    assert df["heat_storage_level_mwh"].to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-6)
    assert df["heat_storage_level_mwh"].iloc[-1] >= opt.heat_storage_final_min_mwh - 1e-6


def test_boiler_backs_up_when_waste_heat_insufficient(tmp_path) -> None:
    """Electrolyzer off (no wind) means no waste heat at all; a real heat
    demand can only be met by the boiler. Heat storage is disabled for this
    scenario: with zero waste heat, storage can only ever discharge or leak
    (q_hs_in is capped by q_wh = 0), so a nonzero standing loss combined
    with a terminal floor would make this genuinely infeasible — the same
    real limitation already covered for hydrogen storage."""
    import yaml

    with open(CONFIG, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    cfg["heat_storage"]["capacity_mwh_th"] = 0
    path = tmp_path / "no_heat_storage.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    opt = ElectrolyzerDispatchOptimizer(str(path))

    result = opt.optimize(
        [0.0, 0.0], [10.0, 10.0], demand_mode="hourly", enable_heat=True, heat_demand_mw=[5.0, 5.0]
    )
    assert result["status"] == "optimal"
    df = result["results_df"]
    assert (df["waste_heat_recovered_mw"] == 0).all()
    assert (df["boiler_output_mw"] > 0).all()
    assert (df["boiler_output_mw"] <= opt.boiler_max_output_mw + 1e-6).all()


def test_coupled_objective_independently_reconstructed() -> None:
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    result = opt.optimize(
        [100.0, 20.0, 80.0, 40.0],
        [10.0, 1.0, 5.0, 1.0],
        demand_mode="hourly",
        enable_heat=True,
        heat_demand_mw=[8.0, 8.0, 8.0, 8.0],
    )
    assert result["status"] == "optimal"
    df = result["results_df"]

    electricity_cost = (df["power_optimized_mw"] * df["electricity_price"] * opt.dt).sum()
    demand_penalty = opt.DEMAND_PENALTY * df["demand_slack_kg_h"].sum() * opt.dt
    boiler_cost = (opt.boiler_fuel_cost_eur_per_mwh * df["boiler_output_mw"] * opt.dt).sum()
    heat_penalty = opt.HEAT_DEMAND_PENALTY * df["heat_slack_mw"].sum() * opt.dt
    h2_value = opt.hydrogen_value_per_kg * (df["h2_produced_kg_h"] * opt.dt).sum()
    heat_delivered = (
        df["waste_heat_recovered_mw"]
        - df["heat_storage_charge_mw"]
        - df["heat_dumped_mw"]
        + df["heat_storage_discharge_mw"]
    )
    heat_value = opt.heat_value_per_mwh * (heat_delivered * opt.dt).sum()

    recomputed = (
        electricity_cost + demand_penalty + boiler_cost + heat_penalty - h2_value - heat_value
    )
    assert result["objective_value"] == pytest.approx(recomputed, abs=1e-4)


def test_model_size_grows_with_heat_enabled() -> None:
    opt = ElectrolyzerDispatchOptimizer(CONFIG)
    baseline = opt.optimize([100.0, 100.0], [1.0, 10.0])
    enlarged = opt.optimize(
        [100.0, 100.0],
        [1.0, 10.0],
        demand_mode="hourly",
        enable_heat=True,
        heat_demand_mw=[5.0, 5.0],
    )
    assert enlarged["variables"] > baseline["variables"]
    assert enlarged["constraints"] > baseline["constraints"]
