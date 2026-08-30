"""Hand-derived dispatch oracles and explicit failure contracts."""

import numpy as np
import pytest
import yaml

from optimization.dispatch import ElectrolyzerDispatchOptimizer


def optimizer(tmp_path, *, dt=1.0, demand_mw=20.0, ramp=0.4, pipeline=None):
    with open("configs/tiny_test.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    base = ElectrolyzerDispatchOptimizer("configs/tiny_test.yaml")
    config["simulation"]["time_step_hours"] = dt
    config["hydrogen_demand"]["daily_average_kg"] = demand_mw * base.h2_coeff * 24
    config["electrolyzer"]["ramp_rate_per_hour"] = ramp
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = ElectrolyzerDispatchOptimizer(str(path))
    if pipeline is not None:
        result.max_h2_kg_s = pipeline * result.h2_coeff / 3600
    return result


@pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
def test_cumulative_demand_is_rate_times_elapsed_hours(tmp_path, dt):
    opt = optimizer(tmp_path, dt=dt)
    result = opt.optimize([100.0, 100.0], [10.0, 10.0])
    assert result["status"] == "optimal"
    assert sum(result["power_schedule"]) * dt == pytest.approx(40.0 * dt)
    assert result["objective_value"] == pytest.approx(400.0 * dt)


def test_two_period_cost_optimum_has_known_dispatch(tmp_path):
    opt = optimizer(tmp_path)
    result = opt.optimize([100.0, 100.0], [1.0, 10.0])
    assert result["power_schedule"] == pytest.approx([40.0, 0.0])
    assert result["objective_value"] == pytest.approx(40.0)


def test_emissions_objective_reverses_cost_preference(tmp_path):
    opt = optimizer(tmp_path)
    result = opt.optimize(
        [100.0, 100.0],
        [1.0, 10.0],
        objective="minimize_emissions",
        carbon_intensity=[10.0, 1.0],
    )
    assert result["power_schedule"] == pytest.approx([0.0, 40.0])
    assert result["objective_value"] == pytest.approx(40.0)
    assert sum(result["cost_profile"]) == pytest.approx(400.0)


def test_ramp_is_scaled_to_timestep_duration(tmp_path):
    opt = optimizer(tmp_path, dt=0.5, demand_mw=30.0)
    result = opt.optimize([100.0, 100.0], [1.0, 10.0])
    assert result["power_schedule"] == pytest.approx([40.0, 20.0])


def test_hourly_shortfall_penalty_is_per_kg_not_per_rate(tmp_path):
    opt = optimizer(tmp_path, dt=0.5)
    result = opt.optimize([0.0], [10.0], demand_mode="hourly")
    assert result["power_schedule"] == pytest.approx([0.0])
    assert result["objective_value"] == pytest.approx(
        opt.hourly_demand_kg * 0.5 * opt.DEMAND_PENALTY
    )


def test_pipeline_capacity_limits_negative_price_dispatch(tmp_path):
    opt = optimizer(tmp_path, demand_mw=0.0, pipeline=25.0)
    result = opt.optimize([100.0], [-10.0])
    assert result["power_schedule"] == pytest.approx([25.0])
    assert result["online_status"] == pytest.approx([1.0])
    assert result["objective_value"] == pytest.approx(-250.0)


def test_wind_and_maximum_load_bound_negative_price_dispatch(tmp_path):
    opt = optimizer(tmp_path, demand_mw=0.0)
    result = opt.optimize([80.0, 150.0], [-10.0, -10.0])
    assert result["power_schedule"] == pytest.approx([80.0, 100.0])


@pytest.mark.parametrize(
    "wind,prices",
    [
        ([], []),
        ([1.0], []),
        ([-1.0], [1.0]),
        ([np.nan], [1.0]),
        ([1.0], [np.inf]),
        ([[1.0]], [1.0]),
    ],
)
def test_invalid_inputs_fail_before_model_build(tmp_path, wind, prices):
    with pytest.raises(ValueError):
        optimizer(tmp_path).optimize(wind, prices)


def test_unknown_demand_mode_does_not_silently_omit_demand(tmp_path):
    with pytest.raises(ValueError, match="demand_mode"):
        optimizer(tmp_path).optimize([100.0], [10.0], demand_mode="typo")


def test_nonfinite_carbon_input_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="carbon"):
        optimizer(tmp_path).optimize(
            [100.0], [10.0], objective="minimize_emissions", carbon_intensity=[np.nan]
        )


@pytest.mark.parametrize("carbon", [[np.nan], [-1.0], [np.inf]])
def test_cost_mode_still_validates_provided_carbon(tmp_path, carbon):
    with pytest.raises(ValueError, match="carbon"):
        optimizer(tmp_path).optimize([100.0], [10.0], carbon_intensity=carbon)


def test_nonhourly_summary_uses_interval_labels(tmp_path, capsys):
    opt = optimizer(tmp_path, dt=0.5)
    result = opt.optimize([100.0, 100.0], [10.0, 10.0])
    opt.print_solution_summary(result)
    output = capsys.readouterr().out
    assert "intervals" in output
    assert "Intervals demand met" in output
