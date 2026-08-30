import numpy as np
import pytest

from optimization.dispatch import ElectrolyzerDispatchOptimizer

CONFIG = "configs/tiny_test.yaml"


def test_four_hour_cumulative_case_has_hand_checkable_energy_solution() -> None:
    optimiser = ElectrolyzerDispatchOptimizer(CONFIG)
    wind = np.full(4, 100.0)
    prices = np.full(4, 10.0)

    result = optimiser.optimize(wind, prices, demand_mode="cumulative")

    assert "optimal" in result["status"].lower()
    required_energy_mwh = optimiser.hourly_demand_kg * 4 / optimiser.h2_coeff
    assert result["power_schedule"].sum() == pytest.approx(required_energy_mwh)
    assert result["objective_value"] == pytest.approx(required_energy_mwh * 10.0)
    assert np.all(result["power_schedule"] <= wind)


def test_below_minimum_wind_forces_shutdown_in_hourly_mode() -> None:
    optimiser = ElectrolyzerDispatchOptimizer(CONFIG)
    result = optimiser.optimize(np.full(2, 5.0), np.full(2, 10.0), demand_mode="hourly")

    assert "optimal" in result["status"].lower()
    assert result["power_schedule"] == pytest.approx(np.zeros(2))
    assert np.all(result["slack_values"] > 0)
    assert not result["results_df"]["demand_met"].any()


def test_infeasible_cumulative_case_returns_status_without_crashing() -> None:
    optimiser = ElectrolyzerDispatchOptimizer(CONFIG)
    result = optimiser.optimize(np.zeros(4), np.full(4, 10.0), demand_mode="cumulative")

    assert result["status"] == "infeasible"
    assert result["results_df"] is None
