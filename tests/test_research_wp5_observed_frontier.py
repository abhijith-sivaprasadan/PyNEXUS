"""Cost-frontier mathematics are unit-tested with conspicuously synthetic toy inputs."""

import copy

import pytest
import yaml

from research.interventions import PROTOCOL, registered_cases
from research.wp5_observed_frontier import PRICE_HASH, evaluate_frontier


def toy_inputs():
    order = yaml.safe_load(PROTOCOL.read_text())["order"]
    cases = [name for name, _ in registered_cases(order)]
    records = []
    for scenario in cases:
        records.append({
            "scenario": scenario,
            "demand_kg": 1000.0,
            "stage_b_shortfall_kg": 100.0,
            "shortfall_fraction": 0.1,
            "net_grid_settlement_eur_OBSERVED_2023": 200.0,
            "wind_nameplate_mw": 150.0,
            "electrolyser_capacity_mw": 125.0,
            "grid_capacity_mw": 50.0,
            "storage_capacity_kg": 20000.0,
        })
    matrix = {
        "schema_version": "pynexus.wp5.full_matrix_synthesis.v1",
        "status": "COMPLETE_VERIFIED_ANNUAL_PHYSICAL_AND_SETTLEMENT_SCREENING",
        "scenario_count": 17,
        "hours_per_case": 8760,
        "source_sha256": {"price_csv_sha256": PRICE_HASH},
        "records": records,
    }
    cost = {
        "schema_version": "pynexus.wp5.cost.v1",
        "currency": "EUR",
        "price_year": 2023,
        "discount_rate": 0.0,
        "discount_rate_source": "https://example.org/SIMULATION_NOT_REAL_SOURCE",
        "electricity_prices": "OBSERVED_NL_A44_2023",
        "includes_grid_tariffs": False,
        "variable_opex_eur_per_kg_delivered": 0.0,
        "variable_opex_source": "https://example.org/SIMULATION_NOT_REAL_SOURCE",
        "cost_scope_reviewed": True,
        "normalisation_method": (
            "UNIT TEST ONLY: all prices are invented simulated EUR 2023 values; "
            "zero discount; all real-source completeness checks must precede research."
        ),
        "assets": {},
    }
    for name in ("wind", "electrolyser", "grid_connection", "hydrogen_storage", "efficiency_upgrade"):
        key = "capex_eur_per_kg" if name == "hydrogen_storage" else "capex_eur_per_mw"
        cost["assets"][name] = {
            key: 2.0, "fixed_opex_fraction": 0.0, "lifetime_years": 10,
            "sources": {
                k: "https://example.org/SIMULATION_NOT_REAL_SOURCE"
                for k in (key, "fixed_opex_fraction", "lifetime_years")
            },
        }
    return matrix, cost


def test_conditional_full_cost_calculation_uses_scenario_specific_assets():
    matrix, costs = toy_inputs()
    output = evaluate_frontier(matrix, costs)
    assert len(output["records"]) == 17
    baseline = output["records"][0]
    assert baseline["annual_delivered_h2_kg"] == 900
    assert baseline["annual_wholesale_settlement_eur"] == 200
    assert baseline["annual_capital_recovery_eur"] > 0
    upgraded = next(x for x in output["records"] if x["scenario"] == "eta")
    assert upgraded["annual_capital_recovery_eur"] > baseline["annual_capital_recovery_eur"]
    assert "efficiency_upgrade" in upgraded["assets"]
    assert output["status"].startswith("CONDITIONAL")
    assert output["non_dominated_cases"]


def test_unpriced_costs_or_wrong_year_are_strictly_rejected():
    matrix, costs = toy_inputs()
    costs["assets"]["wind"]["capex_eur_per_mw"] = None
    with pytest.raises(ValueError, match="numeric"):
        evaluate_frontier(matrix, costs)
    matrix, costs = toy_inputs()
    costs["price_year"] = 2024
    with pytest.raises(ValueError, match="2023 EUR"):
        evaluate_frontier(matrix, costs)


def test_missing_scenarios_or_unreviewed_cost_scope_never_complete_frontier():
    matrix, costs = toy_inputs()
    matrix["records"].pop()
    with pytest.raises(ValueError, match="Changed or incomplete"):
        evaluate_frontier(matrix, costs)
    matrix, costs = toy_inputs()
    costs["cost_scope_reviewed"] = False
    with pytest.raises(ValueError, match="scope"):
        evaluate_frontier(matrix, costs)


def test_untraceable_cost_source_fails():
    matrix, costs = toy_inputs()
    costs = copy.deepcopy(costs)
    costs["assets"]["grid_connection"]["sources"]["capex_eur_per_mw"] = "ILLUSTRATIVE"
    with pytest.raises(ValueError, match="HTTPS"):
        evaluate_frontier(matrix, costs)
