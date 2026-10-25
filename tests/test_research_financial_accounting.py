"""Unit tests for WP5 accounting: no network or observed-market claims."""

import copy

import pytest

from research.financial_accounting import (
    annuity_factor,
    evaluate_case,
    evaluate_matrix,
    pareto_frontier,
    validate_assumptions,
)
from research.interventions import registered_cases


def assumptions():
    def asset(capex_key, capex_value):
        return {
            capex_key: capex_value,
            "fixed_opex_fraction": 0.1,
            "lifetime_years": 2,
            "sources": {
                capex_key: "ILLUSTRATIVE_TEST_INPUT",
                "fixed_opex_fraction": "ILLUSTRATIVE_TEST_INPUT",
                "lifetime_years": "ILLUSTRATIVE_TEST_INPUT",
            },
        }

    return {
        "schema_version": "pynexus.wp5.cost.v1",
        "currency": "EUR",
        "price_year": 2025,
        "discount_rate": 0,
        "discount_rate_source": "ILLUSTRATIVE_TEST_INPUT",
        "electricity_prices": "SYNTHETIC_2023_ARCHIVE",
        "includes_grid_tariffs": False,
        "variable_opex_eur_per_kg_delivered": 1,
        "variable_opex_source": "ILLUSTRATIVE_TEST_INPUT",
        "assets": {
            "wind": asset("capex_eur_per_mw", 1),
            "electrolyser": asset("capex_eur_per_mw", 2),
            "grid_connection": asset("capex_eur_per_mw", 3),
            "hydrogen_storage": asset("capex_eur_per_kg", 0.001),
            "efficiency_upgrade": asset("capex_eur_per_mw", 0.5),
        },
    }


def record(name="baseline", active=()):
    return {
        "scenario": name,
        "active_interventions": list(active),
        "status": "VERIFIED_TWO_STAGE_SOLVE",
        "metrics": {
            "demand_h2_kg": 1000,
            "stage_a_shortfall_kg": 100,
            "stage_b_shortfall_kg": 100,
            "stage_b_grid_settlement_eur_synthetic_prices": 50,
            "wind_nameplate_mw": 150,
            "electrolyser_capacity_mw": 100,
            "grid_capacity_mw": 50,
            "storage_capacity_kg": 20000,
        },
    }


def matrix():
    return {
        "status": "COMPLETE_REGISTERED_MATRIX",
        "hours": 8760,
        "is_full_year": True,
        "price_data_kind": "synthetic_daily_sinusoid_NOT_historical_market_data",
        "cases": [
            record(name, active) for name, active in registered_cases(["W", "E", "G", "S", "eta"])
        ],
    }


def test_annuity_factor_known_values():
    assert annuity_factor(0, 2) == pytest.approx(0.5)
    assert annuity_factor(0.1, 2) == pytest.approx(0.5761904761904761)
    with pytest.raises(ValueError):
        annuity_factor(-0.1, 10)
    with pytest.raises(ValueError):
        annuity_factor(0.1, 0)


def test_one_year_hand_calculated_economics_with_synthetic_price_flag():
    cfg = assumptions()
    rate = validate_assumptions(cfg)
    value = evaluate_case(record(), cfg, rate)
    # Capex: 150 wind + 200 electro + 150 grid + 20 storage = 520.
    # Annuity over two years at 0%: 260, fixed OPEX 52, variable 900,
    # synthetic grid settlement 50, giving 1262 EUR.
    assert value["installed_capex_eur"] == pytest.approx(520)
    assert value["annualised_capex_eur"] == pytest.approx(260)
    assert value["annual_fixed_opex_eur"] == pytest.approx(52)
    assert value["annual_variable_opex_eur"] == pytest.approx(900)
    assert value["annual_cost_eur_ILLUSTRATIVE"] == pytest.approx(1262)
    assert value["unit_cost_eur_per_kg_delivered_ILLUSTRATIVE"] == pytest.approx(1262 / 900)


def test_efficiency_premium_added_not_assumed_free():
    cfg = assumptions()
    base = evaluate_case(record(), cfg, 0)
    upgraded = evaluate_case(record("eta", ("eta",)), cfg, 0)
    # Premium: 100 MW * 0.5 EUR/MW = 50 EUR capex;
    # annualised = 25 EUR/year and fixed OPEX = 5 EUR/year.
    assert upgraded["annual_cost_eur_ILLUSTRATIVE"] == pytest.approx(
        base["annual_cost_eur_ILLUSTRATIVE"] + 30
    )


def test_source_and_cost_completeness_is_mandatory():
    cfg = assumptions()
    del cfg["assets"]["grid_connection"]
    with pytest.raises(ValueError, match="All four"):
        validate_assumptions(cfg)
    cfg = assumptions()
    cfg["assets"]["wind"]["sources"]["capex_eur_per_mw"] = ""
    with pytest.raises(ValueError, match="source"):
        validate_assumptions(cfg)
    cfg = assumptions()
    cfg["assets"]["wind"]["sources"]["capex_eur_per_mw"] = "REPLACE_WITH_TRACEABLE_SOURCE"
    with pytest.raises(ValueError, match="source"):
        validate_assumptions(cfg)
    cfg = assumptions()
    cfg["electricity_prices"] = "REAL_NL_2023"
    with pytest.raises(ValueError, match="re-label"):
        validate_assumptions(cfg)


def test_stage_b_shortfall_must_respect_adequacy_cap():
    cfg = assumptions()
    case = record()
    case["metrics"]["stage_b_shortfall_kg"] = 100.5
    with pytest.raises(ValueError, match="cap"):
        evaluate_case(case, cfg, 0)


def test_complete_registration_and_explicit_acknowledgment_required():
    cfg = assumptions()
    data = matrix()
    with pytest.raises(ValueError, match="acknowledge"):
        evaluate_matrix(data, cfg, acknowledge_synthetic=False)
    result = evaluate_matrix(data, cfg, acknowledge_synthetic=True)
    assert len(result["cases"]) == 17
    assert result["not_a_bankable_lcoh_or_project_feasibility_estimate"]
    assert result["price_data_are_not_historical_market_prices"]
    truncated = copy.deepcopy(data)
    truncated["cases"].pop()
    with pytest.raises(ValueError, match="17-case"):
        evaluate_matrix(truncated, cfg, acknowledge_synthetic=True)


def test_pareto_removes_strictly_dominated_options():
    points = [
        {
            "scenario": "a",
            "annual_cost_eur_ILLUSTRATIVE": 10,
            "adequacy_shortfall_fraction_stage_a": 0.2,
        },
        {
            "scenario": "b",
            "annual_cost_eur_ILLUSTRATIVE": 20,
            "adequacy_shortfall_fraction_stage_a": 0.1,
        },
        {
            "scenario": "c",
            "annual_cost_eur_ILLUSTRATIVE": 30,
            "adequacy_shortfall_fraction_stage_a": 0.3,
        },
    ]
    assert pareto_frontier(points) == ["a", "b"]
