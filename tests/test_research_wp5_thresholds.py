"""WP5 no-invented-input threshold tests."""
import copy

import pytest

from research.interventions import registered_cases
from research.wp5_thresholds import calculate_thresholds


def matrix():
    rows = []
    for name, active in registered_cases(["W", "E", "G", "S", "eta"]):
        shortage = 100.0
        grid_cost = 10.0
        storage = 40.0 if "S" in active else 20.0
        electro = 125.0 if "E" in active else 100.0
        if name == "E":
            shortage, grid_cost = 80.0, 15.0
        if name == "E+S":
            shortage, grid_cost = 60.0, 27.0
        rows.append({
            "scenario": name,
            "active_interventions": list(active),
            "status": "VERIFIED_TWO_STAGE_SOLVE",
            "metrics": {
                "demand_h2_kg": 1000.0,
                "stage_a_shortfall_kg": shortage,
                "stage_b_shortfall_kg": shortage,
                "stage_b_grid_settlement_eur_synthetic_prices": grid_cost,
                "storage_capacity_kg": storage,
                "electrolyser_capacity_mw": electro,
                "wind_nameplate_mw": 150.0,
                "grid_capacity_mw": 50.0,
            },
        })
    return {
        "status": "COMPLETE_REGISTERED_MATRIX",
        "hours": 8760,
        "is_full_year": True,
        "price_data_kind": "synthetic_daily_sinusoid_NOT_historical_market_data",
        "cases": rows,
    }


def test_thresholds_arithmetic_and_scope():
    result = calculate_thresholds(matrix())
    assert result["storage_alone_adequacy_gain_kg"] == 0
    assert result["extra_h2_delivered_E_plus_S_vs_E_kg"] == pytest.approx(20)
    assert result["incremental_grid_settlement_eur_SYNTHETIC"] == pytest.approx(12)
    assert result["incremental_grid_settlement_per_extra_kg_eur_SYNTHETIC"] == pytest.approx(0.6)
    assert result["adequacy_gain_E_plus_S_vs_E_percentage_points"] == pytest.approx(2)
    assert result["not_an_lcoh_or_investment_recommendation"]


@pytest.mark.parametrize("mutation", ["missing", "synthetic_flag", "not_verified", "not_beneficial", "asset_drift"])
def test_fail_closed(mutation):
    record = copy.deepcopy(matrix())
    if mutation == "missing":
        record["cases"].pop()
    elif mutation == "synthetic_flag":
        record["price_data_kind"] = "ACTUAL_NL_PRICES"
    elif mutation == "not_verified":
        record["cases"][0]["status"] = "UNVERIFIED"
    elif mutation == "not_beneficial":
        next(c for c in record["cases"] if c["scenario"] == "E+S")["metrics"][
            "stage_a_shortfall_kg"
        ] = 80
    else:
        next(c for c in record["cases"] if c["scenario"] == "E+S")["metrics"][
            "electrolyser_capacity_mw"
        ] = 150
    with pytest.raises(ValueError):
        calculate_thresholds(record)
