"""WP5 observed-price incremental storage economics: no assumed vendor costs."""

import copy
import json
from zipfile import ZipFile

import pytest

from research.wp5_observed_storage import (
    PRICE_SHA256,
    compare,
    observed_case,
    priced_increment,
    validate_storage_costs,
)


def record(name, *, storage_kg, shortfall_kg, settlement, spent, revenue):
    return {
        "scenario": name,
        "manifest": {
            "source_wind_sha256": "matching_wind",
            "config_sha256": "matching_config",
            "price_csv_sha256": PRICE_SHA256,
        },
        "metrics": {
            "wind_nameplate_mw": 150,
            "electrolyser_capacity_mw": 125,
            "grid_capacity_mw": 50,
            "demand_h2_kg": 1000,
            "storage_capacity_kg": storage_kg,
            "stage_a_shortfall_kg": shortfall_kg,
            "stage_b_shortfall_kg": shortfall_kg,
        },
        "settlement_eur": settlement,
        "import_expense_eur": spent,
        "export_revenue_eur": revenue,
    }


def pair():
    e = record("E", storage_kg=20000, shortfall_kg=300,
               settlement=200, spent=450, revenue=250)
    es = record("E+S", storage_kg=40000, shortfall_kg=200,
                settlement=350, spent=480, revenue=130)
    return e, es


def sourced_costs():
    return {
        "schema_version": "pynexus.wp5.marginal_storage_costs.v1",
        "currency": "EUR",
        "price_year": 2023,
        "discount_rate": 0,
        "capex_eur_per_kg": 2,
        "fixed_opex_fraction": 0.10,
        "lifetime_years": 10,
        "variable_opex_eur_per_additional_kg": 0.50,
        "source_register": {
            k: "https://example.org/NOT-A-VERIFIED-SOURCE-FOR-UNIT-TEST"
            for k in ("capex", "opex", "asset_lifetime", "discount_rate", "variable_opex")
        },
        "engineering_scope_and_2023_normalisation": (
            "UNIT TEST ONLY: 2023 EUR, vessel, balance of plant included; "
            "no actual external cost data are represented here."
        ),
    }


def test_hand_calculated_increment_with_no_cost_assumptions():
    e, es = pair()
    c = compare(e, es)
    assert c["additional_delivered_hydrogen_kg"] == 100
    assert c["shortfall_improvement_stage_a_kg"] == 100
    assert c["additional_import_spend_eur"] == 30
    assert c["forgone_export_revenue_eur"] == 120
    assert c["additional_net_wholesale_grid_settlement_eur"] == 150
    assert c["additional_wholesale_settlement_eur_per_kg"] == 1.5
    assert c["cost_frontier_complete"] is False


def test_conditional_break_even_requires_sourced_inputs():
    base = compare(*pair())
    result = priced_increment(base, sourced_costs())
    # Additional 20,000 kg nameplate * 2 EUR/kg = 40,000 EUR capex.
    # Annualised 4000 at 0% for 10 years + annual fixed 4000,
    # variable 100*0.50 = 50 and wholesale 150 = 8200 / 100 kg.
    assert result["additional_modelled_annual_cost_eur"] == pytest.approx(8200)
    assert result["break_even_additional_hydrogen_value_eur_per_kg"] == pytest.approx(82)
    assert result["cost_frontier_complete"] is False


def test_cost_inputs_without_documented_financial_sources_are_rejected():
    c = sourced_costs()
    c["capex_eur_per_kg"] = None
    with pytest.raises(ValueError, match="numeric"):
        validate_storage_costs(c)
    c = sourced_costs()
    c["source_register"]["capex"] = ""
    with pytest.raises(ValueError, match="source"):
        validate_storage_costs(c)
    c = sourced_costs()
    c["price_year"] = 2025
    with pytest.raises(ValueError, match="2023 EUR"):
        validate_storage_costs(c)


def test_uncontrolled_source_prices_or_non_storage_changes_are_rejected():
    e, es = pair()
    es["manifest"]["price_csv_sha256"] = "different"
    with pytest.raises(ValueError, match="source/config/price"):
        compare(e, es)
    e, es = pair()
    es["metrics"]["grid_capacity_mw"] = 100
    with pytest.raises(ValueError, match="Non-storage"):
        compare(e, es)


def test_fake_relabelled_archives_never_get_a_real_price_claim(tmp_path):
    file = tmp_path / "forged.zip"
    with ZipFile(file, "w") as z:
        z.writestr("manifest.json", json.dumps({"full_year": True, "hours": 8760}))
    with pytest.raises(ValueError, match="correct complete"):
        observed_case(file, "E")


def test_price_register_cannot_be_edited_to_any_other_price_series():
    e, es = pair()
    alt = copy.deepcopy(es)
    alt["manifest"]["price_csv_sha256"] = "synthetic"
    with pytest.raises(ValueError):
        compare(e, alt)
