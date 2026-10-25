"""WP5 research accounting: illustrative annualised cost / adequacy frontier.

No tariff, grid-service, emissions, certification, market-price or bankable LCOH
claims are made here. This is a post-solve calculation over independently
verified WP2 dispatch. The original Stage B price series is synthetic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.attribution import analyse_archive
from research.interventions import registered_cases
from research.run_reference import sha256

ASSET_UNITS = {
    "wind": ("wind_nameplate_mw", "capex_eur_per_mw"),
    "electrolyser": ("electrolyser_capacity_mw", "capex_eur_per_mw"),
    "grid_connection": ("grid_capacity_mw", "capex_eur_per_mw"),
    "hydrogen_storage": ("storage_capacity_kg", "capex_eur_per_kg"),
}
UNPRICED_ITEMS = (
    "grid tariffs and access charges",
    "compression and additional balance-of-plant not separately represented",
    "water supply and treatment",
    "equipment replacement and degradation",
    "hydrogen delivery, transport and certification",
    "taxes, insurance and decommissioning",
    "demand non-delivery consequences",
)


def number(value, label, *, positive=False):
    """Validate scientific inputs without accepting booleans or NaNs."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not np.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'} and finite")
    return result


def annuity_factor(rate, lifetime_years):
    """Capital-recovery factor in 1/year; zero-discount case is defined."""
    r = number(rate, "discount_rate")
    n = number(lifetime_years, "lifetime_years", positive=True)
    if r == 0:
        return 1.0 / n
    return r / (-np.expm1(-n * np.log1p(r)))


def source_text(text, label):
    if not isinstance(text, str) or len(text.strip()) < 6:
        raise ValueError(f"{label}: an identifiable source or 'ILLUSTRATIVE' is required")
    return text.strip()


def validate_assumptions(data):
    """Reject unpriced assets and untraceable reference assumptions.

    Source strings establish declared provenance, not that a source has been
    independently checked or is applicable to the stylised North Sea case.
    """
    if data.get("schema_version") != "pynexus.wp5.cost.v1":
        raise ValueError("Expected WP5 cost schema v1")
    if data.get("currency") != "EUR":
        raise ValueError("WP5 requires EUR values on one common price-year basis")
    year = data.get("price_year")
    if isinstance(year, bool) or not isinstance(year, int) or not 2000 <= year <= 2100:
        raise ValueError("Specify a common numerical price_year")
    rate = number(data.get("discount_rate"), "discount_rate")
    source_text(data.get("discount_rate_source"), "discount_rate_source")
    assets = data.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(ASSET_UNITS) | {"efficiency_upgrade"}:
        raise ValueError("All four base assets and the efficiency-upgrade premium are required")
    for name, (_, capex_key) in {
        **ASSET_UNITS,
        "efficiency_upgrade": ("electrolyser_capacity_mw", "capex_eur_per_mw"),
    }.items():
        spec = assets[name]
        if not isinstance(spec, dict):
            raise ValueError(f"{name} needs a cost definition")
        for key in (capex_key, "fixed_opex_fraction", "lifetime_years"):
            number(spec.get(key), f"{name}.{key}", positive=(key == "lifetime_years"))
        if number(spec["fixed_opex_fraction"], f"{name}.fixed_opex_fraction") > 1:
            raise ValueError("Fixed OPEX fraction must not exceed 1/year")
        sources = spec.get("sources")
        if not isinstance(sources, dict):
            raise ValueError(f"{name}: per-parameter sources are required")
        for key in (capex_key, "fixed_opex_fraction", "lifetime_years"):
            source_text(sources.get(key), f"{name}.{key} source")
    number(data.get("variable_opex_eur_per_kg_delivered"), "variable_opex_eur_per_kg_delivered")
    source_text(data.get("variable_opex_source"), "variable_opex_source")
    if data.get("includes_grid_tariffs") is not False:
        raise ValueError("The present dispatch contains no grid tariffs; flag must be false")
    if data.get("electricity_prices") != "SYNTHETIC_2023_ARCHIVE":
        raise ValueError("Cannot re-label archival Stage B dispatch as observed-market settlement")
    return rate


def evaluate_case(record, assumptions, rate):
    """Compute fully itemized *illustrative* annual cost, no shortfall penalty."""
    if record.get("status") != "VERIFIED_TWO_STAGE_SOLVE":
        raise ValueError("Only verified two-stage dispatch records may be priced")
    m = record["metrics"]
    demand = number(m["demand_h2_kg"], "demand_h2_kg", positive=True)
    unserved_a = number(m["stage_a_shortfall_kg"], "stage_a_shortfall_kg")
    unserved_b = number(m["stage_b_shortfall_kg"], "stage_b_shortfall_kg")
    if unserved_a > demand + 1e-4 or unserved_b > demand + 1e-4:
        raise ValueError("Hydrogen shortfall exceeds total demand")
    if unserved_b + 1e-4 < unserved_a:
        raise ValueError("Stage B appears to outperform the Stage A adequacy minimum")
    allowance = max(1e-4, 1e-8 * demand)
    if unserved_b > unserved_a + allowance + 1e-4:
        raise ValueError("Stage B shortfall exceeds its independently verified adequacy cap")
    settlement = float(m["stage_b_grid_settlement_eur_synthetic_prices"])
    if not np.isfinite(settlement):
        raise ValueError("Invalid synthetic-price grid settlement")
    delivered = demand - unserved_b
    if delivered <= 0:
        raise ValueError("No delivered hydrogen; no unit-cost denominator")
    active = tuple(record.get("active_interventions", ()))
    if "eta" in active and "efficiency_upgrade" not in assumptions["assets"]:
        raise ValueError("Efficiency upgrade cost omitted")

    capital_eur = 0.0
    annualised_capital_eur = 0.0
    fixed_opex_eur = 0.0
    parts = {}
    for asset, (quantity_key, capex_key) in ASSET_UNITS.items():
        spec = assumptions["assets"][asset]
        quantity = number(m[quantity_key], f"{asset} quantity")
        capex = quantity * number(spec[capex_key], f"{asset}.{capex_key}")
        annuity = capex * annuity_factor(rate, spec["lifetime_years"])
        fixed = capex * number(spec["fixed_opex_fraction"], "fixed OPEX fraction")
        parts[asset] = {
            "asset_quantity": quantity,
            "capital_eur": capex,
            "annualised_capital_eur": annuity,
            "annual_fixed_opex_eur": fixed,
        }
        capital_eur += capex
        annualised_capital_eur += annuity
        fixed_opex_eur += fixed
    if "eta" in active:
        spec = assumptions["assets"]["efficiency_upgrade"]
        capex = number(m["electrolyser_capacity_mw"], "rated electro capacity") * number(
            spec["capex_eur_per_mw"], "efficiency upgrade premium"
        )
        annuity = capex * annuity_factor(rate, spec["lifetime_years"])
        fixed = capex * number(spec["fixed_opex_fraction"], "fixed OPEX fraction")
        parts["efficiency_upgrade"] = {
            "capital_eur": capex,
            "annualised_capital_eur": annuity,
            "annual_fixed_opex_eur": fixed,
        }
        capital_eur += capex
        annualised_capital_eur += annuity
        fixed_opex_eur += fixed
    variable = delivered * number(
        assumptions["variable_opex_eur_per_kg_delivered"], "variable OPEX"
    )
    total = annualised_capital_eur + fixed_opex_eur + variable + settlement
    if not np.isfinite(total):
        raise ValueError("Nonfinite annual cost")
    return {
        "scenario": record["scenario"],
        "active_interventions": list(active),
        "adequacy_shortfall_fraction_stage_a": unserved_a / demand,
        "demand_kg": demand,
        "delivered_kg_stage_b": delivered,
        "unserved_kg_stage_b": unserved_b,
        "installed_capex_eur": capital_eur,
        "annualised_capex_eur": annualised_capital_eur,
        "annual_fixed_opex_eur": fixed_opex_eur,
        "annual_variable_opex_eur": variable,
        "grid_settlement_eur_SYNTHETIC": settlement,
        "annual_cost_eur_ILLUSTRATIVE": total,
        "unit_cost_eur_per_kg_delivered_ILLUSTRATIVE": total / delivered,
        "asset_breakdown": parts,
    }


def pareto_frontier(records):
    """Nondominated points minimize annual illustrative cost and unmet fraction."""
    output = []
    for case in records:
        c = case["annual_cost_eur_ILLUSTRATIVE"]
        s = case["adequacy_shortfall_fraction_stage_a"]
        dominated = any(
            (other["annual_cost_eur_ILLUSTRATIVE"] <= c)
            and (other["adequacy_shortfall_fraction_stage_a"] <= s)
            and (
                other["annual_cost_eur_ILLUSTRATIVE"] < c
                or other["adequacy_shortfall_fraction_stage_a"] < s
            )
            for other in records
            if other is not case
        )
        if not dominated:
            output.append(case["scenario"])
    return output


def evaluate_matrix(summary, assumptions, *, acknowledge_synthetic):
    """Only evaluate the complete, original 17-case synthetic-price WP2 archive."""
    if acknowledge_synthetic is not True:
        raise ValueError("Explicit --acknowledge-synthetic-prices required")
    rate = validate_assumptions(assumptions)
    if (
        summary.get("status") != "COMPLETE_REGISTERED_MATRIX"
        or summary.get("hours") != 8760
        or summary.get("is_full_year") is not True
        or summary.get("price_data_kind") != "synthetic_daily_sinusoid_NOT_historical_market_data"
    ):
        raise ValueError("Expected original full-year, complete, synthetic-price WP2 matrix")
    cases = summary.get("cases", [])
    names = [case.get("scenario") for case in cases]
    expected = [name for name, _ in registered_cases(["W", "E", "G", "S", "eta"])]
    if names != expected:
        raise ValueError("Missing/reordered registered scenario; not a 17-case analysis")
    result = [evaluate_case(c, assumptions, rate) for c in cases]
    return {
        "schema_version": "pynexus.wp5.illustrative_frontier.v1",
        "status": "ILLUSTRATIVE_SYNTHETIC_PRICE_FINANCIAL_SENSITIVITY",
        "currency": "EUR",
        "assumed_price_year": assumptions["price_year"],
        "assumptions_source_level": "DECLARED_NOT_EXTERNALLY_VALIDATED",
        "electricity_price_source": "SYNTHETIC_2023_ARCHIVE",
        "price_data_are_not_historical_market_prices": True,
        "not_a_bankable_lcoh_or_project_feasibility_estimate": True,
        "no_capex_or_opex_defaults_inserted": True,
        "excluded_cost_items": list(UNPRICED_ITEMS),
        "cases": result,
        "pareto_scenarios": pareto_frontier(result),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp2-dir", required=True, type=Path)
    parser.add_argument("--assumptions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--acknowledge-synthetic-prices", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Never overwrite an existing research output")
    # Recheck all 17 input, configuration and two-stage dispatch artifacts first.
    analyse_archive(args.wp2_dir)
    summary_path = args.wp2_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assumptions = json.loads(args.assumptions.read_text(encoding="utf-8"))
    result = evaluate_matrix(
        summary, assumptions, acknowledge_synthetic=args.acknowledge_synthetic_prices
    )
    result["wp2_archive_summary_sha256"] = sha256(summary_path)
    result["cost_assumptions_sha256"] = sha256(args.assumptions)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    columns = [
        "scenario",
        "adequacy_shortfall_fraction_stage_a",
        "delivered_kg_stage_b",
        "annualised_capex_eur",
        "annual_fixed_opex_eur",
        "annual_variable_opex_eur",
        "grid_settlement_eur_SYNTHETIC",
        "annual_cost_eur_ILLUSTRATIVE",
        "unit_cost_eur_per_kg_delivered_ILLUSTRATIVE",
    ]
    pd.DataFrame(result["cases"])[columns].to_csv(args.output / "frontier.csv", index=False)
    print("WP5 illustrative accounting verified; synthetic prices; NOT real-world LCOH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
