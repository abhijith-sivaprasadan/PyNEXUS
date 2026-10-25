"""WP5 genuinely annualised observed-price cost/adequacy frontier, source-gated.

Run ONLY with complete validated 2023-EUR cost assumptions over the complete
17-case observed-price annual matrix. Never substitute missing costs as zeros.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.financial_accounting import ASSET_UNITS, annuity_factor, number, validate_assumptions
from research.interventions import PROTOCOL, registered_cases
from research.wp5_market import START

PRICE_HASH = "2dfea4651fce314893201e7b08a4ae0ad24deef6b69ef85d81d9f54428f95bb2"


def validated_observed_costs(data):
    """Require complete declared sources and matching market cost-year basis."""
    if data.get("electricity_prices") != "OBSERVED_NL_A44_2023":
        raise ValueError("Full cost frontier requires 2023 observed NL A44 prices")
    if data.get("price_year") != 2023:
        raise ValueError("All reported capex/opex must be normalized to 2023 EUR")
    check = {**data, "electricity_prices": "SYNTHETIC_2023_ARCHIVE"}
    rate = validate_assumptions(check)
    sources = [
        data.get("discount_rate_source"),
        data.get("variable_opex_source"),
    ]
    for spec in data["assets"].values():
        sources.extend(spec["sources"].values())
    if any(not isinstance(x, str) or not x.startswith("https://") for x in sources):
        raise ValueError("Every financial input must name a traceable HTTPS primary/reviewed source")
    if data.get("cost_scope_reviewed") is not True:
        raise ValueError("Independent asset scope, double counting and currency-year review required")
    if not isinstance(data.get("normalisation_method"), str) or len(data["normalisation_method"]) < 60:
        raise ValueError("Source-specific 2023 EUR conversion and exclusions must be documented")
    if any(number(x["fixed_opex_fraction"], "asset fixed opex fraction") > 1 for x in data["assets"].values()):
        raise ValueError("Invalid annual maintenance factor")
    return rate


def evaluate_frontier(full_matrix, costs):
    """Annualise 17 fixed case asset quantities and real 2023 net wholesale settlement."""
    rate = validated_observed_costs(costs)
    if (
        full_matrix.get("schema_version") != "pynexus.wp5.full_matrix_synthesis.v1"
        or full_matrix.get("status") != "COMPLETE_VERIFIED_ANNUAL_PHYSICAL_AND_SETTLEMENT_SCREENING"
        or full_matrix.get("scenario_count") != 17
        or full_matrix.get("hours_per_case") != 8760
        or full_matrix["source_sha256"].get("price_csv_sha256") != PRICE_HASH
    ):
        raise ValueError("Source-gated complete 2023 A44 matrix unavailable")
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    registered = dict(registered_cases(protocol["order"]))
    records = full_matrix["records"]
    if [rec["scenario"] for rec in records] != list(registered):
        raise ValueError("Changed or incomplete registered intervention ordering")
    result = []
    for original in records:
        scenario = original["scenario"]
        m = original
        if m["demand_kg"] <= 0 or not 0 <= m["shortfall_fraction"] <= 1:
            raise ValueError("Invalid demand or adequacy input")
        demand = number(m["demand_kg"], "demand", positive=True)
        delivered = demand - number(m["stage_b_shortfall_kg"], "shortfall")
        if delivered <= 0:
            raise ValueError("Zero delivered hydrogen")
        annual_capital = 0.0
        annual_fixed = 0.0
        included_assets = {}
        for name, (quantity, key) in ASSET_UNITS.items():
            spec = costs["assets"][name]
            initial = float(m[quantity]) * float(spec[key])
            a = initial * annuity_factor(rate, spec["lifetime_years"])
            fixed = initial * number(spec["fixed_opex_fraction"], "fixed operating rate")
            included_assets[name] = {
                "installed_capex_eur": initial,
                "annualised_capex_eur": a,
                "annual_fixed_opex_eur": fixed,
            }
            annual_capital += a
            annual_fixed += fixed
        if "eta" in registered[scenario]:
            spec = costs["assets"]["efficiency_upgrade"]
            value = float(m["electrolyser_capacity_mw"]) * float(spec["capex_eur_per_mw"])
            a = value * annuity_factor(rate, spec["lifetime_years"])
            fixed = value * float(spec["fixed_opex_fraction"])
            annual_capital += a
            annual_fixed += fixed
            included_assets["efficiency_upgrade"] = {
                "installed_capex_eur": value,
                "annualised_capex_eur": a,
                "annual_fixed_opex_eur": fixed,
            }
        variable = delivered * number(
            costs["variable_opex_eur_per_kg_delivered"], "variable operating cost"
        )
        wholesale = number(
            abs(float(m["net_grid_settlement_eur_OBSERVED_2023"])), "wholesale settlement abs"
        ) * np.sign(float(m["net_grid_settlement_eur_OBSERVED_2023"]))
        annual = wholesale + annual_capital + annual_fixed + variable
        if not np.isfinite(annual):
            raise ValueError("Nonfinite annual total")
        result.append({
            "scenario": scenario,
            "annual_delivered_h2_kg": delivered,
            "unserved_fraction_stage_a": float(m["shortfall_fraction"]),
            "adequacy_percent": 100 * (1 - float(m["shortfall_fraction"])),
            "annual_capital_recovery_eur": annual_capital,
            "annual_fixed_opex_eur": annual_fixed,
            "annual_variable_opex_eur": variable,
            "annual_wholesale_settlement_eur": wholesale,
            "annual_modelled_cost_eur_EXCLUDES_TARIFFS_AND_OTHER_UNPRICED_ITEMS": annual,
            "conditional_unit_cost_eur_per_delivered_kg": annual / delivered,
            "assets": included_assets,
        })
    nondominated = []
    for rec in result:
        cost = rec["annual_modelled_cost_eur_EXCLUDES_TARIFFS_AND_OTHER_UNPRICED_ITEMS"]
        unmet = rec["unserved_fraction_stage_a"]
        dominated = any(
            (other["annual_modelled_cost_eur_EXCLUDES_TARIFFS_AND_OTHER_UNPRICED_ITEMS"] <= cost)
            and (other["unserved_fraction_stage_a"] <= unmet)
            and (
                other["annual_modelled_cost_eur_EXCLUDES_TARIFFS_AND_OTHER_UNPRICED_ITEMS"] < cost
                or other["unserved_fraction_stage_a"] < unmet
            )
            for other in result if other is not rec
        )
        if not dominated:
            nondominated.append(rec["scenario"])
    return {
        "schema_version": "pynexus.wp5.conditional_observed_cost_frontier.v1",
        "status": "CONDITIONAL_2023_SOURCE_DECLARED_COST_ADEQUACY_MODEL_NOT_BANKABLE_LCOH",
        "market_price_calendar": str(START.year),
        "input_price_sha256": PRICE_HASH,
        "financial_price_year": costs["price_year"],
        "currency": "EUR",
        "asset_cost_scope": costs["normalisation_method"],
        "non_dominated_cases": nondominated,
        "records": result,
        "limitations": [
            "Cost sources are declared and independently scoped but not vendor quotations",
            "Original ERA5 hourly wind primary provenance pending independent check",
            "Network tariffs, actual procurement contracts, insurance, taxes, water, degradation, replacements or incentives omitted",
            "This is a conditional modelled subset-cost frontier, not full project LCOH",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Will not overwrite economic research evidence")
    data = json.loads(args.matrix.read_text(encoding="utf-8"))
    costs = json.loads(args.costs.read_text(encoding="utf-8"))
    report = evaluate_frontier(data, costs)
    args.output.mkdir(parents=True)
    (args.output / "frontier.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [{k: v for k, v in rec.items() if k != "assets"} for rec in report["records"]]
    ).to_csv(args.output / "frontier.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
