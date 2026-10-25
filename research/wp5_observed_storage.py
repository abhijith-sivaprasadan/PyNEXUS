"""WP5: independently recompute observed-price E -> E+S storage economics.

Historical processed ENTSO-E A44 prices; this is NOT a full all-assets
levelized hydrogen cost or the original 17-case cost/adequacy frontier.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from research.financial_accounting import annuity_factor, number
from research.run_reference import sha256

PRICE_SHA256 = "2dfea4651fce314893201e7b08a4ae0ad24deef6b69ef85d81d9f54428f95bb2"
SCENARIOS = ("E", "E+S")


def observed_case(path, scenario):
    """Verify saved input lineage, outputs, physical checks and price arithmetic."""
    if scenario not in SCENARIOS:
        raise ValueError("Unregistered comparison scenario")
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if (
            manifest.get("status") != "VERIFIED_2023_DISPATCH_PROCESSED_MARKET_PRICE_INPUT"
            or manifest.get("full_year") is not True
            or manifest.get("hours") != 8760
            or manifest.get("year") != 2023
            or manifest.get("price_csv_sha256") != PRICE_SHA256
            or manifest.get("price_kind") != "PROCESSED_ENTSOE_2023_NL_DAY_AHEAD_NOT_INVOICE"
            or len(manifest.get("cases", [])) != 1
            or manifest["cases"][0].get("scenario") != scenario
        ):
            raise ValueError("Source archive is not the correct complete NL 2023 scenario")
        record_bytes = archive.read(f"{scenario}/record.json")
        record = json.loads(record_bytes)
        if (
            record.get("status") != "VERIFIED_TWO_STAGE_SOLVE"
            or record.get("scenario") != scenario
            or record.get("price_csv_sha256") != PRICE_SHA256
            or any(
                record.get(f"solver_{stage}", {}).get("termination_condition") != "optimal"
                for stage in ("a", "b")
            )
            or any(
                set(record.get(f"verification_{stage}", {}).values()) != {True}
                for stage in ("a", "b")
            )
        ):
            raise ValueError("Source dispatch verification or solver result invalid")
        tables = {}
        for stage in ("a", "b"):
            filename = f"stage_{stage}.csv"
            blob = archive.read(f"{scenario}/{filename}")
            if hashlib.sha256(blob).hexdigest() != record["dispatch_hashes"][filename]:
                raise ValueError("Dispatch bytes do not match source SHA-256")
            frame = pd.read_csv(io.BytesIO(blob))
            if len(frame) != 8760 or not np.array_equal(
                frame["timestep"].to_numpy(dtype=int), np.arange(8760)
            ):
                raise ValueError("Incomplete or misordered 2023 hourly dispatch")
            required = (
                "demand_slack_kg_h",
                "grid_import_mw",
                "grid_export_mw",
                "electricity_price",
            )
            for field in required:
                vals = frame[field].to_numpy(dtype=float)
                if not np.isfinite(vals).all():
                    raise ValueError("Nonfinite hourly dispatch")
            if np.any(np.minimum(frame["grid_import_mw"], frame["grid_export_mw"]) > 1e-6):
                raise ValueError("Simultaneous grid import/export")
            tables[stage] = frame
        metrics = record["metrics"]
        if not np.isclose(
            tables["a"]["demand_slack_kg_h"].sum(),
            metrics["stage_a_shortfall_kg"],
            atol=1e-3,
            rtol=0,
        ) or not np.isclose(
            tables["b"]["demand_slack_kg_h"].sum(),
            metrics["stage_b_shortfall_kg"],
            atol=1e-3,
            rtol=0,
        ):
            raise ValueError("Hourly hydrogen shortfall does not match recorded KPI")
        stage_b = tables["b"]
        settlement = float(
            (
                (stage_b["grid_import_mw"] - stage_b["grid_export_mw"])
                * stage_b["electricity_price"]
            ).sum()
        )
        if not np.isclose(
            settlement,
            metrics["stage_b_grid_settlement_eur_OBSERVED_PRICE_INPUT"],
            atol=0.01,
            rtol=0,
        ):
            raise ValueError("Hourly grid settlement does not match recorded KPI")
        return {
            "scenario": scenario,
            "manifest": manifest,
            "metrics": metrics,
            "settlement_eur": settlement,
            "import_expense_eur": float(
                (stage_b["grid_import_mw"] * stage_b["electricity_price"]).sum()
            ),
            "export_revenue_eur": float(
                (stage_b["grid_export_mw"] * stage_b["electricity_price"]).sum()
            ),
            "artifact_zip_sha256": sha256(path),
            "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        }


def compare(e, es):
    """Calculate storage-only incremental model results, without cost assumptions."""
    if e["scenario"] != "E" or es["scenario"] != "E+S":
        raise ValueError("Expected E and E+S controls")
    m, n = e["metrics"], es["metrics"]
    fixed = ("wind_nameplate_mw", "electrolyser_capacity_mw", "grid_capacity_mw", "demand_h2_kg")
    if any(not np.isclose(m[k], n[k], rtol=0, atol=1e-6) for k in fixed):
        raise ValueError("Non-storage intervention changed in comparison")
    if not np.isclose(m["storage_capacity_kg"], 20000, rtol=0, atol=1e-6) or not np.isclose(
        n["storage_capacity_kg"], 40000, rtol=0, atol=1e-6
    ):
        raise ValueError("Unregistered storage comparison")
    identity_keys = ("source_wind_sha256", "config_sha256", "price_csv_sha256")
    if any(e["manifest"][key] != es["manifest"][key] for key in identity_keys):
        raise ValueError("Different source/config/price hashes: not a controlled comparison")
    extra = (n["demand_h2_kg"] - n["stage_b_shortfall_kg"]) - (
        m["demand_h2_kg"] - m["stage_b_shortfall_kg"]
    )
    if extra <= 0:
        raise ValueError("No incremental hydrogen delivered")
    delta_net = es["settlement_eur"] - e["settlement_eur"]
    delta_import = es["import_expense_eur"] - e["import_expense_eur"]
    forgone_export = e["export_revenue_eur"] - es["export_revenue_eur"]
    if not np.isclose(delta_net, delta_import + forgone_export, atol=0.01, rtol=0):
        raise ValueError("Incorrect wholesale grid settlement decomposition")
    return {
        "status": "VERIFIED_PROCESSED_A44_GRID_SETTLEMENT_ONLY_NOT_TOTAL_COST",
        "scenario_pair": ["E", "E+S"],
        "extra_installed_storage_kg": n["storage_capacity_kg"] - m["storage_capacity_kg"],
        "additional_delivered_hydrogen_kg": extra,
        "shortfall_improvement_stage_a_kg": m["stage_a_shortfall_kg"] - n["stage_a_shortfall_kg"],
        "additional_net_wholesale_grid_settlement_eur": delta_net,
        "additional_import_spend_eur": delta_import,
        "forgone_export_revenue_eur": forgone_export,
        "additional_wholesale_settlement_eur_per_kg": delta_net / extra,
        "price_csv_sha256": PRICE_SHA256,
        "source_wind_hour_alignment_independently_audited": False,
        "raw_a44_xml_archived": False,
        "cost_frontier_complete": False,
    }


def validate_storage_costs(costs):
    """Require 2023-equivalent sourced storage cost evidence; never default zeros."""
    if costs.get("schema_version") != "pynexus.wp5.marginal_storage_costs.v1":
        raise ValueError("Invalid storage-cost assumptions schema")
    if costs.get("currency") != "EUR" or costs.get("price_year") != 2023:
        raise ValueError("Costs must be explicitly converted to 2023 EUR")
    for key in (
        "capex_eur_per_kg",
        "fixed_opex_fraction",
        "lifetime_years",
        "discount_rate",
        "variable_opex_eur_per_additional_kg",
    ):
        number(costs.get(key), key, positive=key == "lifetime_years")
    if number(costs["fixed_opex_fraction"], "fixed OPEX") > 1:
        raise ValueError("Invalid annual fixed OPEX fraction")
    sources = costs.get("source_register")
    if not isinstance(sources, dict) or any(
        not isinstance(sources.get(k), str) or not sources[k].startswith("https://")
        for k in ("capex", "opex", "asset_lifetime", "discount_rate", "variable_opex")
    ):
        raise ValueError("All cost fields require directly identifiable https source links")
    if (
        not isinstance(costs.get("engineering_scope_and_2023_normalisation"), str)
        or len(costs["engineering_scope_and_2023_normalisation"]) < 50
    ):
        raise ValueError("Declare asset scope, inflation/FX normalisation and exclusions")
    return costs


def priced_increment(result, costs):
    costs = validate_storage_costs(costs)
    extra_storage = result["extra_installed_storage_kg"]
    extra_delivery = result["additional_delivered_hydrogen_kg"]
    capex = extra_storage * float(costs["capex_eur_per_kg"])
    recovery = annuity_factor(costs["discount_rate"], costs["lifetime_years"])
    annual_capex = capex * recovery
    annual_fixed = capex * float(costs["fixed_opex_fraction"])
    annual_variable = extra_delivery * float(costs["variable_opex_eur_per_additional_kg"])
    total = (
        result["additional_net_wholesale_grid_settlement_eur"]
        + annual_capex
        + annual_fixed
        + annual_variable
    )
    return {
        **result,
        "status": "CONDITIONAL_SOURCE_DECLARED_STORAGE_INCREMENT_NOT_FULL_LCOH",
        "storage_assumptions": costs,
        "incremental_installed_storage_capex_eur": capex,
        "incremental_annualised_storage_capex_eur": annual_capex,
        "incremental_storage_fixed_opex_eur_per_year": annual_fixed,
        "incremental_variable_opex_eur_per_year": annual_variable,
        "additional_modelled_annual_cost_eur": total,
        "break_even_additional_hydrogen_value_eur_per_kg": total / extra_delivery,
        "cost_frontier_complete": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--E", type=Path, required=True)
    p.add_argument("--ES", type=Path, required=True)
    p.add_argument("--storage-costs", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Refuse to overwrite audited analysis")
    result = compare(observed_case(args.E, "E"), observed_case(args.ES, "E+S"))
    if args.storage_costs:
        result = priced_increment(result, json.loads(args.storage_costs.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
