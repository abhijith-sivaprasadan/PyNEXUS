"""Fail-closed audit and scientific synthesis of completed WP4/WP5 annual matrices.

Never represents partial workflow artifacts as completed research.
No investment costs are silently inferred from price-only dispatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.interventions import PROTOCOL as WP2_PROTOCOL
from research.interventions import registered_cases
from research.run_reference import ROOT, sha256

ENGINEERING_PROTOCOL = ROOT / "research" / "protocol_wp4_engineering.yaml"
EXPECTED_PRICE_SHA = "2dfea4651fce314893201e7b08a4ae0ad24deef6b69ef85d81d9f54428f95bb2"


def expected_names(kind):
    if kind == "wp4":
        protocol = yaml.safe_load(ENGINEERING_PROTOCOL.read_text(encoding="utf-8"))
        return list(protocol["registered_cases"])
    if kind == "wp5":
        protocol = yaml.safe_load(WP2_PROTOCOL.read_text(encoding="utf-8"))
        return [name for name, _ in registered_cases(protocol["order"])]
    raise ValueError("Only wp4 and wp5 registered matrices can be synthesised")


def _table(file, declared_hash):
    if not file.is_file() or sha256(file) != declared_hash:
        raise ValueError(f"Dispatch hash mismatch or absent file: {file.name}")
    data = pd.read_csv(file)
    if len(data) != 8760 or not np.array_equal(
        data["timestep"].to_numpy(dtype=int), np.arange(8760)
    ):
        raise ValueError("An annual dispatch must contain 8760 chronologically ordered hours")
    return data


def verify_wp4(folder, scenario):
    base = folder / f"wp4-engineering-full-year-{scenario}"
    record = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if (
        record["schema_version"] != "pynexus.wp4.engineering.result.v1"
        or record["status"] != "VERIFIED_ANNUAL_ENGINEERING_SCREENING"
        or record["case"] != scenario
        or record["hours"] != 8760
        or record["full_year"] is not True
        or record["stage_a_verified"] is not True
        or record["stage_b_verified"] is not True
        or record["stage_a_status"] != "optimal"
        or record["stage_b_status"] != "optimal"
    ):
        raise ValueError("Incomplete, invalid or unverified full-year WP4 case")
    tables = {
        k: _table(base / f"{k}.csv", record["stage_dispatch_sha256"][k])
        for k in ("stage_a", "stage_b")
    }
    if not np.isclose(
        tables["stage_a"]["demand_slack_kg_h"].sum(),
        record["stage_a_unserved_kg"],
        rtol=0, atol=1e-3,
    ):
        raise ValueError("WP4 hourly hydrogen shortfall differs from manifest")
    net = tables["stage_b"]["grid_import_mw"] - tables["stage_b"]["grid_export_mw"]
    cost = float((net * tables["stage_b"]["electricity_price"]).sum())
    if not np.isclose(
        cost, record["stage_b_grid_settlement_eur_SYNTHETIC"], rtol=0, atol=0.01
    ):
        raise ValueError("WP4 synthetic wholesale settlement differs from hourly dispatch")
    overlap = np.minimum(
        tables["stage_b"]["grid_import_mw"], tables["stage_b"]["grid_export_mw"]
    )
    if float(overlap.max()) > 1e-6:
        raise ValueError("WP4 simultaneous grid import/export observed")
    return {
        "scenario": scenario,
        "demand_kg": float(record["demand_kg"]),
        "shortfall_kg": float(record["stage_a_unserved_kg"]),
        "shortfall_fraction": float(record["stage_a_unserved_fraction"]),
        "adequacy_percent": 100 * (1 - float(record["stage_a_unserved_fraction"])),
        "net_grid_settlement_eur_SYNTHETIC": cost,
        "price_kind": "ARCHIVED_SYNTHETIC_2023_NOT_A44",
        "source_sha256": record["source_sha256"],
        "source_hour_alignment_independently_verified": False,
        "protocol_sha256": record["protocol_sha256"],
        "config_sha256": record["case_config_sha256"],
        "stage_a_dispatch_sha256": record["stage_dispatch_sha256"]["stage_a"],
        "stage_b_dispatch_sha256": record["stage_dispatch_sha256"]["stage_b"],
    }


def verify_wp5(folder, scenario):
    base = folder / f"wp5-observed-full-year-{scenario}"
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["status"] != "VERIFIED_2023_DISPATCH_PROCESSED_MARKET_PRICE_INPUT"
        or manifest["hours"] != 8760
        or manifest["full_year"] is not True
        or manifest["year"] != 2023
        or manifest["price_csv_sha256"] != EXPECTED_PRICE_SHA
        or len(manifest["cases"]) != 1
        or manifest["cases"][0]["scenario"] != scenario
    ):
        raise ValueError("Wrong, incomplete or source-mismatched WP5 run")
    rec = json.loads((base / scenario / "record.json").read_text(encoding="utf-8"))
    if (
        rec["status"] != "VERIFIED_TWO_STAGE_SOLVE"
        or rec["scenario"] != scenario
        or rec["price_csv_sha256"] != EXPECTED_PRICE_SHA
        or any(
            rec[f"solver_{stage}"]["termination_condition"] != "optimal"
            for stage in ("a", "b")
        )
        or any(
            set(rec[f"verification_{stage}"].values()) != {True}
            for stage in ("a", "b")
        )
    ):
        raise ValueError("WP5 source solver/verification failed")
    tables = {
        k: _table(
            base / scenario / f"stage_{k}.csv",
            rec["dispatch_hashes"][f"stage_{k}.csv"],
        )
        for k in ("a", "b")
    }
    m = rec["metrics"]
    if not np.isclose(
        tables["a"]["demand_slack_kg_h"].sum(), m["stage_a_shortfall_kg"], rtol=0, atol=1e-3
    ):
        raise ValueError("WP5 physical adequacy changed after archiving")
    stage_b = tables["b"]
    if not np.isclose(
        stage_b["demand_slack_kg_h"].sum(), m["stage_b_shortfall_kg"], rtol=0, atol=1e-3
    ):
        raise ValueError("WP5 Stage B shortfall differs from original evidence")
    overlap = np.minimum(stage_b["grid_import_mw"], stage_b["grid_export_mw"])
    if float(overlap.max()) > 1e-6:
        raise ValueError("WP5 simultaneous import and export")
    settlement = float(
        (
            (stage_b["grid_import_mw"] - stage_b["grid_export_mw"])
            * stage_b["electricity_price"]
        ).sum()
    )
    if not np.isclose(
        settlement, m["stage_b_grid_settlement_eur_OBSERVED_PRICE_INPUT"],
        rtol=0, atol=0.01,
    ):
        raise ValueError("WP5 annual price settlement not supported by hourly dispatch")
    return {
        "scenario": scenario,
        "demand_kg": float(m["demand_h2_kg"]),
        "shortfall_kg": float(m["stage_a_shortfall_kg"]),
        "shortfall_fraction": float(m["stage_a_shortfall_fraction"]),
        "adequacy_percent": 100 * (1 - float(m["stage_a_shortfall_fraction"])),
        "stage_b_shortfall_kg": float(m["stage_b_shortfall_kg"]),
        "net_grid_settlement_eur_OBSERVED_2023": settlement,
        "grid_import_mwh": float(m["grid_import_mwh"]),
        "grid_export_mwh": float(m["grid_export_mwh"]),
        "electrolyser_capacity_mw": float(m["electrolyser_capacity_mw"]),
        "wind_nameplate_mw": float(m["wind_nameplate_mw"]),
        "grid_capacity_mw": float(m["grid_capacity_mw"]),
        "storage_capacity_kg": float(m["storage_capacity_kg"]),
        "nominal_efficiency_parameter": float(m["nominal_efficiency_parameter"]),
        "price_csv_sha256": manifest["price_csv_sha256"],
        "source_wind_sha256": manifest["source_wind_sha256"],
        "base_config_sha256": manifest["config_sha256"],
        "source_hour_alignment_independently_verified": False,
        "stage_a_dispatch_sha256": rec["dispatch_hashes"]["stage_a.csv"],
        "stage_b_dispatch_sha256": rec["dispatch_hashes"]["stage_b.csv"],
    }


def generate(kind, input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Never overwrite an existing research synthesis")
    names = expected_names(kind)
    if len(names) != 17 or len(set(names)) != 17:
        raise ValueError("Expected exactly the 17 registered cases")
    observed = {
        p.name for p in input_dir.iterdir() if p.is_dir()
    } if input_dir.is_dir() else set()
    prefix = "wp4-engineering-full-year-" if kind == "wp4" else "wp5-observed-full-year-"
    expected = {prefix + name for name in names}
    if observed != expected:
        raise ValueError(
            "Incomplete/extraneous annual matrix; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    fn = verify_wp4 if kind == "wp4" else verify_wp5
    records = [fn(input_dir, name) for name in names]
    hash_keys = (
        ("source_sha256", "protocol_sha256")
        if kind == "wp4" else
        ("price_csv_sha256", "source_wind_sha256", "base_config_sha256")
    )
    if any(len({r[key] for r in records}) != 1 for key in hash_keys):
        raise ValueError("Annual scenarios use different base sources or protocols")
    baseline = records[0]
    for rec in records:
        rec["adequacy_improvement_pp_vs_baseline"] = (
            rec["adequacy_percent"] - baseline["adequacy_percent"]
        )
    output_dir.mkdir(parents=True)
    pd.DataFrame(records).to_csv(output_dir / "results.csv", index=False)
    audit = {
        "schema_version": f"pynexus.{kind}.full_matrix_synthesis.v1",
        "status": "COMPLETE_VERIFIED_ANNUAL_PHYSICAL_AND_SETTLEMENT_SCREENING",
        "scenario_count": len(records),
        "hours_per_case": 8760,
        "scenario_order": names,
        "source_sha256": {key: baseline[key] for key in hash_keys},
        "source_hour_alignment_independently_verified": False,
        "study_scope": (
            "ONE_FACTOR_2023_WITH_SYNTHETIC_PRICES_NOT_EMPIRICAL_UNCERTAINTY"
            if kind == "wp4" else
            "2023_NL_A44_WHOLESALE_PRICE_NO_CAPEX_NO_TARIFF"
        ),
        "fully_costed_adequacy_frontier_established": False,
        "records": records,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["wp4", "wp5"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = generate(args.kind, args.input, args.output)
    print(json.dumps({"status": result["status"], "scenario_count": result["scenario_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
