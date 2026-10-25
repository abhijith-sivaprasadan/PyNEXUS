"""WP6: auditable IAMC-shaped export of the verified WP4 adequacy experiment.

IAMC table compatibility does NOT imply IAM Consortium endorsement, official
variable names, IAM calibration, an emissions scenario or historical H2 output.
The model indexes weather-year counterfactuals at one fixed technology vintage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.run_reference import CONFIG, ROOT, sha256

# This is the exact analysis.json archived by the independently rechecked
# WP4 GitHub Actions synthesis (run 40157038537; artifact 12585572146).
# Pin source bytes, not an easily forged self-declared "verified" status.
WP4_ANALYSIS_SHA256 = "ee1b475d1034a151d1e2e91b85ec3047aed6f0b756d042159d2305cfa466e907"
WP4_RUN_ID = 40157038537
WP4_COMMIT = "127a000e247c494995b0744bdafa0cf290c918f4"
SOURCE = "Open-Meteo ERA5 100m, not exact CDS-2023 original"
YEARS = (2021, 2022, 2023, 2024, 2025)
SCENARIOS = ("baseline", "W", "E", "G", "S", "eta", "E+S")
MODEL = "PyNEXUS|OpenMeteo-ERA5|Fixed-System-Adequacy"
REGION = "North Sea offshore (52.5N, 3.5E)"
IAMC_COLS = ("model", "scenario", "region", "variable", "unit", "year", "value")
VARIABLES_PATH = ROOT / "research" / "wp6_variables.json"


def _finite(value, label, *, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise ValueError(f"{label}: boolean is not a numeric metric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: numeric metric required") from exc
    if not np.isfinite(number):
        raise ValueError(f"{label}: non-finite metric")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label}: below physical minimum")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label}: exceeds physical maximum")
    return number


def _near(actual, expected, label, *, atol=1e-6):
    if not np.isclose(actual, expected, atol=atol, rtol=1e-10):
        raise ValueError(f"{label}: independent arithmetic check failed")


def load_registered_wp4(path):
    """Accept only immutable, identified WP4 synthesis bytes, never arbitrary JSON."""
    path = Path(path)
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != WP4_ANALYSIS_SHA256:
        raise ValueError("WP4 synthesis bytes differ from the independently verified archived artifact")
    result = json.loads(raw)
    if (
        result.get("schema_version") != "pynexus.wp4.interannual.v1"
        or result.get("status") != "COMPLETE_5_YEAR_35_SCENARIO_COMPARISON"
        or result.get("source_revision") != WP4_COMMIT
        or result.get("raw_weather_source") != SOURCE
    ):
        raise ValueError("Wrong WP4 study identity/version/status")
    return result, observed


def validate_synthesis(result, demand_per_day_kg):
    """Recheck annual identities, registration, leap years and finite contrasts.

    This checks a *hash-pinned, earlier independently verified synthesis*.
    It does not reverify raw hourly dispatch in the WP6 process; that was WP4.
    """
    if list(result.get("years", [])) != list(YEARS):
        raise ValueError("Expected registered five weather years")
    if result.get("scenario_count") != 35 or result.get("weather_year_count") != 5:
        raise ValueError("Expected 35 scenario-year observations")
    records = result.get("annual_results")
    years = result.get("annual_metrics")
    if not isinstance(records, list) or len(records) != 35:
        raise ValueError("Missing annual scenario records")
    if not isinstance(years, list) or [x.get("year") for x in years] != list(YEARS):
        raise ValueError("Missing annual year metrics")
    if len({(r.get("year"), r.get("scenario")) for r in records}) != 35:
        raise ValueError("Duplicate scenario-year cell")
    expected_order = [(year, name) for year in YEARS for name in SCENARIOS]
    if [(r.get("year"), r.get("scenario")) for r in records] != expected_order:
        raise ValueError("Scenario-year order no longer matches WP4 registration")
    demand_per_day_kg = _finite(demand_per_day_kg, "registered daily demand", minimum=1)
    for metric in years:
        year = metric["year"]
        rows = [r for r in records if r["year"] == year]
        hours = 8784 if year == 2024 else 8760
        demand = demand_per_day_kg * hours / 24
        baseline = rows[0]["unserved_fraction"]
        resource = rows[0]["wind_resource_mwh"]
        cf = rows[0]["wind_capacity_factor"]
        _near(
            _finite(metric["baseline_unserved_fraction"], "annual baseline"),
            baseline,
            "annual baseline share",
            atol=1e-10,
        )
        _near(
            _finite(metric["wind_capacity_factor"], "annual reference capacity factor"),
            cf,
            "annual reference capacity factor",
            atol=1e-10,
        )
        if metric.get("provider") != "Open-Meteo explicit ERA5 (NOT original CDS-2023 input)":
            raise ValueError("WP4 source identity altered")
        if not isinstance(metric.get("source_api_sha256"), str) or len(
            metric["source_api_sha256"]
        ) != 64:
            raise ValueError("Missing source weather SHA-256")
        for row in rows:
            if row.get("hours") != hours:
                raise ValueError("Leap-year hours or yearly coverage mismatch")
            u = _finite(row["unserved_h2_kg"], "Stage A unserved hydrogen", minimum=0)
            share = _finite(row["unserved_fraction"], "unserved share", minimum=0, maximum=1)
            _near(u, share * demand, "demand and unserved identity", atol=0.02)
            _near(
                _finite(row["wind_resource_mwh"], "reference wind resource", minimum=0),
                resource,
                "shared reference resource",
            )
            _near(
                _finite(row["wind_capacity_factor"], "reference wind CF", minimum=0, maximum=1),
                cf,
                "shared reference capacity factor",
                atol=1e-10,
            )
            _near(
                _finite(row["benefit_percentage_points"], "benefit"),
                (baseline - share) * 100,
                "baseline-relative benefit",
                atol=1e-7,
            )
        lookup = {r["scenario"]: r["unserved_fraction"] for r in rows}
        conditional = 100 * (lookup["E"] - lookup["E+S"])
        interaction = 100 * (
            lookup["E"] + lookup["S"] - lookup["baseline"] - lookup["E+S"]
        )
        _near(
            _finite(metric["storage_conditional_benefit_after_E_pp"], "conditional E+S"),
            conditional,
            "E+S conditional gain",
            atol=1e-7,
        )
        _near(
            _finite(metric["E_plus_S_interaction_pp"], "E+S interaction"),
            interaction,
            "E+S interaction contrast",
            atol=1e-7,
        )
    return True


def variable_dictionary():
    """Load a local, explicitly non-official definition for every exported field."""
    definitions = json.loads(VARIABLES_PATH.read_text(encoding="utf-8"))
    expected = {
        "hydrogen_demand",
        "unserved_hydrogen_stage_a",
        "unserved_fraction_stage_a",
        "delivered_hydrogen_stage_a",
        "reference_wind_resource",
        "reference_wind_capacity_factor",
        "adequacy_benefit_vs_baseline",
        "electrolyser_storage_interaction",
    }
    if definitions.get("schema_version") != "pynexus.wp6.variables.v1":
        raise ValueError("Unregistered local variable dictionary")
    mapping = definitions.get("variables")
    if not isinstance(mapping, dict) or set(mapping) != expected:
        raise ValueError("Missing/extra WP6 variables")
    for key, definition in mapping.items():
        if set(definition) != {"variable", "unit", "description"} or not all(
            isinstance(v, str) and v.strip() for v in definition.values()
        ):
            raise ValueError(f"Invalid variable definition: {key}")
        if not definition["variable"].startswith("PyNEXUS|"):
            raise ValueError("Every local variable must declare its PyNEXUS namespace")
    if len({v["variable"] for v in mapping.values()}) != len(mapping):
        raise ValueError("Duplicate local variable name")
    return definitions


def build_iamc_rows(synthesis, demand_per_day_kg, definitions):
    validate_synthesis(synthesis, demand_per_day_kg)
    names = definitions["variables"]
    year_meta = {r["year"]: r for r in synthesis["annual_metrics"]}
    records = []
    for row in synthesis["annual_results"]:
        year, scenario = row["year"], row["scenario"]
        demand = demand_per_day_kg * row["hours"] / 24
        unserved = row["unserved_h2_kg"]
        values = {
            "hydrogen_demand": demand,
            "unserved_hydrogen_stage_a": unserved,
            "unserved_fraction_stage_a": row["unserved_fraction"],
            "delivered_hydrogen_stage_a": demand - unserved,
            "reference_wind_resource": row["wind_resource_mwh"],
            "reference_wind_capacity_factor": row["wind_capacity_factor"],
            "adequacy_benefit_vs_baseline": row["benefit_percentage_points"],
        }
        if scenario == "E+S":
            values["electrolyser_storage_interaction"] = year_meta[year][
                "E_plus_S_interaction_pp"
            ]
        for key, value in values.items():
            definition = names[key]
            records.append(
                (MODEL, scenario, REGION, definition["variable"], definition["unit"], year, value)
            )
    frame = pd.DataFrame.from_records(records, columns=IAMC_COLS)
    if frame.duplicated(list(IAMC_COLS[:-1])).any():
        raise ValueError("Duplicate IAMC index coordinates")
    if len(frame) != 35 * 7 + 5:
        raise ValueError("Incomplete IAMC metric registration")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite IAMC metrics")
    return frame


def verify_pyam_roundtrip(frame, destination):
    """Write IAMC wide CSV, load with pyam and compare every coordinate and number."""
    try:
        import pyam
    except ImportError as exc:
        raise RuntimeError("WP6 requires optional dependency: pip install -e '.[iamc]'") from exc

    original = pyam.IamDataFrame(frame)
    original.to_csv(destination, iamc_index=True)
    reopened = pyam.IamDataFrame(destination)
    sort_keys = list(IAMC_COLS[:-1])

    def canonical(data):
        result = data[list(IAMC_COLS)].copy()
        result["year"] = pd.to_numeric(result["year"]).astype(int)
        result["value"] = pd.to_numeric(result["value"]).astype(float)
        return result.sort_values(sort_keys).reset_index(drop=True)

    before = canonical(frame)
    after = canonical(reopened.data)
    if not before[sort_keys].equals(after[sort_keys]) or not np.allclose(
        before["value"], after["value"], rtol=1e-12, atol=1e-7
    ):
        raise ValueError("pyam round-trip altered IAMC cell identity or values")
    return len(after), pyam.__version__


def export(source, output):
    source = Path(source)
    output = Path(output)
    if output.exists():
        raise FileExistsError("WP6 does not overwrite any research result")
    synthesis, verified_sha = load_registered_wp4(source)
    base = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if base["hydrogen_demand"]["demand_profile"] != "flat":
        raise ValueError("WP6 annual demand reconstruction requires registered flat demand")
    demand_day = base["hydrogen_demand"]["daily_average_kg"]
    definitions = variable_dictionary()
    frame = build_iamc_rows(synthesis, demand_day, definitions)
    # Complete in a temporary directory: no partial output if pyam rejects data.
    partial = output.with_name(output.name + ".incomplete")
    if partial.exists():
        raise FileExistsError("Previous incomplete WP6 output requires manual audit")
    partial.mkdir(parents=True, exist_ok=False)
    try:
        count, pyam_version = verify_pyam_roundtrip(frame, partial / "iamc_wide.csv")
        frame.to_csv(partial / "iamc_long.csv", index=False, float_format="%.15g")
        (partial / "local_variable_dictionary.json").write_text(
            json.dumps(definitions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": "pynexus.wp6.iamc.v1",
            "status": "HASH_PINNED_WP4_SYNTHESIS_PYAM_ROUNDTRIP_PASSED",
            "source": "WP4 previously independently rechecked 35 full-year dispatches",
            "not_a_new_hourly_dispatch_verification": True,
            "source_run_id": WP4_RUN_ID,
            "source_artifact_id": 12585572146,
            "source_analysis_sha256": verified_sha,
            "source_code_revision": WP4_COMMIT,
            "reference_config_sha256": sha256(CONFIG),
            "variable_dictionary_sha256": sha256(VARIABLES_PATH),
            "pyam_version": pyam_version,
            "model": MODEL,
            "region": REGION,
            "scenarios_per_year": list(SCENARIOS),
            "weather_years_not_technology_trajectories": list(YEARS),
            "row_count": count,
            "variable_count": len(definitions["variables"]),
            "files_sha256": {
                name: sha256(partial / name)
                for name in ("iamc_wide.csv", "iamc_long.csv", "local_variable_dictionary.json")
            },
            "disclaimers": [
                "PyNEXUS variable names are local, not an official IAMC registry.",
                "Weather-year labels are independent dispatch experiments on fixed technology assumptions, not historical H2 production.",
                "The 2023 Open-Meteo ERA5 inputs are not identical to original CDS-2023 inputs.",
                "No empirical plant validation, emissions counterfactual, hydrogen certification or market-price evidence is implied.",
                "WP5 finance assumptions are incomplete; no cost or LCOH values are exported.",
            ],
        }
        (partial / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        partial.rename(output)
    except Exception:
        # Keep explicit failed attempt for audit; refuse silently replacing it.
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp4-synthesis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = export(args.wp4_synthesis, args.output)
    print(
        f"WP6 exported {result['row_count']} IAMC cells; pyam round-trip PASSED; "
        "LOCAL VARIABLES, NOT an IAMC registry"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
