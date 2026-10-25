"""WP4 five-year synthesis: independent rechecks, normalized rankings and scope flags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from optimization.dispatch import ElectrolyzerDispatchOptimizer
from optimization.verification import verify_dispatch
from research.interventions import PROTOCOL as WP2_PROTOCOL
from research.interventions import config_for_case, source_inputs
from research.run_reference import ROOT, sha256
from research.run_weather_year import CASES, PROTOCOL
from research.weather_source import convert_weather_to_wind_2023_model, validate_weather_payload

YEARS = (2021, 2022, 2023, 2024, 2025)


def summarize_year(archive, registration, wp2, base_config):
    path = archive / "year_manifest.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    year = int(result["year"])
    if result["schema_version"] != "pynexus.wp4.year.v1" or result["status"] != "COMPLETE_SEVEN_CASES_YEAR":
        raise ValueError(f"{year}: no complete, verified year manifest")
    if year not in YEARS or result["hours"] != registration["hours_per_year"][str(year)]:
        raise ValueError("Year not registered or wrong leap-year horizon")
    if (result["wp4_protocol_sha256"] != sha256(PROTOCOL)
        or result["wp2_protocol_sha256"] != sha256(WP2_PROTOCOL)):
        raise ValueError(f"{year}: study protocol checksum drift")
    if result["base_config_sha256"] != sha256(ROOT / "outputs/annual_2023/grid_storage_hourly/config.yaml"):
        raise ValueError(f"{year}: reference config changed")
    raw = (archive / "source_weather_api.json").read_bytes()
    if sha256(archive / "source_weather_api.json") != result["raw_api_response_sha256"]:
        raise ValueError(f"{year}: raw provider response hash mismatch")
    times, v100, source_meta = validate_weather_payload(raw, year)
    if source_meta["raw_json_sha256"] != result["source"]["raw_json_sha256"]:
        raise ValueError("Provider response provenance mismatch")
    frame = pd.read_csv(archive / "hourly_weather_and_inputs.csv")
    if sha256(archive / "hourly_weather_and_inputs.csv") != result["hourly_input_csv_sha256"]:
        raise ValueError("Weather and synthetic-price CSV hash mismatch")
    if len(frame) != len(times) or len(result["cases"]) != len(CASES):
        raise ValueError("Wrong number of 2023-source-independent observations/cases")
    if not np.array_equal(frame["datetime_utc"].to_numpy(), times.strftime("%Y-%m-%dT%H:%M:%SZ")):
        raise ValueError("Weather timestamp order changed")
    expected_wind = convert_weather_to_wind_2023_model(
        v100, ROOT / "outputs/annual_2023/grid_storage_hourly/config.yaml"
    )
    if not np.allclose(frame["wind_available_mw"], expected_wind, rtol=1e-9, atol=1e-7):
        raise ValueError(f"{year}: wind preprocessing changed")
    if not np.allclose(frame["wind_speed_100m_ms"], v100, rtol=1e-9, atol=1e-6):
        raise ValueError("Raw weather speed != stored hourly wind")
    config = dict(base_config)
    config["simulation"] = dict(base_config["simulation"], start_date=f"{year}-01-01")
    rows = []
    for recorded, (name, active) in zip(result["cases"], CASES.items(), strict=True):
        if recorded["scenario"] != name or recorded["status"] != "VERIFIED_TWO_STAGE_SOLVE":
            raise ValueError(f"{year}: incorrect case name/status")
        folder = archive / name
        record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
        if recorded != record or list(active) != record["active_interventions"]:
            raise ValueError(f"{year}: altered case record {name}")
        expected_cfg, wind_scale = config_for_case(config, wp2, active)
        expected_cfg["simulation"]["time_horizon_hours"] = len(times)
        expected_cfg["optimization"]["time_limit_seconds"] = 600
        expected_cfg["optimization"]["mip_gap"] = wp2["mip_rel_gap"]
        cfgpath = folder / "config.yaml"
        if sha256(cfgpath) != record["config_sha256"]:
            raise ValueError("Case configuration hash mismatch")
        cfg = yaml.safe_load(cfgpath.read_text(encoding="utf-8"))
        if expected_cfg != cfg:
            raise ValueError(f"{year} {name}: nonregistered modification to physical system")
        opt = ElectrolyzerDispatchOptimizer(str(cfgpath))
        a_path, b_path = (folder / f"stage_{x}.csv" for x in ("a", "b"))
        for p in (a_path, b_path):
            if sha256(p) != record["dispatch_hashes"][p.name]:
                raise ValueError(f"{year} {name}: hourly dispatch hash mismatch")
        a, b = pd.read_csv(a_path), pd.read_csv(b_path)
        wind_scaled = expected_wind * wind_scale
        for exported in (a, b):
            if len(exported) != len(times) or not np.allclose(
                exported["wind_available_mw"], wind_scaled, atol=1e-7, rtol=1e-9
            ):
                raise ValueError(f"{year} {name}: mismatched intervention weather input")
        for exported, obj, value, key, cap in (
            (a, "minimize_unserved", record["metrics"]["stage_a_shortfall_kg"], "verification_a", None),
            (b, "minimize_cash_cost",
             record["metrics"]["stage_b_grid_settlement_eur_synthetic_prices"],
             "verification_b",
             record["metrics"]["stage_a_shortfall_kg"]
             + max(1e-4, 1e-8 * record["metrics"]["demand_h2_kg"])),
        ):
            verification = verify_dispatch(
                opt, exported, value, obj, "hourly", enable_grid=True,
                enable_storage=True, shortfall_cap_kg=cap,
            )
            if not verification["passed"] or verification["checks"] != record[key]:
                raise ValueError(f"{year} {name}: independent verification rejected dispatch")
        fraction = float(a["demand_slack_kg_h"].sum() * opt.dt /
                         (a["h2_demand_kg_h"].sum() * opt.dt))
        if not np.isclose(fraction, record["metrics"]["stage_a_shortfall_fraction"], atol=1e-9):
            raise ValueError("Stored KPI differs from independently computed shortfall")
        rows.append({
            "year": year,
            "scenario": name,
            "unserved_h2_kg": float(record["metrics"]["stage_a_shortfall_kg"]),
            "unserved_fraction": fraction,
            "wind_resource_mwh": float(expected_wind.sum()),
            "wind_capacity_factor": float(result["annual_nameplate_capacity_factor"]),
            "hours": len(times),
        })
    by_name = {row["scenario"]: row for row in rows}
    b = by_name["baseline"]["unserved_fraction"]
    for row in rows:
        row["benefit_percentage_points"] = (b - row["unserved_fraction"]) * 100
    es_conditional = (
        by_name["E"]["unserved_fraction"] - by_name["E+S"]["unserved_fraction"]
    ) * 100
    interaction_pp = (
        by_name["E"]["unserved_fraction"] + by_name["S"]["unserved_fraction"]
        - b - by_name["E+S"]["unserved_fraction"]
    ) * 100
    if not np.isclose(interaction_pp, result["electrolyser_storage_interaction_pp"], atol=1e-6):
        raise ValueError("Wrong archived E+S interaction")
    if not np.isclose(es_conditional, result["storage_conditional_gain_pp"], atol=1e-6):
        raise ValueError("Wrong archived conditional storage benefit")
    return rows, {
        "year": year,
        "hours": len(times),
        "wind_capacity_factor": result["annual_nameplate_capacity_factor"],
        "baseline_unserved_fraction": b,
        "storage_conditional_benefit_after_E_pp": es_conditional,
        "E_plus_S_interaction_pp": interaction_pp,
        "provider": "Open-Meteo explicit ERA5 (NOT original CDS-2023 input)",
        "cds_2023_comparison": result["original_source_2023_comparison"],
        "source_api_sha256": result["raw_api_response_sha256"],
        "run_manifest_sha256": sha256(path),
        "source_commit": result["code_revision"],
    }


def aggregate(root):
    root = Path(root)
    reg = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    wp2 = yaml.safe_load(WP2_PROTOCOL.read_text(encoding="utf-8"))
    _, base, _ = source_inputs(wp2, hours=8760)
    all_rows, years = [], []
    for year in YEARS:
        rows, stats = summarize_year(root / f"wp4-year-{year}", reg, wp2, base)
        all_rows.extend(rows)
        years.append(stats)
    revisions = {r["source_commit"] for r in years}
    if len(revisions) != 1:
        raise ValueError("Mixed code versions across weather-year scenario runs")
    if len(all_rows) != 35:
        raise ValueError("35 controlled year×scenario results required")
    for y in years:
        if y["hours"] not in (8760, 8784):
            raise ValueError("Bad year normalization")
    df = pd.DataFrame(all_rows)
    yr = pd.DataFrame(years)
    columns = ["baseline", "W", "E", "G", "S", "eta"]
    rankings = {}
    for year in YEARS:
        temp = df[(df["year"] == year) & (df["scenario"].isin(columns))].copy()
        rankings[year] = temp.sort_values(
            ["benefit_percentage_points", "scenario"],
            ascending=[False, True],
        )["scenario"].tolist()
    unique_winners = {
        y: rankings[y][0] for y in YEARS
    }
    return {
        "schema_version": "pynexus.wp4.interannual.v1",
        "status": "COMPLETE_5_YEAR_35_SCENARIO_COMPARISON",
        "years": list(YEARS),
        "raw_weather_source": "Open-Meteo ERA5 100m, not exact CDS-2023 original",
        "source_revision": next(iter(revisions)),
        "scenario_count": len(all_rows),
        "weather_year_count": 5,
        "annual_results": all_rows,
        "annual_metrics": years,
        "individual_intervention_rankings": rankings,
        "best_single_intervention_by_year": unique_winners,
        "baseline_shortfall_fraction_range": [
            float(yr["baseline_unserved_fraction"].min()),
            float(yr["baseline_unserved_fraction"].max()),
        ],
        "ES_interaction_pp_range": [
            float(yr["E_plus_S_interaction_pp"].min()),
            float(yr["E_plus_S_interaction_pp"].max()),
        ],
        "limits": [
            "Years are contiguous weather history, not independent random trials.",
            "Open-Meteo's ERA5 serving/selection is not the identical CDS grid-cell extraction used for 2023 WP2.",
            "No ERA5 year has been used as an observed price input: 2023 synthetic prices repeat modulo 8760 hours.",
            "Stage A shortfall fractions and conditional E+S contrast are primary endpoints.",
            "No economic cost/profit, hydrogen certification or life-cycle emissions interpretation is supported.",
            "Results are author-run numerical findings, not external industrial validation.",
        ],
    }


def report(result):
    lines = [
        "# WP4 — Interannual weather robustness (2021–2025)",
        "",
        "**Evidence:** complete-year Open-Meteo explicitly selected ERA5 100 m wind, "
        "converted through the same PyNEXUS turbine model. "
        "**NOT the same archive/extraction as CDS-backed 2023 WP2.**",
        "",
        "| Year | Wind CF | Baseline shortfall | E+S interaction | Conditional storage gain after E |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in result["annual_metrics"]:
        lines.append(
            f"| {x['year']} | {x['wind_capacity_factor'] * 100:.2f}% "
            f"| {x['baseline_unserved_fraction'] * 100:.3f}% "
            f"| {x['E_plus_S_interaction_pp']:+.3f} pp "
            f"| {x['storage_conditional_benefit_after_E_pp']:.3f} pp |"
        )
    lines.extend([
        "",
        "## Single intervention ranking by year",
        "",
        "| Year | Ranking, largest to smallest reduction in unmet H2 |",
        "|---|---|",
    ])
    for year, ranking in result["individual_intervention_rankings"].items():
        lines.append(f"| {year} | {' > '.join(ranking)} |")
    lines.extend([
        "",
        "## 2023 cross-source check",
        "",
        "The 2023 Open-Meteo ERA5 run is recomputed in the same data family as every other "
        "WP4 year. It is **not substituted for**, and should not be pooled with, "
        "the earlier CDS-backed 2023 WP2 control.",
    ])
    ctrl = next(x["cds_2023_comparison"] for x in result["annual_metrics"] if x["year"] == 2023)
    lines.extend([
        f"- CDS-2023 wind availability: {ctrl['historical_cds_wind_available_mwh']:.2f} MWh",
        f"- Open-Meteo ERA5-2023 wind availability: "
        f"{ctrl['open_meteo_era5_wind_available_mwh']:.2f} MWh",
        f"- Hourly two-source wind-power MAE: {ctrl['hourly_power_mae_mw']:.3f} MW",
        f"- Hourly two-source wind-power RMSE: {ctrl['hourly_power_rmse_mw']:.3f} MW",
        "",
        "## Important limitations",
        "",
        "- This is a 5-year sensitivity screening, not a probabilistic prediction "
        "or evidence of future wind-climate frequencies.",
        "- Wind is from Open-Meteo's explicitly selected ERA5 route. "
        "Observed power-curve/farm representation remains simplified.",
        "- The price series is a deterministic *synthetic 2023* sequence reused each "
        "year. Only physical adequacy metrics are used for intervention ranking.",
        "- Grid-imported electricity meeting hydrogen demand does not imply it is renewable.",
        "- An interaction in this numerical scenario set is a finite contrast, "
        "not proof of a field-observed technology synergy.",
    ])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("WP4 comparison archives are immutable")
    result = aggregate(args.root)
    args.output.mkdir(parents=True)
    (args.output / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(result["annual_results"]).to_csv(
        args.output / "scenario_comparison.csv", index=False
    )
    pd.DataFrame(result["annual_metrics"]).to_csv(
        args.output / "year_comparison.csv", index=False
    )
    (args.output / "RESEARCH_FINDINGS.md").write_text(
        report(result), encoding="utf-8"
    )
    print(
        f"WP4 VERIFIED: {result['weather_year_count']} years "
        f"{result['scenario_count']} cases; E+S interaction range "
        f"{result['ES_interaction_pp_range']} pp",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
