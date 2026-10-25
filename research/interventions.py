"""Registered WP2 intervention builder, execution and independently checked results.

Research status: full-year conclusions require actual GitHub Actions solve artifacts.
This module never modifies archived WP1 results or chooses values after seeing output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from optimization.dispatch import ElectrolyzerDispatchOptimizer
from research.adequacy import solve_adequacy_first
from research.run_reference import CONFIG, HISTORICAL, INPUTS, ROOT, sha256

PROTOCOL = ROOT / "research" / "protocol_wp2.yaml"


def registered_cases(order):
    """Baseline, five singles, ten pairs, and the full intervention set."""
    if len(order) != 5 or len(set(order)) != 5:
        raise ValueError("WP2 requires five distinct intervention identifiers")
    cases = [("baseline", ())]
    cases.extend((name, (name,)) for name in order)
    cases.extend(("+".join(pair), pair) for pair in itertools.combinations(order, 2))
    cases.append(("all", tuple(order)))
    if len(cases) != 17 or len({case[0] for case in cases}) != 17:
        raise RuntimeError("Invalid 17-case experimental registration")
    return cases


def config_for_case(base_config, protocol, active):
    """Return fresh config and source-wind scaling; reject baseline drift."""
    cfg = copy.deepcopy(base_config)
    order = protocol["order"]
    if not set(active).issubset(set(order)):
        raise ValueError("Unregistered intervention selected")
    interventions = protocol["interventions"]
    for name in order:
        definition = interventions[name]
        section, field = definition["field"].split(".", 1)
        observed = base_config[section][field]
        if not np.isclose(observed, definition["reference"], rtol=0, atol=1e-10):
            raise ValueError(f"{name} reference mismatch: config={observed}")
        if name in active:
            cfg[section][field] = definition["target"]
    # The observed reference annual wind series is an already-computed farm
    # output. For W, identical turbines and fixed per-unit loss factors imply
    # proportional scaling without inventing new weather observations.
    wind_scale = cfg["wind_turbine"]["n_turbines"] / base_config["wind_turbine"]["n_turbines"]
    if not np.isfinite(wind_scale) or wind_scale <= 0:
        raise ValueError("Nonpositive wind capacity multiplier")
    # Critical isolation: changing storage capacity DOES NOT create free
    # initial hydrogen nor alter the terminal inventory or charge/discharge.
    for key in (
        "initial_level_kg",
        "final_level_min_kg",
        "max_charge_rate_kg_h",
        "max_discharge_rate_kg_h",
    ):
        if cfg["hydrogen_storage"][key] != base_config["hydrogen_storage"][key]:
            raise ValueError(f"Storage control changed unexpectedly: {key}")
    if cfg["hydrogen_demand"] != base_config["hydrogen_demand"]:
        raise ValueError("Demand must remain fixed across primary interventions")
    return cfg, wind_scale


def _verified_archived_bytes(path, expected):
    """Prove byte identity or a CRLF/LF-only serialization change."""
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    lf = raw.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    candidates = {
        "BYTE_IDENTICAL": actual,
        "POSIX_LF_ONLY": hashlib.sha256(lf).hexdigest(),
        "WINDOWS_CRLF_ONLY": hashlib.sha256(crlf).hexdigest(),
    }
    matching = [kind for kind, digest in candidates.items() if digest == expected]
    if not matching:
        raise RuntimeError(
            "Archive provenance failure; expected manifest hash does not match "
            f"source or LF/CRLF equivalent: expected={expected}, actual={actual}"
        )
    return matching[0]


def source_inputs(protocol, *, hours):
    historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    provenance = _verified_archived_bytes(INPUTS, historical["input_sha256"])
    _verified_archived_bytes(CONFIG, historical["config_sha256"])
    frame = pd.read_csv(INPUTS)
    cols = ("wind_available_mw", "electricity_price", "carbon_intensity")
    if len(frame) != protocol["full_horizon_hours"] or not set(cols).issubset(frame.columns):
        raise ValueError("Unexpected source-year rows/columns")
    if not 1 <= hours <= len(frame):
        raise ValueError("Study horizon outside complete archived reference")
    for name in cols:
        values = frame[name].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite archived {name}")
    base_cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if base_cfg["simulation"]["time_step_hours"] != 1:
        raise ValueError("This study requires hourly reference inputs")
    return frame.iloc[:hours].copy(), base_cfg, provenance


def _record_dispatch(run, destination):
    """Persist all dispatch hours from both stages for independent checking."""
    a = destination / "stage_a.csv"
    b = destination / "stage_b.csv"
    run["stage_a"]["results_df"].to_csv(a, index=False)
    run["stage_b"]["results_df"].to_csv(b, index=False)
    return {"stage_a.csv": sha256(a), "stage_b.csv": sha256(b)}


def evaluate_case(case, base_cfg, source, protocol, destination, *, time_limit):
    """Two solves with isolated parameters and identical fixed source signals."""
    scenario_id, active = case
    cfg, wind_scale = config_for_case(base_cfg, protocol, active)
    cfg["simulation"]["time_horizon_hours"] = len(source)
    cfg["optimization"]["time_limit_seconds"] = time_limit
    cfg["optimization"]["mip_gap"] = protocol["mip_rel_gap"]
    destination.mkdir(parents=True, exist_ok=False)
    config_path = destination / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    opt = ElectrolyzerDispatchOptimizer(str(config_path))
    wind = source["wind_available_mw"].to_numpy(dtype=float) * wind_scale
    prices = source["electricity_price"].to_numpy(dtype=float)
    expected_capacity = cfg["wind_turbine"]["n_turbines"] * cfg["wind_turbine"]["rated_power_mw"]
    if np.any(wind < 0) or np.any(wind > expected_capacity + 1e-6):
        raise ValueError("Wind availability outside registered installed nameplate")
    run = solve_adequacy_first(opt, wind, prices, enable_storage=True)
    a = run["stage_a"]
    b = run["stage_b"]
    df = b["results_df"]
    dt = opt.dt
    # Primary outcome from Stage A only, avoiding the Stage B allowance.
    metrics = {
        "demand_h2_kg": float(run["total_demand_kg"]),
        "stage_a_shortfall_kg": float(run["stage_a_unserved_kg"]),
        "stage_a_shortfall_fraction": float(run["stage_a_unserved_kg"] / run["total_demand_kg"]),
        "stage_b_shortfall_kg": float(run["stage_b_unserved_kg"]),
        "independent_lower_bound_kg": float(run["optimistic_shortfall_lower_bound_kg"]),
        "stage_b_grid_settlement_eur_synthetic_prices": float(run["stage_b_grid_settlement_eur"]),
        "wind_available_mwh": float(df["wind_available_mw"].sum() * dt),
        "electrolyser_input_mwh": float(df["power_optimized_mw"].sum() * dt),
        "h2_produced_kg": float(df["h2_produced_kg_h"].sum() * dt),
        "grid_import_mwh": float(df["grid_import_mw"].sum() * dt),
        "grid_export_mwh": float(df["grid_export_mw"].sum() * dt),
        "grid_net_import_mwh": float((df["grid_import_mw"] - df["grid_export_mw"]).sum() * dt),
        "curtailment_mwh": float(df["curtailment_mw"].sum() * dt),
        "shortfall_hours_stage_b": int((df["demand_slack_kg_h"] > 1e-5).sum()),
        "electrolyser_capacity_mw": float(opt.p_rated),
        "grid_capacity_mw": float(opt.grid_connection_mw),
        "wind_nameplate_mw": float(expected_capacity),
        "nominal_efficiency_parameter": float(opt.nominal_eta),
        "linearized_h2_kg_per_mwh": float(opt.h2_coeff),
        "storage_capacity_kg": float(opt.storage_capacity_kg),
    }
    artifacts = _record_dispatch(run, destination)
    result = {
        "scenario": scenario_id,
        "active_interventions": list(active),
        "status": "VERIFIED_TWO_STAGE_SOLVE",
        "metrics": metrics,
        "config_sha256": sha256(config_path),
        "dispatch_hashes": artifacts,
        "solver_a": {
            k: a[k]
            for k in (
                "solver_status",
                "termination_condition",
                "solve_wall_time_s",
                "variables",
                "binary_variables",
                "constraints",
            )
        },
        "solver_b": {
            k: b[k]
            for k in (
                "solver_status",
                "termination_condition",
                "solve_wall_time_s",
                "variables",
                "binary_variables",
                "constraints",
            )
        },
        "verification_a": run["verification_a"]["checks"],
        "verification_b": run["verification_b"]["checks"],
    }
    (destination / "record.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def analyse_contrasts(results, order):
    """Finite difference effects on Stage A shortfall fraction."""
    lookup = {r["scenario"]: r for r in results}
    if any(r["status"] != "VERIFIED_TWO_STAGE_SOLVE" for r in results):
        raise ValueError("Cannot analyse a matrix with failed solves")
    base = lookup["baseline"]["metrics"]["stage_a_shortfall_fraction"]
    benefit = {
        name: base - result["metrics"]["stage_a_shortfall_fraction"]
        for name, result in lookup.items()
    }
    pairs = []
    for a, b in itertools.combinations(order, 2):
        name = a + "+" + b
        pairs.append(
            {
                "pair": name,
                "benefit_fraction": benefit[name],
                "interaction_fraction": benefit[name] - benefit[a] - benefit[b],
            }
        )
    return {
        "baseline_shortfall_fraction": base,
        "single_benefits_fraction": {name: benefit[name] for name in order},
        "pairwise_interactions": pairs,
        "all_benefit_fraction": benefit["all"],
        "interpretation": "Finite scenario contrasts; not an identified causal effect",
    }


def run_matrix(destination, *, hours=8760, scenario_filter=None, time_limit=600):
    """Fail closed on bad inputs; record errors instead of deleting failures."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("An archived study directory cannot be overwritten")
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["schema_version"] != "pynexus.wp2.v1":
        raise ValueError("Unsupported WP2 protocol version")
    order = protocol["order"]
    cases = registered_cases(order)
    if scenario_filter is not None:
        cases = [c for c in cases if c[0] == scenario_filter]
        if len(cases) != 1:
            raise ValueError("Unknown registered scenario")
    source, base_cfg, provenance = source_inputs(protocol, hours=hours)
    destination.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": "pynexus.wp2.result.v1",
        "status": "INCOMPLETE",
        "registered_scenarios": 17,
        "executed_scenarios": len(cases),
        "is_full_year": hours == 8760,
        "hours": hours,
        "weather_data_kind": "archived_ERA5_2023_wind",
        "price_data_kind": "synthetic_daily_sinusoid_NOT_historical_market_data",
        "historical_input_provenance_mode": provenance,
        "input_sha256": sha256(INPUTS),
        "source_manifest_sha256": sha256(HISTORICAL),
        "protocol_sha256": sha256(PROTOCOL),
        "research_version": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "research_objectives": [
            "minimize_unserved_kg",
            "minimize_grid_settlement_eur_subject_to_adequacy_cap",
        ],
        "cases": [],
        "failure_count": 0,
        "results_are_not_independent_real_world_validation": True,
    }
    errors = []
    for case in cases:
        label = case[0]
        try:
            record = evaluate_case(
                case,
                base_cfg,
                source,
                protocol,
                destination / label,
                time_limit=time_limit,
            )
            summary["cases"].append(record)
            m = record["metrics"]
            print(
                f"{label}: verified | unserved={m['stage_a_shortfall_kg']:.3f} kg "
                f"| grid_settlement={m['stage_b_grid_settlement_eur_synthetic_prices']:.2f} EUR",
                flush=True,
            )
        except Exception as exc:
            errors.append({"scenario": label, "error_type": type(exc).__name__, "error": str(exc)})
            summary["cases"].append(
                {
                    "scenario": label,
                    "active_interventions": list(case[1]),
                    "status": "ERROR_NOT_A_RESULT",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"{label}: FAILED: {type(exc).__name__}: {exc}", flush=True)
        (destination / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    summary["failure_count"] = len(errors)
    complete = len(cases) == 17 and not errors and hours == 8760
    if not errors:
        if len(cases) == 17:
            summary["contrasts"] = analyse_contrasts(summary["cases"], order)
        summary["status"] = (
            "COMPLETE_REGISTERED_MATRIX" if complete else "VERIFIED_PARTIAL_OR_SMOKE"
        )
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError(f"{len(errors)} of {len(cases)} scenario solves failed")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=int, default=8760)
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--time-limit", type=int, default=600)
    args = parser.parse_args(argv)
    result = run_matrix(
        args.output,
        hours=args.hours,
        scenario_filter=args.scenario,
        time_limit=args.time_limit,
    )
    print(
        f"WP2: {result['status']} with {len(result['cases'])} "
        f"verified scenarios; data type: ERA5 wind + synthetic prices"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
