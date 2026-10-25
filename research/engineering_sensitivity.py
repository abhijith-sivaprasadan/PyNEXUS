"""Isolated WP4 engineering sensitivity screening over the frozen CDS-2023 inputs.

These input values are illustrative perturbations, NOT probability intervals.
Do not compare different demand levels by absolute unmet kilograms alone.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from optimization.dispatch import ElectrolyzerDispatchOptimizer
from research.adequacy import solve_adequacy_first
from research.interventions import PROTOCOL as WP2_PROTOCOL
from research.interventions import source_inputs
from research.run_reference import CONFIG, INPUTS, ROOT, sha256

PROTOCOL = ROOT / "research" / "protocol_wp4_engineering.yaml"


def perturb_config(base, entry, source):
    """Change exactly one registered control; original caller data stay unchanged."""
    config = copy.deepcopy(base)
    data = source.copy(deep=True)
    name = entry["path"]
    if name is None:
        if entry["value"] is not None:
            raise ValueError("Baseline must have no perturbation value")
        return config, data
    value = entry["value"]
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
        raise ValueError("A sensitivity requires a finite numeric control")
    if name == "wind_availability_multiplier":
        if not 0 < value <= 1.5:
            raise ValueError("Wind scaling must be positive and bounded")
        installed = base["wind_turbine"]["rated_power_mw"] * base["wind_turbine"]["n_turbines"]
        wind = data["wind_available_mw"].to_numpy(dtype=float) * value
        # Keep fixed nameplate, clipping an optimistic recovery-of-losses
        # multiplier rather than introducing an unregistered capacity increase.
        data["wind_available_mw"] = np.minimum(wind, installed)
        return config, data
    parts = name.split(".")
    if len(parts) != 2 or parts[0] not in config or parts[1] not in config[parts[0]]:
        raise ValueError("Unregistered configuration field")
    before = config[parts[0]][parts[1]]
    if isinstance(before, bool) or not isinstance(before, (float, int)):
        raise ValueError("Sensitivity may change numeric configuration fields only")
    if value <= 0:
        raise ValueError("Perturbed technology/demand parameter must be positive")
    config[parts[0]][parts[1]] = value
    return config, data


def run_case(case, output, *, hours=24, time_limit=600):
    """Run, verify and preserve one source-isolated parameter perturbation."""
    if isinstance(hours, bool) or hours not in (24, 8760):
        raise ValueError("Registered sensitivity supports a 24h smoke or 8760h annual horizon")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Research evidence cannot be overwritten")
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["schema_version"] != "pynexus.wp4.engineering.v1":
        raise ValueError("Incorrect engineering sensitivity protocol")
    if protocol["grid_exchange_mode"] != "legacy":
        raise ValueError("Legacy base runs and exclusive-grid experiments must not be conflated")
    cases = protocol["registered_cases"]
    if case not in cases:
        raise ValueError("Unknown registered sensitivity")
    wp2 = yaml.safe_load(WP2_PROTOCOL.read_text(encoding="utf-8"))
    source, base, provenance = source_inputs(wp2, hours=hours)
    config, data = perturb_config(base, cases[case], source)
    config["simulation"]["time_horizon_hours"] = hours
    config["optimization"]["time_limit_seconds"] = time_limit
    config["optimization"]["mip_gap"] = wp2["mip_rel_gap"]
    output.mkdir(parents=True, exist_ok=False)
    cpath = output / "config.yaml"
    cpath.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    input_file = output / "inputs.csv"
    data.to_csv(input_file, index=False)
    manifest = {
        "schema_version": "pynexus.wp4.engineering.result.v1",
        "status": "INCOMPLETE_NO_VERIFIED_RESULT",
        "case": case,
        "intervention": cases[case],
        "hours": hours,
        "full_year": hours == 8760,
        "source_kind": "ARCHIVED_CDS_2023_WITH_SYNTHETIC_PRICE_ONLY",
        "grid_exchange_mode": "legacy",
        "source_hour_alignment_independently_verified": False,
        "wind_provenance_mode": provenance,
        "source_sha256": sha256(INPUTS),
        "baseline_config_sha256": sha256(CONFIG),
        "protocol_sha256": sha256(PROTOCOL),
        "case_config_sha256": sha256(cpath),
        "case_input_sha256": sha256(input_file),
        "revision": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "not_claimed": "Not empirical uncertainty, not real market price, not LCOH",
    }
    try:
        opt = ElectrolyzerDispatchOptimizer(str(cpath))
        result = solve_adequacy_first(
            opt,
            data["wind_available_mw"].to_numpy(dtype=float),
            data["electricity_price"].to_numpy(dtype=float),
            enable_storage=True,
        )
        a, b = result["stage_a"], result["stage_b"]
        stage_hashes = {}
        for label, stage in (("stage_a", a), ("stage_b", b)):
            file = output / f"{label}.csv"
            stage["results_df"].to_csv(file, index=False)
            stage_hashes[label] = sha256(file)
        demand = result["total_demand_kg"]
        manifest.update(
            {
                "status": (
                    "VERIFIED_ANNUAL_ENGINEERING_SCREENING"
                    if hours == 8760
                    else "VERIFIED_PARTIAL_24H_SMOKE_NOT_ANNUAL"
                ),
                "stage_dispatch_sha256": stage_hashes,
                "stage_a_status": a["status"],
                "stage_b_status": b["status"],
                "stage_a_verified": result["verification_a"]["passed"],
                "stage_b_verified": result["verification_b"]["passed"],
                "stage_a_unserved_kg": result["stage_a_unserved_kg"],
                "stage_a_unserved_fraction": result["stage_a_unserved_kg"] / demand,
                "stage_b_grid_settlement_eur_SYNTHETIC": result["stage_b_grid_settlement_eur"],
                "demand_kg": demand,
                "grid_import_mwh": float(b["results_df"]["grid_import_mw"].sum() * opt.dt),
                "grid_export_mwh": float(b["results_df"]["grid_export_mw"].sum() * opt.dt),
            }
        )
    except Exception as exc:
        manifest["failure_classification"] = type(exc).__name__
        raise
    finally:
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--hours", type=int, choices=(24, 8760), default=24)
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_case(args.case, args.output, hours=args.hours, time_limit=args.time_limit),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
