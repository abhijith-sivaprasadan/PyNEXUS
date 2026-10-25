"""Re-run WP1 two-stage research objective on archived 2023 annual inputs.

Inputs include ERA5-derived 2023 wind and synthetic price/carbon signals.
This is a controlled computational experiment, not a market valuation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import pandas as pd

from optimization.dispatch import ElectrolyzerDispatchOptimizer
from research.adequacy import solve_adequacy_first

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "outputs" / "annual_2023" / "grid_storage_hourly"
INPUTS = REFERENCE / "inputs.csv"
CONFIG = REFERENCE / "config.yaml"
HISTORICAL = REFERENCE / "run_manifest.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_run(output, *, time_limit_seconds=600, mip_rel_gap=1e-4):
    """Generate a new research comparison, never overwrite historical evidence."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Output already exists; research runs never overwrite results")
    historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    expected = historical["input_sha256"]
    source_bytes = INPUTS.read_bytes()
    observed = hashlib.sha256(source_bytes).hexdigest()
    lf_bytes = source_bytes.replace(b"\\r\\n", b"\\n")
    windows_line_ending_sha256 = hashlib.sha256(
        lf_bytes.replace(b"\\n", b"\\r\\n")
    ).hexdigest()
    posix_line_ending_sha256 = hashlib.sha256(lf_bytes).hexdigest()
    if observed == expected:
        historical_input_mode = "BYTE_IDENTICAL"
    elif windows_line_ending_sha256 == expected:
        historical_input_mode = "LINE_ENDINGS_ONLY_WINDOWS_CRLF"
    elif posix_line_ending_sha256 == expected:
        historical_input_mode = "LINE_ENDINGS_ONLY_POSIX_LF"
    else:
        raise RuntimeError(
            "Historical input digest mismatch, NOT ACCEPTED: "
            f"expected={expected}, checked_out={observed}, "
            f"lf={posix_line_ending_sha256}, crlf={windows_line_ending_sha256}"
        )
    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)

    frame = pd.read_csv(INPUTS)
    required = {"wind_available_mw", "electricity_price", "carbon_intensity"}
    if not required.issubset(frame.columns) or len(frame) != 8760:
        raise ValueError("The archived input must contain 8,760 rows and the known schema")
    opt = ElectrolyzerDispatchOptimizer(str(CONFIG))
    opt.time_limit = float(time_limit_seconds)
    opt.mip_gap = float(mip_rel_gap)
    if opt.dt != 1 or len(frame) != 8760:
        raise ValueError("Reference experiment requires a complete 1-hour 2023 year")
    report = solve_adequacy_first(
        opt,
        frame["wind_available_mw"].to_numpy(),
        frame["electricity_price"].to_numpy(),
        enable_storage=True,
    )
    if not report["verification_a"]["passed"] or not report["verification_b"]["passed"]:
        raise RuntimeError("The two independent verifications must both pass")
    output.mkdir(parents=True)
    report["stage_a"]["results_df"].to_csv(output / "dispatch_adequacy.csv", index=False)
    report["stage_b"]["results_df"].to_csv(output / "dispatch_grid_settlement.csv", index=False)

    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    manifest = {
        "schema_version": "pynexus.research.wp1.v1",
        "research_status": "AUTHOR_REPORTED_COMPUTATIONAL_RUN_NOT_EXTERNAL_VALIDATION",
        "historical_2023_record_reused_as_input_not_edited": True,
        "input_kind": "historical_2023_ERA5_wind_and_synthetic_prices",
        "reference_manifest_sha256": sha256(HISTORICAL),
        "reference_config_sha256": sha256(CONFIG),
        "input_sha256": sha256(INPUTS),
        "original_manifest_input_sha256": expected,
        "historical_input_verification_mode": historical_input_mode,
        "git_sha": revision.stdout.strip() if revision.returncode == 0 else None,
        "python_packages": {
            x: importlib.metadata.version(x) for x in ("pyomo", "highspy", "numpy", "pandas")
        },
        "solver": {
            "name": "highs",
            "mip_rel_gap_limit": mip_rel_gap,
            "time_limit_seconds": time_limit_seconds,
        },
        "result": {
            "total_demand_kg": report["total_demand_kg"],
            "optimistic_shortfall_lower_bound_kg": report["optimistic_shortfall_lower_bound_kg"],
            "stage_a_unserved_kg": report["stage_a_unserved_kg"],
            "stage_b_unserved_kg": report["stage_b_unserved_kg"],
            "numerical_allowance_kg": report["numerical_allowance_kg"],
            "stage_b_grid_settlement_eur": report["stage_b_grid_settlement_eur"],
            "adequacy_objective": report["stage_a"]["objective_definition"],
            "cashflow_objective": report["stage_b"]["objective_definition"],
        },
        "stage_a_solver": {
            k: report["stage_a"][k]
            for k in (
                "solver_status",
                "termination_condition",
                "solve_wall_time_s",
                "variables",
                "binary_variables",
                "constraints",
            )
        },
        "stage_b_solver": {
            k: report["stage_b"][k]
            for k in (
                "solver_status",
                "termination_condition",
                "solve_wall_time_s",
                "variables",
                "binary_variables",
                "constraints",
            )
        },
        "independent_verification": {
            "adequacy": report["verification_a"]["checks"],
            "grid_settlement": report["verification_b"]["checks"],
        },
        "artifacts": {
            "dispatch_adequacy_sha256": sha256(output / "dispatch_adequacy.csv"),
            "dispatch_grid_settlement_sha256": sha256(output / "dispatch_grid_settlement.csv"),
        },
        "limitations": [
            "The archived 2023 electricity prices are synthetic, not ENTSO-E.",
            "The economic objective is grid settlement, not total project cost/LCOH.",
            "No independent plant calibration, external replication or multi-IAM validation.",
            "Legacy 15.9% and EUR 2.94bn objectives are from another formulation.",
        ],
    }
    (output / "research_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=int, default=600)
    parser.add_argument("--mip-rel-gap", type=float, default=1e-4)
    args = parser.parse_args(argv)
    manifest = reference_run(
        args.output,
        time_limit_seconds=args.time_limit_seconds,
        mip_rel_gap=args.mip_rel_gap,
    )
    print(json.dumps(manifest["result"], sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
