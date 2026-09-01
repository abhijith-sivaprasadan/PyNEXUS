"""Non-interactive reference runner with explicit inputs and reproducibility evidence."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.era5 import CoverageError, fetch_and_load
from optimization.dispatch import ElectrolyzerDispatchOptimizer
from optimization.verification import verify_dispatch


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git_state():
    root = Path(__file__).resolve().parents[1]

    def run(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=normal")
    return {"git_sha": sha, "working_tree_dirty": bool(status) if status is not None else None}


def solve(args):
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    opt = ElectrolyzerDispatchOptimizer(str(config_path))
    periods = cfg["simulation"]["time_horizon_hours"] / opt.dt
    if periods <= 0 or not float(periods).is_integer():
        raise ValueError("horizon must be a positive integer multiple of time_step_hours")
    count = int(periods)
    if args.synthetic:
        # Explicit screening inputs, not ENTSO-E/ERA5 data. Preserve original
        # engineering parameters; the generator does not tune model constraints.
        t = np.arange(count) * opt.dt
        inputs = pd.DataFrame(
            {
                "wind_available_mw": 140.0 + 10.0 * np.sin(2 * np.pi * t / 24),
                "electricity_price": 50.0 + 20.0 * np.cos(2 * np.pi * t / 24),
                "carbon_intensity": 200.0 + 100.0 * np.sin(2 * np.pi * t / 24),
            }
        )
        source = {"kind": "synthetic", "generator": "deterministic_daily_sines_v1"}
    else:
        inputs = pd.read_csv(args.input)
        source = {"kind": "user_csv", "original_sha256": sha256(Path(args.input))}
    if len(inputs) != count:
        raise ValueError("Input row count does not match the configured horizon")
    required = {"wind_available_mw", "electricity_price"}
    if not required.issubset(inputs):
        raise ValueError(f"Input requires columns {sorted(required)}")
    carbon = inputs["carbon_intensity"].to_numpy() if "carbon_intensity" in inputs else None
    out = Path(args.output)
    # Never mix a failed run with old successful tables.
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_bytes(config_path.read_bytes())
    inputs.to_csv(out / "inputs.csv", index=False)
    result = opt.optimize(
        inputs["wind_available_mw"],
        inputs["electricity_price"],
        objective=args.objective,
        demand_mode=args.demand_mode,
        carbon_intensity=carbon,
    )
    manifest = {
        "schema_version": 1,
        **git_state(),
        "snapshots": count,
        "time_step_hours": opt.dt,
        "input_source": source,
        "config_sha256": sha256(out / "config.yaml"),
        "input_sha256": sha256(out / "inputs.csv"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["pyomo", "highspy", "numpy", "pandas", "pyyaml"]
            },
        },
        "solver": {
            "name": "highs",
            "threads": opt.threads,
            "random_seed": opt.random_seed,
            "time_limit_s": opt.time_limit,
            "mip_rel_gap": opt.mip_gap,
        },
        "model": {key: result[key] for key in ["variables", "binary_variables", "constraints"]},
        "result": {
            key: result[key]
            for key in [
                "solver_status",
                "termination_condition",
                "objective_value",
                "solve_wall_time_s",
            ]
        },
        "objective": args.objective,
        "demand_mode": args.demand_mode,
        "benchmark_scope": "reference run; not the full annual benchmark protocol",
    }
    if result["results_df"] is None:
        write_json(out / "run_manifest.json", manifest)
        print(f"Solve rejected: {result['status']}", file=sys.stderr)
        return 2
    result["results_df"].to_csv(out / "dispatch.csv", index=False)
    manifest["dispatch_sha256"] = sha256(out / "dispatch.csv")
    evidence = verify_dispatch(
        opt,
        result["results_df"],
        result["objective_value"],
        args.objective,
        args.demand_mode,
        carbon,
    )
    write_json(out / "verification.json", evidence)
    manifest["verified"] = evidence["passed"]
    write_json(out / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(out),
                "verified": evidence["passed"],
                "snapshots": count,
                "objective": result["objective_value"],
            }
        )
    )
    return 0 if evidence["passed"] else 3


def fetch_era5_command(args):
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    df, record = fetch_and_load(
        cfg,
        args.start_date,
        args.end_date,
        cache_dir=Path(args.cache_dir),
        provenance_dir=Path(args.provenance_dir),
    )
    print(
        json.dumps(
            {
                "rows": len(df),
                "provenance_dir": args.provenance_dir,
                "file_sha256": record.file_sha256,
                "grid_cell": record.spatial_point,
            }
        )
    )
    return 0


def verify(args):
    out = Path(args.run)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["result"]["termination_condition"] != "optimal":
        raise ValueError("Run did not terminate optimally")
    for key, name in [
        ("config_sha256", "config.yaml"),
        ("input_sha256", "inputs.csv"),
        ("dispatch_sha256", "dispatch.csv"),
    ]:
        if manifest.get(key) != sha256(out / name):
            raise ValueError(f"Hash mismatch: {name}")
    opt = ElectrolyzerDispatchOptimizer(str((out / "config.yaml").resolve()))
    inputs = pd.read_csv(out / "inputs.csv")
    frame = pd.read_csv(out / "dispatch.csv")
    if len(frame) != manifest["snapshots"] or len(inputs) != len(frame):
        raise ValueError("Result horizon mismatch")
    for column in ["wind_available_mw", "electricity_price"]:
        if not np.allclose(inputs[column], frame[column], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Exported input mismatch: {column}")
    evidence = verify_dispatch(
        opt,
        frame,
        manifest["result"]["objective_value"],
        manifest["objective"],
        manifest["demand_mode"],
        inputs["carbon_intensity"].to_numpy() if "carbon_intensity" in inputs else None,
    )
    print(json.dumps(evidence))
    return 0 if evidence["passed"] else 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("solve")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="CSV with explicit input arrays")
    source.add_argument("--synthetic", action="store_true")
    run.add_argument(
        "--objective", choices=["minimize_cost", "minimize_emissions"], default="minimize_cost"
    )
    run.add_argument("--demand-mode", choices=["cumulative", "hourly"], default="cumulative")
    check = sub.add_parser("verify")
    check.add_argument("--run", required=True)
    fetch = sub.add_parser("fetch-era5")
    fetch.add_argument("--config", required=True)
    fetch.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    fetch.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    fetch.add_argument("--cache-dir", default="data/raw")
    fetch.add_argument("--provenance-dir", default="data/provenance_records")
    args = parser.parse_args(argv)
    commands = {"solve": solve, "verify": verify, "fetch-era5": fetch_era5_command}
    try:
        return commands[args.command](args)
    except (ValueError, KeyError, OSError, CoverageError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
