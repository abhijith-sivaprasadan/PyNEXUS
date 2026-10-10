"""Independent synthetic demand/price sensitivities, serially or over MPI ranks.

Each scenario constructs a fresh optimiser/model. This distributes whole solves;
it does not parallelise the mathematical solver or represent observed weather.
"""

import argparse
import copy
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from optimization.dispatch import ElectrolyzerDispatchOptimizer
from optimization.verification import verify_dispatch

ROOT = Path(__file__).resolve().parents[1]


def scenarios():
    return [
        {"id": f"demand-{demand:g}_price-{price:g}", "demand_factor": demand, "price_factor": price}
        for demand in (0.8, 1.0, 1.2)
        for price in (0.5, 1.0, 1.5)
    ]


def solve_scenario(spec, config, hours):
    cfg = copy.deepcopy(config)
    cfg["simulation"]["time_horizon_hours"] = hours
    cfg["optimization"]["threads"] = 1
    cfg["optimization"]["random_seed"] = 0
    cfg["hydrogen_demand"]["daily_average_kg"] *= spec["demand_factor"]
    dt = cfg["simulation"]["time_step_hours"]
    count = hours / dt
    if count <= 0 or not float(count).is_integer():
        raise ValueError("hours must be a positive multiple of timestep")
    t = np.arange(int(count)) * dt
    # Same daily screening-input convention as the existing synthetic CLI.
    wind = 140.0 + 10.0 * np.sin(2 * np.pi * t / 24)
    price = spec["price_factor"] * (50.0 + 20.0 * np.cos(2 * np.pi * t / 24))
    with tempfile.TemporaryDirectory(prefix="pynexus-scenario-") as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        optimiser = ElectrolyzerDispatchOptimizer(str(path))
        result = optimiser.optimize(wind, price, demand_mode="cumulative")
        if result["status"] == "infeasible":
            # Preserve overload sensitivities rather than dropping/tuning them.
            # A hard cumulative demand above the theoretical maximum is an
            # independent infeasibility certificate, without trusting the solver.
            maximum = (
                min(optimiser.p_max * optimiser.h2_coeff, optimiser.max_h2_kg_s * 3600) * hours
            )
            required = optimiser.hourly_demand_kg * hours
            if required <= maximum + 1e-5:
                raise RuntimeError(f"{spec['id']}: infeasible without capacity certificate")
            return {
                "scenario": spec,
                "status": "infeasible",
                "objective_value": None,
                "table": {},
                "verification": {
                    "passed": True,
                    "checks": {"demand_exceeds_theoretical_capacity": True},
                    "required_hydrogen_kg": required,
                    "maximum_hydrogen_kg": maximum,
                },
            }
        if result["status"] != "optimal":
            raise RuntimeError(f"{spec['id']}: {result['status']}")
        check = verify_dispatch(
            optimiser,
            result["results_df"],
            result["objective_value"],
            "minimize_cost",
            "cumulative",
        )
        if not check["passed"]:
            raise RuntimeError(f"{spec['id']}: independent verification failed: {check}")
        return {
            "scenario": spec,
            "status": result["status"],
            "objective_value": float(result["objective_value"]),
            "table": result["results_df"].to_dict(orient="list"),
            "verification": check,
        }


def compare_results(actual, expected):
    """Require matching scenario IDs, objective, and every exported result column."""
    actual = {r["scenario"]["id"]: r for r in actual}
    expected = {r["scenario"]["id"]: r for r in expected}
    if actual.keys() != expected.keys():
        raise AssertionError("missing or unexpected scenarios")
    for key, result in actual.items():
        ref = expected[key]
        if result["scenario"] != ref["scenario"] or result["status"] != ref["status"]:
            raise AssertionError(f"{key}: scenario/status mismatch")
        if result["objective_value"] is None or ref["objective_value"] is None:
            if result["objective_value"] != ref["objective_value"]:
                raise AssertionError(f"{key}: objective presence mismatch")
        else:
            np.testing.assert_allclose(
                result["objective_value"], ref["objective_value"], rtol=1e-8, atol=1e-5
            )
        if result["table"].keys() != ref["table"].keys():
            raise AssertionError(f"{key}: column mismatch")
        for column, values in result["table"].items():
            np.testing.assert_allclose(
                values, ref["table"][column], rtol=1e-8, atol=1e-5, err_msg=f"{key}: {column}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tiny_test.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(argv)
    if args.hours <= 0:
        parser.error("hours must be positive")
    if args.serial:
        comm, rank, size = None, 0, 1
    else:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    # Each rank reads the same immutable configuration; failures are gathered too.
    if comm:
        comm.Barrier()
    started = time.perf_counter()
    local, errors = [], []
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        for i, spec in enumerate(scenarios()):
            if i % size == rank:
                local.append(solve_scenario(spec, config, args.hours))
    except Exception as exc:
        errors.append(f"rank {rank}: {type(exc).__name__}: {exc}")
    gathered = comm.gather((local, errors), root=0) if comm else [(local, errors)]
    failure = None
    if rank == 0:
        try:
            failures = [error for _, items in gathered for error in items]
            if failures:
                raise RuntimeError("; ".join(failures))
            results = sorted(
                [r for items, _ in gathered for r in items], key=lambda r: r["scenario"]["id"]
            )
            wall = time.perf_counter() - started
            if len(results) != len(scenarios()):
                raise RuntimeError("incomplete scenario set")
            if args.reference:
                ref = json.loads(args.reference.read_text(encoding="utf-8"))
                if ref["hours"] != args.hours:
                    raise ValueError("reference has a different horizon")
                compare_results(results, ref["results"])
            report = {
                "schema_version": 1,
                "ranks": size,
                "hours": args.hours,
                "input_source": "synthetic daily screening signals; demand/price sensitivity",
                "solver_threads_per_rank": 1,
                "wall_seconds": wall,
                "reference_matches": True if args.reference else None,
                "results": results,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(f"{size} rank(s), {len(results)} scenarios, {wall:.6f} s; verified")
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    if comm:
        failure = comm.bcast(failure, root=0)
    if failure:
        raise RuntimeError(failure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
