"""Measure full process wall time (including MPI startup) and preserve every trial."""

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/mpi_benchmark.json")
    parser.add_argument("--launcher", default="mpiexec")
    args = parser.parse_args()
    if args.hours < 1 or args.repeats < 1:
        parser.error("hours and repeats must be positive")
    launcher = shutil.which(args.launcher)
    if not launcher:
        parser.error("an MPI runtime (mpiexec/mpirun) is required")
    directory = ROOT / ".benchmark-mpi"
    directory.mkdir(exist_ok=True)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    reference = directory / "reference.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "optimization.scenario_sweep",
            "--serial",
            "--hours",
            str(args.hours),
            "--output",
            str(reference),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=600,
    )
    rows = []
    for ranks in (0, 2, 4):
        total, compute = [], []
        for trial in range(args.repeats):
            output = directory / f"ranks-{ranks}-trial-{trial}.json"
            command = [
                sys.executable,
                "-m",
                "optimization.scenario_sweep",
                "--hours",
                str(args.hours),
                "--output",
                str(output),
                "--reference",
                str(reference),
            ]
            if ranks == 0:
                command.append("--serial")
            else:
                command = [launcher, "-n", str(ranks)] + command
            started = time.perf_counter()
            subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=600)
            total.append(time.perf_counter() - started)
            result = json.loads(output.read_text(encoding="utf-8"))
            if not result["reference_matches"]:
                raise RuntimeError("standalone reference mismatch")
            compute.append(result["wall_seconds"])
        rows.append(
            {
                "mode": "serial" if ranks == 0 else "mpi",
                "ranks": ranks or 1,
                "process_seconds": total,
                "compute_seconds": compute,
                "median_process_seconds": statistics.median(total),
                "median_compute_seconds": statistics.median(compute),
            }
        )
    for row in rows:
        row["speedup"] = rows[0]["median_process_seconds"] / row["median_process_seconds"]
        row["ideal_speedup"] = row["ranks"]
    report = {
        "platform": platform.platform(),
        "python": sys.version,
        "mpi_launcher": launcher,
        "hours": args.hours,
        "repeats": args.repeats,
        "scenario_count": 9,
        "reference_matches": True,
        "timing_scope": "process seconds include imports and MPI startup; compute includes model construction, solve and gather",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("mode ranks median_process_s median_compute_s speedup ideal_speedup")
    for row in rows:
        print(
            f"{row['mode']} {row['ranks']} {row['median_process_seconds']:.6f} "
            f"{row['median_compute_seconds']:.6f} {row['speedup']:.4f} {row['ideal_speedup']}"
        )


if __name__ == "__main__":
    main()
