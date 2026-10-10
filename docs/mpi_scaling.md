# MPI scenario-level parallelism

`optimization.scenario_sweep` distributes nine whole, independent Pyomo/HiGHS solves across MPI ranks by scenario index. It is independent-task distribution, not a distributed linear-system or PDE solver. Every scenario constructs a fresh configuration, optimiser and Pyomo model; HiGHS uses one thread per rank and a fixed seed. There is no shared output directory for model state. Rank failures are gathered and broadcast so all ranks exit with a failure rather than leaving peers waiting at a collective.

The scenarios combine hydrogen demand multipliers 0.8/1.0/1.2 with electricity-price multipliers 0.5/1.0/1.5. These modify supported model inputs and use the CLI's existing deterministic daily synthetic wind/price convention. They are screening sensitivities, not observed ERA5 weather-years or historical forecasts. All nine scenarios are retained: six have optimal dispatches; the three 120%-demand cases are infeasible. An independent hydrogen-capacity bound certifies those overload cases. Infeasibility is reported, not silently converted into a successful dispatch or removed from the sample.

Install an MPI runtime first (Open MPI on Linux/macOS or Microsoft MPI on Windows), then:

```sh
python -m pip install -e '.[mpi]'
python -m optimization.scenario_sweep --serial --hours 168 --output standalone.json
mpiexec -n 4 python -m optimization.scenario_sweep --hours 168 --output mpi.json --reference standalone.json
python scripts/benchmark_mpi.py --hours 168 --repeats 3
```

On Open MPI, `mpirun` is an alternative to `mpiexec`. On machines with fewer available slots, an explicitly labelled small correctness run can use `--oversubscribe`; the local performance measurements below did not use oversubscription.

## Correctness

All feasible scenarios pass the existing independent dispatch verifier, including objective recomputation and physical constraints. Distributed output is compared with standalone output by scenario ID, status, objective, and every exported result-table column (rtol 1e-8, atol 1e-5). The new regression tests solve scenarios in forward and reverse order, compare results and confirm that the source configuration remains unchanged. MPI CI explicitly checks 2 and 4 ranks against a standalone reference.

## Local measurements

Measured 11 October 2026, Windows, Python 3.10, Microsoft MPI, nine scenarios of 168 hourly snapshots, three trials per row. All distributed trials matched their standalone reference. Raw trial timings are in [mpi_benchmark.json](mpi_benchmark.json).

The primary wall clock includes process startup, imports, MPI launch and the entire scenario set. A second timer covers model construction, solving and result gathering after imports/MPI setup. Keeping both prevents MPI startup overhead from being hidden.

| Mode | Ranks | Total wall time (s) | Compute/gather (s) | Total-time speedup | Ideal speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| serial | 1 | 5.234137 | 1.904120 | 1.0000 | 1 |
| mpi | 2 | 4.416798 | 0.970590 | 1.1851 | 2 |
| mpi | 4 | 4.221301 | 0.669762 | 1.2399 | 4 |

Startup/import overhead dominates this small scenario set, so four ranks yield only about 1.24x end-to-end speedup even though compute time falls more substantially. This is a bounded workstation demonstration; it establishes neither HPC-cluster scalability nor domain-specific fusion competence. Results depend on the machine, scenario count and solve duration. Existing verification suites have not been weakened or replaced.
