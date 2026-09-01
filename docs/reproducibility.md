# Reference workflow and verification

Install with `pip install -e ".[dev]"`. From the checkout:

```bash
pynexus solve --config config.yaml --synthetic --output outputs/reference-week
pynexus verify --run outputs/reference-week
```

The reference generator creates explicit daily sinusoids for wind, price and
carbon intensity. These are synthetic numerical inputs, not a North Sea forecast.
The model's engineering parameters are not altered to obtain a preferred result.

The manifest records git revision/dirty status (null if Git metadata cannot be
read), config/input/output SHA256, versions, solver settings and termination,
model counts and solve wall time. The verifier recomputes wind, binary/load,
ramp, pipeline, demand, hydrogen, curtailment, cost and objective checks from CSV.
An optimal solver result alone is not the verification gate.

## 2026-08-30 correction

Tests exposed demand/ramp/shortfall-penalty scaling defects for non-hourly steps,
invalid input acceptance, and an unsafe substring-based solver-status check.
Ten new regression cases failed before the fix. The corrected contracts use
elapsed hours, finite 1-D vectors and exact optimal termination.

For a hand-checkable 20 MW average demand over two intervals: required energy is
20 MWh at dt=0.5, 40 MWh at dt=1, and 80 MWh at dt=2. Previously all three
incorrectly required 40 MWh. At dt=1 the formulation is unchanged.

The 168-hour synthetic cost reference objective is 825902.3904382455 EUR in both
baseline `8c2ccd6` and the corrected model in the local review environment
(16598.60557768922 MWh dispatched). Re-run to obtain your own manifest; this is not a
locked-environment or annual benchmark release. No 8,760-hour claim is made.

## 2026-09-01 — first real 8,760-hour annual run

Phase C5 (`REVAMP_PLAN.md`). `configs/annual.yaml` (`time_horizon_hours: 8760`)
against real ERA5 wind for 2023 at the configured North Sea site
(`data/raw/era5_2023_wind.csv`, Phase C1/C5, no gaps — see `docs/data_provenance.md`),
paired with the same synthetic price/carbon generator used elsewhere (ENTSO-E
is Phase A3, not yet wired to `solve` — see `docs/data_provenance.md`).
Three runs, all with recorded solver status per the project's own rule that
an infeasible or time-limited solve is a result to report, not to hide —
artefacts committed under `outputs/annual_2023/`:

| Run | Flags | Status | Wall time | Model size (vars / bin / constraints) |
|---|---|---|---|---|
| `baseline` | none (wind-only, cumulative demand) | **infeasible** | 4.7 s | 26,280 / 8,760 / 52,559 |
| `grid_cumulative` | `--enable-grid` | **infeasible** | 5.9 s | 52,560 / 8,760 / 52,559 |
| `grid_storage_hourly` | `--enable-grid --enable-storage --demand-mode hourly` | **optimal, verified** | 16.7 s | 78,840 / 8,760 / 78,839 |

**Why the first two are infeasible, and why that's the correct answer, not a
bug**: `config.yaml`'s hydrogen demand (50 t/day → 18,250,000 kg/year at
`h2_coeff` ≈ 21.09 kg/MWh) needs about 865,499 MWh of electrolyser input
over the year. Real 2023 wind at this site only delivers about 522,097 MWh
total (annual capacity factor 0.397, see `docs/validation.md`), and even
capping delivery at the electrolyser's own 100 MW rating, wind alone can
supply at most 459,873 MWh — nowhere near enough for `demand_mode="cumulative"`'s
hard constraint. Adding the grid (`grid.connection_capacity_mw: 50`) raises
the ceiling to 728,175 MWh — still 137,324 MWh (16%) short of what full
cumulative satisfaction needs. Both are genuine infeasibilities of the
*as-configured* system against *real* wind, not solver or model defects —
confirmed by checking that even an unrealistic 100 MW grid connection
(matching the electrolyser's full rating) only just barely closes the gap
(876,000 MWh available vs 865,499 MWh needed, a 1.2% margin). No parameter
was changed to force a feasible cumulative result.

**The third run is the honest, meaningful annual result**: switching to
`demand_mode="hourly"` (soft, penalised shortfall) with both grid and
storage enabled solves to a *verified* optimum. Even with everything the
model offers turned on: 268,299 MWh imported from the grid, 62,278 MWh
exported, and **15.9% of annual hydrogen demand still goes unmet**
(2,897,283 of 18,250,000 kg, absorbed as penalised `demand_slack`) — the
configured 50 MW grid connection and 100 MW electrolyser, against this
specific site's real 2023 wind, cannot fully satisfy the configured demand
target at any price. That is a real capacity-planning finding this model
surfaced by actually being run at annual scale against real data, not
something visible from the 168-hour synthetic reference.

The objective value (≈2.94 billion EUR) is dominated by the `DEMAND_PENALTY`
term on that unmet 2.9 million kg — consistent with the same penalty-domination
finding already documented for the heat-value sweep in `docs/results.md`.

## Remaining release gates

A real annual run against real ERA5 wind now exists (above) with recorded
solver status, wall time, and model size, and real-data adapters (ERA5,
partially ENTSO-E) exist — closing two items previously listed here. Still
outstanding: peak RSS/hardware reporting, objective bound/gap detail beyond
what HiGHS's own termination reports, a repeated clean run and its archived
evidence, exact dependency locking, and a conventional src-layout migration.
Reference-run manifests do not satisfy that full benchmark protocol.
