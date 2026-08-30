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

## Remaining release gates

An annual benchmark still needs peak RSS/hardware, objective bound/gap, detailed
solver statistics, a repeated clean run and its archived evidence. Exact dependency
locking, a conventional src-layout migration and real-data adapters remain separate
work. Reference-run manifests do not satisfy that full benchmark protocol.
