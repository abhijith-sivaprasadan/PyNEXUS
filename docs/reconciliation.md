# Data reconciliation and gross-error detection

`optimization/reconciliation.py` (Phase A4). Same statistical method already
used in this portfolio's ThermoTwin repository — single-constraint weighted
least squares plus a chi-square(1) global test — applied here to the
hydrogen mass balance instead of a thermal energy balance.

**This operates on synthetic measurements** (see `tests/test_reconciliation.py`).
It demonstrates the method; it is not validated against plant instrumentation.

## The balance

Four redundant "sensor" measurements of hydrogen flow at the point where the
electrolyser's output splits toward storage and the pipeline:

```
production == to_store - from_store + to_pipeline
```

i.e. everything produced either goes into storage, comes back out of
storage, or heads toward demand via the pipeline. Given noisy measurements
of all four with stated variances, weighted least squares finds the
minimum-adjustment correction (weighted by each sensor's stated variance)
that closes the balance exactly.

## The gross-error test

`test_statistic = raw_imbalance^2 / (a^T diag(variance) a)` is compared
against `CHI_SQUARE_1_CRIT_999 = 10.83` (chi-square(1) at 99.9% confidence,
matching ThermoTwin's threshold). This correctly detects *that* the four
measurements are mutually inconsistent beyond what their stated
uncertainties would explain — `tests/test_reconciliation.py` covers both a
clean small-noise fixture (no flag) and a deliberately 50 kg/h-biased sensor
(flagged).

## What this does NOT do: name a "likely culprit" sensor

An earlier version of this module tried to rank the four sensors by their
WLS adjustment size (raw, and separately standardized by each sensor's own
variance) and report the largest as the probable fault. Both versions are
provably wrong, not just imprecise, and were removed rather than shipped
with a caveat:

```
adjustment_i = lam * var_i * a_i,   lam = raw_imbalance / sum_j(var_j)
```

`lam` is a single scalar shared by every sensor. So the *ratio* between any
two sensors' adjustments is fixed by their variances alone
(`adjustment_i / adjustment_j = ±var_i/var_j`) and does not depend on which
sensor's measurement actually contains the error. Verified directly in
`test_per_sensor_fault_is_not_identifiable_from_one_constraint`: biasing
each of the four sensors in turn, holding variances fixed, produces the
*exact same* adjustment ratios every time — a "largest adjustment" ranking
would report the identical sensor as the culprit regardless of which one
was actually wrong.

This is a real, general property of gross-error localization with a single
redundant constraint (the classic reference is Mah & Tamhane 1982's
measurement-test literature on the "smearing effect"), not a defect
specific to this implementation. Reliable fault isolation needs either
repeated/historical readings per sensor (to see which one is
*persistently* the outlier) or additional independent constraints (more
sensors/equations than this single balance provides) — both out of scope
here. The honest deliverable is: detect that something is wrong, and hand a
human the full measured/adjusted/adjustment breakdown to investigate, not a
false-precision pointer at one sensor.
