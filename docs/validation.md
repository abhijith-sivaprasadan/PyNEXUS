# Capacity-factor sanity check

**This is a comparison, not a calibration.** No model parameter was adjusted
to make this number land closer to the published range — see
`CONTRIBUTING.md` and `REVAMP_PLAN.md`'s governing rules.

## What is compared

The model's own annual capacity factor for the configured North Sea site
(52.5°N, 3.5°E, 15 MW turbines × 10, 120 m hub height), computed from real
ERA5 wind (Phase C1) run through `components.wind_turbine.OffshoreWindFarm`
including the shear correction, the analytic cubic power curve, and the
configured wake/electrical-loss/availability factors — against a published
figure for North Sea offshore capacity factors at comparable turbine scale.

## Published reference

Elizalde, A., Akhtar, N., Geyer, B., and Schrum, C.: "Uncertainty in North
Sea offshore wind power: contributions of reanalysis forcing, turbine type,
and wake parameterization", *Wind Energy Science*, 11, 1077–1095, 2026.
https://doi.org/10.5194/wes-11-1077-2026 (peer-reviewed, Copernicus).

This paper is an unusually close match to our own setup: it simulates a
150 GW North Sea wind farm cluster driven by ERA5, at two turbine ratings —
3.6 MW and **15 MW**, the same rating configured in this repository's
`config.yaml`. Reported mean annual load factors (their Table 7, ERA5-driven
scenario):

| Scenario | Load factor (no wake) | Load factor (with wake) |
|---|---|---|
| 3.6 MW turbines | 0.57 | 0.42 |
| **15 MW turbines** | **0.61** | **0.49** |

They also cite a broader literature range for context: 0.23–0.52 (mean
0.35) across US and North Sea offshore installations of varying turbine
size and vintage (Cassa, 2024; Smith, 2024, cited therein).

## This model's result

Computed from the real Phase C5 annual ERA5 fetch (52.5°N, 3.5°E, 2023,
8,760 hours, no gaps) run through `OffshoreWindFarm.power_output_mw_from_hub_height`
with `config.yaml`'s configured turbine/loss parameters:

| Quantity | Value |
|---|---|
| Annual mean farm power | 59.6 MW (of 150 MW rated) |
| **Annual capacity factor (as configured, all losses applied)** | **0.397** |
| CF with wake loss only (electrical/availability losses removed) | 0.427 |
| CF with no losses at all (raw power curve output) | 0.474 |

## Comparison

| Scenario | Published (15 MW, Elizalde et al.) | This model |
|---|---|---|
| No wake / no losses | 0.61 | 0.474 |
| With wake (/ with losses) | 0.49 | 0.397 (all losses) / 0.427 (wake only) |
| Broader literature range | 0.23–0.52 (mean 0.35) | 0.397 — inside the range, above the mean |

**Against the broader literature range, this is a clean match** — 0.397 sits
comfortably inside 0.23–0.52 and above its 0.35 mean, for either the
all-losses or wake-only figure.

**Against the specific 15 MW Elizalde figures, this model runs lower** — by
roughly 6 points (0.427 vs 0.49) on the wake-comparable basis, and by
roughly 14 points (0.474 vs 0.61) on the no-loss basis. The no-loss gap is
larger than the wake-comparable gap, which points away from this model's
loss-factor choices being the main cause (removing them entirely still
leaves the biggest gap) and toward two more likely, already-documented
structural differences: (1) this model's power curve is a cubic
approximation between cut-in and rated (`docs/assumptions.md` already flags
that real turbine curves are not cubic near rated, and a cubic curve is
known to underestimate output in the upper-mid range compared to a real
manufacturer curve); and (2) this is one grid point for one calendar year
(2023), not Elizalde et al.'s 150 GW regional cluster averaged, which
smooths over single-point/single-year wind variability that can run below a
multi-site multi-year mean.

**Per REVAMP_PLAN.md's own instruction** ("if the model CF is far off,
investigate before adjusting"): a 0.397 result against a 0.23–0.61
published spread across these sources is not far off, and the plausible
causes above are already-documented modelling choices, not a defect to
patch by tuning a loss factor to move the number closer to 0.49. No
parameter was changed after seeing this result.

## Interpretation

Our farm-level model nets out wake loss (`wake_loss_fraction`), electrical
loss (`electrical_loss_fraction`), and unavailability
(`unavailability_fraction`) as a single combined multiplicative factor
applied uniformly across the year (see `docs/assumptions.md`) — not a
spatially resolved wake simulation like Elizalde et al.'s mesoscale model.
Agreement with the broader published range is a plausibility check on the
right order of magnitude for a screening model; it is not validation of
this model's specific loss-factor values or power curve, which remain
literature-sourced/analytic assumptions rather than a project-specific
engineering or measurement-based study.
