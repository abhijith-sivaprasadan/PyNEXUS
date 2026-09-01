# Phase B results: does heat coupling change dispatch?

**Data**: synthetic (deterministic-seeded), not real ERA5/ENTSO-E — the
question here is about the coupled optimisation model's own behaviour, which
is separable from Phase C's real-data work. 168 hourly timesteps, `config.yaml`'s
150 MW / 100 MW electrolyser/wind sizing. Wind and heat demand were both
given a diurnal cycle with heat demand shifted to peak overnight/morning
(the scenario REVAMP_PLAN.md's B4 calls "winter-peaking"); price and carbon
intensity use the same daily-sine shape as the rest of this repo's synthetic
demos. Full generating code and raw sweep output: `docs/figures/heat_value_sensitivity.png`
and the sweep script reproduced below.

## The finding: heat value has essentially zero effect on electrolyser dispatch, at any tested level

Swept `heat_value_per_mwh` from 0 to 500 EUR/MWh-th (config's default is 30)
against two synthetic wind scenarios — one abundant, one scarce — and with
`hydrogen_value_per_kg` both at its configured 4.0 and zeroed out to isolate
heat's own effect. In every case: **total electrolyser dispatch and online
hours were bit-for-bit identical across the entire heat-value range.** Only
the backup boiler's dispatch moved (10.2 → 21.3 MWh of boiler output across
0 → 120 EUR/MWh-th in the abundant-wind case) — heat value changes how much
gas the boiler is willing to burn to cover a gap, never whether the
electrolyser runs.

## Why: the demand-slack penalty structurally dominates

`ElectrolyzerDispatchOptimizer.DEMAND_PENALTY = 1000` EUR per kg of unmet
hourly hydrogen demand (`optimization/dispatch.py`) is calibrated to make
missing hydrogen demand practically prohibited — the model's original design
intent, unrelated to Phase B. At the configured `h2_coeff` (~21 kg/MWh), that
penalty is equivalent to roughly 21,000 EUR/MWh of foregone electrolyser
output whenever demand would go unmet. No heat value in any plausible
range (or even the 500 EUR/MWh-th tested, which is well above real
district-heating prices) comes close to competing with that. So in `hourly`
demand mode, the electrolyser's dispatch is driven almost entirely by (a)
wind availability and (b) avoiding the demand-slack penalty — the coupled
objective's hydrogen-value and heat-value terms are real and correctly
wired into the objective (`test_coupled_objective_independently_reconstructed`
confirms this numerically) but are simply too small to ever be the deciding
factor for the electrolyser's own on/off or load decision.

**Confirmed by isolating the mechanism**: re-running the same scarce-wind
sweep with `DEMAND_PENALTY` lowered to 0.5 EUR/kg (a scratch override, not
a config change — the real value stays 1000) makes heat value immediately
and strongly change dispatch: online hours rise from 7 to 125 (of 168) as
heat value goes from 0 to 500 EUR/MWh-th. See `docs/figures/heat_value_sensitivity.png` —
the flat blue line is the real, as-configured model; the rising pink line is
the same model with the demand penalty deliberately weakened, showing the
coupling *would* work as REVAMP_PLAN.md's B4 anticipated, if the penalty
weren't the dominant term.

```python
# Reproduction (abundant-wind default-penalty case; see docs/figures/ script
# history for the full sweep including the scarce-wind and lowered-penalty runs)
import numpy as np
from optimization.dispatch import ElectrolyzerDispatchOptimizer

opt = ElectrolyzerDispatchOptimizer("config.yaml")
T = 168
np.random.seed(11)
t = np.arange(T)
wind = np.clip(25 + 20 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 12, T), 0, 150)
price = np.clip(50 + 20 * np.cos(2 * np.pi * t / 24) + np.random.normal(0, 8, T), 5, 150)
carbon = np.clip(200 + 80 * np.sin(2 * np.pi * t / 24), 50, 400)
heat_demand = np.clip(12 + 6 * np.cos(2 * np.pi * (t % 24 - 7) / 24) + np.random.normal(0, 1, T), 3, 25)

for heat_value in [0.0, 30.0, 60.0, 120.0, 250.0, 500.0]:
    opt.heat_value_per_mwh = heat_value
    r = opt.optimize(
        wind, price, demand_mode="hourly", enable_heat=True,
        heat_demand_mw=heat_demand, carbon_intensity=carbon,
    )
    df = r["results_df"]
    print(heat_value, int(df["online_status"].sum()), float((df["power_optimized_mw"]).sum()))
```

## Seasonality: mistimed, but it doesn't matter here

REVAMP_PLAN.md's B4 also asks whether a winter-peaking heat demand paired
with winter-peaking wind helps or is mistimed. In this synthetic setup they
are **not** mistimed — both peak together on the same diurnal cycle by
construction (heat demand peaks overnight/morning; the wind profile's own
diurnal shape means more wind is often available then too, in the specific
random seed used). But per the finding above, this alignment is moot: since
heat value never changes electrolyser dispatch in the current configuration,
timing alignment between wind and heat demand has no measurable effect on
the coupled objective's outcome either. A real seasonality study — a full
year with a genuinely winter-vs-summer heat demand shape, once Phase C5's
annual ERA5 data exists — would be needed to say anything about the real
North Sea site; this synthetic week cannot support that claim.

## What this means for using the coupled objective

If the goal is to see heat/hydrogen value actually influence dispatch
decisions, `DEMAND_PENALTY` (and its heat equivalent, `HEAT_DEMAND_PENALTY`,
which is the same 1000 EUR-equivalent order of magnitude by construction)
needs to be deliberately tuned down from its "never miss demand" default, or
demand needs to be modelled as genuinely elastic rather than penalised. That
tension — guaranteed delivery vs. economically optimal dispatch — is a real
modelling choice this repo has not resolved, and REVAMP_PLAN.md did not
anticipate it. Recording it here rather than only in a sweep script is the
point of this document: a model that runs and shows no effect is a finding
worth keeping, not a failed experiment to bury.
