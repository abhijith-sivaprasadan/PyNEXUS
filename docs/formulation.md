# Formulation

For every hourly timestep, the dispatch model chooses electrolyser electrical input `p[t]` in MW, binary online status `u[t]`, and non-negative hydrogen-demand slack in kg/h.

The model minimises either electricity cost, `sum(price[t] * p[t] * dt)`, or the analogous grid-carbon-intensity expression. Hourly-demand mode adds a high penalty for unmet hydrogen. Cumulative-demand mode imposes total hydrogen production across the horizon.

Constraints enforce wind availability, online-dependent minimum and maximum electrolyser load, consecutive-hour ramp limits, a simplified pipeline mass-flow limit, and hourly or cumulative demand. Hydrogen production uses a linear coefficient in kg/MWh derived from the configured nominal LHV efficiency. The binary status makes this a mixed-integer linear model.

Units: power is MW, energy is MWh per timestep, hydrogen flow is kg/h or kg/s where explicitly stated, prices are EUR/MWh, and carbon intensity is kg CO2/MWh.

## Time and demand conventions

With timestep duration `dt` hours and demand rate `d` kg/h, cumulative demand is
`sum(p[t] * h2_coeff * dt) >= d * T * dt`. This is a horizon-total constraint,
not an hourly delivery/storage-balance model: it assumes temporal aggregation of
demand and does not model a hydrogen store. Hourly mode instead uses
`p[t] * h2_coeff + slack[t] >= d`; overproduction is allowed.

Ramps satisfy `abs(p[t]-p[t-1]) <= ramp_per_hour * rated_power * dt`.
The initial on/off and power state are unconstrained; startup/minimum up/down
times are not implemented. Pipeline capacity applies to a rate (kg/h), not energy.

Slack is a rate; its penalty is integrated as `1000 * sum(slack[t] * dt)`.
The penalty is a numerical modelling choice: EUR/kg in cost mode and kgCO2/kg
in emissions mode. It is not a sourced economic/environmental damage factor.
Cost and emissions objectives therefore have different units and must not be
compared as if they were the same scalar quantity.

## Hydrogen storage (Phase A1, opt-in: `optimize(enable_storage=True)`)

Off by default — the pre-Phase-A model and its locked reference objective
values (`docs/reproducibility.md`) are unchanged unless a caller explicitly
opts in. When enabled, adds `s[t]` (storage level, kg), `h_in[t]` (charge
rate, kg/h), `h_out[t]` (discharge rate, kg/h), with:

```
h_in[t] <= p[t] * h2_coeff                       # can't divert more than was produced
s[t] = s[t-1] + (h_in[t] - h_out[t]) * dt - loss[t]
loss[t] = loss_fraction_per_hour * s[t-1] * dt    # boil-off/leakage, see docs/assumptions.md
0 <= s[t] <= capacity_kg
0 <= h_in[t] <= max_charge_rate_kg_h
0 <= h_out[t] <= max_discharge_rate_kg_h
s[0] uses s[-1] = initial_level_kg
s[T-1] >= final_level_min_kg                      # terminal condition
```

The terminal condition exists because without it the optimiser drains the
store for free on the last timestep (discharging has no direct cost) and
never restocks — a classic storage-model artefact, not a real operating
strategy. `configs/tiny_test.yaml`'s `test_storage_terminal_condition_prevents_end_of_horizon_dumping`
exercises exactly this.

**Only meaningful in `demand_mode="hourly"`.** In hourly mode the demand
balance becomes `p[t]*h2_coeff - h_in[t] + h_out[t] + slack[t] >= demand_kg`
— production can now be diverted to storage or supplemented from it. In
`demand_mode="cumulative"`, storage variables still exist (and their own
constraints/terminal condition still bind) but do not appear in the demand
constraint at all, since cumulative demand is already a horizon-total
figure storage timing cannot affect; combining the two is not incorrect,
just not useful — storage has nothing to optimise for and the solver will
typically leave `h_in`/`h_out` at zero.

Cumulative-demand mode is retained unchanged (see "Time and demand
conventions" above) rather than replaced, per REVAMP_PLAN.md's Phase A1
instruction to keep both modes.

## Grid import/export (Phase A2, opt-in: `optimize(enable_grid=True)`)

Also off by default. When enabled, adds `g_imp[t]`, `g_exp[t]` (MW, both
bounded by `grid.connection_capacity_mw`) and `curtail[t]` (MW, unbounded
above), replacing the plain `p[t] <= wind[t]` constraint with a single
energy balance:

```
wind[t] + g_imp[t] == p[t] + g_exp[t] + curtail[t]
```

(REVAMP_PLAN.md's Phase A2 sketch gives two separate lines,
`wind[t] = p[t] + g_exp[t] + curtail[t]` and
`p[t] <= wind_to_electrolyser[t] + g_imp[t]`, with an undefined
`wind_to_electrolyser[t]`; the single balance above is the physically
correct combination — sources equal sinks — and is what's implemented.)

Objective gains `+ price[t]*g_imp[t]*dt` (cost) and `- price[t]*g_exp[t]*dt`
(revenue) in cost mode; `+ carbon[t]*g_imp[t]*dt` in emissions mode.

**No anti-simultaneous-import/export binary.** Checked whether it's needed,
per the plan's own instruction, and it is not free of a real caveat: this
model uses one scalar `price[t]` for both directions at each hour, so
importing and exporting the same amount at the same hour is cost-neutral —
`price[t]*g_imp[t]*dt - price[t]*g_imp[t]*dt = 0` — and not excluded by the
LP. The *net* exchange `g_imp[t] - g_exp[t]` is still uniquely pinned down
by the energy balance and by the objective (whenever price != 0), so
aggregate results (total cost/emissions, curtailment) are unaffected; only
the individual `g_imp[t]`/`g_exp[t]` values can be non-unique in a degenerate
solution. Add the binary if per-direction values need to be interpretable in
a specific run.

**Export is not credited as avoided grid emissions** in the emissions
objective (`self.credit_export_emissions = False`, hardcoded). Grid
displacement factors depend on whether you use marginal or average
generation-mix carbon intensity, which is genuinely contested and would
require a sourced, situation-specific factor this repo does not have.
Crediting it with an invented number would be exactly the kind of unsourced
figure the project's honesty rules forbid.

## Heat coupling (Phase B1-B3, opt-in: `optimize(enable_heat=True, heat_demand_mw=...)`)

Off by default. Requires `heat_demand_mw`, a time series (MW-th) the same
length as `wind_power_mw`, and `boiler.max_output_mw_th > 0` in config.

**B1 — waste heat.** `q_wh[t]` is not a decision variable; it's a Pyomo
`Expression` fixed by `p[t]`:

```
q_wh[t] = p[t] * (1 - eta_linearized) * recoverable_heat_fraction
```

`eta_linearized` is the SAME linearised efficiency the MILP's hydrogen
output (`h2_coeff`) already assumes (fixed at the 80%-load point via
`components.electrolyzer.efficiency_at_load`), not the full nonlinear
load-dependent curve — using a different efficiency for heat than for
hydrogen would silently violate the electrolyser's own energy balance.
`recoverable_heat_fraction` (default 0.5) reflects that PEM stack coolant
is typically 50-80°C — useful for district heating or low-temperature
industrial demand, not high-grade process heat; it is a stated assumption,
not a sourced figure for a specific stack (see `docs/assumptions.md`).

**B2 — heat storage and backup boiler.** Same SOC structure as hydrogen
storage (`e_hs`/`q_hs_in`/`q_hs_out`, standing loss, terminal condition),
plus `q_boiler[t]` (bounded by `boiler.max_output_mw_th`) and `q_dump[t]`
(heat recovered but not used — unbounded above, since waste heat exceeding
both demand and storage headroom has nowhere else to go). The balance is
soft (a `heat_slack[t]` absorbs shortfall, mirroring `demand_slack`):

```
q_hs_in[t] + q_dump[t] <= q_wh[t]                                    # can't divert more than recovered
(q_wh[t] - q_hs_in[t] - q_dump[t]) + q_hs_out[t] + q_boiler[t] + heat_slack[t] >= heat_demand[t]
```

When `heat_storage.capacity_mwh_th <= 0`, `q_hs_in`/`q_hs_out` are fixed
`Param`s at zero (no storage) rather than omitted, so the balance constraint
above is unconditional.

**B3 — the coupled objective.** Only affects `objective="minimize_cost"`
(the emissions objective, when `enable_heat=True`, gains only the boiler's
own combustion emissions and a heat-shortfall penalty — it does not value
heat or hydrogen, since "emissions" has no revenue concept). The cost
objective gains:

```
+ boiler_fuel_cost_eur_per_mwh_th * q_boiler[t] * dt        (fuel cost)
+ HEAT_DEMAND_PENALTY * heat_slack[t] * dt                  (shortfall penalty, mirrors DEMAND_PENALTY)
- hydrogen_value_per_kg * h2_delivered[t] * dt              (revenue)
- heat_value_per_mwh * heat_delivered[t] * dt                (revenue)
```

**This is a deliberate, consequential weighting choice, not a detail**: once
delivered hydrogen and heat carry a value, "minimise cost" stops meaning
"cheapest way to meet a fixed demand" and starts meaning "maximise net
economic value of both outputs" — the electrolyser can now be economically
incentivised to run beyond the bare minimum needed to satisfy `demand_kg`,
because doing so is profitable. `hydrogen_value_per_kg` (config default 4.0
EUR/kg) and `heat_value_per_mwh` (default 30 EUR/MWh-th) are both
illustrative, not sourced offtake prices.

**Real finding, not just a run**: sweeping `heat_value_per_mwh` from 0 to
500 EUR/MWh-th produces *zero* change in electrolyser dispatch at any
tested value, because `DEMAND_PENALTY` (1000 EUR/kg) structurally dominates
both value terms by orders of magnitude — see `docs/results.md` for the
full analysis, the mechanism, and the figure confirming it (and confirming
the coupling *does* work once the penalty is deliberately weakened).

## Why heat storage/boiler live in `optimization/dispatch.py`, not `components/`

REVAMP_PLAN.md's target structure lists `components/heat_storage.py` and
`components/boiler.py` as new files. This implementation puts them inline
in the optimizer instead, for the same reason hydrogen storage (Phase A1)
also went inline rather than into `components/hydrogen_storage.py`:
`OffshoreWindFarm`, `PEMElectrolyzer`, and `HydrogenPipeline` are physics
evaluators usable standalone, outside the MILP, computing a real physical
quantity from an input. Storage/boiler here are pure MILP decision
variables and linear balance constraints with no standalone behaviour
beyond exposing config parameters — there is no meaningful class to extract
that isn't just the optimizer's own state.
