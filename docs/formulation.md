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
