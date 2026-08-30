# Formulation

For every hourly timestep, the dispatch model chooses electrolyser electrical input `p[t]` in MW, binary online status `u[t]`, and non-negative hydrogen-demand slack in kg/h.

The model minimises either electricity cost, `sum(price[t] * p[t] * dt)`, or the analogous grid-carbon-intensity expression. Hourly-demand mode adds a high penalty for unmet hydrogen. Cumulative-demand mode imposes total hydrogen production across the horizon.

Constraints enforce wind availability, online-dependent minimum and maximum electrolyser load, consecutive-hour ramp limits, a simplified pipeline mass-flow limit, and hourly or cumulative demand. Hydrogen production uses a linear coefficient in kg/MWh derived from the configured nominal LHV efficiency. The binary status makes this a mixed-integer linear model.

Units: power is MW, energy is MWh per timestep, hydrogen flow is kg/h or kg/s where explicitly stated, prices are EUR/MWh, and carbon intensity is kg CO2/MWh.
