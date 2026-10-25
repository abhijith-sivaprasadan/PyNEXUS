# PyNEXUS — controlled technology adequacy-gaps study (protocol v1.0)

**Status:** registered research plan. No new 8,760-hour intervention results exist merely because this file or the WP1 code has been committed. The historical 2023 annual outputs remain unchanged.

## Research question and evidence boundaries

Which of wind supply, electrolyser power/efficiency, grid connection and hydrogen storage constrains delivery in a stylised North Sea wind-to-hydrogen system, and how do physically feasible interventions interact?

This is an **energy-technology adequacy** experiment, not a global IAM or a physical residual-gross-emissions study. Do not call improvements in hydrogen-demand satisfaction avoided CO2 without a sourced counterfactual and emissions boundary.

## Historical reference (unaltered)

- Source model: `configs/annual.yaml`; real ERA5-derived 2023 wind, 8,760 hourly timesteps; synthetic price/carbon arrays.
- Existing `outputs/annual_2023/` records: wind-only cumulative **infeasible**; grid cumulative **infeasible**; grid + storage hourly **optimal** under a legacy penalty-based objective.
- Last historical run: 2,897,283 kg unmet H2 of 18,250,000 kg, about 15.9% for this specific configuration. This is not a new WP1 result and does not establish a technology ranking.
- Historical `objective_value` (~EUR 2.94 billion) is dominated by an arbitrary EUR 1,000/kg demand-shortfall penalty, not a standalone grid bill, OPEX or annualized cost.
- The archived annual manifest declares a dirty working tree; WP1 requires a subsequent clean rerun with locked dependencies to move beyond that record.

## Two-stage research objective (WP1)

**Stage A:** minimize `sum(demand_slack[t] * timestep_hours)`, yielding a physically interpretable H2 shortfall in kg. Stage A does not use price and must not be called a monetary objective.

**Stage B:** minimize `sum(price[t] * (grid_import[t] - grid_export[t]) * timestep_hours)`, subject to `total_slack <= stage_A_total + numerical_allowance`. This is net wholesale grid **settlement** only: it excludes wind CAPEX/OPEX, hydrogen product revenue, storage investment, fees, market rules and grid charges. It is NOT levelized cost of hydrogen. Stage B explicitly requires a grid connection and hourly demand; heat coupling is deferred.

The `research.adequacy.solve_adequacy_first` wrapper calls these objectives on identical inputs, with a separate check of each solution's solver status, hourly mass and energy balances, objective reconstruction, and final shortfall cap.

For research runs, HiGHS `mip_rel_gap` must be at most 1e-4. A solver reporting `optimal` within its configured gap is still not a mathematical proof of a global optimum without qualification; record and interpret its termination criterion.

## Model audit priorities and hard stop rules

1. Preserve the legacy objective and historical outputs byte-for-byte; do not silently rewrite those records.
2. Audit current `price*p + price*g_imp - price*g_exp`: under a cash-payment interpretation the first term counts consumed onsite wind at the same price as externally purchased electricity and thereby double-counts part of grid consumption. A wind opportunity-cost interpretation is *possible*, but it is not a grid bill. The new Stage B avoids the term and clearly scopes its result.
3. A high penalty does not make missed H2 demand physically impossible. Distinguish infeasibility under hard demand from minimum shortfall under soft demand.
4. Do not add a physically questionable grid-export emissions credit.
5. Current grid formulation permits simultaneous buy/sell at one common price, potentially non-unique individual flows. Use net exchange for money, but flag per-direction flows for scientific reporting and consider an explicit anti-simultaneous exchange constraint before publishable power-flow claims.
6. Store the actual configuration, input hashes, solver versions, exit status, MIP tolerances, objective definition and independent verification for every new result.
7. A regenerated 2023 run (and later weather years) requires source/provenance of wind inputs. The committed provenance is evidence of a historical fetch, not a substitute for actual input availability during reruns.

## Registered primary interventions (to implement only after WP1 gate)

Reference: 150 MW offshore wind; 100 MW electrolyser; 50 MW grid connection; 20 t hydrogen storage; 50 t/day demand. Use exactly the same 2023 ERA5 wind series, demand profile and comparison objective for all primary interventions.

| Code | Single intervention | Registered screening value | Important controlled boundary |
| --- | --- | --- | --- |
| W | Wind nameplate | 150 -> 225 MW | Same turbine power curve and hourly per-unit wind |
| E | Electrolyser nameplate | 100 -> 125 MW | Update ramp and any rating-dependent coefficients |
| G | Grid connection | 50 -> 100 MW | No presumption of an approved grid connection |
| S | Hydrogen inventory | 20 -> 40 t | Keep charge/discharge limits separately visible |
| eta | Nominal LHV efficiency | 70 -> 75% | Recompute actual linearized kg/MWh conversion |

The numeric interventions are initially **illustrative screening assumptions** and must undergo a literature/engineering justification review before the final paper. A proposed increase in nominal efficiency is not a demonstrated technology innovation.

Experiment matrix: 1 baseline + 5 individual + 10 pairs + 1 all-interventions = **17 scenarios**. A complete 2^5 factorial (32 scenarios) is the preferred extension for higher-order interactions.

## Metrics and analysis plan

For each scenario, report `unserved_h2_kg / demanded_h2_kg`, electrolyser utilization, curtailment (MWh), net grid exchange (MWh), hydrogen storage utilization, hours with shortfall, solver gap and independently checked arithmetic identities.

Individual effect: `benefit(A) = shortfall_fraction(base) - shortfall_fraction(A)`. Pairwise finite contrast: `I(A,B) = benefit(A+B) - benefit(A) - benefit(B)`. A positive interaction is synergy **in this finite controlled screening definition**, not a real-world causal mechanism without further evidence.

Document constraint-limited hours, year/season distribution and a separate optimistic capacity bound. A bound that ignores ramping cannot prove feasible dispatch, but can disprove certain overoptimistic claims.

## Extension gates

- **WP1:** research objective, test coverage, independent balance checks, clean annual re-execution and a locked environment.
- **WP2:** recorded 17-scenario table with failures/infeasibilities retained and registered inputs unchanged.
- **WP3:** intervention effects and pairwise contrasts independently recalculated from archived results.
- **WP4:** weather-year, economic source, price-penalty and engineering parameter sensitivities, with no synthetic prices marketed as historical market prices.
- **WP5:** separately annualized cost–adequacy frontier, with capital-cost, lifetime, discount-rate and numerator/denominator semantics.
- **WP6:** IAMC-compatible output from verified metrics, local variable dictionary, round-trip through pyam; no false claim of an official IAMC variable registry.
- **WP7:** manuscript with code-to-figure reproducibility, limitations, and explicit AI-assisted implementation disclosure.

**Publication criterion:** Each headline numerical result must be reproducible from archived inputs, a pinned environment and an independent verification routine; no proposed experiment should be written up as completed.
