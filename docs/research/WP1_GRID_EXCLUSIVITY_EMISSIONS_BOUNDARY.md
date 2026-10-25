# WP1 research-only grid exclusivity and emissions-boundary addendum

**Status:** implementation and unit tests only; historical annual results unchanged.
The prior 17-case WP2 and 2023 NL observed-price baseline/E/E+S records are
still *legacy grid-mode* runs. The new constraint must be applied in a
separately registered experiment and compared with the original results
before describing any prior findings as anti-circulation-constrained.

## Physical grid exchange

`ElectrolyzerDispatchOptimizer.optimize(..., grid_exchange_mode="exclusive")`
adds an hourly binary `z_t` and the two exact MILP inequalities:

```text
0 <= grid_import[t] <= grid_capacity_MW * z[t]
0 <= grid_export[t] <= grid_capacity_MW * (1 - z[t])
z[t] in {0, 1}
```

This prohibits simultaneous positive import and export **as a mathematical
constraint**, even under negative electricity prices or objective
degeneracy. The default is `grid_exchange_mode="legacy"` to avoid altering
previously published artifacts or locked test oracle values. It is available
only for grid-enabled research Stage A `minimize_unserved` and Stage B
`minimize_cash_cost`. Call
`solve_adequacy_first(..., grid_exchange_mode="exclusive")`
to apply identical model physics in both stages. An additional postsolve
overlap check remains in place.

**Not established by this code alone:** equivalence between exclusive and
legacy annual results, market clearing or permissions, explicit settlement
tariffs and charges. Those require new annual solves.

## Operational grid-emissions audit

The original `minimize_emissions` objective combines `carbon_intensity * p`
with `carbon_intensity * g_imp` and the monetary shortfall penalty, which
mixes unlike units (kg CO2 with EUR-valued penalty terms). It remains a
**historical legacy objective and is not approved for emissions claims**.

The separate `research.emissions_accounting.grid_import_emissions` is a
postsolve numerical boundary: `sum(grid_import_MW * kg_CO2/MWh * hour)`.
It has NO credit for grid exports and does NOT charge all electrolyser
consumption again. It rejects a missing/misaligned factor vector, nonfinite
factors, negative flows and simultaneous grid import/export.

It intentionally makes **no** renewable-hydrogen certification, plant total
life-cycle CO2, embodied offshore-wind, avoided-emissions or real Netherlands
carbon claim. A named time-aligned source for 2023 grid-carbon intensity
is required before presenting a historical result. There is **no new
emissions-minimizing solver objective** here; such a study needs an explicitly
registered adequacy cap and units-consistent second objective plus testing.

## Reproducibility gate

This patch passes or fails on unit/regression CI. Future annual exclusivity
runs must preserve a mode-labelled config, input hashes, solver termination
and complete CSVs and independently compare shortfall and settlement across
legacy versus exclusive modes before any new economic claim.
