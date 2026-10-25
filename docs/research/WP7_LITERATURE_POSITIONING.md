# WP7 evidence-based literature comparison and defensible contribution statement

**Literature screening date:** 2026-10-25. This is a **focused research-positioning review**, not a systematic search with exhaustive database coverage, PRISMA screening, bias appraisal or a proven absence of prior similar work. Primary article pages and DOI/abstracts should be inspected again before publication.

## Relevant existing work (avoid misleading novelty claims)

| Study | Verified research scope from public article/abstract | Relevance / distinction |
|---|---|---|
| Guichard et al. (2024), IET Renewable Power Generation, DOI [10.1049/rpg2.13134](https://doi.org/10.1049/rpg2.13134) | PyPSA offshore wind + reversible SOFC technology, sensitivity over conversion, hydrogen inventory, subsea connection and power/hydrogen prices | Offshore electrolysis/storage/cable sizing and price sensitivity are **not new**. PyNEXUS examines instead fixed intervention contrasts under a deliberately adequacy-first objective and a PEM/limited-grid buffer, not general profit-maximising design |
| [Eniola et al. (2025)](https://doi.org/10.1016/j.ijhydene.2025.01.444), International Journal of Hydrogen Energy | Wind-speed variability, component sizing and energy storage in a wind-to-hydrogen microgrid; seven-day case | Storage can be useful as resource intermittency increases. PyNEXUS offers fixed annual hourly adequacy and a registered **conditional** E+S interaction in a grid-coupled model, not universal optimal sizing |
| [MISOCP storage dimensioning (2025)](https://doi.org/10.1016/j.ijhydene.2024.11.449), International Journal of Hydrogen Energy | Grid-connected wind/PV/hydrogen/battery joint storage sizing and energy management | Broader operational physics and design objective; do not claim PyNEXUS is physically more detailed |
| [Spatially explicit offshore hydrogen supply chains (2025/2026)](https://doi.org/10.1007/s11081-025-10062-5), Optimization and Engineering | Hourly MILP supply chains with electrolysers, offshore platforms, compressors, pipes, storage, cables and LCOH | More complete full-chain techno-economics. PyNEXUS currently lacks explicit compression/transport investment scope and cannot claim comparable commercial cost fidelity |
| [Joint offshore wind + hydrogen platform transmission planning (2025)](https://doi.org/10.1016/j.apenergy.2025.126383), Applied Energy | Capacity, cable layout and hydrogen platform optimisation with efficiency-dependent electrolyser operation | Shows why PyNEXUS's single-site linear conversion and assumed grid-cost coefficients are simplifications |
| [Wind-to-hydrogen curtailment optimisation (2026)](https://www.mdpi.com/2673-4141/7/4/151), Journal of Marine Science and Engineering | Full 8,760-hour model with electrolyser, compressor and hydrogen store and cost parameters | A contemporaneous close analogue. A 17-case model or 8,760-hour comparison alone is insufficient as a novelty claim |

## Precise positioning to defend

PyNEXUS does **not** invent offshore-wind hydrogen modelling, time-series dispatch, storage sizing or techno-economic optimization. Its narrower deliverable is a transparent **constraint/adequacy screening protocol** in which a small pre-registered intervention matrix diagnoses bottleneck complementarity (20 vs 40 t H2 storage becoming useful only after additional electrolyser capacity), disentangles hydrogen service from wholesale grid settlement, and cross-checks observed 2023 NL day-ahead prices directly against source A44 XML—including curve type A03 source-defined constant blocks.

This should be presented as a **replicable case study and modelling-methods demonstration**, not a proven universally novel theoretical optimisation framework or an industrial site design.

## Compulsory comparisons before submitting

- Verify exact authors, study datasets and modelling formulations for the six papers above, and use precise citations with DOI, year and page numbers.
- Describe what PyNEXUS includes/excludes against each study: temporal horizon, installed capacities, objective priority, weather provenance, observed electricity prices, constraints, conversion efficiency, H2 inventory, tariff and full-system CAPEX coverage.
- State distinctly that cross-year WP4 ERA5/Open-Meteo wind inputs are not exact matches to original CDS-2023 inputs and are not plant measurements.
- Add the full engineering-sensitivity synthesis only after all 17 annual runs pass, not after the workflow is merely defined.
- Insert a capital-inclusive cost–adequacy figure only after every source coefficient is normalized and an observed-price 17-case archive passes independent audit.
- Audit the original legacy `minimize_emissions` mixed-units objective and do not claim actual carbon-optimal dispatch; an added postsolve import-emissions boundary does not fix the legacy solver objective.
- Independent domain reviewers should challenge the one-hour alignment, interpolation-free A03 source semantics, adequacy cap, electrolyser startup/min-load, storage compression and 30-bar realism, grid exclusivity and annualization.

**Status:** a defensible context and comparator register is prepared; full systematic novelty review and independent peer critique are NOT yet performed.
