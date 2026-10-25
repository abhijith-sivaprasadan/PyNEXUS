# WP1 preliminary adequacy-first findings — archived 2023 hourly case

**Status:** AUTHOR-REPORTED COMPUTATIONAL RESULT, independently checked by code; not external validation, not peer reviewed, not a completed intervention study.

**Source workflow:** [GitHub Actions PR run 40146853931](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40146853931), artifact `wp1-annual-adequacy-reference` (research manifest and two hourly dispatch CSVs). A prior CI run reproduced the same printed metrics but lacked artifact upload owing to a hidden output directory; the later run retained the evidence.

## Setup

- Research code on PR #3; `research/run_reference.py` consumes `outputs/annual_2023/grid_storage_hourly/inputs.csv` and its stored configuration.
- The existing 2023 wind inputs are **ERA5-derived** but the prices and carbon intensities are **synthetic**, not Dutch historical market data. The original archived reference record remains unmodified.
- Historical input provenance is fail-closed: the runner accepts only the original recorded SHA-256 or SHA-256 equality after an LF/CRLF line-ending-only normalization, and records which condition matched in the output manifest. No unconditional bypass or approximate-value matching.
- HiGHS MIP tolerance requested: `mip_rel_gap=1e-4`. Individual solution quality is reported as the solver's termination under that configured tolerance, not mathematical proof of absolute global optimality.
- Stage A minimizes total unmet hydrogen independently of prices; Stage B minimizes net grid settlement while permitting no more than a 0.1825 kg absolute increase in unmet hydrogen.

## CI-reported results (two consistent executions)

| Quantity | Recorded value |
| --- | ---: |
| Total demand [kg H2] | 18,250,000 |
| Optimistic unavoidable-shortfall lower bound [kg H2] | 2,895,630.079501 |
| Stage A minimum unmet hydrogen [kg H2] | **2,897,283.360921** |
| Stage B unmet hydrogen [kg H2] | **2,897,283.543421** |
| Stage B shortfall increase from Stage A [kg H2] | 0.182500 |
| Numerical allowance [kg H2] | 0.1825 |
| Stage B grid-settlement objective [EUR] | **10,229,890.199867** |

Shortfall fractions are approximately 15.8755% in both stages. The optimistic lower bound excludes ramp and minimum-load constraints and cannot certify an achievable system dispatch. The observed gap between the bound and Stage A is around 1,653 kg; it is not attributed to a particular constraint without further diagnostics.

The Stage A unmet hydrogen is consistent with the historic penalty-based result, but Stage A uses an explicit physical adequacy objective. The EUR 10.23 million Stage B objective is **not** a production-cost estimate or LCOH: it is synthetic-price wholesale net grid settlement, excluding wind generation investment and OPEX, grid tariffs and fees, electrolyser CAPEX/OPEX, storage, hydrogen revenue and financial assumptions.

## Verification and evidence

- The two objectives are independently reconstructed from exported dispatch quantities.
- Electrical balance, storage balance/terminal inventory, hourly hydrogen balances, ramps, nameplate and pipeline constraints, shortfall cap and reported solver termination are subject to the wrapper's verification gates.
- The CI artifact contains the archived-input hashes and provenance verification mode, actual solver environment versions, source Git revision, run-manifest output, checked KPI summary and complete stage-A / stage-B hourly dispatch tables.
- The old baseline remains separately archived; this new objective is not compared to EUR 2.94 billion as if those were two versions of the same cost metric.

## Open gates

1. Record a fully pinned environment; the CI package installation currently resolves versions at run time.
2. Investigate exchange-flow degeneracy and non-simultaneous grid import/export rules before treating individual directional flows as a market forecast.
3. Obtain real market prices and define capital/operating cost boundaries before techno-economic generalization.
4. Run the registered interventions and explain remaining constraint effects; these have **not** yet been executed.
5. Repeat for additional weather years and quantify uncertainty; 2023 alone is not climatologically representative.

**Next scientific step:** WP2's registered baseline + 5 individual interventions + 10 pairwise combinations + combined portfolio, with all assumptions and output evidence kept comparable.
