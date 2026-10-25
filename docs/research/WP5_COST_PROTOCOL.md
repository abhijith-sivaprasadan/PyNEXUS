# WP5 — cost/adequacy accounting protocol (implementation gate)

**State:** code and toy-unit verification; no financial result is claimed.
This WP5 branch follows the verified but unmerged WP4 research branch.

## Mathematical boundary

For each of the 17 complete, independently verified WP2 scenarios, the
post-processing module computes:

- Component capital cost = installed asset quantity x assumed EUR/unit.
  Wind, electrolyser and grid capacities are MW; compressed H2 storage is kg.
- Annualised capital cost = capital x CRF(rate, lifetime), where
  CRF = r / (1 - (1+r)^(-n)); at r=0 it is 1/n.
- Annual fixed OPEX = capital x sourced annual OPEX fraction.
- Annual variable OPEX = H2 delivered in verified Stage B x sourced EUR/kg.
- Net electricity settlement = verified Stage B synthetic-price net exchange.
- Illustrative annual cost = annualised capital + fixed OPEX + variable OPEX
  + net electricity settlement.
- Illustrative EUR/kg = above numerator / verified **delivered** Stage B H2.
  Do not divide by nominal demand when there is unserved demand.
- The Pareto set minimises annual illustrative cost and physical Stage A
  unmet-demand fraction. This is an illustrative multi-objective trade-off,
  **not an engineering or commercial optimum**.

Adding an efficiency improvement must explicitly include an extra efficiency
premium per MW of electrolyser nameplate, as a separate annualised asset
assumption; it cannot be treated as a costless retrofit. Costs use one common
nominal EUR price year. The code does not automatically inflation-adjust or
combine mixed-year supplier estimates.

## Non-negotiable safeguards

1. Inputs require sourced costs for every base asset, upgrade, lifetime,
   fixed OPEX rate, discount rate and variable OPEX.
2. The current program takes the original WP2 2023 synthetic-price study.
   It explicitly refuses observed-price provenance labels and requires the
   CLI user to acknowledge the synthetic electricity series.
3. It calls WP3's independent WP2 archive re-verification before pricing:
   the full 17-case hourly dispatch, config registration, hashes, balances and
   solver objectives must pass again.
4. No default CAPEX, grid-upgrade cost, storage cost, efficiency-premium,
   lifetime or discount rate is generated.
5. Missing source assumptions, incomplete arrays, partial 17-case archives,
   inaccurate shortfall caps, nonfinite values and non-positive delivered
   hydrogen prevent a result, rather than being silently imputed.
6. Negative net electricity settlement is permissible as a narrow market
   dispatch term but **never means project profit**.

Important exclusions include real transmission access charges, water supply
and treatment, hydrogen compression and delivery beyond modeled equipment,
asset replacements, tax, insurance and decommissioning. Even a completed
illustrative analysis is not a bankable levelised hydrogen production cost.
The software verifies arithmetic and file provenance; it does not verify that
a declared literature or vendor cost source is credible or comparable.

## How to execute, once explicit assumptions have been collected

Copy the WP5 template and replace every null value with a traceable estimate
in a single common EUR price year. Document what each source measures,
including location, technology maturity and whether balance-of-plant is
included, and preserve the actual input file in the research archive.

Retrieve the **full original verified 17-case WP2 archive** via GitHub Actions
run 40148299968 into a new directory, preserving each case's hourly CSV and
metadata. From the repository root run:

    python -m research.financial_accounting \
      --wp2-dir path/to/wp2-2023-17-case-study \
      --assumptions path/to/assumptions.json \
      --output path/to/wp5-illustrative-output \
      --acknowledge-synthetic-prices

Outputs: structured analysis.json and frontier.csv, with the hashes of
the verified WP2 summary and cost assumptions.

## Research actions still needed before publishing economic findings

- Source price-year-normalised CAPEX/OPEX for offshore wind, electrolyser,
  storage, grid reinforcement, and the incremental efficiency improvement.
- Establish asset lifetime, residual-value/degradation and replacement logic.
- Obtain actual Netherlands hourly power prices covering the chosen year,
  with named bidding zone, market zone and timezone conversion evidence.
- Re-optimise Stage B against these price observations (not revalue
  synthetic-price-optimised dispatch at historical prices).
- Validate non-circulating grid exchange, transmission tariffs, storage costs,
  wind operating costs and all equipment boundary assumptions.
- Only then compare market-derived, explicitly scoped economics; separately
  examine grid-power carbon intensity and renewable electricity eligibility.

A successful finance calculation **only** establishes arithmetic consistency
under supplied assumptions, not scientific validity of those assumptions.
