# WP5 completion track — priced scenarios, marginal thresholds and observed-price redispatch

**Implementation and evidence status (2026-10-25):** The original financial accounting framework remains unit-tested but financial coefficients are unsourced. `research.wp5_thresholds` generates a synthetic-price marginal accounting comparison. **Real processed NL A44 2023 prices have now been acquired and validated in a manual GitHub Actions run, and full-year observed-price Stage A/B redispatch has successfully completed for baseline, E and E+S.** The regular pull-request CI remains hermetic and does not query ENTSO-E. See [the independently audited 2023 storage comparison](WP5_OBSERVED_PRICE_2023_E_VS_ES.md) for run links, checksums and the limited €5.12/kg incremental wholesale-settlement interpretation. None of these amounts constitutes an LCOH, a contractual price or bankable economics.

## A. Marginal threshold before vendor cost evidence

Use archived GitHub Actions WP2 17-scenario artifact from run `40148299968`:

```bash
python -m research.wp5_thresholds --wp2-dir wp2-archive --output wp5-thresholds
```

The routine first runs `research.attribution.analyse_archive` (independent archived hourly/metadata checks). It calculates incremental hydrogen delivered from E to E+S, extra 20,000 kg nameplate, Stage A adequacy improvement, incremental Stage B synthetic grid settlement, and the **minimum incremental value per extra kg before capital, OPEX and other unpriced costs**. It intentionally does not set a sale price, discount rate, electrolyser CAPEX, storage vessel CAPEX, or production LCOH. Storage alone's physical benefit is separately reported. Interpret the ratio as a **synthetic-price accounting component only**, never a real marginal hydrogen cost.

## B. Obtain and verify processed historical Dutch prices

The project already has optional `entsoe-py` (`pip install -e '.[data]'`); to fetch a complete local-market-date bracket around the UTC 2023 horizon:

```bash
export ENTSOE_API_KEY='your_personal_key'
python -m research.wp5_market fetch --output wp5-nl-2023-market-prices
```

The fetch requests NL day-ahead A44 from `2022-12-31` through `2024-01-01` local market days, clips to **2023-01-01 00:00 UTC .. 2023-12-31 23:00 UTC** inclusive, and accepts exactly 8,760 ordered timestamps with no missing, duplicated, nonfinite or subhourly samples. Amsterdam DST transitions are addressed by conversion to UTC before validation; negative prices are accepted. Any 15/30-minute data require a separately registered and independently tested aggregation method. It records the transformed-price CSV SHA256 and provider-request metadata, but **does not preserve the ENTSO-E raw XML**. Consequently, it remains a processed-data provenance level. The repository does not contain an API token or market price archive.

The existing `data.entsoe` module requests whole local market days rather than an exact UTC year; the new strict adapter provides an independent UTC gate.

**Confirmed real-data status:** The first whole-year A44 response omitted one timestamp, `2023-12-29 23:00 UTC` (`2023-12-30 00:00 CET`). The manual acquisition workflow issued an independent official NL A44 requery for the surrounding local dates, retrieved the missing observation explicitly, verified all 72 overlapping prices and preserved supplementary observations with a checksum. All 8,760 processed hourly prices then passed strict UTC validation. Price CSV SHA-256: `2dfea4651fce314893201e7b08a4ae0ad24deef6b69ef85d81d9f54428f95bb2`. **The original ENTSO-E A44 XML is still not archived.**

## C. Run fresh dispatch against observed price input

```bash
# Short test horizon is explicitly not an annual result:
python -m research.wp5_market dispatch --snapshot wp5-nl-2023-market-prices \
  --scenario baseline --hours 24 --output wp5-market-smoke
# Genuine annual rerun; 17 registered cases may be costly:
python -m research.wp5_market dispatch --snapshot wp5-nl-2023-market-prices \
  --scenario all --hours 8760 --output wp5-market-2023-all
```

The pipeline reconstructs the original 2023 wind/config source with hash checks, swaps **only** the price input, and solves Stage A (physical shortfall minimum) and Stage B (net grid settlement at the new price series under adequacy constraint). It uses the existing independent dispatch verification and refuses any Stage B simultaneous grid import/export above the declared tolerance. An overlap rejection does not by itself fix the mixed-integer anti-circulation formulation; a future explicitly registered grid logic revision is needed for that stronger guarantee.

**Calendar assumption requiring independent audit:** the original archival CDS-derived 2023 8,760-row wind file is interpreted as chronological UTC beginning 2023-01-01 00:00. Preserve and check the original weather ingestion manifest before assigning market dispatch as definitive hourly alignment.

The internal WP2 writer names the settlement field SYNTHETIC; WP5B rewrites the output record to `stage_b_grid_settlement_eur_OBSERVED_PRICE_INPUT` and stores the input-price checksum. The underlying optimization objective is **net wholesale grid settlement only**; no tariffs, balancing, renewable hydrogen eligibility, measured output, actual plant contract, or total project economics is inferred.

**Completed annual cases:** baseline [run 40207056315](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40207056315), E [run 40217860564](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40217860564), and E+S [run 40216993086](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40216993086). Independently checked exact hourly rows and saved SHA-256, solver statuses, physical verifiers, no simultaneous import/export and hourly net-settlement arithmetic. **E+S improves annual delivery by 343.79 tonnes against E at +€1.759 million modeled net wholesale grid settlement (≈ €5.12/additional kg).** This is a conditional one-year wholesale-price accounting comparison *excluding storage capex and operating cost*. The other 14 registered cases have not yet been run using 2023 observed prices.

## D. External cost evidence register, current source screen

- **Offshore wind:** IRENA, *Renewable Power Generation Costs in 2025* (July 2026), https://www.irena.org/Publications/2026/Jul/Renewable-Power-Generation-Costs-in-2025 . Distinguish global plant installed-cost data from Dutch North Sea project-specific CAPEX and prevent interconnection double counting.
- **PEM electrolysis (100 MW scope):** European Hydrogen Observatory, *Electrolyser cost*, 2025 data, https://observatory.clean-hydrogen.europa.eu/hydrogen-landscape/production-trade-and-cost/electrolyser-cost . Technology-specific CAPEX in EUR/kW and OPEX in EUR/kW/y, with stack, BoP, utilities and other cost boundaries; the underlying spreadsheet must be reviewed before coefficients are entered.
- **Hydrogen production economics context:** European Hydrogen Observatory, *Cost of hydrogen production*, https://observatory.clean-hydrogen.europa.eu/hydrogen-landscape/production-trade-and-cost/cost-hydrogen-production . A cross-check, **not** direct validation of the PyNEXUS simulated plant.
- **Official day-ahead price source:** ENTSO-E Transparency Platform, https://transparency.entsoe.eu ; authentication and A44 data are required for the source-year archive.
- **Compressed hydrogen vessels, network reinforcement, efficiency-retrofit premium, stack replacement, finance, tariffs:** No justified scope-matched coefficients yet. Mark these **BLOCKED** rather than filling them with guesses.

**Cost-model gate:** all five assets (including the hypothetical efficiency gain) have required null costs in `research/protocol_wp5.template.json`. The existing `research.financial_accounting` intentionally refuses incomplete source-labelled assumptions and already independently verifies the archived WP2 records before calculating its conditional frontier. Do not present a complete 17-case financial frontier until the efficiency premium and all common-scope inputs are resolved. A reduced scenario set excluding eta can be a separately registered economic experiment only if its analysis code independently validates the subset; never silently drop cases within the original 17-case accounting routine.

## E. Remaining milestones

1. Audit EHO's actual 2025 100 MW PEM spreadsheet and source comparable per-asset capital, lifetime, maintenance and replacement intervals. Source and document exchange/price-year normalisation; do not reinterpret placeholders as prices.
2. Obtain Dutch-specific grid marginal reinforcement cost and 20–40 t compressed H2 storage engineering-cost evidence with non-overlapping system boundaries. Investigate real technology basis and marginal expense of moving 70% to 75% LHV electrolyser efficiency.
3. Acquire and permanently archive original ENTSO-E A44 XML with hashes, and externally audit archival wind-hour timing relative to actual UTC market hours.
4. Run all 17 annual scenarios with actual prices and verify solver outputs, including no simultaneous import/export; compare physical adequacy to original archival solution.
5. Produce price-year-harmonised conditional cost–adequacy frontier, at least three financial sensitivity cases, replacement/degradation schedule and corrected storage marginal delivered-hydrogen cost.
6. Review engineering validity and update the manuscript only after reproducibility gates. Do not call annualised accounting output bankable LCOH.

**Evidence boundary:** the software verifies completeness and arithmetic; source quality and scope matching require scientific review and project-specific data.
