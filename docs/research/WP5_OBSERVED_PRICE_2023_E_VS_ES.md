# WP5 — Verified 2023 Netherlands observed-price dispatch: E versus E+S

**Result provenance (2026-10-25).** These are independently rechecked outputs from three successful GitHub Actions annual runs on **commit `7207e803a4d004ec62ea4a2f26ff81459fe4c092`**. Source data are processed ENTSO-E NL A44 day-ahead prices, **not** original source XML, invoices, or an electricity contract. No assumptions for CAPEX, OPEX, tariff, H₂ sale price, or discount rate were introduced.

## Main finding — incremental storage only

With offshore wind fixed at 150 MW, electrolyser fixed at **125 MW**, grid connection fixed at 50 MW, and 2023 demand fixed at 50 t/day, raising hydrogen storage from **20 t (E)** to **40 t (E+S)**:

- Reduces annual unmet demand by **343,792.77 kg** (343.79 tonnes), from **1,914.72 t** to **1,570.93 t**; annual demand served rises from **89.51%** to **91.39%** (**+1.88 percentage points**).
- Increases modeled **net wholesale grid settlement** by **€1,759,091.43** from €25,748,968.93 to €27,508,060.36.
- Implies **€5.12/additional kg delivered**, an **incremental wholesale-grid-settlement component only**; this is **not** the total incremental cost of hydrogen or an LCOH.
- Increases electrolyser energy use by **16.564 GWh**. Grid imports rise by **2.053 GWh** and grid exports fall by **14.516 GWh**.

## Audited scenario comparison

| Metric | Baseline | E | E+S |
|---|---:|---:|---:|
| Electrolyser nameplate (MW) | 100 | 125 | 125 |
| Storage nameplate (t) | 20 | 20 | 40 |
| Annual demand (t) | 18,250 | 18,250 | 18,250 |
| Delivered H₂, demand − Stage B shortfall (t) | 15,352.72 | 16,335.28 | 16,679.07 |
| Annual demand met (%) | 84.12 | 89.51 | 91.39 |
| Stage A minimum shortfall (t) | 2,897.28 | 1,914.72 | 1,570.93 |
| Stage B unmet demand (t) | 2,897.28 | 1,914.72 | 1,570.93 |
| H₂ produced (t) | 15,353.18 | 16,340.92 | 16,690.18 |
| Electrolyser electricity input (GWh) | 728.12 | 774.96 | 791.53 |
| Grid imports (GWh) | 281.61 | 300.59 | 302.64 |
| Grid exports (GWh) | 56.88 | 36.31 | 21.80 |
| Net grid settlement (€ million) | 23.8637 | 25.7490 | 27.5081 |

### Why the wholesale settlement increases

Stage B cost is recomputed from all 8,760 hourly rows as **Σ [grid imports (MW) − grid exports (MW)] × hourly day-ahead price (€/MWh) × 1 hour**, with no tariff, tax, network charge, storage capex or financing. Relative to E, E+S changes:

| Mechanism | Difference (EUR/year) | Share of additional net grid settlement |
|---|---:|---:|
| Extra priced grid-import expenditure | €189,760.01 | 10.79% |
| Forgone priced export revenue | €1,569,331.42 | 89.21% |
| **Total incremental grid settlement** | **€1,759,091.43** | **100.00%** |

The dominant component (**89.21%**) is reduced wholesale revenue from electricity exports. This is a modeling result: storage allows more hydrogen production from electricity that could otherwise have been sold, with only a modest increase in grid purchases. The economic decision depends on a defensible realized **incremental hydrogen value**, storage investment/operating costs, and contractual/tariff constraints not currently provided.

## Verification and provenance

- Successful baseline annual Actions [run 40207056315](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40207056315/artifacts/12603934314); E annual [run 40217860564](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40217860564/artifacts/12607969199); E+S annual [run 40216993086](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40216993086/artifacts/12607758194). These Actions artifacts have limited retention; preserve an independent immutable copy for publication.
- The baseline, E and E+S `manifest.json` records all indicate **8,760-hour full-year runs**, shared wind/source config hashes, and the same processed-price CSV checksum.
- Price CSV SHA-256: `2dfea4651fce314893201e7b08a4ae0ad24deef6b69ef85d81d9f54428f95bb2`.
- Wind input SHA-256: `1e1ef87daf7ae2d8f0545fb3fb163a71b5f49c08af8da7c5495124e0ed1cf3ed`. The original wind-hour temporal alignment is **assumed** (2023-01-01 00:00 UTC first row), not independently audited against CDS raw ingest metadata.
- Independently read each zipped `record.json` and all six `stage_a.csv` / `stage_b.csv` hourly files; verified **SHA-256 dispatch hashes**, exact 8,760 timesteps per stage, optimal solver terminations, all recorded physical checks, no simultaneous hourly import/export above 1e-6 MW, annual shortfall sums, import/export sums and exact independent net-price settlement arithmetic (tolerance €0.0001).
- Stage A and Stage B both solved optimally in the registered adequacy-first dispatch protocol. Stage B shortfall may differ from Stage A minimum by a tiny registered numerical feasibility allowance; the **difference between E and E+S** is consistent across stages.
- **Important limitation:** the original grid-exchange MILP does not explicitly forbid simultaneous import/export with binary anti-circulation constraints. No simultaneous flows were observed in these outputs, but this is an output check, not a formulation guarantee.

## Scope boundaries before claims of economic feasibility

1. Retain original ENTSO-E A44 XML and validate any compressed-curve construction; current prices are verified *processed* timeseries, source XML is **not** archived.
2. Audit CDS wind-hour alignment, time standards and physical model assumptions including simplified linear electrolyser efficiency and hydrogen storage/pipeline behavior.
3. Source nonoverlapping capital/operating costs for storage, electrolysis, offshore wind, grid reinforcement and the efficiency intervention; include financing, replacements/degradation, power purchase and sale conditions, tariffs and source year/FX normalization.
4. Perform explicitly scoped sensitivity studies and alternative market years. One observed 2023 wholesale-price year cannot establish bankability or a long-term decision.

**Do not label €5.12/kg as LCOH, a realized selling price, or an all-in cost.** It is only the increment in the optimized wholesale grid-settlement component per increment in delivered hydrogen for this conditional 2023 comparison.
