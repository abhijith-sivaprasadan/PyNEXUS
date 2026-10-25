# WP5 source register — first evidence screen (25 October 2026)

**Status: preliminary literature evidence, not a populated financial model.**
These sources establish plausible cost evidence to interrogate. They are NOT
automatically inserted into the WP5 cost template. The technology, region,
boundary, currency, price year and cost definitions must be checked first.

| Cost block | Candidate reference | Evidence/interpretation | Model readiness |
| --- | --- | --- | --- |
| Offshore wind investment | IRENA, Renewable Power Generation Costs in 2024 (2025), https://www.irena.org/Digital-Report/Renewable-Power-Generation-Costs-in-2024 | 2024 globally weighted offshore-wind total installed cost: **USD 2,852/kW**. IRENA highlights large geographic spread and costs of complete projects rather than a turbine-only quote. | **Not directly usable** without currency/price-year normalisation, site adjustment and grid-interface boundary disaggregation |
| Electrolyser installation | IEA, Global Hydrogen Review 2025, executive summary, https://www.iea.org/reports/global-hydrogen-review-2025/executive-summary | 2024 installed costs outside China roughly **USD 2,000–2,600/kW**, reflecting a range of technologies/projects; IEA distinguishes manufacturing from engineering, procurement, construction and contingencies. | **Not directly PEM-specific** and cannot be substituted for a justified installed-cost scenario without source-scope review |
| Current hydrogen market context | IEA, Global Hydrogen Review 2026, https://www.iea.org/reports/global-hydrogen-review-2026 | 2026 synthesis highlights material regional CAPEX differences, investment delays and uncertainties. | Qualitative context; not an asset-level cost coefficient |
| Electricity market data | ENTSO-E Transparency Platform, official support on day-ahead A44 requests, https://transparencyplatform.zendesk.com/hc/en-us/articles/17703493179028-Guidance-on-Creating-a-Support-Ticket | Official guidance illustrates A44 day-ahead query and zone identifiers; API requests require source-provenance, zone and UTC/time interval audit. | **Missing**: actual Netherlands matching-year price archive and a new Stage B solve |
| Grid expansion | No validated project-specific source yet | Network reinforcement cost is geography-, topology- and permitting-specific. Doubling the modeled grid allowance is not a priced approved connection. | **Blocked** |
| Compressed hydrogen storage | No scope-matched source yet | kg nameplate cost must distinguish storage vessels, compression, BOP, operating pressure, safety, charge/discharge hardware and scaling behaviour. | **Blocked** |
| Efficiency upgrade (70% -> 75% nominal) | No validated incremental investment source yet | Screening is a parameter intervention, not a proven efficiency-retrofit technology; the marginal efficiency premium must not be invented or assumed zero. | **Blocked** |
| Discount rate, asset lives, OPEX, currency conversions | No harmonised source package yet | Different sources combine unlike asset lifetimes, WACC assumptions, vendor scopes and price years. | **Blocked** |

## Non-overlap rule for financial boundaries

The **IRENA offshore-wind total installed cost includes categories such as
turbine, foundation, installation and electrical interconnection**. The
additional grid-connection asset in PyNEXUS must reflect genuinely incremental
network upgrade expenditure rather than reprice export cables/assets already
included in the wind CAPEX. Report and audit any allocation.

Likewise, an installed electrolyser cost may include engineering, procurement,
construction and balance-of-plant. Do not add them a second time as generic
variable OPEX or plant investment. Costs will ultimately have to use a common
EUR price year and documented exchange/inflation assumptions.

## Economic validation gate

1. Lock a non-overlapping cost boundary, including which offshore assets are
   within each CAPEX block.
2. Obtain source-defined plant cost/lifetime/fixed OPEX intervals, with
   uncertainty bands and evidence-tier labels; investigate the 75% efficiency
   upgrade premium rather than interpreting it as free.
3. Acquire complete historical matching-year Dutch day-ahead prices with
   UTC timestamps, zero-duplicate timeline, documented market zone and
   DST/market time unit handling; verify source and missing data.
4. Re-run physical two-stage dispatch, especially Stage B, under those
   prices; retain original synthetic-price results separately.
5. Only then construct and discuss an *externally sourced, conditional*
   cost–adequacy frontier. The current WP5 code intentionally emits merely
   illustration-grade costs if explicit finance assumptions are supplied.

The presence of a cost figure in this register is not sufficient to license
an economic result. Unfilled financial parameters remain null by design.
