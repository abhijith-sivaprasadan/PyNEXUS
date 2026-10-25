# When does hydrogen storage improve wind-to-hydrogen adequacy? An auditable intervention-matrix and weather-year screening study

**Research manuscript — computational screening; version 0.1, 25 October 2026. Not submitted or peer-reviewed.**

**Author:** Abhijith Sivaprasadan  
**Affiliation:** To be confirmed by the author prior to submission.  
**Keywords:** offshore wind; electrolytic hydrogen; hydrogen storage; adequacy; bottleneck screening; ERA5; reproducible optimisation; PyNEXUS.

## Abstract

Wind-coupled hydrogen production is constrained by the joint sizing of renewable electricity supply, electrolyser capacity, grid access and hydrogen storage. This paper asks whether hydrogen inventory is useful in isolation or only after complementary production capacity is introduced. We evaluate a **stylised** 150 MW offshore wind, 100 MW PEM electrolyser, 50 MW grid connection, 20 t hydrogen store and 50 t/day flat hydrogen-demand configuration using two-stage hourly optimisation. First, hydrogen shortfall is minimised; second, net electricity settlement is minimised subject to preserving the first-stage shortfall within numerical tolerance. Seventeen pre-registered single-, pair- and combined-parameter interventions are computed over original CDS-derived 2023 ERA5 wind, with independent post-solve diagnostics. Seven selected interventions are also assessed across five weather years (2021–2025) from a **distinct Open-Meteo-served ERA5 product**. The 2023 original-source baseline leaves **{baseline_shortfall_t} t/year ({baseline_shortfall_pct}%)** of configured hydrogen demand unserved. Doubling the grid connection to 100 MW removes modelled shortfall, while doubling storage capacity alone leaves it unchanged. Increasing electrolyser capacity to 125 MW makes expanded storage useful: the E+S combination provides **{es_extra_delivery_t} t/year** of additional hydrogen demand satisfaction relative to E alone, a **{es_interaction_pp}-percentage-point** positive finite contrast. Across five Open-Meteo ERA5 weather years, the baseline shortfall spans **{wp4_min_shortfall_pct}–{wp4_max_shortfall_pct}%**, and E+S interaction spans **{wp4_min_interaction_pp}–{wp4_max_interaction_pp} percentage points**. These are controlled model outcomes, not plant observations, causal treatment effects, economic optima, or evidence of renewable certification. A hash-pinned manuscript pipeline recomputes 17 archived hourly case outcomes, links the five-year synthesis and checks the IAMC-compatible exchange before rebuilding every figure and table.

## 1. Introduction

The operational reliability of electrolytic-hydrogen production depends on renewable variability, conversion nameplate, network import capacity, and the quantity and rate at which hydrogen can be stored and withdrawn. Increasing a single device's nameplate does not necessarily improve demand satisfaction if another element remains limiting. Moreover, storage can shift the time of delivery without increasing the total electricity-to-hydrogen conversion resource. Understanding **when additional storage volume has operational value** is therefore a distinct question from estimating a commercially optimum storage capacity.

This study examines that question with the open-source PyNEXUS dispatch model and controlled hypothetical parameter changes. It deliberately evaluates **adequacy**, measured by the hydrogen demand that can be served within model constraints; it does not equate hydrogen adequacy with avoided greenhouse-gas emissions, a low-carbon hydrogen label, or an economically feasible investment. The IEA has emphasised electricity, conversion technology and hydrogen-project economics as distinct considerations [1]. Recent offshore-wind–hydrogen modelling likewise evaluates systems with multiple interacting device constraints [2]; that literature provides **context**, not external validation of the PyNEXUS screening results.

**Research questions.** (RQ1) In the registered 2023 configuration, which isolated capacity changes reduce hydrogen shortfall? (RQ2) Does expanding hydrogen inventory have a conditional benefit after electrolyser expansion, and what does a separate optimistic production bound reveal? (RQ3) Does the ranking and sign of that finite storage interaction persist across 2021–2025 historical weather-year inputs? (RQ4) Can every claimed numerical result be regenerated from fixed and independently checked model artifacts, without treating the model's synthetic electricity-price inputs as historical prices?

**Contributions.** We present an exact registered 17-scenario primary matrix; a non-causal physical-bound analysis of constraint co-occurrence; a seven-scenario five-year weather-screening replication using the same weather-product family within that replication; an explicit separation of original CDS and separately served Open-Meteo 2023 data; and an executable research-output pipeline with verified source hashes and reproducible figures. No design or capacity value is purported to be a universal engineering optimum.

## 2. Model and methods

### 2.1. Stylised reference system

The reference system is situated at approximately 52.5° N, 3.5° E, an offshore North Sea location represented by a single modelled wind-weather point. Its wind farm comprises ten generic 15 MW turbines (150 MW in total), connected to a nominal 100 MW proton-exchange-membrane electrolyser, a 50 MW electricity grid interface, and compressed gaseous hydrogen storage with 20,000 kg rated inventory. Daily hydrogen demand is fixed at 50,000 kg, resulting in **{baseline_h2_demand_t} tonnes of annual configured demand** in non-leap years and 18,300 tonnes in 2024. The nominal LHV conversion-efficiency parameter is 70%. The model includes electrolyser loading and ramp restrictions, hydrogen pipeline flow capacity, storage inventory and charge/discharge limits, and electricity import/export. The hourly wind power availability is a derived model input, not a turbine measurement. Engineering parameters are stylised screening assumptions rather than an approved network connection or supplier quote.

The registered intervention definitions are: **W** (wind 150→225 MW), **E** (electrolyser 100→125 MW), **G** (grid connection 50→100 MW), **S** (hydrogen inventory 20→40 t), and **eta** (nominal efficiency 70→75%). All are hypothetical and use otherwise fixed operating definitions. Storage inventory initial and terminal requirements and charge/discharge limits do **not** increase automatically when storage nameplate doubles; an extra capacity intervention is not the same as a faster hydrogen transfer system. The 2023 experiment comprises the baseline, five singles, all ten two-way pairs, and one all-intervention case: **17 scenarios**, not a full 2^5 factorial. Higher-order interactions beyond the all-intervention case cannot be identified as a factorial decomposition.

### 2.2. Objectives, verification, and accounting boundaries

The core model uses Pyomo and HiGHS [3–5]. Its research wrapper first solves **Stage A**, minimising total unmet hydrogen across all 1-hour intervals, in kilograms. **Stage B** minimises the sum of hourly wholesale price times **net** grid imports (imports minus exports), while requiring total unserved hydrogen to remain within a tightly specified numerical allowance of the Stage A optimum. This lexicographic separation prevents an arbitrary financial penalty from redefining hydrogen shortfall as a monetary output. Solver-reported optimality, numerical tolerances, independent hourly energy/hydrogen checks, input hashes and configuration files are archived in WP2 and WP4.

The WP2 and WP4 manuscript figures use the **explicitly synthetic** repeated 2023 Stage B hourly price series. These specific historical experiments are **not** an electricity tariff, audited project cash flow, observed Dutch spot-price history, or levelised cost of hydrogen. Separately, follow-up WP5 work has now re-solved the original CDS-derived 2023 baseline, E and E+S cases against verified **processed** Dutch A44 day-ahead observations: see the [source-linked WP5 supplement](https://github.com/abhijith-sivaprasadan/PyNEXUS/blob/main/docs/research/WP5_OBSERVED_PRICE_2023_E_VS_ES.md). Its distinct accounting boundary and source limitations must not be silently pooled into these original WP2/WP4 figures. Stage A's adequacy results are not optimised against prices. The source study also provides synthetic carbon-intensity inputs; none is used in this manuscript to infer historical emissions or renewable-hydrogen eligibility. Although the independently archived primary solutions show no simultaneous grid import/export and no simultaneous storage charge/discharge under reported thresholds, the model formulation does not guarantee their exclusion in every conceivable case.

The primary response is the dimensionless **unserved fraction**: annual Stage A unmet hydrogen divided by annual configured demand. A positive intervention benefit is the baseline fraction minus the intervention fraction (reported in percentage points). The finite E+S contrast equals [benefit(E+S) − benefit(E) − benefit(S)]. Positive contrast means that the combined decrease in shortfall is larger than the sum of the two individual decreases in the same controlled screening model. This is **not** evidence of an exogenous causal effect in operating industry.

### 2.3. Optimistic production ceiling and constraint diagnostics

For the 2023 archive, WP3 separately computes an optimistic instantaneous upper bound on hydrogen production using available wind plus allowed grid power, electrolyser power and pipeline capacity, without treating all chronological ramp/inventory constraints as simultaneously limiting. Accounting for initial-to-final inventory credit, this yields a mathematical **lower bound** on unavoidable unserved hydrogen. Actual Stage A shortfall cannot improve beyond this optimistic lower bound without changing assumptions. The difference between calculated shortfall and the bound is *not* a uniquely attributable ramping/storage/operational loss; several constraints may overlap and the bound intentionally relaxes model limitations. The study also counts storage-capacity hits and shortfall hours, but those co-occurrence counts are not mixed-integer dual values or a causal allocation.

### 2.4. Weather experiments and source comparability

The original 2023 full-year input is based on a Copernicus Climate Data Store ERA5 retrieval. The five-year WP4 analysis instead requests **Open-Meteo's explicit ERA5 100 m wind-speed series** for 2021, 2022, 2023, 2024 and 2025 with 8,760 or 8,784 hourly UTC observations as appropriate [6,7]. It uses consistent API selection and the same turbine model across years; this produces an internally comparable weather sensitivity but does **not** establish byte equivalence between the two providers' processed 2023 series. ERA5 is documented by Hersbach and colleagues [6], and the Open-Meteo API documents the selected model, nearest-cell selection and disabled elevation downscaling [7]. The common environmental series permits comparison of seven pre-registered cases per year: baseline, W, E, G, S, eta and E+S. Weather years are temporally related historic samples, **not** independent random draws or a climate-change forecast.

The year 2024 is treated explicitly as a 366-day year with 8,784 hourly observations and greater annual hydrogen demand. Primary cross-year endpoints are **percent of annual configured demand** and within-year finite interactions, rather than unnormalised kilograms alone. For each year Stage A remains the adequacy objective; Stage B's artificial price sequence is unchanged in definition and excluded from all economic conclusions.

### 2.5. Evidence lineage and analytical reproducibility

The WP2 annual experiment, WP3 diagnostic synthesis, WP4 weather synthesis and WP6 IAMC-compatible exchange were generated by distinct prior GitHub Actions workflows, with archived inputs/results and independent checks. WP6 adopts **local PyNEXUS variable names**, not official IAMC identifiers; it was round-tripped with the open-source pyam package [8]. WP7 does not rerun all 35 annual Stage A/B optimisations. Instead, the manuscript generator enforces the archived SHA-256 hashes of the original WP2/3/4 analyses and WP6 exchange manifest, verifies the link between stages, recalculates Stage A shortfall and Stage B settlement from the **hourly 17-case WP2 CSVs**, checks 35 year-scenario identities and 250 corresponding IAMC exchange cells, and regenerates the paper figures and tables. This preserves the distinction between (a) a new independent recalculation of saved values, (b) earlier verification of full-year solver output, and (c) empirical physical validation, which is **not available**.

## 3. Results

### 3.1. Original CDS-derived 2023 intervention matrix

**Figure 1** and **Table 1** cover every registered 2023 scenario; no scenarios were omitted from the plotted primary matrix. The baseline has **{baseline_shortfall_t} t/year ({baseline_shortfall_pct}%)** unserved hydrogen. Storage capacity expansion on its own changes this to **{storage_single_pct}%** (no measurable gain at the registered precision). Expanding wind alone reduces it to **{wind_single_pct}%**, electrolyser alone to **{electrolyser_single_pct}%**, and nominal conversion efficiency alone to **{efficiency_single_pct}%**. Doubling the grid connection reduces **modelled** unmet hydrogen to **{grid_single_pct}%**. This latter result is contingent on permission to obtain the additional grid capacity and sufficient electrical supply; it is **not** a renewable-energy or low-emissions finding.

![Fig. 1 — Original CDS-2023 interventions](figures/fig1_wp2_2023_interventions.png)

*Figure 1.* Stage A unmet demand across all 17 registered parameter combinations (original CDS-derived 2023 wind). Percentages are of the same 18.25 million kg annual configured demand. Grid-related zero-shortfall cases are outcomes of the stylised allowed-import capacity, not costed or permitted projects. **Machine-readable values:** `tables/table1_2023_17_scenarios.csv`.

{table_2023_markdown}

### 3.2. Conditional value of hydrogen inventory

The combination of increased electrolyser power and increased storage (E+S) reduces annual shortfall to **{es_shortfall_t} t/year ({es_shortfall_pct}%)**. Relative to E alone, the larger store enables **{es_extra_delivery_t} t/year** of additional delivered hydrogen demand and a finite E+S contrast of **{es_interaction_pp} percentage points**. Since S alone delivers zero incremental adequacy in 2023, its conditional value after E expansion numerically equals the computed pairwise contrast. This is evidence of modelled **complementarity between rated conversion power and buffer volume**, not an identified commercial return on storage investment.

Independent WP3 diagnostics clarify the mechanism. In the baseline, the optimistic bound alone accounts for **{optimistic_bound_t} tonnes** of unavoidable unserved demand, or **{baseline_bound_share_pct}%** of the actual shortfall; only **{baseline_gap_above_bound_t} tonnes** lie above that particular optimistic bound. Original storage inventory peaks at about **{baseline_max_storage_t} tonnes** (below 20 t nameplate), so doubling storage volume cannot overcome the underlying annual energy/production ceiling in that configuration. After raising electrolyser nameplate, the smaller store reaches its limit and expanded storage offers additional flexibility. **Figure 4** separates the lower bound and remainder mathematically; it does not assign the remainder to one physical constraint.

![Fig. 4 — Optimistic production bound](figures/fig4_optimistic_bound_diagnostics.png)

*Figure 4.* An optimistic model-derived annual shortfall lower bound and the amount of actual Stage A shortfall above it. These parts are **not** a causal decomposition of energy, storage, ramping, or market effects. Original 2023 weather, three registered scenarios only. Values: `tables/table3_constraint_bounds.csv`.

{table_bounds_markdown}

### 3.3. 2021–2025 weather sensitivity

In the source-consistent Open-Meteo ERA5 weather-year screening, the baseline unmet-demand share spans **{wp4_min_shortfall_pct}–{wp4_max_shortfall_pct}%** (Figure 2). The 100 MW grid-intervention case eliminates model shortfall in all five years, while S alone does not change it in any year. E+S's positive finite interaction persists each year, ranging **{wp4_min_interaction_pp}–{wp4_max_interaction_pp} percentage points**; the incremental demand satisfaction relative to E ranges approximately **{wp4_min_extra_t}–{wp4_max_extra_t} tonnes per weather year**, with leap-year annual demand treated explicitly. A stable *sign in these five connected historical weather years* is not proof of stability under extreme futures, other sites or a probabilistic resource distribution.

![Fig. 2 — Weather-year adequacy](figures/fig2_wp4_weather_adequacy.png)

*Figure 2.* Within-source weather-year sensitivity of registered configurations. Each point is a full-year hourly model result using the separately served Open-Meteo ERA5 product. All intervention definitions remain fixed across weather years. Data: `tables/table2_weather_robustness.csv`.

{table_weather_markdown}

![Fig. 3 — E+S interaction](figures/fig3_storage_interaction_provenance.png)

*Figure 3.* E+S interaction on the five-year **Open-Meteo** series. The **original CDS-based 2023** result is marked separately, not silently averaged with the other source family. The finite contrast is computed from Stage A adequacy, not synthetic electricity price or capital costs.

## 4. Discussion

### 4.1. A conditional storage bottleneck, not a universal rank

The baseline shortfall is overwhelmingly close to an independently computed optimistic annual power-to-hydrogen production ceiling. An enlarged inventory vessel alone cannot increase total energy availability or the maximum amount converted over the year. Once electrolyser capacity increases, however, additional output during high-availability periods can encounter the original storage ceiling, and inventory expansion then improves chronology-sensitive delivery. The E+S contrast therefore provides a **conditional design-screening insight**: first test whether additional upstream productive capacity creates hydrogen that can use an enlarged buffer. This does **not** imply storage is universally ineffective; other demand shapes, cycling limits, solar/wind mixes, outage profiles, end-of-year constraints and delivery schedules may give storage independent value.

An expanded grid allowance achieved the lowest modelled Stage A shortfall. Yet its apparent adequacy dominance should not be interpreted as an investment recommendation: imports require physical connection rights, may face binding network/market restrictions and tariffs, and may not meet renewable or emissions-related hydrogen eligibility. WP5's annualised-cost accounting framework has intentionally unfilled capital-cost and price-year inputs; without appropriately sourced, non-overlapping CAPEX/OPEX, realistic energy prices and new price-optimised dispatch, the cost/adequacy Pareto frontier is **not established**. Consequently this paper draws no LCOH, net present value, payback or abatement ranking.

### 4.2. What the weather sensitivity supports

Replication within one consistently extracted wind product is useful because a single calendar year can overstate precision of a modelled adequacy ranking. The sign of the conditional E+S contrast is positive in each of the five analysed weather years, while the absolute magnitude shifts. These years are only five contiguous historical periods; without a longer climatology, alternative meteorological products, multiple offshore sites and systematic parameter uncertainty, the numerical range should **not** be misrepresented as a predictive confidence interval. Original 2023 CDS and Open-Meteo 2023 have separate retrieval and processing histories. Their agreement or disagreement is a **provenance check**, not evidence of perfect identity.

### 4.3. Numerical scope versus engineering validation

The solver solution satisfies the modelled conditions under recorded tolerances, not field-measured equipment performance. Turbine wakes are simplified; electrolyser efficiency is parameterised; degradation is not dynamically validated; hydrogen gas storage pressure/thermodynamics and pipeline behaviour are simplified; electric power-flow constraints and actual transmission rights are not represented at engineering fidelity. The anti-circulating flow issue remains a potential modelling concern despite no simultaneous import/export detected in the archived studied solutions. Active constraint hours from Stage A should not be interpreted as economic scarcity rents, because Stage A does not seek least imported electricity and may use nonunique feasible dispatch. Neither the optimistic bound nor coincident operational counters uniquely identifies causality or dual prices.

### 4.4. Follow-up observed-price and exclusive-grid evidence status

The reproducible primary manuscript remains grounded in the frozen synthetic-price WP2 and WP4 scenario archives. **New, separately audited 2023 A44 follow-up evidence** verifies historical wholesale-price-based baseline, E and E+S dispatch, and the E-to-E+S marginal storage grid-settlement component. The numbers and scope are documented in [WP5 observed-price research findings](https://github.com/abhijith-sivaprasadan/PyNEXUS/blob/main/docs/research/WP5_OBSERVED_PRICE_2023_E_VS_ES.md); they are **not imported as extra original-WP2 observations or a completed capital-inclusive frontier**. The optional exclusive buy/sell MILP constraints and grid-emissions reporting boundary introduced later also do not retroactively apply to archived original runs. None of these software additions imply real-plant validation, a fully costed commercial technology choice, or an independently peer-reviewed manuscript.

### 4.5. Independent NL 2023 source equivalence and interim economics

After the original WP2/WP4 experiments were archived, the NL 2023 day-ahead
series was retrieved from ENTSO-E's Transparency Platform, with all 8,760
hourly processed prices independently reproduced **exactly** from 14 original
A44 XML provider responses using the documented A03 variable-sized-block
rules. The 2023 electricity price mean was €95.82/MWh and 315 price hours
were negative. The original year-spanning client returned 8,759 hourly
observations and omitted **2023-12-29 23:00 UTC**; a separate XML A03
compressed period at **2023-12-30 08:00 UTC** must not be misidentified
as the same timestamp. Full source and parser limitations are documented
in the [WP1 original-price evidence report](WP1_2023_A03_PRIMARY_SOURCE_EQUIVALENCE.md).

Against these observed electricity prices, a separately modelled 125 MW
electrolyser configuration with 20 t hydrogen storage (E) fulfils 89.51% of
configured hydrogen demand, while doubling storage to 40 t (E+S) fulfils
91.39%. The difference is approximately **343.79 additional tonnes
delivered**, at **€1.759 million additional annual net wholesale grid
settlement**, or **€5.12 per additional kilogram** for the wholesale
settlement component alone. Most of the grid-settlement increase relates to
reduced opportunities to export electricity. This observation is
**not LCOH, total annual system cost, a proven storage profit/loss or a
vendor-calibrated design**. Capital, O&M, network tariffs and hydrogen
offtake price remain unpriced in the observed-price runs. Source:
[WP5 observed-price E/E+S evidence](WP5_OBSERVED_PRICE_2023_E_VS_ES.md).

Full 17-case observed-price and single-factor engineering robustness
workflows have subsequently been added, but their **outputs are not
part of this manuscript until all registered annual cases have passed
independent verification and been archived**. The investment cost evidence
register and conditional observed-price accounting module likewise do
not establish a complete 17-case economic frontier without validated
2023-EUR coefficients and equipment scope. Literature positioning
is described in [WP7 literature comparison](WP7_LITERATURE_POSITIONING.md).
Original 2023 CDS NetCDF provenance, externally durable data deposits
and independent engineering review remain acceptance gates.

### 4.6. Limitations and research priorities

Key limitations are: (1) stylised fixed capacity interventions without vendor engineering endorsement; (2) 17 cases rather than a complete five-factor factorial, with no systematic higher-order attribution; (3) one point weather source and five consecutive years, without confidence intervals; (4) synthetic Stage B prices for the *primary* WP2/WP4 runs; separately verified observed Netherlands 2023 A44 prices (exactly matched against 14 source XML files) for only baseline, E and E+S, without electricity network tariffs or real procurement contracts; (5) no investment, replacement, degradation or finance validation; (6) missing grid-access, hydrogen certification and carbon-emissions counterfactual; (7) no independently observed plant data; (8) no independent scientific peer review; and (9) dependence on ephemeral hosted artifacts, mitigated by local byte hashes but requiring long-term archival before Actions retention expiry.

Before a journal submission, we recommend: (a) a full 32-case five-factor matrix and finer engineering-relevant sweeps, (b) sensitivity to electrolyser ramp/minimum-load and hydrogen flow-rate limits, (c) location-replicated multi-decade meteorological input with a second independent data processing pipeline, (d) explicitly constrained anti-circulation grid exchange and sensitivity to network permissions, (e) separately sourced engineering/economic model with cost/market reoptimisation, (f) physical validation if plant data become accessible, and (g) a stable citable research snapshot/DOI with complete hourly archive and machine-readable environment lock file. These tasks remain **future work**, not completed analyses.

## 5. Conclusion

Within a fixed stylised North Sea wind-to-hydrogen configuration, the archived 2023 Stage A optimisation leaves **{baseline_shortfall_pct}%** of demand unmet. Storage expansion from 20 to 40 t alone does not reduce it; increased electrolyser nameplate enables a positive E+S storage contrast of **{es_interaction_pp} percentage points** and **{es_extra_delivery_t} tonnes** of added 2023 demand delivery relative to electrolyser expansion alone. That sign persists across the five weather years tested with a distinct, internally consistent ERA5 source family. Doubling modelled grid connection eliminates shortfall, **without** establishing permission to import, renewable status, low-carbon status or economic optimality. The methodological contribution is a registered, hash-audited, figure-reproducible path from hourly optimisation evidence to bounded scholarly claims. Broad real-world design prescriptions require additional empirical and market validation.

## Data, code and reproducibility statement

Code: [PyNEXUS repository](https://github.com/abhijith-sivaprasadan/PyNEXUS). A WP7 generator recomputes 17 original hourly Stage A/B case outputs, verifies hashes for WP2 (run 40148299968), WP3 (run 40150436630), WP4 (run 40157038537), and the WP6 pyam exchange (run 40190690775), and emits this rendered manuscript, figures, tables and the claim/asset SHA manifest. The exact input-file hashes are included in the emitted `manifest.json`; original raw hourly WP4 source inputs and dispatches remain in the separate WP4 run archives. GitHub Actions artifacts are time-limited; no permanent DOI has yet been issued. Figures and tables should not be reproduced by manually copying numbers from this draft. **CI success verifies a reproducibility workflow and arithmetic, not independent industrial model validation.**

## Declaration of AI-assisted preparation and author verification

This manuscript draft and parts of its analysis, code, tests, documentation and figure-generation pipeline were prepared with assistance from OpenAI language-model tools. The human author remains responsible for verifying original source materials, scientific claims, implementation correctness, figures, authorship, citation suitability, conflict-of-interest declarations and submission compliance. Automated source-hash and numerical checks are reported where performed; **no claim is made of a completed independent human peer review**. This disclosure should be refined to match the requirements of any target journal. Funding, conflicts of interest, institutional affiliation, and individual author contributions require author confirmation before external submission.

## References

[1] International Energy Agency (IEA). *The Future of Hydrogen: Seizing Today's Opportunities* (2019). https://www.iea.org/reports/the-future-of-hydrogen

[2] *Capacity configuration of offshore wind–hydrogen system with coupled AWE/PEM for improving power resilience under extreme weather*. *Clean Energy* 10(2), 96–109 (2026). https://doi.org/10.1093/ce/zkaf073 (Contextual comparison only; this paper does not externally validate PyNEXUS.)

[3] M. L. Bynum et al. *Pyomo — Optimization Modeling in Python*. Third edition, Springer (2021). https://doi.org/10.1007/978-3-030-68928-5

[4] W. E. Hart, J.-P. Watson and D. L. Woodruff. “Pyomo: modeling and solving mathematical programs in Python.” *Mathematical Programming Computation* 3, 219–260 (2011). https://www.pyomo.org/citing-pyomo

[5] Q. Huangfu and J. A. J. Hall. “Parallelizing the dual revised simplex method.” *Mathematical Programming Computation* 10, 119–142 (2018). https://doi.org/10.1007/s12532-017-0130-5 (General HiGHS solver citation; not a proof of the mixed-integer model's optimality beyond reported solver tolerances.)

[6] H. Hersbach et al. “The ERA5 global reanalysis.” *Quarterly Journal of the Royal Meteorological Society* 146, 1999–2049 (2020). https://doi.org/10.1002/qj.3803

[7] Open-Meteo. *Historical Weather API Documentation*, ERA5, cell selection and elevation options. https://open-meteo.com/en/docs/historical-weather-api (Accessed 25 October 2026; original WP4 source bytes retained separately.)

[8] D. Huppmann et al. “pyam: Analysis and visualisation of integrated assessment and macro-energy scenarios.” *Open Research Europe* (2021). https://doi.org/10.12688/openreseurope.13633.2

---

**Review gate:** This document is a source-verified **working paper draft**, not a submission-ready independent empirical or financial study. All numeric result placeholders above must be filled by the WP7 artifact pipeline; a non-rendered source template is not the authoritative values table.
