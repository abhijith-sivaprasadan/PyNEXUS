# WP3 — 2023 chronological constraint attribution (verified analysis release)

**Evidence status:** author-executed, computationally verified post-solve diagnostic study. The analysis is **not** independently calibrated, peer reviewed, a causal dual analysis, a market-price estimate, or a certification of renewable hydrogen.

**Frozen source:** [WP2 17-scenario annual experiment (run 40148299968)](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40148299968), artifact [`wp2-2023-17-case-study`](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40148299968/artifacts/12582550521). **Independent WP3 verification:** [run 40150224190](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40150224190), artifact [`wp3-2023-constraint-attribution`](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40150224190/artifacts/12581732617).

The archived WP3 output contains `analysis.json`, `diagnostics.csv`, `monthly.csv`, and a machine-generated `RESEARCH_FINDINGS.md`. Per-case stage A and B CSVs and configurations remain in WP2's larger artifact. All 17 WP2 scenario configurations, hash-linked dispatch CSVs, numerical balances, objectives and interaction formulas were rechecked; the frozen WP2 protocol and 2023 source inputs were not modified.

## Study question

**Why did doubling storage capacity fail to improve adequacy in the baseline, yet improve adequacy by about 344 tonnes when combined with 25% more electrolyser nameplate capacity?**

Only the 2023 stylised North Sea system is considered. The existing fixed hydrogen demand is 50,000 kg/day. Wind comes from ERA5-derived input; prices are synthetic.

## Independent physical bound

The optimistic bound constructs the *largest possible* annual hydrogen production from every hour's wind-plus-grid power, subject to electrolyser and simplified pipeline caps, but **ignores ramping, binary minimum-load restrictions, storage losses and chronological availability of stored hydrogen**:

```text
p_possible[t] = min(wind_available[t] + grid_connection_MW,
                    electrolyser_max_MW, pipeline_limit_kg_h / h2_coeff)
h2_possible_kg = sum(p_possible[t] * h2_coeff * timestep_h)
credit_kg = max(0, initial_storage_kg - required_terminal_storage_kg)
unavoidable_unserved_lower_bound_kg = max(0, annual_demand_kg - h2_possible_kg - credit_kg)
```

This is a **lower bound on required shortfall**, not a feasible operation schedule. The difference between observed shortfall and the bound cannot be allocated exclusively to any one additional constraint: chronological mismatch, ramping, minimum load, storage flows/losses and other restrictions can overlap.

### Principal results

| Configuration | Unserved H₂ (tonnes/year) | Optimistic unavoidable shortfall (tonnes/year) | Additional beyond bound (tonnes/year) | Hours with unmet demand | Peak storage (tonnes) | Hours at storage cap |
|---|---:|---:|---:|---:|---:|---:|
| Reference: 100 MW electrolyser, 50 MW grid, 20 t H₂ storage | 2,897.28 | 2,895.63 | 1.65 | 4,094 | 10.51 | 0 |
| Storage-only: 40 t | 2,897.28 | 2,895.63 | 1.65 | 4,094 | 10.51 | 0 |
| Electrolyser: 125 MW, 20 t H₂ storage | 1,914.72 | 896.21 | 1,018.51 | 2,545 | 20.00 | 27 |
| Electrolyser + storage: 125 MW, 40 t H₂ | 1,570.93 | 896.21 | 674.72 | 2,103 | 40.00 | 12 |
| Grid: 100 MW, other references unchanged | 0 | 0 | 0 | 0 | 10.00 | 0 |

In the reference case, **99.9429% of the observed unmet hydrogen** is already at least the annual upper-production-bound deficit. Adding storage volume cannot eliminate that *structural annual* shortage. In addition, its original 20 t inventory ceiling is not even approached: the highest stored mass is about **10.51 t**.

With the 125 MW electrolyser, the system can produce substantially more hydrogen at high-availability times; the original 20 t store reaches its ceiling for **27 hours**. Increasing that storage ceiling to 40 t creates additional shifting room; the larger store also reaches its capacity for **12 hours**. Stage A storage discharge increases from approximately **1,092.7 t/year** in E to **1,437.0 t/year** in E+S.

The conditional storage effect is **343.793 tonnes/year** additional delivered hydrogen. The additional benefit is **+1.8838 percentage points of configured annual demand** beyond the sum of the E and S single-factor effects. This is a controlled finite contrast in the *model*, not a real-world claim of universal technological synergy.

### Chronology

The reference had **4,094 hours with unmet demand**, with the longest uninterrupted interval **143 hours**. Its independently computed instantaneous power ceiling implies **4,415 hours** when even a hypothetical unrestricted dispatch *without stored hydrogen discharge* could not meet that hour's demand.

In the E case, **2,545** hours had unmet demand; in E+S, **2,103** hours. The instantaneous supply-deficit-hour count remains **4,415** in both E and E+S because changing storage volume does not change wind, grid capacity or hydrogen conversion. This supports *temporal shifting*, rather than improved instantaneous power availability, as the mechanism of E+S's added delivery.

Monthly results are archived in `monthly.csv`. Under the historical weather pattern, September is the largest-shortfall month in both reference and E/E+S cases (reference about 415.3 tonnes; E about 354.2 tonnes; E+S about 334.3 tonnes). This is a weather-year-specific result, not a seasonal distribution estimate for future years.

## Constraint coincidence is NOT a causal decomposition

The Stage A optimum only minimises unserved hydrogen. **It does not penalise importing grid power or rewarding exports.** Therefore grid import is **at the 50 MW connection cap for all 8,760 hours in the baseline Stage A dispatch**, including hours when imported power is not marginally necessary to meet hydrogen demand. Counting these active-cap hours as proof of an economically or physically binding grid constraint would be scientifically incorrect.

Reported "shortfall coincident with grid cap" metrics are deliberately overlapping descriptive statistics; they do **not** uniquely allocate hydrogen shortfall to grid, electrolysers, ramping or storage. The optimistic production bound and registered parameter interventions, not an endogenous Stage A import count, support the reference-case conclusion.

To estimate marginal constraint values rigorously, additional controlled relaxations/dual-safe LP sensitivity diagnostics would be needed; unqualified mixed-integer shadow prices are not valid.

## Grid and storage flow checks

For all 17 registered scenarios, the independently analysed exports showed **zero hours of simultaneous grid import/export** and **zero hours of simultaneous storage charge/discharge**, in both the Stage A and B dispatch outputs, at explicit nonzero diagnostic thresholds.

This is a property of these saved optimal solutions, **not proof that the current mathematical formulation forbids those patterns in other runs**. An explicit anti-circulation market rule remains recommended before making market or engineering predictions.

The G intervention (50→100 MW permitted imports) reached zero unmet hydrogen in the stylised case. **That does not establish renewable or low-carbon hydrogen**, grid permits, realistic prices, dispatchability or commercial feasibility.

## Interpretation limits and next research

1. The single weather year and synthetic electricity prices prevent interannual robustness or real-market conclusions.
2. Storage volume changes held initial and terminal inventories, charge/discharge rates, and demand fixed; the observed E+S effect applies only to this parameterization.
3. Grid expansion, electrolyser expansion and storage capacity changes are stylised numerical interventions, not forecastable capital projects.
4. Losses, pipeline feasibility and electrolyser conversion are simplified; no measured plant validation has occurred.
5. Objective A establishes adequacy, Objective B estimates only wholesale net-grid settlement with synthetic hourly prices (neither LCOH nor profit).
6. The initial registration tested baseline, singles, pairs and all interventions; it does not identify all higher-order factor interactions.

**Next proposed work:** WP4 (additional years/weather sensitivity and external data provenance), then a separately costed adequacy frontier. A manuscript should distinguish independent mathematical bounds, author-reported numerical solver results and unverified engineering assumptions.
