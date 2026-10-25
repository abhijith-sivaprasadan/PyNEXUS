# WP4 verified findings — 2021–2025 weather-year screening

**Status:** independently rechecked full-year scenario output; GitHub Actions run
[40157038537](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40157038537)
completed successfully on 25 October 2026. The archived synthesis is
[artifact 12585572146](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40157038537/artifacts/12585572146).
These are author-computed numerical results, not independent empirical validation.

## Reproduction and provenance

- Five complete calendar years, 2021–2025, with 35 verified annual two-stage
  scenario solves. 2024 has 8,784 hours; the other years each have 8,760.
- Site: nominal 52.5 N, 3.5 E, Open-Meteo explicitly selected ERA5 100 m wind, UTC.
  Original PyNEXUS hub-height correction, turbine curve and loss factors retained.
- Seven pre-registered scenarios per year: baseline, W (wind), E (electrolyser),
  G (grid), S (storage), eta (efficiency), E+S.
- Stage A minimum annual unserved H2 is the primary endpoint; Stage B synthetic
  price settlement is not observed market cost.
- Provider responses, hourly source inputs, configurations and both dispatch
  outputs are hash-archived and were independently checked by the synthesis.
- The Open-Meteo ERA5 extraction is distinct from the original CDS 2023 series.

## Verified outcomes

| Year | Wind CF | Baseline unserved H2 | S-alone benefit | E+S interaction | Additional E+S delivery over E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2021 | 34.87% | 18.287% | 0.000 pp | +1.784 pp | 325.59 t |
| 2022 | 35.46% | 18.465% | 0.000 pp | +1.469 pp | 268.08 t |
| 2023 | 39.73% | 15.878% | 0.000 pp | +1.882 pp | 343.53 t |
| 2024 | 38.71% | 16.043% | 0.000 pp | +1.374 pp | 251.49 t |
| 2025 | 36.17% | 17.120% | 0.000 pp | +1.276 pp | 232.85 t |

Rounded from archived year_comparison.csv and scenario_comparison.csv.
The E+S interaction is a finite scenario contrast, not proof of real-world
technology synergy. Since S alone has zero effect in all five years,
the interaction equals the conditional gain from S after E.

## Findings

1. The baseline unserved-demand fraction varied between **15.878% and 18.465%**.
2. **G ranked first in every year** and eliminated modeled shortfall in all five.
   This does not establish that the supplied hydrogen is renewable, that the
   grid upgrade would be permitted, or that imported electricity is emission-free.
3. **S had no benefit alone** across all five years, consistent with the earlier
   WP3 account of limited baseline storage utilisation.
4. **E+S beat E alone in every year**, providing an additional **232.85–343.53
   tonnes/year** of hydrogen delivery.
5. Among the other single interventions, eta usually ranked above E;
   **2024 reversed this order** slightly. This illustrates some rank sensitivity
   without constituting a significance test.

## 2023 provider comparison

Open-Meteo ERA5 2023 wind availability was **521,988.60 MWh** versus
**522,096.98 MWh** for original CDS-derived ERA5 2023. Hourly power
MAE was **0.224 MW**, RMSE **0.371 MW**. Small aggregate differences do not
establish identical hourly samples or source interchangeability.

## Boundaries and follow-on research

**Established for the frozen stylised model:** the sign of the E+S interaction
and the leading G adequacy ranking persist across these five weather years.

**Not established:** actual project profitability, future-climate probabilities,
independence of serial weather years, external engineering validation,
renewable-hydrogen certification or verified economic performance.

**WP5 gate:** Capital and fixed/variable OPEX require sourced assumptions and
explicit boundaries. Observed Netherlands market prices need timezone/DST,
tariff and provenance work and a NEW Stage B dispatch optimisation; the archived
synthetic-price settlement must never be relabelled as historical market data.
