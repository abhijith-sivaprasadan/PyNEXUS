# WP6 — IAMC-compatible exchange of independently verified adequacy results

**Implementation status:** WP6 exports the **existing WP4 2021–2025 seven-scenario, five-year weather experiment** into an IAMC-shaped annual table with a transparent, **local** variable dictionary. It uses `pyam.IamDataFrame` to save a standard wide-format IAMC CSV, reloads the CSV through pyam, and compares every coordinate and numerical value. This is a data-exchange and reproducibility contribution, **not** a new optimisation, an IAM Consortium scenario, an official variable registry, or a new published hydrogen production time series.

## Evidence and trust boundary

- Source: the WP4 study synthesis from **GitHub Actions run [40157038537](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40157038537)**, artifact `wp4-2021-2025-interannual-synthesis` (artifact ID `12585572146`).
- Expected source file SHA-256: `ee1b475d1034a151d1e2e91b85ec3047aed6f0b756d042159d2305cfa466e907`.
- Source code version: `127a000e247c494995b0744bdafa0cf290c918f4`.
- At the original WP4 run, **35 complete-year hourly Stage A + Stage B dispatches were independently rechecked** against the archived source data, configurations, hours and results. WP6 consumes the exact hash-pinned **resulting synthesis**; it checks arithmetic and registration but does **not** itself independently rerun or reverifiy those original 35 hourly dispatch files.
- Source 2023 weather in this series is **Open-Meteo's explicitly selected ERA5 product**, which differs from the historical original CDS-2023 series. The latter is **not** combined or relabelled. The five weather years are **not independent iid climate samples**.
- Stage A minimises unmet hydrogen demand. The repeated Stage B hourly synthetic price and carbon signals are outside this IAMC exchange. No observed Dutch spot prices, validated life-cycle emissions, hydrogen certification or cost result exists in this WP.
- Installed technology and model parameters are held at controlled screening values; successive years represent **weather-year sensitivity experiments**, **not a projected evolution of hydrogen supply in those years**.

### Exported data

The full, registered WP4 grid includes 5 calendar weather years × 7 configurations = **35 scenario-year experiments** (2024 contains 8,784 hourly observations, other years 8,760). WP6 exports seven variables per experiment, plus a separate E+S interaction metric for each year: **250 IAMC cells**.

| Local variable | Unit | Meaning |
| --- | --- | --- |
| `PyNEXUS|Hydrogen|Demand` | kg H2/yr | Configured annual flat H₂ demand; 18.25 million kg in ordinary years, 18.30 million kg in leap year |
| `PyNEXUS|Hydrogen|Unserved Demand|Stage A` | kg H2/yr | Annual Stage A H₂ shortfall |
| `PyNEXUS|Hydrogen|Unserved Demand|Stage A|Share` | 1 | Shortfall / configured demand |
| `PyNEXUS|Hydrogen|Delivered Demand|Stage A` | kg H2/yr | Demand less unserved Stage A demand; not plant output |
| `PyNEXUS|Wind|Reference Available Resource` | MWh/yr | Weather-specific **150 MW reference** available wind, even in the W capacity-expansion scenario |
| `PyNEXUS|Wind|Reference Capacity Factor` | 1 | Capacity factor of the same unchanged reference farm, not of W expansion |
| `PyNEXUS|Hydrogen|Avoided Shortfall vs Baseline` | percentage point | Per-year controlled adequacy difference relative to baseline |
| `PyNEXUS|Hydrogen|Electrolyser Storage Interaction` | percentage point | Per-year finite interaction of E+S, **only on scenario E+S** |

These names belong to the **PyNEXUS local dictionary** in `research/wp6_variables.json`. Do not present them as officially registered IAMC variables. The dimensions `model`, `scenario`, `region`, `variable`, `unit`, `year` and `value` are structurally compatible with IAMC / pyam. `model` names the specific PyNEXUS weather product; `scenario` is an engineering configuration and **not** a global mitigation scenario; `region` is a **stylised offshore point**, not an IAM administrative aggregation.

### Execute

From the repository root, on Python 3.10+:

```sh
python -m pip install -e '.[dev,iamc]'
# Download exact WP4 synthesis archive from the GitHub Actions run (gh authenticated):
gh run download 40157038537 -n wp4-2021-2025-interannual-synthesis -D wp4-verified-source
python -m research.iamc_export --wp4-synthesis wp4-verified-source/analysis.json --output wp6-iamc-output
```

**Outputs:** `iamc_wide.csv` (standard pyam wide annual IAMC table), `iamc_long.csv` (same cells in explicit seven-column long form), `local_variable_dictionary.json` (descriptions/units), `manifest.json` (source hash, repository configuration and dictionary hashes, export file hashes, pyam version and scope warnings). Existing output directories cannot be overwritten. The export writes a `.incomplete` folder if validation or pyam fails, so the attempted output remains inspectable.

**Independent export verification:** `pyam.IamDataFrame(iamc_wide.csv)` must reconstruct exactly the same `model/scenario/region/variable/unit/year` coordinates and values to rounding tolerance. Additional tests reject duplicate cells, missing annual cases, 2024 calendar drift, nonfinite values, changed H₂ demand/shortfall identities, baseline benefit mismatches, invalid E+S arithmetic, incorrect source family and an arbitrary document claiming to be verified.

**Reproducibility note:** the original WP4 run's GitHub Actions archive has a finite retention period. Preserve the artifact permanently outside Actions (with original SHA and chain-of-custody) before expiry. The exporter deliberately fails if the hash-pinned input is unavailable; it never silently substitutes generated data.

**Reference:** the `pyam` IAMC table interface and round-trip read/write API: https://pyam-iamc.readthedocs.io/en/stable/api/iamdataframe.html

## Scientific boundaries and next stage

1. This format does not establish IAMC scenario registry membership or scientific comparability with published IAM ensembles.
2. No WP5 finance data are exported: no externally qualified cost inputs, observed electricity-price data or claimed LCOH estimates are available.
3. Quantified Stage A adequacy changes here reflect one stylised North Sea system with fixed technology assumptions and historical wind patterns. No real plant metering or hydrogen contract validation exists.
4. The `PyNEXUS|...` terms are nonstandard local identifiers. Mapping to a controlled external variable standard would require a separate reviewed ontology/schema change, not cosmetic renaming.
5. WP7 can use this portable data exchange for publication figures **only with input hashes, figure generation scripts, versioned variable definitions, limitations, and AI-assistance disclosure**.

**Interpretation:** successful pyam ingestion establishes technical interoperability of scenario timeseries tables. It neither independently validates the physical model nor upgrades illustrative technical experiments into empirically measured outcomes.
