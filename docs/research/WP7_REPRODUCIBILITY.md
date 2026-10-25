# WP7 — reproducible working paper and scientific claim gate

**State:** manuscript draft, not peer-reviewed; no new hourly dispatch study or experimentally validated plant performance. The rendered `MANUSCRIPT.md`, 4 figures (PNG+SVG), tables, `claim_audit.json` and SHA-256 manifest are generated **only after** verification of the original WP2/WP3/WP4/WP6 source archive bytes.

## Registered sources

| Stage | Original GitHub Actions run | Artifact name | WP7 verification |
|---|---|---|---|
| WP2, 17 annual cases | 40148299968 | `wp2-2023-17-case-study` | Byte-locked `summary.json`, 17 hourly Stage A + Stage B CSV files and config hashes; recompute Stage A H2 shortfall and Stage B synthetic settlement |
| WP3 physical bounds | 40150436630 | `wp3-2023-constraint-attribution` | Byte-locked `analysis.json`, independent bound sum and matching WP2 summary SHA |
| WP4 five-year weather study | 40157038537 | `wp4-2021-2025-interannual-synthesis` | Byte-locked `analysis.json`, 35 registered cells, calendar/demand arithmetic and within-year E+S interaction |
| WP6 IAMC exchange | 40190690775 | `wp6-iamc-verified-exchange` | Byte-locked `manifest.json`, its file hashes, 250 IAMC cells, Stage A unserved H2 agrees with WP4 |

**Do not conflate verification types:** WP2 original saved hourly outputs are independently arithmetically rechecked in WP7; WP4 hourly dispatches were independently verified in their *prior* workflow, whereas WP7 consumes their fixed synthesis; neither process establishes field-level industrial model validation.

## How to generate

Install `pip install -e '.[dev]'` and obtain the four original artifact ZIPs from their exact run IDs. Unzip each to `wp7-source/wp2`, `wp3`, `wp4`, and `wp6` (top-level `summary.json` and `analysis.json` should be within the expected paths). From the repository root:

```bash
python -m research.paper_wp7 \
  --wp2 wp7-source/wp2 --wp3 wp7-source/wp3 \
  --wp4 wp7-source/wp4 --wp6 wp7-source/wp6 \
  --template docs/research/WP7_MANUSCRIPT_TEMPLATE.md \
  --output wp7-manuscript
```

The dedicated GitHub Actions workflow `.github/workflows/wp7.yml` retrieves those artifacts, runs integrity/format tests, creates all manuscript tables and plots, and archives the generated package. The script refuses to replace an existing directory; failed attempts leave an `.incomplete` directory for investigation. The output `manifest.json` hashes each reproduced figure, table and rendered paper. No cost, price, carbon or safety claims are added when evidence is missing.

## Read this before submission

The manuscript distinguishes fixed-technology weather years from historical industrial hydrogen output; original CDS 2023 from Open-Meteo ERA5 weather 2023; an optimistic physical bound from causal constraint allocation; Stage A adequacy from synthetic Stage B settlement; and numerical solver checks from field validation. The positive interaction is a finite *model contrast*, not an experimentally identified causal effect. The 17-case matrix is **not** a complete factorial. The five years are not independent random samples and do not justify inferential confidence intervals.

Required before publication or external release: confirm author name/affiliation and contributor credit; verify all primary literature citations, author funding/COI declarations and AI-assisted drafting disclosure; replicate inputs with a persistent DOI-backed raw-hourly archive and a locked environment; obtain independent engineering-domain critique; add longer weather and engineering sensitivities; and avoid marketing WP5's as-yet incomplete cost assumptions as finished economics. **A passing CI workflow does not make the manuscript peer-reviewed.**