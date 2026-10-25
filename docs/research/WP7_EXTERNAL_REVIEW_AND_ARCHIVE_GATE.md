# WP7 — independent reproducibility, external review and archival gate

**Status (2026-10-25): NOT publication complete.** Passing GitHub Actions and creating a reviewer issue do not constitute review by an independent domain scientist.

## Preserved original study evidence

- Frozen WP2 2023 17-case hourly dataset and WP3 constraint attribution.
- Distinct five-year WP4 weather-screening synthesis, with separate original CDS versus Open-Meteo product definitions.
- Independently validated 2023 NL A44 raw source XML (14 archived original documents, source SHA-256).
- Independently reconstructed processed 2023 NL day-ahead prices, 8,760/8,760 exact numerical equality to original A03 XML.
- Full-year actual-price dispatch baseline/E/E+S, independently verified by separate WP5 audit.
- New complete 17-case WP4 engineering and WP5 real-price **workflows**, but their annual synthesis must pass before results are included.
- Manuscript version 0.1 and associated reproducibility figures/tables; these are **drafts**.

## Immutable release manifest: minimum content

The publication package must include a manifest with absolute file list, original source URLs and retrieval dates, per-file SHA-256, unmodified source XML and native CDS NetCDF, code commit SHA, environment/dependency lock, solver exact versions/options, main config/source interventions, hourly Stage A/Stage B dispatch and diagnostic checks, finance-source register and generated tables/figures. It must preserve original 2023 price and weather source differences and include license/reuse restrictions.

GitHub Actions artifact retention is temporary. Original A44 XML [run 40225702504](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40225702504) currently expires **2026-12-24**. Copy its original ZIP and manifest unchanged to durable research storage promptly; likewise preserve the full WP2/WP3/WP4/WP5 datasets.

An external DOI-backed deposit (e.g. Zenodo, institutional KTH archive or trusted research data repository) must be created by the author under applicable data redistribution licenses. A hypothetical repository identifier must **not** be invented or added here.

## Reviewer signoff

[External review tracking issue #22](https://github.com/abhijith-sivaprasadan/PyNEXUS/issues/22) contains separate mathematical and hydrogen-equipment checklists, reviewer independence/affiliation requirements, a clean-room reproduction criterion and final conflict-of-interest/signoff requirements.

At least one independent energy optimisation researcher and one independent hydrogen engineering specialist must review the frozen manuscript and complete outputs. An author-generated technical critique is an internal audit, not independent peer review.

**Publication gate conditions:** all 17 sensitivity/market scenarios pass, original CDS timestamp source is verified, all financial assumptions are sourced and normalized or all economic claims remain explicitly limited, the original mixed-unit emissions objective is never presented as emissions-optimal, reviewer responses are addressed, and the true fixed-source manuscript and all key artifacts receive a stable archive record.
