> **Primary-source price equality now established, 2026-10-25:** The original 14 XML A44 files have been downloaded and SHA-256 checked. Independent application of official A03 block rules reconstructs all 8,760 hourly 2023 prices **exactly**, with zero disagreement against processed observed-price artifact. See [original A03 equivalence audit](WP1_2023_A03_PRIMARY_SOURCE_EQUIVALENCE.md). The raw-byte archive still needs long-term preservation and the **original CDS wind alignment remains unverified**.

# WP1 primary ENTSO-E A44 XML preservation and wind calendar audit

**Implemented; original-source verification is pending a live manual source run.**

The existing WP5 processed Dutch 2023 hourly price series is independently
complete (8,760 hourly UTC values) and was validated against a second A44
query that explicitly returned a missing local-midnight hour. However
`entsoe-py` PandasClient has already transformed the official documents;
previous workflow runs did **not** preserve the original provider XML.

Run GitHub Actions **WP1 original NL 2023 A44 XML evidence archive** on
`main` using the already configured `ENTSOE_API_KEY` secret. This performs
14 intact local-month `EntsoeRawClient.query_day_ahead_prices` requests
(December 2022, January–December 2023, January 2024), stores the exact
official XML response bytes, and SHA-256 hashes every file in
`raw_manifest.json`. It checks official A44 publication documents, detects
curve types (including A03 compact-price curves), resolutions and provider
point counts. The key and upstream URLs are never written to the evidence.

**An A03 curve is NOT automatically expanded** into hourly values by this
script. Actual 2023 hourly reconstruction from raw XML must be a separately
tested standards-grounded step and compared to the processed 2023 price
CSV SHA `2dfea4651fce314893201e7b08a4ae0ad24deef6b69ef85d81d9f54428f95bb2`.
Only that step can close the end-to-end *primary-source-value* equivalence
gate. A downloaded XML archive alone closes the raw-byte preservation gate.

The workflow's GitHub Actions artifact expires (60-day retention). Download
it and retain the **unchanged XML files plus manifest hashes** in a durable
storage system or research repository. Do not claim Zenodo DOI, independent
replication, or publication-grade preservation until a deposited record
actually exists.

## Archived wind timestamp boundary: still open

The current original 2023 Stage A/B dispatch input CSV contains hourly
power without its originating source timestamps; the model configuration
uses `Europe/Amsterdam` for its site metadata while ERA5 native times are
UTC. The `data.era5.check_calendar_coverage` function correctly operates
in UTC, but the original 2023 power CSV cannot, by its row count alone,
certify that row 0 corresponds to 2023-01-01 00:00 UTC.

**Required evidence to close the gate:** original CDS NetCDF files from the
source request, source hashes/provenance manifests, the exact 100m-to-hub
wind and turbine conversion, and independent timestamp-to-row equality
comparison with the saved 2023 `inputs.csv`. No replacement with merely
similar 2023 Open-Meteo values is acceptable; these are separate products.

There is no original CDS NetCDF in `data/raw` on the committed main branch,
so this cannot truthfully be marked verified. Original GRIB/NetCDF bytes
must be recovered or re-requested from CDS using explicitly documented
request parameters and validated against the originally recorded hourly
wind production before claiming exact source-level alignment.
