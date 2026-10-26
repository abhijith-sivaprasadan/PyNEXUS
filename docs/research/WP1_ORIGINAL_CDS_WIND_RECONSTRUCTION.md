## 2026-10-26 — Completed source-hour audit

**Verified source-value equivalence; original NetCDF byte identity not recovered.** This result supersedes the historic "pending" status in this document.

[GitHub Actions run 40230838449](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40230838449) re-requested 12 months of authentic CDS ERA5 NetCDF fields. All 8,760 original 2023 UTC wind-power input values were independently reconstructed with the frozen turbine model, without temporal shifting or imputation. The max and mean absolute hourly differences are 8.526512829121202e-14 and 2.9534172149379998e-15 MW. Annual original and reconstructed wind are 522096.9756187392 and 522096.97561873915 MWh.

Audit status: `VERIFIED_REDELIVERED_CDS_UTC_WIND_VALUES_ORIGINAL_BYTES_DIFFER`. All **12 of 12** recovered monthly NetCDF files have new SHA-256 checksums versus the earlier source records; the original raw files are **not** byte-identical. The verified 2023 CDS wind series must not be silently substituted with the separately served WP4 Open-Meteo ERA5 data.

Evidence: [UTC audit and per-month digests](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40230838449/artifacts/12653138512) and [12 monthly redelivered NetCDF files](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40230838449/artifacts/12653508461), both 60-day Actions artifacts. Preserve them in durable storage. This verifies computational source-to-power equivalence; it does not independently validate commercial turbine or wind farm measurements.

---

# WP1 — original 2023 ERA5 wind primary-source audit and reconstruction

**Status: ready to run, NOT YET PRIMARY-VERIFIED.**

The frozen annual 2023 dispatch records a complete 8,760-row wind array,
but original 12 month Copernicus CDS NetCDF files are not stored in the
GitHub repository. The twelve monthly provenance records
(`data/provenance_records/era5_2023-*.json`) do preserve expected original
NetCDF checksums, request area/variables, nearest grid point, and UTC-hour
calendar checks.

The new `research.wp1_cds_wind_audit` uses original hourly source values,
not proxy Open-Meteo weather data:

1. Recover each exact month by replaying the registered CDS request or
   supply the existing original NetCDF files unchanged.
2. Confirm 12 source provenance records, expected source-file SHA256,
   spatial grid cell (52.5 N, 3.5 E), monthly UTC calendar and row count.
3. Reconstruct hub wind speed from actual 100m u/v vectors using the
   registered 0.11 offshore shear exponent; avoid a second 10m shear
   correction.
4. Use the repository's unchanged turbine power curve and loss/availability
   parameters to reconstruct every original installed-farm megawatt hour.
5. Compare the 8,760 reconstructed MW values to the archived 2023 wind
   input in the **same chronological UTC order**, with a hard 1e-6 MW
   absolute numerical tolerance. No resampling, shifting, or interpolation.
6. Distinguish exact original raw-byte matches from semantically matching
   CDS redeliveries with different NetCDF packaging.

A `VERIFIED_ORIGINAL_CDS_BYTES_AND_UTC_WIND_VALUES` result means all source
file hashes match. A `VERIFIED_REDELIVERED_CDS_UTC_WIND_VALUES_ORIGINAL_BYTES_DIFFER`
result means source MWs align but original byte-level continuity is not
established. A failure/no report closes neither gate.

## Run options

**A. If original monthly NetCDF files exist** (preferred), collect them
under one directory with the original filenames
`era5_YYYY-MM-DD_YYYY-MM-DD_LAT_LON.nc`. Run:

```bash
python -m research.wp1_cds_wind_audit --raw-dir <original_monthly_nc_dir> \
  --output wp1-cds-original-hour-proof
```

**B. If those raw bytes were lost**, configure the GitHub Actions repository
secret `CDSAPI_KEY` and manually run
[WP1 recover and audit primary 2023 CDS ERA5 wind](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/workflows/wp1-cds-primary-wind.yml)
on `main`. It reissues the 12 original monthly requests, archives their
checksums and hourly audit, and preserves the fetched NetCDF separately.
A new redelivery may not have the exact original NetCDF SHA256 even when
its decoded weather data and the archived dispatch agree.

The original files and successful audit must be deposited to durable
access-controlled research storage, not left exclusively in expiring
GitHub Actions artifacts. A change in CDS data version or source response
must be reported, not smoothed over. This is a computational-source
verification, not independent physical wind-farm measurement validation.
