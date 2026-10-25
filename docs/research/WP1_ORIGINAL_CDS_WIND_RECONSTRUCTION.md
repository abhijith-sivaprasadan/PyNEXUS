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
