# Data provenance

## ERA5 wind (implemented, not yet executed against live data in this repo)

`data/era5.py` implements a real ingestion path against the Copernicus
Climate Data Store (CDS): building the exact API request, fetching the raw
NetCDF file, extracting the nearest grid cell to the configured site,
computing wind speed from the u/v components, correcting it to hub height,
and writing a committed provenance record. This is real code with unit-test
coverage (`tests/test_era5.py`), not a placeholder.

**What has not happened**: no live CDS fetch has been run in this
environment. There is no `~/.cdsapirc` configured here, so no committed
provenance record under `data/provenance_records/` was produced from a real
download, and no capacity-factor comparison or annual run (Phase C4/C5) has
been done. Configured source names in `config.yaml` still describe an
implemented-and-tested path, not proof that live data was fetched — that
distinction is exactly what the provenance record exists to settle, once one
is produced.

### Setup, to actually run a fetch

1. Register for a free account at https://cds.climate.copernicus.eu.
2. Accept the ERA5 licence terms on the dataset page (required once, via the
   web UI, before API access works).
3. Create `~/.cdsapirc` with your URL and personal access token, per the CDS
   API documentation. Never commit this file or its contents.
4. `pip install -e ".[data]"` to get `cdsapi`, `xarray`, `netCDF4`.
5. `pynexus fetch-era5 --config config.yaml --start-date 2023-01-01 --end-date 2023-01-07`

### What is fetched

10m and 100m u/v wind components, `reanalysis-era5-single-levels`, hourly,
at the grid cell nearest `location.latitude`/`location.longitude` in the
config. Wind speed is `sqrt(u^2 + v^2)`. Hub-height wind speed is computed
from the 100m components (not 10m — 100m is much closer to a typical
offshore hub height, which shortens the extrapolation) via
`data.era5.wind_speed_at_hub_height_from_100m`, default power-law
(offshore α = 0.11) with a log-law alternative also implemented; both are
documented with their source assumptions in that module's docstring.

### Fail-loud gap handling

`check_calendar_coverage` in `data/era5.py` raises `CoverageError` naming the
specific missing hours (or duplicate timestamps) if a fetched series does not
fully cover the requested range at hourly resolution. It does not
interpolate, forward-fill, or otherwise paper over a gap — this mirrors the
gap-handling discipline already used in gb-flexabm. `pynexus fetch-era5`
exits non-zero with the specific missing hours printed if this happens.

### Provenance record

Every fetch writes `data/provenance_records/era5_<start>_<end>.json`
(schema in `data/provenance.py`) recording: source, variables, requested vs.
actual grid-cell centre, temporal range and timezone, retrieval timestamp,
the exact CDS request dict, the raw file's SHA-256, expected vs. actual row
count, and the calendar check result (leap day / DST / gaps / duplicates).
The raw NetCDF itself is not committed (`data/raw/` is gitignored) — the
record plus checksum is what lets someone else verify a re-fetch matches.

## ENTSO-E prices

Not yet implemented (Phase A of `REVAMP_PLAN.md`). `config.yaml`'s
`economics.electricity_price_source: "entso_e"` and
`electricity_network.load_profile: "entso_e"` remain configured-but-not-
implemented, same as before this change. The deterministic examples and
tests continue to use synthetic price arrays.

## General rule

Any dataset this repository fetches from an external source must record its
provider, exact variable, geography, period, access time, licence,
transformations, missing-data treatment, and checksum — see
`data.provenance.ProvenanceRecord` for the enforced schema. Credentials
belong in environment variables or `~/.cdsapirc`-style files outside the
repo and must never be committed.
