# Data provenance

## ERA5 wind (implemented and verified against the live CDS API)

`data/era5.py` fetches real wind data from the Copernicus Climate Data Store
(CDS): building the exact API request, fetching the raw NetCDF file,
extracting the nearest grid cell to the configured site, computing wind
speed from the u/v components, correcting it to hub height, and writing a
committed provenance record. A small validation fetch (2 days, North Sea
site) has been run successfully against the live API; the full annual fetch
is Phase C5, tracked separately below.

### Setup, to run a fetch yourself

1. Register for a free account at https://cds.climate.copernicus.eu.
2. Accept the ERA5 licence terms on the dataset page (required once, via the
   web UI, before API access works).
3. Create `~/.cdsapirc` with your URL and personal access token, per the CDS
   API documentation. Never commit this file or its contents.
4. `pip install -e ".[data]"` to get `cdsapi`, `xarray`, `netCDF4`.
5. `pynexus fetch-era5 --config config.yaml --start-date 2023-01-01 --end-date 2023-01-07`
   for a short range, or `pynexus fetch-era5-year --config config.yaml --year 2023`
   for a full year (see "Fetching a full year" below — this cannot be one request).

### What is fetched

10m and 100m u/v wind components, `reanalysis-era5-single-levels`, hourly,
at the grid cell nearest `location.latitude`/`location.longitude` in the
config. Wind speed is `sqrt(u^2 + v^2)`. Hub-height wind speed is computed
from the 100m components (not 10m — 100m is much closer to a typical
offshore hub height, which shortens the extrapolation) via
`data.era5.wind_speed_at_hub_height_from_100m`, default power-law
(offshore α = 0.11) with a log-law alternative also implemented; both are
documented with their source assumptions in that module's docstring.

### Three things the live API taught us that the docs originally got wrong

Each of these was found by actually running against CDS, not anticipated in
advance — worth recording because the failure mode in each case was subtle
enough to produce a wrong answer silently rather than an obvious crash, had
it gone unnoticed:

1. **The time coordinate is `valid_time`, not `time`.** The unified CDS API
   (2024+) renamed it; the legacy API used `time`. `load_era5_wind` checks
   for both.
2. **Coverage must be checked in UTC, not the configured local timezone.**
   `build_cds_request`'s year/month/day fields are UTC-native — ERA5 itself
   is a UTC-referenced dataset. An earlier version of `check_calendar_coverage`
   built its expected-hours range in `simulation.timezone` (Europe/Amsterdam,
   UTC+1/+2). That shifted local midnight away from the UTC day boundary CDS
   actually returns, so a genuinely complete UTC day was reported as missing
   its first two hours and carrying two extra hours at the end — a false
   positive, not a real gap. Caught by running the loader against a real
   downloaded file, not by the unit tests (which used synthetic indices
   built the same wrong way). Fixed by defining "expected coverage" in UTC
   throughout; `timezone_name` is now used only for the informational
   DST-transition flag, never for defining what counts as complete.
3. **A full year is one request too many.** A single request for 8,760 hours
   × 4 variables is rejected by CDS ("cost limits exceeded... request is too
   large"). See "Fetching a full year" below.

### Fetching a full year

`pynexus fetch-era5-year --config config.yaml --year <YYYY>` chunks the
year into 12 monthly requests (`data.era5.fetch_era5_year_monthly`) — CDS's
own documented workaround for the size limit above. Submitting all 12 at
once was tried first and is also wrong: CDS rejected most of them with
"Number queued requests for this dataset is temporarily limited" (also
discovered live, not documented anywhere we could find in advance). Requests
are now submitted in small batches (`max_concurrent`, default 3) — enough to
get real parallelism without tripping the undocumented per-user queue cap;
lower it with `--max-concurrent` if a batch still comes back rejected for
the same reason.

**The real limit turned out to be tighter still, and outside anyone's
control.** CDS's own "Queue status" panel (visible on a request's Details
page) states the CDS-MARS backend — the archive ERA5 is served from —
allows exactly **1 concurrent request per user**, regardless of
`max_concurrent`; the batching above governs how many requests we *submit*
together, not how many CDS actually runs at once for us. Observed live: at
one point the global queue showed 460 requests running and 4,713 queued
system-wide. This is shared infrastructure under heavy load, not a
parameter we can tune around — an annual fetch can take anywhere from
roughly 20 minutes to several hours per month depending on congestion at
the time, and there is no way to speed up an individual request once
queued (retrying or resubmitting does not help and just adds another
queued request).

### Fail-loud gap handling

`check_calendar_coverage` in `data/era5.py` raises `CoverageError` naming the
specific missing hours (or duplicate timestamps) if a fetched series does not
fully cover the requested range at hourly resolution. It does not
interpolate, forward-fill, or otherwise paper over a gap — this mirrors the
gap-handling discipline already used in gb-flexabm. `pynexus fetch-era5`
exits non-zero with the specific missing hours printed if this happens. For
the annual fetch, both each individual month and the full concatenated year
are checked — a clean month-by-month result does not by itself guarantee no
boundary issue between months.

### Provenance record

Every fetch writes `data/provenance_records/era5_<start>_<end>.json`
(schema in `data/provenance.py`) recording: source, variables, requested vs.
actual grid-cell centre, temporal range and timezone, retrieval timestamp,
the exact CDS request dict, the raw file's SHA-256, expected vs. actual row
count, the calendar check result (leap day / DST / gaps / duplicates), and
the `expver` value(s) present — ERA5 marks each hour as final ("0001") or
preliminary, not-yet-back-filled ERA5T ("0005") data, and ERA5T values can be
revised later, which matters for anyone re-running the same request near the
present and expecting an identical answer. The raw NetCDF itself is not
committed (`data/raw/` is gitignored) — the record plus checksum is what lets
someone else verify a re-fetch matches. A full-year fetch additionally writes
`era5_<year>_annual_summary.json` listing the 12 monthly records and the
whole-year calendar check.

## ENTSO-E prices (Phase A3: implemented and unit-tested, no live fetch run yet)

`data/entsoe.py` implements a real ingestion path against the ENTSO-E
Transparency Platform's day-ahead price API (`entsoe-py`), with the same
discipline as `data/era5.py`: fail-loud coverage checking, a cached raw
file, and a committed provenance record per fetch. **No live ENTSO-E fetch
has been run** — this needs a separate credential from the CDS one used for
ERA5 (a Web API security token, requested from within the ENTSO-E
Transparency Platform's own account settings; register at
https://transparency.entsoe.eu). Set it as the `ENTSOE_API_KEY` environment
variable; never put it in `config.yaml` or commit it.

`config.yaml`'s bidding zone defaults to `NL` (`data.entsoe.DEFAULT_BIDDING_ZONE`),
matching this repo's configured North Sea/Netherlands site — not a generic
or wrong-country price, which would be silently wrong rather than obviously
broken.

**Real bug caught building this, worth recording alongside the ERA5 ones**:
the first version reused `data.era5.check_calendar_coverage` (UTC-day
aligned) for ENTSO-E's local-market-day fetches, converting the local day
boundary to a UTC date string. That silently drops the fractional-day
offset whenever the local timezone isn't UTC — Amsterdam is UTC+1 in
January, so day-ahead prices for "2023-01-01" actually start at 23:00 UTC
on 2022-12-31, and truncating to a UTC date string loses that, making a
genuinely complete local day look like it's missing 23 hours. Caught by a
test using a fake client returning a real, complete 48-hour series before
any live credential was involved — the same *class* of bug fixed once
already in ERA5's own coverage check, reintroduced by reusing that
function outside the UTC-day assumption it depends on. Fixed with a
dedicated `data.entsoe.check_local_day_coverage` that builds its expected
range directly in local time.

`pynexus solve` is not yet wired to use real ENTSO-E prices (only
`--wind-csv`'s synthetic-price pairing exists today) — that wiring is
deferred until `ENTSOE_API_KEY` is actually available to test against
live, the same gate ERA5 was behind before `~/.cdsapirc` existed.

## Running the dispatch model on real ERA5 wind

`pynexus solve --wind-csv <csv from fetch-era5-year --output-csv> --config ...`
pairs real ERA5 wind (already hub-height corrected) with the same
deterministic synthetic price/carbon generator used by `--synthetic`, since
ENTSO-E prices don't exist yet. The run manifest's `input_source.kind` is
`"era5_wind_synthetic_price"`, distinct from `"synthetic"` and `"user_csv"`
— this is a hybrid input source and must always be described as one. It is
real wind, not a real dispatch-cost result; the cost/emissions objective
values from such a run reflect a real wind resource against a synthetic
price signal, not a real market outcome.

## General rule

Any dataset this repository fetches from an external source must record its
provider, exact variable, geography, period, access time, licence,
transformations, missing-data treatment, and checksum — see
`data.provenance.ProvenanceRecord` for the enforced schema. Credentials
belong in environment variables or `~/.cdsapirc`-style files outside the
repo and must never be committed.
