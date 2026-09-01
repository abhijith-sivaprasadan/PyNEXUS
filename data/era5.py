"""ERA5 reanalysis wind ingestion.

Fetches 10m and 100m u/v wind components from the Copernicus Climate Data
Store (CDS) for the configured site, extracts the nearest grid cell, corrects
100m wind speed to turbine hub height, and writes a committed provenance
record for every fetch (see `data/provenance.py`).

Requires a free CDS account and `~/.cdsapirc` (or `CDSAPI_URL`/`CDSAPI_KEY`
env vars). Setup is documented in docs/data_provenance.md. The `cdsapi`,
`xarray`, and `netCDF4` packages are only imported inside the functions that
need them, so importing this module (e.g. for the shear-correction or
calendar-check helpers) does not require the CDS credentials or those heavy
optional dependencies to be present.

Governing rule: this loader never silently fills a data gap. Incomplete
coverage is a hard failure, not something to interpolate past.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.provenance import ProvenanceRecord, build_provenance_record, write_provenance_record

ERA5_SOURCE = "ERA5 reanalysis-era5-single-levels (hourly)"
ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
]

# Wind shear correction: extrapolate ERA5's 100m wind speed to hub height.
# 100m is used as the base (not 10m) because it is much closer to typical
# offshore hub heights (~100-150m), which shortens the extrapolation and its
# associated uncertainty relative to correcting from 10m.
#
# Power law:  v(h) = v_ref * (h / h_ref)^alpha
#   Offshore alpha = 0.11 is the IEC 61400-3 / open-sea literature value
#   also used for the 10m-based correction already in components/wind_turbine.py.
#
# Log law:    v(h) = v_ref * ln(h / z0) / ln(h_ref / z0)
#   z0 = 0.0002 m is a standard open-sea roughness length (Charnock-relation
#   order-of-magnitude value used in offshore wind resource literature).
OFFSHORE_SHEAR_EXPONENT = 0.11
OFFSHORE_ROUGHNESS_LENGTH_M = 0.0002
ERA5_HUB_REFERENCE_HEIGHT_M = 100.0


def wind_speed_at_hub_height_from_100m(
    v100: "np.ndarray | float",
    hub_height_m: float,
    law: str = "power_law",
    shear_exponent: float = OFFSHORE_SHEAR_EXPONENT,
    roughness_length_m: float = OFFSHORE_ROUGHNESS_LENGTH_M,
):
    """Correct ERA5 100m wind speed to turbine hub height.

    Parameters
    ----------
    v100 : array-like or float
        Wind speed at 100m (m/s), i.e. sqrt(u100^2 + v100^2) from ERA5.
    hub_height_m : float
        Turbine hub height (m).
    law : str
        "power_law" (default) or "log_law".
    shear_exponent : float
        Only used for "power_law". Default is the offshore literature value.
    roughness_length_m : float
        Only used for "log_law". Default is a standard open-sea value.

    Returns
    -------
    array-like or float
        Wind speed at hub height (m/s).
    """
    v100 = np.asarray(v100, dtype=float)
    if law == "power_law":
        result = v100 * (hub_height_m / ERA5_HUB_REFERENCE_HEIGHT_M) ** shear_exponent
    elif law == "log_law":
        result = (
            v100
            * np.log(hub_height_m / roughness_length_m)
            / np.log(ERA5_HUB_REFERENCE_HEIGHT_M / roughness_length_m)
        )
    else:
        raise ValueError(f"Unknown shear law: {law!r}. Use 'power_law' or 'log_law'.")
    return float(result) if result.ndim == 0 else result


def build_cds_request(
    latitude: float,
    longitude: float,
    grid_resolution: float,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Build the exact CDS API request dict for the configured site/period.

    A single-point request in `cdsapi`/ERA5 still requires an `area` box
    (north/west/south/east); we use the configured grid resolution to draw
    the smallest box that reliably contains one grid cell around the point,
    and let `load_era5_wind` record which grid cell centre was actually
    returned.
    """
    pad = max(grid_resolution, 0.25)
    dates = pd.date_range(start_date, end_date, freq="D")
    return {
        "product_type": "reanalysis",
        "variable": list(ERA5_VARIABLES),
        "year": sorted({str(d.year) for d in dates}),
        "month": sorted({f"{d.month:02d}" for d in dates}),
        "day": sorted({f"{d.day:02d}" for d in dates}),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [
            round(latitude + pad, 3),
            round(longitude - pad, 3),
            round(latitude - pad, 3),
            round(longitude + pad, 3),
        ],
        "format": "netcdf",
    }


def fetch_era5(
    latitude: float,
    longitude: float,
    grid_resolution: float,
    start_date: str,
    end_date: str,
    cache_dir: Path = Path("data/raw"),
) -> tuple[Path, dict[str, Any]]:
    """Submit a CDS API request and download the raw NetCDF file.

    Requires `cdsapi` to be installed (`pip install -e ".[data]"`) and a
    valid `~/.cdsapirc`. Raises a clear error naming the missing piece
    instead of a bare stack trace from inside the `cdsapi` package.
    """
    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError(
            "fetch_era5 requires the 'cdsapi' package: pip install -e '.[data]'"
        ) from exc

    request = build_cds_request(latitude, longitude, grid_resolution, start_date, end_date)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"era5_{start_date}_{end_date}_{latitude}_{longitude}.nc"

    try:
        client = cdsapi.Client()
    except Exception as exc:
        raise RuntimeError(
            "CDS API client could not be created - check ~/.cdsapirc "
            "(see docs/data_provenance.md for setup). "
            f"Underlying error: {exc}"
        ) from exc
    client.retrieve("reanalysis-era5-single-levels", request, str(target))

    return target, request


def month_boundaries(year: int) -> list[tuple[str, str]]:
    """[(first_day, last_day), ...] for each calendar month of `year`, as UTC-day strings."""
    boundaries = []
    for month in range(1, 13):
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthEnd(0)
        boundaries.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    return boundaries


def fetch_era5_year_monthly(
    latitude: float,
    longitude: float,
    grid_resolution: float,
    year: int,
    cache_dir: Path = Path("data/raw"),
    max_concurrent: int = 3,
) -> list[tuple[Path, dict[str, Any]]]:
    """Fetch a full year of ERA5 wind as 12 monthly requests.

    A single request for a full year of hourly, multi-variable ERA5 data is
    rejected by CDS ("cost limits exceeded... request is too large") — this
    was discovered against the live API, not assumed. Chunking by calendar
    month is CDS's own documented workaround and keeps each request well
    within the size that already succeeded in a smaller (2-day) request.

    Requests are submitted in batches of `max_concurrent`
    (`wait_until_complete=False`, so `client.retrieve` returns immediately
    after submission rather than polling), so CDS processes each batch in
    parallel server-side; this function then blocks on downloading that
    batch before submitting the next. Submitting all 12 months at once was
    tried first and is wrong: CDS rejected most of them outright with
    "Number queued requests for this dataset is temporarily limited" —
    also discovered against the live API. `max_concurrent=3` is a
    conservative guess at that undocumented per-user limit; lower it if a
    request comes back rejected for the same reason.
    """
    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError(
            "fetch_era5_year_monthly requires the 'cdsapi' package: pip install -e '.[data]'"
        ) from exc

    try:
        client = cdsapi.Client(wait_until_complete=False)
    except Exception as exc:
        raise RuntimeError(
            "CDS API client could not be created - check ~/.cdsapirc "
            "(see docs/data_provenance.md for setup). "
            f"Underlying error: {exc}"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    months = month_boundaries(year)

    fetched = []
    for batch_start in range(0, len(months), max_concurrent):
        batch = months[batch_start : batch_start + max_concurrent]

        handles = []
        for start_date, end_date in batch:
            request = build_cds_request(latitude, longitude, grid_resolution, start_date, end_date)
            result = client.retrieve("reanalysis-era5-single-levels", request)
            handles.append((start_date, end_date, request, result))

        for start_date, end_date, request, result in handles:
            target = cache_dir / f"era5_{start_date}_{end_date}_{latitude}_{longitude}.nc"
            result.download(str(target))
            fetched.append((target, request, start_date, end_date))

    return fetched


class CoverageError(Exception):
    """Raised when a fetched time series does not fully cover its expected range."""


def check_calendar_coverage(
    timestamps: pd.DatetimeIndex,
    start_date: str,
    end_date: str,
    timezone_name: str,
    freq: str = "1h",
) -> dict[str, Any]:
    """Verify an hourly time series has no gaps or duplicates over the full range.

    `start_date`/`end_date` are UTC calendar days — this matches what
    `build_cds_request` actually requests from CDS (its year/month/day
    fields are UTC-native; ERA5 itself is fundamentally a UTC-referenced
    dataset) and what a fetched file's `time`/`valid_time` coordinate
    actually contains. `timezone_name` is used only to report whether a
    local DST transition falls within the range, for the audit trail — it
    does not change what counts as a "complete" fetch. Checking coverage in
    local time instead of UTC was tried first and is wrong: a fixed local
    UTC offset (e.g. Amsterdam's +1/+2) shifts local midnight away from the
    UTC day boundary CDS actually returns, which makes a genuinely complete
    UTC day look like it has missing hours at one end and extra hours at
    the other.

    Fails loudly: raises CoverageError naming the missing hours rather than
    returning a report the caller might ignore, and rather than the loader
    silently interpolating or forward-filling past a hole.

    Returns a `calendar_check` dict (for the provenance record) only when
    coverage is complete.
    """
    expected = pd.date_range(
        start=pd.Timestamp(start_date, tz="UTC"),
        end=pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23),
        freq=freq,
    )

    actual = pd.DatetimeIndex(timestamps)
    actual = actual.tz_localize("UTC") if actual.tz is None else actual.tz_convert("UTC")

    duplicates = actual[actual.duplicated()]
    if len(duplicates) > 0:
        raise CoverageError(
            f"Duplicate timestamps in fetched series: {list(duplicates.astype(str))[:10]}"
            + (" ..." if len(duplicates) > 10 else "")
        )

    missing = expected.difference(actual)
    if len(missing) > 0:
        raise CoverageError(
            f"Incomplete ERA5 coverage: {len(missing)} of {len(expected)} expected hours "
            f"missing. First missing: {list(missing.astype(str)[:10])}"
            + (" ..." if len(missing) > 10 else "")
        )

    has_leap_day = ((expected.month == 2) & (expected.day == 29)).any()
    local_expected = expected.tz_convert(timezone_name)
    # A DST transition in the local timezone shows up as a change in UTC
    # offset partway through the range, not as a gap or duplicate in the
    # UTC-native `expected`/`actual` series compared above — this is purely
    # informational for the audit trail.
    local_offsets = {ts.utcoffset() for ts in local_expected}
    dst_transition_in_range = len(local_offsets) > 1
    return {
        "leap_day_in_range": bool(has_leap_day),
        "dst_transition_in_range": bool(dst_transition_in_range),
        "no_gaps": True,
        "no_duplicates": True,
        "expected_hours": int(len(expected)),
        "actual_hours": int(len(actual)),
    }


def _nearest_grid_point(dataset, latitude: float, longitude: float) -> tuple[float, float]:
    lat_name = "latitude" if "latitude" in dataset.coords else "lat"
    lon_name = "longitude" if "longitude" in dataset.coords else "lon"
    lat_vals = dataset[lat_name].values
    lon_vals = dataset[lon_name].values
    nearest_lat = float(lat_vals[np.argmin(np.abs(lat_vals - latitude))])
    nearest_lon = float(lon_vals[np.argmin(np.abs(lon_vals - longitude))])
    return nearest_lat, nearest_lon


def load_era5_wind(
    nc_path: Path,
    latitude: float,
    longitude: float,
    timezone_name: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a fetched ERA5 NetCDF file and return (wind_df, spatial_point).

    wind_df has a UTC DatetimeIndex (ERA5's native timezone — the `time`/
    `valid_time` coordinate carries no timezone label but represents UTC
    instants) and columns: wind_speed_10m_ms, wind_speed_100m_ms.
    `start_date`/`end_date` are UTC calendar days; `timezone_name` is used
    only for the informational DST-transition flag in the calendar check.

    Raises CoverageError (via check_calendar_coverage) if the file does not
    fully cover [start_date, end_date] at hourly resolution — this function
    does not interpolate or fill gaps.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "load_era5_wind requires the 'xarray' package: pip install -e '.[data]'"
        ) from exc

    ds = xr.open_dataset(nc_path)
    nearest_lat, nearest_lon = _nearest_grid_point(ds, latitude, longitude)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    point = ds.sel({lat_name: nearest_lat, lon_name: nearest_lon}, method="nearest")

    # The unified CDS API (2024+) names the time coordinate "valid_time";
    # the legacy CDS API used "time". Support both rather than assuming one.
    time_name = "time" if "time" in point.coords else "valid_time"

    u10 = point["u10"].values
    v10 = point["v10"].values
    u100 = point["u100"].values
    v100 = point["v100"].values
    times = pd.to_datetime(point[time_name].values)

    check_calendar_coverage(times, start_date, end_date, timezone_name)

    wind_speed_10m = np.sqrt(u10**2 + v10**2)
    wind_speed_100m = np.sqrt(u100**2 + v100**2)

    df = pd.DataFrame(
        {
            "wind_speed_10m_ms": wind_speed_10m,
            "wind_speed_100m_ms": wind_speed_100m,
        },
        index=pd.DatetimeIndex(times, name="time"),
    )

    # expver marks whether each hour is final ERA5 ("0001") or preliminary,
    # not-yet-back-filled ERA5T ("0005") data. ERA5T values can be revised
    # after the fact, so this matters for anyone re-running the same request
    # later and getting a (slightly) different answer near the present.
    expver_values = (
        sorted(set(np.atleast_1d(point["expver"].values))) if "expver" in point.coords else []
    )

    return df, {
        "latitude": nearest_lat,
        "longitude": nearest_lon,
        "expver_values": [str(v) for v in expver_values],
    }


def _load_and_record(
    cfg: dict[str, Any],
    nc_path: Path,
    request: dict[str, Any],
    start_date: str,
    end_date: str,
    provenance_dir: Path,
) -> tuple[pd.DataFrame, ProvenanceRecord]:
    """Shared load+shear-correct+provenance step for a single fetched file."""
    loc = cfg["location"]
    wt = cfg["wind_turbine"]

    df, spatial_point = load_era5_wind(
        nc_path,
        loc["latitude"],
        loc["longitude"],
        cfg["simulation"]["timezone"],
        start_date,
        end_date,
    )
    calendar_check = check_calendar_coverage(
        df.index, start_date, end_date, cfg["simulation"]["timezone"]
    )
    df["wind_speed_hub_ms"] = wind_speed_at_hub_height_from_100m(
        df["wind_speed_100m_ms"].to_numpy(), wt["hub_height_m"]
    )

    record = build_provenance_record(
        source=ERA5_SOURCE,
        variables=ERA5_VARIABLES,
        requested_latitude=loc["latitude"],
        requested_longitude=loc["longitude"],
        actual_latitude=spatial_point["latitude"],
        actual_longitude=spatial_point["longitude"],
        start_date=start_date,
        end_date=end_date,
        timezone_name=cfg["simulation"]["timezone"],
        cds_request=request,
        raw_file=nc_path,
        expected_row_count=calendar_check["expected_hours"],
        actual_row_count=len(df),
        calendar_check=calendar_check,
        extra={"expver_values": spatial_point.get("expver_values", [])},
    )
    provenance_path = provenance_dir / f"era5_{start_date}_{end_date}.json"
    write_provenance_record(record, provenance_path)

    return df, record


def fetch_and_load(
    cfg: dict[str, Any],
    start_date: str,
    end_date: str,
    cache_dir: Path = Path("data/raw"),
    provenance_dir: Path = Path("data/provenance_records"),
) -> tuple[pd.DataFrame, ProvenanceRecord]:
    """Full pipeline: fetch from CDS, load, shear-correct, and record provenance.

    This is the function `pynexus fetch-era5` calls. It is the only place
    that ties fetch + load + provenance together; each piece is independently
    testable without live CDS access. For a request larger than CDS accepts
    in one call (e.g. a full year), use `fetch_and_load_year` instead.
    """
    loc = cfg["location"]
    nc_path, request = fetch_era5(
        loc["latitude"],
        loc["longitude"],
        loc["era5_grid_resolution"],
        start_date,
        end_date,
        cache_dir,
    )
    return _load_and_record(cfg, nc_path, request, start_date, end_date, provenance_dir)


def fetch_and_load_year(
    cfg: dict[str, Any],
    year: int,
    cache_dir: Path = Path("data/raw"),
    provenance_dir: Path = Path("data/provenance_records"),
    max_concurrent: int = 3,
) -> tuple[pd.DataFrame, list[ProvenanceRecord], dict[str, Any]]:
    """Full pipeline for a full year: 12 monthly CDS fetches, concatenated.

    Writes one provenance record per month plus a summary record
    (`era5_<year>_annual_summary.json`) covering the concatenated year as a
    whole. Raises CoverageError if any individual month, or the full
    concatenated year, has a gap or duplicate — a clean month-by-month
    result does not by itself guarantee no boundary issue between months.
    """
    loc = cfg["location"]
    fetched = fetch_era5_year_monthly(
        loc["latitude"],
        loc["longitude"],
        loc["era5_grid_resolution"],
        year,
        cache_dir,
        max_concurrent=max_concurrent,
    )

    frames = []
    monthly_records = []
    for nc_path, request, start_date, end_date in fetched:
        df, record = _load_and_record(cfg, nc_path, request, start_date, end_date, provenance_dir)
        frames.append(df)
        monthly_records.append(record)

    full_df = pd.concat(frames).sort_index()
    full_calendar_check = check_calendar_coverage(
        full_df.index, f"{year}-01-01", f"{year}-12-31", cfg["simulation"]["timezone"]
    )

    summary = {
        "year": year,
        "monthly_provenance_files": [f"era5_{start}_{end}.json" for _, _, start, end in fetched],
        "calendar_check": full_calendar_check,
        "total_rows": len(full_df),
    }
    (provenance_dir / f"era5_{year}_annual_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return full_df, monthly_records, summary
