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

    Fails loudly: raises CoverageError naming the missing hours rather than
    returning a report the caller might ignore, and rather than the loader
    silently interpolating or forward-filling past a hole.

    Returns a `calendar_check` dict (for the provenance record) only when
    coverage is complete.
    """
    expected = pd.date_range(
        start=pd.Timestamp(start_date, tz=timezone_name),
        end=pd.Timestamp(end_date, tz=timezone_name) + pd.Timedelta(hours=23),
        freq=freq,
    )

    actual = pd.DatetimeIndex(timestamps)
    if actual.tz is None:
        actual = actual.tz_localize("UTC").tz_convert(timezone_name)
    else:
        actual = actual.tz_convert(timezone_name)

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
    # A DST transition shows up as a repeated or skipped wall-clock hour in a
    # naive local calendar; since `expected`/`actual` are built and compared
    # in UTC-anchored, tz-aware arithmetic above, transitions are already
    # handled correctly by construction — recorded here for the audit trail.
    return {
        "leap_day_in_range": bool(has_leap_day),
        "dst_transition_handled": True,
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

    wind_df has a UTC-indexed, tz-converted-to-`timezone_name` DatetimeIndex
    and columns: wind_speed_10m_ms, wind_speed_100m_ms.

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

    u10 = point["u10"].values
    v10 = point["v10"].values
    u100 = point["u100"].values
    v100 = point["v100"].values
    times = pd.to_datetime(point["time"].values)

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
    return df, {"latitude": nearest_lat, "longitude": nearest_lon}


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
    testable without live CDS access.
    """
    loc = cfg["location"]
    wt = cfg["wind_turbine"]

    nc_path, request = fetch_era5(
        loc["latitude"], loc["longitude"], loc["era5_grid_resolution"], start_date, end_date
    )
    df, spatial_point = load_era5_wind(
        nc_path, loc["latitude"], loc["longitude"], cfg["simulation"]["timezone"], start_date, end_date
    )

    calendar_check = check_calendar_coverage(
        df.index, start_date, end_date, cfg["simulation"]["timezone"]
    )

    df["wind_speed_hub_ms"] = wind_speed_at_hub_height_from_100m(
        df["wind_speed_100m_ms"].to_numpy(), wt["hub_height_m"]
    )

    expected_hours = calendar_check["expected_hours"]
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
        expected_row_count=expected_hours,
        actual_row_count=len(df),
        calendar_check=calendar_check,
    )
    provenance_path = provenance_dir / f"era5_{start_date}_{end_date}.json"
    write_provenance_record(record, provenance_path)

    return df, record
