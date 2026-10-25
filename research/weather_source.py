"""WP4 complete-year wind input from an explicitly selected Open-Meteo ERA5 series.

DO NOT relabel this API route as the original CDS single-grid-cell NetCDF
retrieval. The Open-Meteo processing/point selection may differ. All 5 years
(including the 2023 control) use this SAME source, power curve and shear
assumptions; the 2023 original CDS-derived run remains a separate benchmark.

Fail closed on missing/duplicated/non-finite observations and unexpected units.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from components.wind_turbine import OffshoreWindFarm
from data.era5 import wind_speed_at_hub_height_from_100m

API_URL = "https://archive-api.open-meteo.com/v1/archive"
DATASET = "Open-Meteo Historical Weather API: explicit ERA5 selection"
VALID_YEARS = (2021, 2022, 2023, 2024, 2025)


def make_request_url(year, *, latitude=52.5, longitude=3.5):
    if year not in VALID_YEARS:
        raise ValueError("WP4 pre-registers only complete years 2021 through 2025")
    query = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "hourly": "wind_speed_100m",
            "wind_speed_unit": "ms",
            "timezone": "GMT",
            "models": "era5",
            "cell_selection": "nearest",
            "elevation": "nan",
        }
    )
    return API_URL + "?" + query


def fetch_weather(year, *, fetcher=None, retries=3):
    """Fetch actual API bytes. No interpolations or synthetic fallback."""
    url = make_request_url(year)

    def request():
        req = urllib.request.Request(
            url, headers={"User-Agent": "PyNEXUS-reproducible-research/0.1"}
        )
        with urllib.request.urlopen(req, timeout=150) as reply:
            return reply.read()

    fetcher = fetcher or request
    for i in range(retries):
        try:
            raw = fetcher()
            if not isinstance(raw, bytes) or not raw:
                raise ValueError("Weather provider returned empty/nonbyte response")
            return raw, url
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if i == retries - 1:
                raise RuntimeError(f"Could not download actual ERA5 source: {exc}") from exc
            time.sleep(4 * (i + 1))
    raise RuntimeError("Unreachable weather retrieval state")


def validate_weather_payload(raw, year):
    """Verify coverage in UTC, leap-day rules, unique hourly timestamps and units."""
    data = json.loads(raw)
    if data.get("error"):
        raise ValueError(f"Weather API rejected ERA5 request: {data.get('reason')}")
    if int(data.get("utc_offset_seconds", 0)) != 0:
        raise ValueError("Weather timestamps must have zero UTC offset")
    units = data.get("hourly_units", {})
    if units.get("wind_speed_100m") != "m/s":
        raise ValueError(f"Unexpected 100m wind unit: {units.get('wind_speed_100m')!r}")
    hourly = data.get("hourly", {})
    times = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True))
    expected = pd.date_range(
        f"{year}-01-01",
        f"{year + 1}-01-01",
        freq="h",
        inclusive="left",
        tz="UTC",
    )
    if len(times) != len(expected) or times.has_duplicates or not times.equals(expected):
        raise ValueError(
            f"Invalid {year} ERA5 hourly UTC calendar: observed={len(times)}, "
            f"expected={len(expected)}, duplicate={times.has_duplicates}"
        )
    values = np.asarray(hourly["wind_speed_100m"], dtype=float)
    if len(values) != len(expected) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("100m wind speeds must be finite, nonnegative, gap-free")
    if not np.isfinite(float(data.get("latitude", np.nan))) or not np.isfinite(
        float(data.get("longitude", np.nan))
    ):
        raise ValueError("Provider must identify returned spatial coordinates")
    if not (50.0 < float(data["latitude"]) < 55.0 and 1.0 < float(data["longitude"]) < 6.0):
        raise ValueError("Unexpected returned ERA5 location")
    return (
        times,
        values,
        {
            "provider": DATASET,
            "explicit_model": "era5",
            "reported_latitude": float(data["latitude"]),
            "reported_longitude": float(data["longitude"]),
            "utc_calendar": True,
            "leap_year": bool(len(times) == 8784),
            "hour_count": len(times),
            "raw_json_sha256": hashlib.sha256(raw).hexdigest(),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "processing_note": (
                "Open-Meteo ERA5 uses explicit nearest-cell selection and "
                "elevation=nan to avoid default land-biased cell choice; "
                "this wind_speed_100m is a separately served product, "
                "not the exact historical CDS u/v grid-cell extraction."
            ),
        },
    )


def convert_weather_to_wind_2023_model(v100, config_path):
    """Reuse PyNEXUS's exact original 120m shear and turbine farm-loss model."""
    farm = OffshoreWindFarm(str(config_path))
    hub = wind_speed_at_hub_height_from_100m(v100, farm.hub_height_m)
    power = np.asarray(farm.power_output_mw_from_hub_height(hub), dtype=float)
    if power.shape != np.asarray(v100).shape or not np.isfinite(power).all():
        raise ValueError("Unexpected wind farm power series dimensions/values")
    if (power < -1e-8).any() or (power > farm.farm_rated_mw + 1e-6).any():
        raise ValueError("Wind power exceeds physical farm installed capacity")
    return power


def frozen_synthetic_signal(values_2023, hours):
    """Repeat the identical archived 2023 synthetic signal for every study year.

    In leap years the 2023 sequence wraps for the additional 24 hours. This
    series is for Stage B numerical comparability ONLY. Stage A ignores prices.
    """
    values = np.asarray(values_2023, dtype=float)
    if len(values) != 8760 or not np.isfinite(values).all():
        raise ValueError("Expected archived synthetic 8760-hour price series")
    return values[np.arange(hours) % 8760]
