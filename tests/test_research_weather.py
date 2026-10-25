"""WP4: offline checks for actual-provider semantics, leap years and scope."""

from __future__ import annotations

import json
import urllib.parse

import numpy as np
import pandas as pd
import pytest

from research.aggregate_weather import aggregate
from research.run_weather_year import CASES
from research.weather_source import (
    DATASET,
    VALID_YEARS,
    convert_weather_to_wind_2023_model,
    frozen_synthetic_signal,
    make_request_url,
    validate_weather_payload,
)


def payload(year, *, missing=False, duplicated=False, unit="m/s"):
    idx = pd.date_range(
        f"{year}-01-01",
        f"{year + 1}-01-01",
        inclusive="left",
        tz="UTC",
        freq="h",
    )
    times = idx.strftime("%Y-%m-%dT%H:%M").tolist()
    if missing:
        times.pop(10)
    if duplicated:
        times[10] = times[9]
    obj = {
        "latitude": 52.5,
        "longitude": 3.5,
        "utc_offset_seconds": 0,
        "hourly_units": {"wind_speed_100m": unit},
        "hourly": {"time": times, "wind_speed_100m": [9.0] * len(times)},
    }
    return json.dumps(obj).encode("utf-8")


@pytest.mark.parametrize("year,expected", [(2021, 8760), (2024, 8784), (2025, 8760)])
def test_complete_utc_year_and_leap_day(year, expected):
    times, wind, meta = validate_weather_payload(payload(year), year)
    assert len(times) == expected
    assert len(wind) == expected
    assert meta["hour_count"] == expected
    assert meta["leap_year"] is (year == 2024)
    assert DATASET.startswith("Open-Meteo")
    assert "not the exact" in meta["processing_note"]


@pytest.mark.parametrize("problem", ["missing", "duplicated", "wrong_unit"])
def test_unusable_provider_data_rejected(problem):
    kwargs = {
        "missing": {"missing": True},
        "duplicated": {"duplicated": True},
        "wrong_unit": {"unit": "km/h"},
    }[problem]
    with pytest.raises(ValueError):
        validate_weather_payload(payload(2023, **kwargs), 2023)


def test_era5_only_utc_and_registered_site():
    url = make_request_url(2021)
    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert parsed["models"] == ["era5"]
    assert parsed["timezone"] == ["GMT"]
    assert parsed["wind_speed_unit"] == ["ms"]
    assert parsed["cell_selection"] == ["nearest"]
    assert parsed["elevation"] == ["nan"]
    assert parsed["latitude"] == ["52.5"]
    assert parsed["longitude"] == ["3.5"]
    assert VALID_YEARS == (2021, 2022, 2023, 2024, 2025)
    with pytest.raises(ValueError):
        make_request_url(2026)


def test_synthetic_price_signal_wraps_explicitly_for_leap_day():
    original = np.arange(8760, dtype=float)
    out = frozen_synthetic_signal(original, 8784)
    assert np.array_equal(out[:8760], original)
    assert np.array_equal(out[-24:], original[:24])
    with pytest.raises(ValueError):
        frozen_synthetic_signal(np.array([1, 2]), 8784)


def test_wind_power_uses_existing_physical_farm_curve():
    x = convert_weather_to_wind_2023_model(
        np.array([0.0, 8.0, 13.0, 20.0, 30.0]),
        "outputs/annual_2023/grid_storage_hourly/config.yaml",
    )
    assert x.shape == (5,)
    assert x[0] == pytest.approx(0)
    assert x[1] > 0
    assert x[2] > x[1]
    assert x[-1] == pytest.approx(0)
    assert np.all(x <= 150.0)


def test_matrix_is_explicit_and_seven_cases_only():
    assert list(CASES) == ["baseline", "W", "E", "G", "S", "eta", "E+S"]
    assert CASES["E+S"] == ("E", "S")


def test_incomplete_interannual_archive_cannot_be_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        aggregate(tmp_path)
