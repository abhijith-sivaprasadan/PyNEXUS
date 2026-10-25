"""Hermetic ERA5 primary wind source calendar audit tests."""

import numpy as np
import pandas as pd
import pytest

from components.wind_turbine import OffshoreWindFarm
from data.era5 import wind_speed_at_hub_height_from_100m
from research.run_reference import CONFIG
from research.wp1_cds_wind_audit import audit_cds_year, compare_utc_hourly_wind


def test_utc_hour_alignment_and_double_shear_avoidance():
    times = pd.date_range("2023-01-01T00:00:00Z", periods=8760, freq="h")
    wind = pd.DataFrame({"wind_speed_100m_ms": np.full(8760, 9.0)}, index=times)
    turbine = OffshoreWindFarm(str(CONFIG))
    speed = wind_speed_at_hub_height_from_100m(
        wind["wind_speed_100m_ms"].to_numpy(), turbine.hub_height_m
    )
    expected_mw = turbine.power_output_mw_from_hub_height(speed)
    result = compare_utc_hourly_wind(wind, expected_mw)
    assert result["hour_count"] == 8760
    assert result["max_abs_wind_difference_mw"] == 0
    assert result["exact_same_source_hour_alignment_verified"] is True


def test_wrong_utc_origin_rejected_even_with_exact_8760_values():
    times = pd.date_range("2023-01-01T01:00:00Z", periods=8760, freq="h")
    wind = pd.DataFrame({"wind_speed_100m_ms": np.full(8760, 9.0)}, index=times)
    with pytest.raises(ValueError, match="coverage|ordered"):
        compare_utc_hourly_wind(wind, np.full(8760, 1.0))


def test_unverified_wind_values_not_silently_accepted():
    times = pd.date_range("2023-01-01T00:00:00Z", periods=8760, freq="h")
    wind = pd.DataFrame({"wind_speed_100m_ms": np.full(8760, 9.0)}, index=times)
    with pytest.raises(ValueError, match="does not match"):
        compare_utc_hourly_wind(wind, np.zeros(8760))


def test_missing_primary_cds_files_fail_without_publishing_result(tmp_path):
    with pytest.raises(FileNotFoundError, match="monthly NetCDF"):
        audit_cds_year(tmp_path / "no-raw", tmp_path / "no-report")
    assert not (tmp_path / "no-report").exists()
