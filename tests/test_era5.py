import sys

import numpy as np
import pandas as pd
import pytest

from data.era5 import (
    CoverageError,
    build_cds_request,
    check_calendar_coverage,
    fetch_era5,
    wind_speed_at_hub_height_from_100m,
)

# --- Shear correction -----------------------------------------------------


def test_power_law_is_identity_at_reference_height() -> None:
    # ERA5's 100m reading, corrected to a 100m hub, must be unchanged.
    assert wind_speed_at_hub_height_from_100m(9.5, hub_height_m=100.0) == pytest.approx(9.5)


def test_log_law_is_identity_at_reference_height() -> None:
    assert wind_speed_at_hub_height_from_100m(
        9.5, hub_height_m=100.0, law="log_law"
    ) == pytest.approx(9.5)


def test_power_law_uplifts_for_taller_hub() -> None:
    v = wind_speed_at_hub_height_from_100m(10.0, hub_height_m=150.0)
    assert v > 10.0


def test_unknown_shear_law_rejected() -> None:
    with pytest.raises(ValueError):
        wind_speed_at_hub_height_from_100m(10.0, hub_height_m=120.0, law="nonsense")


def test_shear_correction_is_vectorised() -> None:
    v = wind_speed_at_hub_height_from_100m(np.array([5.0, 10.0, 15.0]), hub_height_m=120.0)
    assert v.shape == (3,)
    assert np.all(v > np.array([5.0, 10.0, 15.0]))  # 120m > 100m reference => uplift


# --- CDS request construction ----------------------------------------------


def test_build_cds_request_structure() -> None:
    request = build_cds_request(
        latitude=52.5, longitude=3.5, grid_resolution=0.25, start_date="2023-01-01", end_date="2023-01-02"
    )
    assert set(request["variable"]) == {
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
    }
    assert request["year"] == ["2023"]
    assert request["month"] == ["01"]
    assert request["day"] == ["01", "02"]
    assert len(request["time"]) == 24
    assert request["format"] == "netcdf"
    north, west, south, east = request["area"]
    assert north > 52.5 > south
    assert west < 3.5 < east


# --- Calendar coverage: the fail-loud contract ------------------------------


def _complete_hourly_index(start, end, tz):
    return pd.date_range(
        start=pd.Timestamp(start, tz=tz), end=pd.Timestamp(end, tz=tz) + pd.Timedelta(hours=23), freq="1h"
    )


def test_complete_coverage_passes_across_leap_day() -> None:
    idx = _complete_hourly_index("2024-02-27", "2024-03-01", "Europe/Amsterdam")
    result = check_calendar_coverage(idx, "2024-02-27", "2024-03-01", "Europe/Amsterdam")
    assert result["leap_day_in_range"] is True
    assert result["no_gaps"] is True
    assert result["no_duplicates"] is True
    assert result["actual_hours"] == result["expected_hours"] == len(idx)


def test_complete_coverage_passes_across_dst_transition() -> None:
    # Europe/Amsterdam springs forward on the last Sunday in March.
    idx = _complete_hourly_index("2024-03-30", "2024-03-31", "Europe/Amsterdam")
    result = check_calendar_coverage(idx, "2024-03-30", "2024-03-31", "Europe/Amsterdam")
    assert result["no_gaps"] is True
    assert result["no_duplicates"] is True


def test_missing_hour_fails_loudly() -> None:
    idx = _complete_hourly_index("2023-06-01", "2023-06-02", "Europe/Amsterdam")
    gapped = idx.delete(5)  # drop one hour partway through

    with pytest.raises(CoverageError, match="Incomplete ERA5 coverage"):
        check_calendar_coverage(gapped, "2023-06-01", "2023-06-02", "Europe/Amsterdam")


def test_duplicate_hour_fails_loudly() -> None:
    idx = _complete_hourly_index("2023-06-01", "2023-06-01", "Europe/Amsterdam")
    duplicated = idx.insert(3, idx[3])

    with pytest.raises(CoverageError, match="Duplicate timestamps"):
        check_calendar_coverage(duplicated, "2023-06-01", "2023-06-01", "Europe/Amsterdam")


def test_no_silent_fill_on_gap() -> None:
    """The loader must never return a report for incomplete data — it must raise."""
    idx = _complete_hourly_index("2023-06-01", "2023-06-01", "Europe/Amsterdam")
    gapped = idx.delete(0)

    with pytest.raises(CoverageError):
        check_calendar_coverage(gapped, "2023-06-01", "2023-06-01", "Europe/Amsterdam")


# --- fetch_era5 failure paths (hermetic: no real network/credentials used) --


def test_fetch_era5_missing_cdsapi_package_raises_clear_import_error(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cdsapi", None)  # forces `import cdsapi` to raise

    with pytest.raises(ImportError, match=r"pip install -e '\.\[data\]'"):
        fetch_era5(52.5, 3.5, 0.25, "2023-01-01", "2023-01-02")


def test_fetch_era5_missing_credentials_raises_clear_runtime_error(monkeypatch) -> None:
    cdsapi = pytest.importorskip("cdsapi")  # only meaningful with the optional [data] extra

    def _raise_missing_config(*args, **kwargs):
        raise Exception("Missing/incomplete configuration file: ~/.cdsapirc")

    monkeypatch.setattr(cdsapi, "Client", _raise_missing_config)

    with pytest.raises(RuntimeError, match="check ~/.cdsapirc"):
        fetch_era5(52.5, 3.5, 0.25, "2023-01-01", "2023-01-02")
