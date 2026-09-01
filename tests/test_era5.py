import sys

import numpy as np
import pandas as pd
import pytest

from data.era5 import (
    CoverageError,
    build_cds_request,
    check_calendar_coverage,
    fetch_era5,
    month_boundaries,
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
        latitude=52.5,
        longitude=3.5,
        grid_resolution=0.25,
        start_date="2023-01-01",
        end_date="2023-01-02",
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


# --- Monthly chunking (CDS rejects a full-year single request) -------------


def test_month_boundaries_covers_all_twelve_months_no_overlap() -> None:
    boundaries = month_boundaries(2023)
    assert len(boundaries) == 12
    assert boundaries[0] == ("2023-01-01", "2023-01-31")
    assert boundaries[1] == ("2023-02-01", "2023-02-28")  # non-leap year
    assert boundaries[11] == ("2023-12-01", "2023-12-31")


def test_month_boundaries_handles_leap_year_february() -> None:
    boundaries = month_boundaries(2024)
    assert boundaries[1] == ("2024-02-01", "2024-02-29")


def test_month_boundaries_are_contiguous() -> None:
    boundaries = month_boundaries(2023)
    for (_, end), (next_start, _) in zip(boundaries, boundaries[1:]):
        assert pd.Timestamp(end) + pd.Timedelta(days=1) == pd.Timestamp(next_start)


# --- Calendar coverage: the fail-loud contract ------------------------------
#
# start_date/end_date are UTC calendar days (matching what CDS actually
# returns), so the reference index used to build "actual" here is always
# constructed in UTC — never in a local timezone. An earlier version of this
# module (and these tests) built the expected range in local time instead;
# that made a genuinely complete UTC day look incomplete once Amsterdam's
# +1/+2 offset was applied, and was only caught against a real downloaded
# file. See data.era5.check_calendar_coverage's docstring.


def _complete_hourly_index_utc(start, end):
    return pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23),
        freq="1h",
    )


def test_complete_coverage_passes_across_leap_day() -> None:
    idx = _complete_hourly_index_utc("2024-02-27", "2024-03-01")
    result = check_calendar_coverage(idx, "2024-02-27", "2024-03-01", "Europe/Amsterdam")
    assert result["leap_day_in_range"] is True
    assert result["no_gaps"] is True
    assert result["no_duplicates"] is True
    assert result["actual_hours"] == result["expected_hours"] == len(idx)


def test_dst_transition_is_detected_and_does_not_break_coverage() -> None:
    # Europe/Amsterdam springs forward on the last Sunday in March: a
    # genuinely complete UTC range must still pass, and the transition
    # must be flagged informationally.
    idx = _complete_hourly_index_utc("2024-03-30", "2024-03-31")
    result = check_calendar_coverage(idx, "2024-03-30", "2024-03-31", "Europe/Amsterdam")
    assert result["no_gaps"] is True
    assert result["no_duplicates"] is True
    assert result["dst_transition_in_range"] is True


def test_no_dst_transition_reports_false() -> None:
    idx = _complete_hourly_index_utc("2024-06-01", "2024-06-02")
    result = check_calendar_coverage(idx, "2024-06-01", "2024-06-02", "Europe/Amsterdam")
    assert result["dst_transition_in_range"] is False


def test_missing_hour_fails_loudly() -> None:
    idx = _complete_hourly_index_utc("2023-06-01", "2023-06-02")
    gapped = idx.delete(5)  # drop one hour partway through

    with pytest.raises(CoverageError, match="Incomplete ERA5 coverage"):
        check_calendar_coverage(gapped, "2023-06-01", "2023-06-02", "Europe/Amsterdam")


def test_duplicate_hour_fails_loudly() -> None:
    idx = _complete_hourly_index_utc("2023-06-01", "2023-06-01")
    duplicated = idx.insert(3, idx[3])

    with pytest.raises(CoverageError, match="Duplicate timestamps"):
        check_calendar_coverage(duplicated, "2023-06-01", "2023-06-01", "Europe/Amsterdam")


def test_no_silent_fill_on_gap() -> None:
    """The loader must never return a report for incomplete data — it must raise."""
    idx = _complete_hourly_index_utc("2023-06-01", "2023-06-01")
    gapped = idx.delete(0)

    with pytest.raises(CoverageError):
        check_calendar_coverage(gapped, "2023-06-01", "2023-06-01", "Europe/Amsterdam")


def test_naive_timestamps_are_treated_as_utc() -> None:
    """ERA5's valid_time coordinate is tz-naive but represents UTC instants."""
    idx = _complete_hourly_index_utc("2023-06-01", "2023-06-01").tz_localize(None)
    result = check_calendar_coverage(idx, "2023-06-01", "2023-06-01", "Europe/Amsterdam")
    assert result["no_gaps"] is True


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
