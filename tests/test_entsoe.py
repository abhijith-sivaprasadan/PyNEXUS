"""Hermetic tests for data/entsoe.py — no live ENTSO-E network calls.

Mirrors tests/test_era5.py's approach to the equivalent CDS failure paths:
monkeypatch the optional dependency and the client construction/query, and
verify errors surface as clear, typed exceptions rather than bare tracebacks.
"""

import sys

import pandas as pd
import pytest

from data.entsoe import (
    DEFAULT_BIDDING_ZONE,
    _get_api_key,
    check_local_day_coverage,
    fetch_entsoe_prices,
)
from data.era5 import CoverageError as Era5CoverageError


def test_local_day_coverage_passes_for_a_complete_local_day() -> None:
    """A day fully covered in LOCAL time must pass even though, converted to
    UTC, its boundary sits at 23:00 the previous day (Amsterdam is UTC+1 in
    January) — this is exactly the boundary a UTC-day-only check gets wrong."""
    idx = pd.date_range(
        start=pd.Timestamp("2023-01-01", tz="Europe/Amsterdam"), periods=24, freq="1h"
    )
    result = check_local_day_coverage(idx, "2023-01-01", "2023-01-01", "Europe/Amsterdam")
    assert result["no_gaps"] is True
    assert result["expected_hours"] == result["actual_hours"] == 24


def test_local_day_coverage_fails_on_a_genuine_gap() -> None:
    idx = pd.date_range(
        start=pd.Timestamp("2023-01-01", tz="Europe/Amsterdam"), periods=24, freq="1h"
    ).delete(5)
    with pytest.raises(Era5CoverageError, match="Incomplete ENTSO-E coverage"):
        check_local_day_coverage(idx, "2023-01-01", "2023-01-01", "Europe/Amsterdam")


def test_get_api_key_raises_clear_error_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ENTSOE_API_KEY"):
        _get_api_key()


def test_get_api_key_rejects_blank_value(monkeypatch) -> None:
    monkeypatch.setenv("ENTSOE_API_KEY", "   ")
    with pytest.raises(RuntimeError, match="ENTSOE_API_KEY"):
        _get_api_key()


def test_get_api_key_returns_stripped_value(monkeypatch) -> None:
    monkeypatch.setenv("ENTSOE_API_KEY", "  abc123  ")
    assert _get_api_key() == "abc123"


def test_fetch_missing_entsoe_package_raises_clear_import_error(monkeypatch) -> None:
    monkeypatch.setenv("ENTSOE_API_KEY", "fake-key-for-this-test")
    monkeypatch.setitem(sys.modules, "entsoe", None)  # forces `from entsoe import ...` to raise

    with pytest.raises(ImportError, match=r"pip install -e '\.\[data\]'"):
        fetch_entsoe_prices(DEFAULT_BIDDING_ZONE, "2023-01-01", "2023-01-02", "Europe/Amsterdam")


def test_fetch_missing_api_key_fails_before_any_network_call(monkeypatch) -> None:
    pytest.importorskip("entsoe")  # only meaningful with the optional [data] extra
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ENTSOE_API_KEY"):
        fetch_entsoe_prices(DEFAULT_BIDDING_ZONE, "2023-01-01", "2023-01-02", "Europe/Amsterdam")


def test_fetch_client_construction_failure_raises_clear_runtime_error(monkeypatch) -> None:
    entsoe = pytest.importorskip("entsoe")
    monkeypatch.setenv("ENTSOE_API_KEY", "fake-key-for-this-test")

    def _raise(*args, **kwargs):
        raise Exception("boom")

    monkeypatch.setattr(entsoe, "EntsoePandasClient", _raise)

    with pytest.raises(RuntimeError, match="ENTSO-E client could not be created"):
        fetch_entsoe_prices(DEFAULT_BIDDING_ZONE, "2023-01-01", "2023-01-02", "Europe/Amsterdam")


def test_fetch_query_failure_raises_clear_runtime_error(monkeypatch) -> None:
    entsoe = pytest.importorskip("entsoe")
    monkeypatch.setenv("ENTSOE_API_KEY", "fake-key-for-this-test")

    class FakeClient:
        def __init__(self, api_key):
            pass

        def query_day_ahead_prices(self, *args, **kwargs):
            raise Exception("upstream 503")

    monkeypatch.setattr(entsoe, "EntsoePandasClient", FakeClient)

    with pytest.raises(RuntimeError, match="ENTSO-E query failed"):
        fetch_entsoe_prices(DEFAULT_BIDDING_ZONE, "2023-01-01", "2023-01-02", "Europe/Amsterdam")


def test_fetch_succeeds_with_a_fake_client(monkeypatch, tmp_path) -> None:
    """End-to-end through fetch_and_load_entsoe with a fake client returning
    a real, complete hourly series — checks provenance/caching/coverage
    wiring without ever touching the network."""
    import pandas as pd

    entsoe = pytest.importorskip("entsoe")
    monkeypatch.setenv("ENTSOE_API_KEY", "fake-key-for-this-test")

    idx = pd.date_range(
        start=pd.Timestamp("2023-01-01", tz="Europe/Amsterdam"),
        periods=48,
        freq="1h",
    )
    fake_prices = pd.Series(50.0, index=idx)

    class FakeClient:
        def __init__(self, api_key):
            pass

        def query_day_ahead_prices(self, country_code, start, end):
            return fake_prices

    monkeypatch.setattr(entsoe, "EntsoePandasClient", FakeClient)

    from data.entsoe import fetch_and_load_entsoe

    cfg = {
        "simulation": {"timezone": "Europe/Amsterdam"},
        "location": {"latitude": 52.5, "longitude": 3.5},
    }
    df, record = fetch_and_load_entsoe(
        cfg,
        "2023-01-01",
        "2023-01-02",
        cache_dir=tmp_path / "raw",
        provenance_dir=tmp_path / "provenance",
    )
    assert len(df) == 48
    assert record.source.startswith("ENTSO-E")
    assert (tmp_path / "raw" / "entsoe_NL_2023-01-01_2023-01-02.csv").exists()
    assert (tmp_path / "provenance" / "entsoe_NL_2023-01-01_2023-01-02.json").exists()
