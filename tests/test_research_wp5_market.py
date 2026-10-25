"""Fail-closed UTC and provenance tests for WP5B; no remote requests."""

import copy

import numpy as np
import pandas as pd
import pytest

from research.wp5_market import (
    END,
    EXPECTED,
    GAP_2023_UTC,
    START,
    TIME_COLUMN,
    fetch_official_snapshot,
    load_snapshot,
    normalise_hourly_prices,
    parse_snapshot_csv,
    price_coverage_diagnostics,
    recover_from_independent_a44_query,
    select_market_cases,
    write_snapshot,
)

META = {"bidding_zone": "NL", "document_type": "A44", "request": "UNIT_TEST_ONLY"}


def synthetic_utc_prices():
    return pd.Series(np.linspace(-50, 400, len(EXPECTED)), index=EXPECTED)


def test_exact_utc_year_negative_prices_and_hash(tmp_path):
    root = tmp_path / "archive"
    m = write_snapshot(root, synthetic_utc_prices(), metadata=META)
    frame, loaded = load_snapshot(root)
    assert len(frame) == 8760
    assert frame[TIME_COLUMN].iloc[0] == "2023-01-01T00:00:00Z"
    assert frame[TIME_COLUMN].iloc[-1] == "2023-12-31T23:00:00Z"
    assert m["negative_price_hours"] > 0
    assert loaded["csv_sha256"] == m["csv_sha256"]
    assert m["provider_response_xml_archived"] is False


def test_amsterdam_dst_and_utc_bracketing(tmp_path):
    # Amsterdam calendar has 23- and 25-hour days. These are nevertheless
    # 8,760 unique UTC market hours when the UTC year is bracketed correctly.
    local = pd.date_range(
        START - pd.Timedelta(days=2),
        END + pd.Timedelta(days=2),
        freq="h",
        inclusive="left",
    ).tz_convert("Europe/Amsterdam")
    series = pd.Series(np.ones(len(local)), index=local)
    frame = normalise_hourly_prices(series)
    assert len(frame) == 8760
    assert frame[TIME_COLUMN].iloc[0] == "2023-01-01T00:00:00Z"
    assert frame[TIME_COLUMN].iloc[-1] == "2023-12-31T23:00:00Z"


@pytest.mark.parametrize("change", ["drop", "duplicate", "shuffle", "nan", "subhour", "naive"])
def test_reject_incomplete_or_ambiguous_year(change):
    prices = synthetic_utc_prices()
    if change == "drop":
        prices = prices.iloc[:-1]
    elif change == "duplicate":
        prices = pd.concat([prices.iloc[:1], prices])
    elif change == "shuffle":
        prices = prices.iloc[::-1]
    elif change == "nan":
        prices.iloc[1] = np.nan
    elif change == "subhour":
        prices.index = prices.index + pd.Timedelta(minutes=15)
    else:
        prices.index = prices.index.tz_localize(None)
    with pytest.raises(ValueError):
        normalise_hourly_prices(prices)


def test_tampering_detected(tmp_path):
    root = tmp_path / "archive"
    write_snapshot(root, synthetic_utc_prices(), metadata=META)
    path = root / "prices.csv"
    path.write_text(path.read_text().replace("-50.0", "-300.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        load_snapshot(root)


def test_bad_snapshot_metadata_rejected(tmp_path):
    with pytest.raises(ValueError, match="Netherlands"):
        write_snapshot(
            tmp_path / "no",
            synthetic_utc_prices(),
            metadata={**META, "bidding_zone": "DE"},
        )
    root = tmp_path / "archive"
    write_snapshot(root, synthetic_utc_prices(), metadata=META)
    path = root / "manifest.json"
    import json

    m = json.loads(path.read_text())
    m["currency"] = "USD"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="currency"):
        load_snapshot(root)


def test_bad_timestamp_z_rejected_after_sha_validation(tmp_path):
    root = tmp_path / "archive"
    write_snapshot(root, synthetic_utc_prices(), metadata=META)
    csv = root / "prices.csv"
    frame = pd.read_csv(csv)
    frame.loc[0, TIME_COLUMN] = "2023-01-01 00:00"
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="explicit"):
        parse_snapshot_csv(csv)


def test_no_reuse_output(tmp_path):
    root = tmp_path / "archive"
    write_snapshot(root, synthetic_utc_prices(), metadata=META)
    with pytest.raises(FileExistsError):
        write_snapshot(root, synthetic_utc_prices(), metadata=copy.deepcopy(META))


def test_diagnostics_pinpoint_single_missing_utc_hour():
    prices = synthetic_utc_prices().drop(EXPECTED[145])
    diagnosis = price_coverage_diagnostics(prices)
    assert diagnosis["status"].startswith("REJECTED_PROVIDER")
    assert diagnosis["in_year_rows"] == 8759
    assert diagnosis["missing_exact_hour_count"] == 1
    assert diagnosis["first_missing_exact_hours_utc"] == [EXPECTED[145].isoformat()]
    assert diagnosis["unexpected_in_year_timestamp_count"] == 0
    assert diagnosis["nonfinite_or_nonnumeric_in_year_prices"] == 0
    assert diagnosis["imputation_or_resampling_performed"] is False


def test_diagnostics_reveal_fully_quarter_hourly_source():
    quarter_hours = pd.date_range(START, END, freq="15min", inclusive="left")
    prices = pd.Series(40.0, index=quarter_hours)
    diagnosis = price_coverage_diagnostics(prices)
    assert diagnosis["in_year_rows"] == 35040
    assert diagnosis["missing_exact_hour_count"] == 0
    assert diagnosis["unexpected_in_year_timestamp_count"] == 26280
    assert diagnosis["successive_interval_minutes_counts"]["15.0"] == 35039
    with pytest.raises(ValueError, match="nonhourly"):
        normalise_hourly_prices(prices)


def test_rejected_provider_calendar_archives_diagnostics_only(monkeypatch, tmp_path, capsys):
    import json

    import research.wp5_market as market

    prices = synthetic_utc_prices().drop(EXPECTED[300])

    def fake_provider(*args, **kwargs):
        return prices, {"bidding_zone": "NL", "document_type": "A44"}

    monkeypatch.setattr(market, "fetch_entsoe_prices", fake_provider)
    source = tmp_path / "rejected"
    diagnostics_dir = tmp_path / "diagnostics"
    with pytest.raises(ValueError, match="Missing/extra"):
        fetch_official_snapshot(source, diagnostics_dir=diagnostics_dir)
    assert not source.exists()
    result = json.loads((diagnostics_dir / "coverage_diagnostics.json").read_text())
    assert result["missing_exact_hour_count"] == 1
    assert result["first_missing_exact_hours_utc"] == [EXPECTED[300].isoformat()]
    assert "securityToken" not in capsys.readouterr().out
    assert list(diagnostics_dir.iterdir()) == [diagnostics_dir / "coverage_diagnostics.json"]


def test_valid_provider_calendar_writes_only_verified_prices(monkeypatch, tmp_path):
    import research.wp5_market as market

    def fake_provider(*args, **kwargs):
        return synthetic_utc_prices(), META

    monkeypatch.setattr(market, "fetch_entsoe_prices", fake_provider)
    source = tmp_path / "valid"
    diagnostics_dir = tmp_path / "diagnostics"
    result = fetch_official_snapshot(source, diagnostics_dir=diagnostics_dir)
    assert result["row_count"] == 8760
    assert not diagnostics_dir.exists()
    assert (source / "prices.csv").is_file()


def test_independent_a44_source_requery_restores_exact_missing_observation(tmp_path, monkeypatch):
    import research.wp5_market as market

    master = synthetic_utc_prices()
    original = master.drop(GAP_2023_UTC).tz_convert("Europe/Amsterdam")
    targeted = master.loc["2023-12-29":"2023-12-31"].tz_convert("Europe/Amsterdam")
    calls = []

    def fake_provider(zone, start, end, timezone):
        calls.append((zone, start, end, timezone))
        if start == "2022-12-31":
            return original, META
        return targeted, META

    monkeypatch.setattr(market, "fetch_entsoe_prices", fake_provider)
    out = tmp_path / "verified"
    result = fetch_official_snapshot(out, diagnostics_dir=tmp_path / "failure")
    assert len(calls) == 2
    assert calls[1] == ("NL", "2023-12-29", "2023-12-31", "Europe/Amsterdam")
    recovery = result["verified_source_gap_recovery"]
    assert recovery["recovered_observation_count"] == 1
    assert recovery["status"].startswith("RECOVERED_EXPLICITLY_OBSERVED")
    assert recovery["independent_overlap_hour_count"] >= 12
    assert recovery["maximum_overlap_price_difference_eur_mwh"] == 0
    assert recovery["price_interpolation_or_forward_fill"] is False
    assert "supplementary_a44_request" in result["provider_request"]
    assert (out / "independent_a44_targeted_observations.csv").exists()
    assert not (tmp_path / "failure").exists()
    data, verified = load_snapshot(out)
    observed = data.loc[data[TIME_COLUMN] == "2023-12-29T23:00:00Z", "price_eur_mwh"]
    assert len(observed) == 1
    assert observed.iloc[0] == pytest.approx(master.loc[GAP_2023_UTC])
    assert verified["verified_source_gap_recovery"] == recovery | {
        "independent_targeted_observations_sha256": recovery[
            "independent_targeted_observations_sha256"
        ]
    }


def test_independent_requery_rejects_conflicting_overlap():
    source = synthetic_utc_prices()
    annual = source.drop(GAP_2023_UTC)
    targeted = source.loc["2023-12-29":"2023-12-31"].copy()
    targeted.iloc[0] += 0.01
    with pytest.raises(ValueError, match="disagree"):
        recover_from_independent_a44_query(annual, targeted)


def test_independent_requery_rejects_missing_target_hour():
    source = synthetic_utc_prices()
    annual = source.drop(GAP_2023_UTC)
    targeted = source.loc["2023-12-29":"2023-12-31"].drop(GAP_2023_UTC)
    with pytest.raises(ValueError, match="did not return"):
        recover_from_independent_a44_query(annual, targeted)


def test_independent_requery_rejects_gap_elsewhere():
    source = synthetic_utc_prices()
    annual = source.drop(EXPECTED[151])
    targeted = source.loc["2023-12-29":"2023-12-31"]
    with pytest.raises(ValueError, match="registered"):
        recover_from_independent_a44_query(annual, targeted)


def test_failed_official_requery_keeps_research_gate_closed(tmp_path, monkeypatch):
    import json

    import research.wp5_market as market

    master = synthetic_utc_prices()
    original = master.drop(GAP_2023_UTC)
    calls = []

    def fake_provider(zone, start, end, timezone):
        calls.append((start, end))
        if start == "2022-12-31":
            return original, META
        return original.loc["2023-12-29":"2023-12-31"], META

    monkeypatch.setattr(market, "fetch_entsoe_prices", fake_provider)
    with pytest.raises(ValueError, match="Missing/extra"):
        fetch_official_snapshot(
            tmp_path / "must_not_exist", diagnostics_dir=tmp_path / "diagnostics"
        )
    assert len(calls) == 2
    assert not (tmp_path / "must_not_exist").exists()
    diagnosis = json.loads((tmp_path / "diagnostics/coverage_diagnostics.json").read_text())
    assert diagnosis["missing_exact_hour_count"] == 1
    assert diagnosis["independent_a44_requery_outcome"].startswith("REJECTED_ValueError")


def test_combined_all_intervention_is_one_registered_case():
    from research.interventions import registered_cases

    choices = registered_cases(["W", "E", "G", "S", "eta"])
    assert select_market_cases(choices, "all") == [
        ("all", ("W", "E", "G", "S", "eta"))
    ]
    assert select_market_cases(choices, "all-cases") == choices
    assert len(select_market_cases(choices, "all-cases")) == 17
    with pytest.raises(ValueError, match="Unregistered scenario"):
        select_market_cases(choices, "ALL")
