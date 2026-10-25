"""Fail-closed UTC and provenance tests for WP5B; no remote requests."""

import copy

import numpy as np
import pandas as pd
import pytest

from research.wp5_market import (
    END,
    EXPECTED,
    START,
    TIME_COLUMN,
    load_snapshot,
    normalise_hourly_prices,
    parse_snapshot_csv,
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
