"""ENTSO-E day-ahead electricity price ingestion (Phase A3).

Fetches real day-ahead prices from the ENTSO-E Transparency Platform for the
configured bidding zone and writes a committed provenance record, mirroring
`data/era5.py`'s discipline exactly: fail-loud on incomplete coverage, a
committed provenance record per fetch, and no live fetch executed until a
credential is actually configured.

Requires a free ENTSO-E Transparency Platform account and a Web API security
token, requested from within the platform's account settings (this is a
separate credential from the CDS one used for ERA5 — different provider,
different registration flow). Set it as the `ENTSOE_API_KEY` environment
variable; never put it in `config.yaml` or commit it. The `entsoe-py` package
is only imported inside the functions that need it, matching `data/era5.py`.

STATUS: implemented and unit-tested (hermetically — no live network calls in
tests), but no live ENTSO-E fetch has been run against this code. Deferred
during Phase A per REVAMP_PLAN.md's own fallback ("if the API proves
awkward, keep synthetic but say so clearly and make the interface ready for
real data") — this module is that interface, built the same way the ERA5 one
was before real CDS credentials existed. `pynexus solve` still uses
synthetic prices only; wiring this into the CLI is left for when the
ENTSOE_API_KEY is actually available to test against, the same gate ERA5
was behind before you set up ~/.cdsapirc.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from data.era5 import CoverageError
from data.provenance import ProvenanceRecord, build_provenance_record, write_provenance_record

ENTSOE_SOURCE = "ENTSO-E Transparency Platform (day-ahead prices, A44)"

# Netherlands bidding zone. This repo's configured site (52.5N, 3.5E) is
# Dutch North Sea, so day-ahead price should be the NL zone, not a
# generic/German/UK price — a wrong zone would be a silently wrong price
# series, not an obviously broken one.
DEFAULT_BIDDING_ZONE = "NL"


def check_local_day_coverage(
    timestamps: pd.DatetimeIndex, start_date: str, end_date: str, timezone_name: str
) -> dict[str, Any]:
    """Local-day-aligned coverage check for ENTSO-E's local-market-day fetches.

    `data.era5.check_calendar_coverage` is UTC-day-aligned by design — that
    is correct for ERA5 (a UTC-native dataset) but was tried here first and
    is wrong: converting a local day's boundary to UTC and truncating to a
    UTC date string silently drops the fractional-day offset whenever the
    local timezone isn't UTC (e.g. Amsterdam's +1/+2), so a genuinely
    complete local-day fetch gets compared against the wrong UTC window and
    reports hours "missing" that were never requested. Caught by a test
    using a fake client returning a real, complete 48-hour series — the
    same class of local/UTC boundary bug already fixed once in
    `data.era5.check_calendar_coverage`, reintroduced here by reusing that
    function outside the UTC-day assumption it depends on.

    Builds `expected` directly in `timezone_name` (local midnight to local
    midnight, matching what was actually requested from ENTSO-E) rather
    than going through any UTC conversion at all.
    """
    expected = pd.date_range(
        start=pd.Timestamp(start_date, tz=timezone_name),
        end=pd.Timestamp(end_date, tz=timezone_name) + pd.Timedelta(hours=23),
        freq="1h",
    )
    actual = pd.DatetimeIndex(timestamps)
    actual = actual.tz_localize(timezone_name) if actual.tz is None else actual.tz_convert(timezone_name)

    duplicates = actual[actual.duplicated()]
    if len(duplicates) > 0:
        raise CoverageError(
            f"Duplicate timestamps in fetched series: {list(duplicates.astype(str))[:10]}"
            + (" ..." if len(duplicates) > 10 else "")
        )

    missing = expected.difference(actual)
    if len(missing) > 0:
        raise CoverageError(
            f"Incomplete ENTSO-E coverage: {len(missing)} of {len(expected)} expected hours "
            f"missing. First missing: {list(missing.astype(str)[:10])}"
            + (" ..." if len(missing) > 10 else "")
        )

    return {
        "no_gaps": True,
        "no_duplicates": True,
        "expected_hours": int(len(expected)),
        "actual_hours": int(len(actual)),
    }


def _get_api_key() -> str:
    key = os.environ.get("ENTSOE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ENTSOE_API_KEY environment variable is not set. Register at "
            "https://transparency.entsoe.eu and request a Web API security "
            "token from your account settings (see docs/data_provenance.md)."
        )
    return key


def fetch_entsoe_prices(
    bidding_zone: str,
    start_date: str,
    end_date: str,
    timezone_name: str,
) -> tuple[pd.Series, dict[str, Any]]:
    """Fetch real day-ahead prices for one bidding zone and date range.

    `start_date`/`end_date` are local-timezone calendar days (unlike ERA5's
    UTC convention) because ENTSO-E's day-ahead market itself operates on
    local gate-closure days for the bidding zone, not UTC — using UTC days
    here would misalign with what "day-ahead" actually means for this
    market. Returns (price_series_eur_per_mwh, request_metadata).
    """
    try:
        from entsoe import EntsoePandasClient
    except ImportError as exc:
        raise ImportError(
            "fetch_entsoe_prices requires the 'entsoe-py' package: pip install -e '.[data]'"
        ) from exc

    api_key = _get_api_key()
    try:
        client = EntsoePandasClient(api_key=api_key)
    except Exception as exc:
        raise RuntimeError(f"ENTSO-E client could not be created: {exc}") from exc

    start = pd.Timestamp(start_date, tz=timezone_name)
    end = pd.Timestamp(end_date, tz=timezone_name) + pd.Timedelta(days=1)

    try:
        prices = client.query_day_ahead_prices(bidding_zone, start=start, end=end)
    except Exception as exc:
        raise RuntimeError(
            f"ENTSO-E query failed for {bidding_zone} {start_date}..{end_date}: {exc}"
        ) from exc

    request_metadata = {
        "bidding_zone": bidding_zone,
        "document_type": "A44",
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    return prices, request_metadata


def fetch_and_load_entsoe(
    cfg: dict[str, Any],
    start_date: str,
    end_date: str,
    cache_dir: Path = Path("data/raw"),
    provenance_dir: Path = Path("data/provenance_records"),
    bidding_zone: str = DEFAULT_BIDDING_ZONE,
) -> tuple[pd.DataFrame, ProvenanceRecord]:
    """Full pipeline: fetch from ENTSO-E, cache raw, check coverage, record provenance.

    Mirrors `data.era5.fetch_and_load`, including caching the raw fetched
    series to `cache_dir` (gitignored, same as ERA5's raw NetCDF files) and
    hashing that actual cached file for the provenance record — the record's
    SHA-256 must correspond to real fetched data, not a placeholder.
    Raises CoverageError (fail-loud, no interpolation) if the returned
    series does not fully cover the requested range at hourly resolution.
    """
    timezone_name = cfg["simulation"]["timezone"]
    prices, request_metadata = fetch_entsoe_prices(
        bidding_zone, start_date, end_date, timezone_name
    )

    df = pd.DataFrame({"electricity_price_eur_mwh": prices.to_numpy()}, index=prices.index)

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_file = cache_dir / f"entsoe_{bidding_zone}_{start_date}_{end_date}.csv"
    df.to_csv(raw_file)

    calendar_check = check_local_day_coverage(df.index, start_date, end_date, timezone_name)

    record = build_provenance_record(
        source=ENTSOE_SOURCE,
        variables=["day_ahead_price"],
        requested_latitude=cfg["location"]["latitude"],
        requested_longitude=cfg["location"]["longitude"],
        actual_latitude=cfg["location"]["latitude"],
        actual_longitude=cfg["location"]["longitude"],
        start_date=start_date,
        end_date=end_date,
        timezone_name=timezone_name,
        cds_request=request_metadata,
        raw_file=raw_file,
        expected_row_count=calendar_check["expected_hours"],
        actual_row_count=len(df),
        calendar_check=calendar_check,
        extra={"bidding_zone": bidding_zone},
    )
    provenance_path = provenance_dir / f"entsoe_{bidding_zone}_{start_date}_{end_date}.json"
    write_provenance_record(record, provenance_path)

    return df, record


__all__ = [
    "CoverageError",
    "DEFAULT_BIDDING_ZONE",
    "ENTSOE_SOURCE",
    "check_local_day_coverage",
    "fetch_and_load_entsoe",
    "fetch_entsoe_prices",
]
