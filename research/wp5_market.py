"""WP5B: strict UTC price snapshots and independent market-price redispatch.

Run with an ENTSO-E API token to obtain a processed A44 series; no observed
electricity data are bundled. Raw ENTSO-E XML is NOT retained by this adapter.
A price-informed dispatch does NOT constitute a full technoeconomic LCOH.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.entsoe import fetch_entsoe_prices
from research.interventions import PROTOCOL, evaluate_case, registered_cases, source_inputs
from research.run_reference import CONFIG, INPUTS, sha256

START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2024-01-01T00:00:00Z")
EXPECTED = pd.date_range(START, END, freq="h", inclusive="left")
SCHEMA = "pynexus.wp5.nl_day_ahead_hourly_utc.v1"
PRICE_COLUMN = "price_eur_mwh"
TIME_COLUMN = "timestamp_utc"


def normalise_hourly_prices(prices):
    """Require an exact chronological 8,760-hour UTC grid; no silent resampling."""
    if not isinstance(prices, pd.Series) or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Expected pandas Series with DatetimeIndex")
    if prices.index.tz is None:
        raise ValueError("Naive timestamps are forbidden: supply explicit timezone")
    utc = prices.index.tz_convert("UTC")
    if utc.has_duplicates:
        raise ValueError("Duplicate market timestamps")
    if not utc.is_monotonic_increasing:
        raise ValueError("Market timestamps must be monotonically increasing")
    # ENTSO-E requests include bracketing local market days to capture the
    # 2023 UTC year. Only that precise UTC interval is admissible downstream.
    keep = (utc >= START) & (utc < END)
    clipped_index = utc[keep]
    values = pd.to_numeric(prices.iloc[np.flatnonzero(keep)], errors="raise").to_numpy(
        dtype=float
    )
    if not clipped_index.equals(EXPECTED):
        raise ValueError(
            "Missing/extra or nonhourly records in 2023 UTC; "
            "no hourly interpolation or implicit quarter-hour aggregation"
        )
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite day-ahead price")
    return pd.DataFrame(
        {
            TIME_COLUMN: EXPECTED.strftime("%Y-%m-%dT%H:%M:%SZ"),
            PRICE_COLUMN: values,
        }
    )


def parse_snapshot_csv(path):
    """Reject ambiguous time formats, duplicates and tampered or incomplete data."""
    frame = pd.read_csv(path, dtype={TIME_COLUMN: str})
    if list(frame.columns) != [TIME_COLUMN, PRICE_COLUMN] or len(frame) != len(EXPECTED):
        raise ValueError("Snapshot must have exact two-column, 8,760-hour schema")
    stamps = frame[TIME_COLUMN]
    if not stamps.str.fullmatch(r"2023-\d\d-\d\dT\d\d:00:00Z").all():
        raise ValueError("Expected explicit 2023 ISO UTC hourly timestamps with Z")
    parsed = pd.to_datetime(stamps, utc=True, errors="raise")
    prices = pd.Series(
        pd.to_numeric(frame[PRICE_COLUMN], errors="raise").to_numpy(dtype=float),
        index=pd.DatetimeIndex(parsed),
    )
    return normalise_hourly_prices(prices)


def write_snapshot(output, prices, *, metadata):
    """Write a new immutable processed-price archive, hash-lock its CSV."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Do not overwrite an existing market evidence archive")
    normalised = normalise_hourly_prices(prices)
    if metadata.get("bidding_zone") != "NL" or metadata.get("document_type") != "A44":
        raise ValueError("Only Netherlands day-ahead A44 prices are permitted")
    output.mkdir(parents=True, exist_ok=False)
    target = output / "prices.csv"
    normalised.to_csv(target, index=False)
    manifest = {
        "schema_version": SCHEMA,
        "status": "PROCESSED_PROVIDER_TIMESERIES_NOT_RAW_XML",
        "source_claim": "ENTSOE_TRANSPARENCY_A44_RETURNED_BY_ENTSOE_PY",
        "provider_response_xml_archived": False,
        "bidding_zone": "NL",
        "document_type": "A44",
        "currency": "EUR",
        "price_unit": "EUR/MWh",
        "resolution_minutes": 60,
        "timezone": "UTC",
        "year": 2023,
        "row_count": 8760,
        "start_utc": normalised[TIME_COLUMN].iloc[0],
        "end_utc": normalised[TIME_COLUMN].iloc[-1],
        "negative_price_hours": int((normalised[PRICE_COLUMN] < 0).sum()),
        "csv_sha256": sha256(target),
        "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "provider_request": metadata,
        "limitations": [
            "ENTSOE PandasClient-parsed provider response; original A44 XML not archived",
            "Not an invoice, PPA, electricity purchase cost or carbon-intensity record",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_snapshot(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    mandatory = {
        "schema_version": SCHEMA,
        "status": "PROCESSED_PROVIDER_TIMESERIES_NOT_RAW_XML",
        "bidding_zone": "NL",
        "document_type": "A44",
        "currency": "EUR",
        "price_unit": "EUR/MWh",
        "resolution_minutes": 60,
        "timezone": "UTC",
        "year": 2023,
        "row_count": 8760,
        "provider_response_xml_archived": False,
    }
    for name, value in mandatory.items():
        if manifest.get(name) != value:
            raise ValueError(f"Snapshot metadata mismatch: {name}")
    file = directory / "prices.csv"
    if manifest.get("csv_sha256") != sha256(file):
        raise ValueError("Market snapshot SHA256 mismatch")
    result = parse_snapshot_csv(file)
    if int((result[PRICE_COLUMN] < 0).sum()) != manifest.get("negative_price_hours"):
        raise ValueError("Negative-price count mismatch")
    return result, manifest


def fetch_official_snapshot(output):
    # Local market request deliberately brackets the 2023 UTC year. Query
    # includes extra local days at both ends so time-zone offsets cannot
    # silently omit the 2023-12-31 23:00 UTC interval.
    series, metadata = fetch_entsoe_prices(
        "NL", "2022-12-31", "2024-01-01", "Europe/Amsterdam"
    )
    return write_snapshot(output, series, metadata=metadata)


def run_market_case(snapshot, output, *, scenario="baseline", hours=8760, time_limit=600):
    """Re-solve Stage A + Stage B from original wind, using observed-price input.

    For smoke horizons shorter than 8,760 hours the output explicitly cannot
    be presented as the annual real-price comparison.
    """
    if output.exists():
        raise FileExistsError("Research output directory already exists")
    if isinstance(hours, bool) or not isinstance(hours, int) or not 1 <= hours <= 8760:
        raise ValueError("Hours must be integer in 1..8760")
    prices, price_manifest = load_snapshot(snapshot)
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    choices = registered_cases(protocol["order"])
    selected = choices if scenario == "all" else [x for x in choices if x[0] == scenario]
    if not selected:
        raise ValueError("Unregistered scenario")
    wind_data, base_cfg, verified_mode = source_inputs(protocol, hours=hours)
    if base_cfg["simulation"]["start_date"] != "2023-01-01":
        raise ValueError("Original wind data start date does not match UTC price year")
    source = wind_data.copy()
    source["electricity_price"] = prices[PRICE_COLUMN].to_numpy(dtype=float)[:hours]
    output.mkdir(parents=True, exist_ok=False)
    results = {
        "schema_version": "pynexus.wp5.market_dispatch.v1",
        "status": "INCOMPLETE",
        "price_kind": "PROCESSED_ENTSOE_2023_NL_DAY_AHEAD_NOT_INVOICE",
        "provider_raw_xml_archived": False,
        "cost_claim": "NONE_GRID_SETTLEMENT_ONLY",
        "year": 2023,
        "hours": hours,
        "full_year": hours == 8760,
        "source_wind_sha256": sha256(INPUTS),
        "config_sha256": sha256(CONFIG),
        "price_csv_sha256": price_manifest["csv_sha256"],
        "price_manifest_sha256": sha256(Path(snapshot) / "manifest.json"),
        "original_2023_wind_provenance_mode": verified_mode,
        "hourly_alignment_assumption": (
            "Archival ERA5 2023 wind input row 0 is interpreted as "
            "2023-01-01 00:00 UTC; compare source manifest before publication."
        ),
        "limitations": [
            "No costs beyond net grid settlement; not LCOH",
            "Prices are processed and original provider XML has not been archived",
            "No explicit antisimultaneous buy/sell constraint in original solver",
            "No tariffs, purchase contracts, hydrogen certification or emissions attribution",
        ],
        "cases": [],
    }
    for item in selected:
        try:
            record = evaluate_case(
                item, base_cfg, source, protocol, output / item[0], time_limit=time_limit
            )
            record["metrics"]["stage_b_grid_settlement_eur_OBSERVED_PRICE_INPUT"] = (
                record["metrics"].pop("stage_b_grid_settlement_eur_synthetic_prices")
            )
            record["price_csv_sha256"] = price_manifest["csv_sha256"]
            record["price_kind"] = results["price_kind"]
            stage_b = pd.read_csv(output / item[0] / "stage_b.csv")
            overlap = np.minimum(
                stage_b["grid_import_mw"].to_numpy(dtype=float),
                stage_b["grid_export_mw"].to_numpy(dtype=float),
            )
            if np.any(overlap > 1e-6):
                raise ValueError(
                    "Grid import/export simultaneous: solver formulation needs "
                    "anti-circulation modification before economic interpretation"
                )
            (output / item[0] / "record.json").write_text(
                json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            results["cases"].append(record)
        except Exception:
            (output / "manifest.json").write_text(
                json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            raise
    results["status"] = (
        "VERIFIED_2023_DISPATCH_PROCESSED_MARKET_PRICE_INPUT"
        if hours == 8760
        else "VERIFIED_PARTIAL_SMOKE_ONLY_NOT_ANNUAL_RESULT"
    )
    (output / "manifest.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    fetch = subs.add_parser("fetch", help="Fetch NL 2023 A44 using ENTSOE_API_KEY")
    fetch.add_argument("--output", type=Path, required=True)
    run = subs.add_parser("dispatch", help="Re-solve one or all registered cases")
    run.add_argument("--snapshot", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--scenario", default="baseline")
    run.add_argument("--hours", type=int, default=8760)
    run.add_argument("--time-limit", type=int, default=600)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        result = fetch_official_snapshot(args.output)
    else:
        result = run_market_case(
            args.snapshot, args.output, scenario=args.scenario,
            hours=args.hours, time_limit=args.time_limit,
        )
    print(json.dumps({
        key: result[key] for key in ("status", "schema_version")
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
