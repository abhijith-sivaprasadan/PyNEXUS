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
GAP_2023_UTC = pd.Timestamp("2023-12-29T23:00:00Z")


def recover_from_independent_a44_query(annual_prices, targeted_prices):
    """Accept only a second, matching ENTSO-E observation for the one known gap.

    A wider yearly and an independently requested local-day A44 response must
    agree on every overlapping hour. No interpolation or forward filling.
    """
    for label, series in (("annual", annual_prices), ("targeted", targeted_prices)):
        if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f"{label} A44 source must be a datetime-indexed Series")
        if series.index.tz is None or series.index.has_duplicates:
            raise ValueError(f"{label} A44 source timestamps must be aware and unique")
        if not series.index.is_monotonic_increasing:
            raise ValueError(f"{label} A44 source timestamps must be sorted")

    def to_utc_year(prices):
        utc = prices.index.tz_convert("UTC")
        keep = (utc >= START) & (utc < END)
        data = pd.Series(
            pd.to_numeric(prices.iloc[np.flatnonzero(keep)], errors="raise").to_numpy(dtype=float),
            index=utc[keep],
        )
        if not np.isfinite(data.to_numpy()).all():
            raise ValueError("A44 requery contained nonfinite price observations")
        if len(data.index.difference(EXPECTED)):
            raise ValueError("A44 requery contained off-hour observations")
        return data

    annual = to_utc_year(annual_prices)
    candidate = to_utc_year(targeted_prices)
    missing = EXPECTED.difference(annual.index)
    if len(missing) != 1 or missing[0] != GAP_2023_UTC:
        raise ValueError("Recovery permits only the registered 2023-12-29 23:00 UTC gap")
    if GAP_2023_UTC not in candidate.index:
        raise ValueError("Targeted independent ENTSO-E A44 query did not return the missing hour")
    # Two immediate adjacent observations are mandatory, as well as at
    # least 12 matching independent price observations overall.
    neighbors = (GAP_2023_UTC - pd.Timedelta(hours=1), GAP_2023_UTC + pd.Timedelta(hours=1))
    if any(ts not in candidate.index or ts not in annual.index for ts in neighbors):
        raise ValueError("Targeted A44 query omitted a neighboring control hour")
    overlap = candidate.index.intersection(annual.index)
    if len(overlap) < 12:
        raise ValueError("Targeted A44 query has too few independent overlap controls")
    differences = abs(candidate.loc[overlap].to_numpy() - annual.loc[overlap].to_numpy())
    largest_difference = float(differences.max())
    if not np.all(differences <= 1e-6):
        raise ValueError(
            "Original and targeted official A44 responses disagree on overlapping prices"
        )
    merged = pd.concat([annual, candidate.loc[[GAP_2023_UTC]]]).sort_index()
    normalise_hourly_prices(merged)  # independent strict 8,760-hour check
    audit = {
        "status": "RECOVERED_EXPLICITLY_OBSERVED_A44_PRICE_NOT_INTERPOLATED",
        "method": "SECOND_OFFICIAL_NL_A44_QUERY_WITH_FULL_OVERLAP_CONSISTENCY_CHECK",
        "recovered_utc_timestamp": GAP_2023_UTC.isoformat(),
        "recovered_local_timestamp": GAP_2023_UTC.tz_convert("Europe/Amsterdam").isoformat(),
        "recovered_price_eur_per_mwh": float(candidate.loc[GAP_2023_UTC]),
        "recovered_observation_count": 1,
        "independent_overlap_hour_count": int(len(overlap)),
        "maximum_overlap_price_difference_eur_mwh": largest_difference,
        "price_interpolation_or_forward_fill": False,
    }
    return merged, audit


def price_coverage_diagnostics(prices):
    """Explain a rejected provider calendar without assuming or repairing missing data.

    Contains UTC timestamps and counts only, never the token, request URLs,
    or non-public provider exception messages. The result is NOT accepted data.
    """
    if not isinstance(prices, pd.Series) or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Provider diagnostics require a datetime-indexed Series")
    if prices.index.tz is None:
        raise ValueError("Provider diagnostics require timezone-aware timestamps")
    utc = prices.index.tz_convert("UTC")
    inside = utc[(utc >= START) & (utc < END)]
    unique = inside.unique().sort_values()
    missing = EXPECTED.difference(unique)
    unexpected = unique.difference(EXPECTED)
    # Pandas 2/3 may store datetime64 at different internal resolutions.
    # Timedelta arithmetic is unit-aware, unlike dividing .asi8 by nanoseconds.
    deltas = pd.Series(inside).diff().dropna().dt.total_seconds().to_numpy() / 60
    counts = pd.Series(deltas).value_counts().sort_index()
    within_year = prices.iloc[np.flatnonzero((utc >= START) & (utc < END))]
    numerical = pd.to_numeric(within_year, errors="coerce").to_numpy(dtype=float)

    def sample(stamps):
        return [stamp.isoformat() for stamp in stamps[:8]]

    return {
        "status": "REJECTED_PROVIDER_CALENDAR_DIAGNOSTIC_NOT_VERIFIED_DATA",
        "expected_hour_count_2023_utc": len(EXPECTED),
        "provider_rows_total_including_bracketing_days": len(prices),
        "provider_timezone": str(prices.index.tz),
        "provider_first_timestamp_utc": utc[0].isoformat() if len(utc) else None,
        "provider_last_timestamp_utc": utc[-1].isoformat() if len(utc) else None,
        "in_year_rows": len(inside),
        "in_year_unique_timestamps": len(unique),
        "in_year_duplicate_count": len(inside) - len(unique),
        "missing_exact_hour_count": len(missing),
        "first_missing_exact_hours_utc": sample(missing),
        "unexpected_in_year_timestamp_count": len(unexpected),
        "first_unexpected_timestamps_utc": sample(unexpected),
        "first_in_year_timestamps_utc": sample(inside),
        "last_in_year_timestamps_utc": sample(inside[-8:]),
        "successive_interval_minutes_counts": {
            str(float(interval)): int(count) for interval, count in counts.items()
        },
        "nonfinite_or_nonnumeric_in_year_prices": int((~np.isfinite(numerical)).sum()),
        "negative_in_year_price_count": int((numerical < 0).sum()),
        "provider_raw_xml_archived": False,
        "imputation_or_resampling_performed": False,
    }


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
    values = pd.to_numeric(prices.iloc[np.flatnonzero(keep)], errors="raise").to_numpy(dtype=float)
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


def write_snapshot(output, prices, *, metadata, targeted_evidence=None, recovery_audit=None):
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
    evidence_hash = None
    if targeted_evidence is not None:
        if recovery_audit is None:
            raise ValueError("Additional A44 source evidence requires its verification audit")
        evidence_index = targeted_evidence.index.tz_convert("UTC")
        extra = pd.DataFrame(
            {
                TIME_COLUMN: evidence_index.strftime("%Y-%m-%dT%H:%M:%SZ"),
                PRICE_COLUMN: targeted_evidence.to_numpy(dtype=float),
            }
        )
        evidence_path = output / "independent_a44_targeted_observations.csv"
        extra.to_csv(evidence_path, index=False)
        evidence_hash = sha256(evidence_path)
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
        "verified_source_gap_recovery": (
            {
                **recovery_audit,
                "independent_targeted_observations_sha256": evidence_hash,
            }
            if recovery_audit is not None
            else None
        ),
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


def fetch_official_snapshot(output, *, diagnostics_dir=None):
    # Bracket the UTC 2023 year using complete Dutch local market dates.
    series, metadata = fetch_entsoe_prices("NL", "2022-12-31", "2024-01-01", "Europe/Amsterdam")
    try:
        normalise_hourly_prices(series)
    except ValueError:
        diagnosis = price_coverage_diagnostics(series)
        # The first real 2023 A44 run revealed one exact missing local midnight
        # (2023-12-30 00:00 CET). Only attempt an independent source requery
        # for this known gap, never invent a price.
        if (
            diagnosis["missing_exact_hour_count"] == 1
            and diagnosis["first_missing_exact_hours_utc"] == [GAP_2023_UTC.isoformat()]
            and diagnosis["unexpected_in_year_timestamp_count"] == 0
            and diagnosis["in_year_duplicate_count"] == 0
        ):
            try:
                targeted, targeted_request = fetch_entsoe_prices(
                    "NL", "2023-12-29", "2023-12-31", "Europe/Amsterdam"
                )
                complete, recovery = recover_from_independent_a44_query(series, targeted)
                print(
                    "Official targeted A44 requery recovered an observed hour; "
                    f"cross-checked {recovery['independent_overlap_hour_count']} overlapping "
                    "observations; no interpolation or assumed prices.",
                    flush=True,
                )
                return write_snapshot(
                    output,
                    complete,
                    metadata={**metadata, "supplementary_a44_request": targeted_request},
                    targeted_evidence=targeted,
                    recovery_audit=recovery,
                )
            except (RuntimeError, ValueError) as exc:
                # No API-token-bearing upstream error strings reach the log.
                diagnosis["independent_a44_requery_outcome"] = (
                    f"REJECTED_{type(exc).__name__}; no market price was inserted"
                )
        print("Rejected ENTSO-E A44 2023 calendar:", json.dumps(diagnosis), flush=True)
        if diagnostics_dir is not None:
            diagnostics_dir = Path(diagnostics_dir)
            diagnostics_dir.mkdir(parents=True, exist_ok=False)
            (diagnostics_dir / "coverage_diagnostics.json").write_text(
                json.dumps(diagnosis, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        raise
    return write_snapshot(output, series, metadata=metadata)


def select_market_cases(choices, scenario):
    """Preserve the registered 'all' combined intervention as one case.

    The explicit 'all-cases' token is the only batch-mode selector.
    """
    if scenario == "all-cases":
        return choices
    selected = [item for item in choices if item[0] == scenario]
    if len(selected) != 1:
        raise ValueError(f"Unregistered scenario: {scenario!r}")
    return selected


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
    selected = select_market_cases(choices, scenario)
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
            record["metrics"]["stage_b_grid_settlement_eur_OBSERVED_PRICE_INPUT"] = record[
                "metrics"
            ].pop("stage_b_grid_settlement_eur_synthetic_prices")
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
    fetch.add_argument("--diagnostics-dir", type=Path)
    run = subs.add_parser("dispatch", help="Re-solve one case, or all with --scenario all-cases")
    run.add_argument("--snapshot", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--scenario", default="baseline")
    run.add_argument("--hours", type=int, default=8760)
    run.add_argument("--time-limit", type=int, default=600)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        result = fetch_official_snapshot(args.output, diagnostics_dir=args.diagnostics_dir)
    else:
        result = run_market_case(
            args.snapshot,
            args.output,
            scenario=args.scenario,
            hours=args.hours,
            time_limit=args.time_limit,
        )
    print(json.dumps({key: result[key] for key in ("status", "schema_version")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
