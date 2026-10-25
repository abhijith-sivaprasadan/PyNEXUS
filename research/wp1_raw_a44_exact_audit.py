"""Independent ENTSO-E A44 A03 raw-XML vs processed 2023 NL price audit.

Standards-based A03 variable-sized blocks are expanded ONLY within each
published Period. This is not interpolation: ENTSO-E specifies that the
value remains constant until the next published position or Period end.

No ENTSO-E token, network call or pandas/entsoe-py XML converter is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from research.wp1_raw_a44_archive import REGION, SCHEMA, months
from research.wp5_market import EXPECTED, PRICE_COLUMN, TIME_COLUMN, load_snapshot

MAX_RAW_BYTES = 12_000_000
DOMAIN_NL = "10YNL----------L"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _child(node, name):
    matches = [child for child in node if child.tag.rsplit("}", 1)[-1] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one A44 XML {name} element")
    return matches[0]


def _text(node, name):
    value = _child(node, name).text
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing A44 XML {name} value")
    return value.strip()


def _timestamp(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        raise ValueError("Official A44 timestamps must include UTC offset/Z")
    if timestamp.minute or timestamp.second or timestamp.microsecond:
        raise ValueError("NL 2023 hourly period boundary not on the hour")
    return timestamp.tz_convert("UTC")


def expand_official_a03_xml(xml_bytes):
    """Return UTC->EUR/MWh observations and source-defined compression counts.

    Mandatory positions, start/end, bidding zone, resolution, unit, and
    domain checks prevent an unsourced carry across malformed Periods.
    """
    if (
        not isinstance(xml_bytes, bytes)
        or not xml_bytes
        or len(xml_bytes) > MAX_RAW_BYTES
        or b"<!DOCTYPE" in xml_bytes.upper()
        or b"<!ENTITY" in xml_bytes.upper()
    ):
        raise ValueError("Unsafe or empty original A44 XML")
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        raise ValueError("Malformed original A44 XML") from None
    if root.tag.rsplit("}", 1)[-1] != "Publication_MarketDocument":
        raise ValueError("Original XML is not a Publication_MarketDocument")
    if _text(root, "type") != "A44":
        raise ValueError("Original XML document is not A44")
    series = [x for x in root if x.tag.rsplit("}", 1)[-1] == "TimeSeries"]
    if not series:
        raise ValueError("A44 document has no TimeSeries")
    hourly = {}
    compressed_periods = 0
    compressed_positions = 0
    for ts in series:
        if _text(ts, "in_Domain.mRID") != DOMAIN_NL or _text(ts, "out_Domain.mRID") != DOMAIN_NL:
            raise ValueError("Original source is not NL bidding-zone day-ahead prices")
        if _text(ts, "currency_Unit.name") != "EUR":
            raise ValueError("NL A44 source currency must be EUR")
        if _text(ts, "price_Measure_Unit.name") != "MWH":
            raise ValueError("NL A44 source price unit must be EUR/MWh")
        if _text(ts, "curveType") != "A03":
            raise ValueError("Only official A03 price-block curves supported in this audit")
        period = _child(ts, "Period")
        if _text(period, "resolution") != "PT60M":
            raise ValueError("Source resolution is not PT60M")
        interval = _child(period, "timeInterval")
        start = _timestamp(_text(interval, "start"))
        end = _timestamp(_text(interval, "end"))
        span = (end - start) / pd.Timedelta(hours=1)
        if span not in (23, 24, 25):
            raise ValueError("Unexpected Dutch local-market delivery-day length")
        hours = int(span)
        points = []
        for point in period:
            if point.tag.rsplit("}", 1)[-1] == "Point":
                pos = int(_text(point, "position"))
                price = float(_text(point, "price.amount"))
                if not np.isfinite(price):
                    raise ValueError("Nonfinite official A44 price")
                points.append((pos, price))
        if not points or points[0][0] != 1:
            raise ValueError("No valid A03 block at first market position")
        if any(
            index < 1 or index > hours
            for index, _ in points
        ) or any(left[0] >= right[0] for left, right in zip(points, points[1:])):
            raise ValueError("Invalid, unordered or duplicated A03 block positions")
        if len(points) < hours:
            compressed_periods += 1
            compressed_positions += hours - len(points)
        for k, (position, price) in enumerate(points):
            stop = points[k + 1][0] if k + 1 < len(points) else hours + 1
            for index in range(position - 1, stop - 1):
                instant = start + pd.Timedelta(hours=index)
                if instant in hourly:
                    raise ValueError("Overlapping or duplicated official UTC hour")
                hourly[instant] = price
    result = pd.Series(hourly, dtype=float).sort_index()
    return result, {
        "source_timeseries_count": len(series),
        "source_a03_compressed_periods": compressed_periods,
        "source_a03_implicit_constant_price_hours": compressed_positions,
    }


def verify_primary_source(raw_dir, processed_dir):
    """Check 14 original XML SHA256s and all 8,760 processed prices exactly."""
    raw_dir = Path(raw_dir)
    raw_manifest_path = raw_dir / "raw_manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    if (
        raw_manifest.get("schema_version") != SCHEMA
        or raw_manifest.get("status") != "RAW_A44_SOURCE_ARCHIVED_WITHOUT_INTERPRETATION"
        or raw_manifest.get("source_domain") != REGION
        or raw_manifest.get("contains_api_token") is not False
        or raw_manifest.get("raw_xml_is_original_provider_response") is not True
    ):
        raise ValueError("Original raw XML source manifest incomplete or unsupported")
    entries = raw_manifest.get("segments")
    expected_filenames = [f"NL_A44_{year}{month:02d}.xml" for year, month in months()]
    if not isinstance(entries, list) or [s.get("file") for s in entries] != expected_filenames:
        raise ValueError("Incomplete or out-of-order original 14-month A44 manifest")
    source = {}
    period_count = 0
    compressed_periods = 0
    inherited_hours = 0
    for segment in entries:
        source_file = raw_dir / segment["file"]
        data = source_file.read_bytes()
        if (
            len(data) != segment.get("bytes")
            or hashlib.sha256(data).hexdigest() != segment.get("sha256")
        ):
            raise ValueError("Original A44 source document hash mismatch")
        values, metadata = expand_official_a03_xml(data)
        if metadata["source_timeseries_count"] != segment["timeseries_count"]:
            raise ValueError("Raw A44 source TimeSeries count mismatch")
        if segment.get("curve_types") != ["A03"] or segment.get("resolutions") != ["PT60M"]:
            raise ValueError("Raw source manifest indicates incompatible curve or interval")
        period_count += metadata["source_timeseries_count"]
        compressed_periods += metadata["source_a03_compressed_periods"]
        inherited_hours += metadata["source_a03_implicit_constant_price_hours"]
        for stamp, amount in values.items():
            if stamp in source:
                raise ValueError("Official monthly document hour overlap")
            source[stamp] = float(amount)
    raw = pd.Series(source, dtype=float).sort_index().reindex(EXPECTED)
    if raw.isna().any() or len(raw) != 8760:
        raise ValueError("Original A44 XML does not cover exactly 8,760 UTC year-hours")
    processed, pmanifest = load_snapshot(processed_dir)
    observed_index = pd.DatetimeIndex(processed[TIME_COLUMN])
    if not observed_index.equals(EXPECTED):
        raise ValueError("Processed prices have a different UTC calendar")
    processed_values = processed[PRICE_COLUMN].to_numpy(dtype=float)
    source_values = raw.to_numpy(dtype=float)
    differences = np.abs(source_values - processed_values)
    if not np.isfinite(differences).all() or not np.array_equal(source_values, processed_values):
        raise ValueError(
            f"Original source price mismatch: {int(np.count_nonzero(differences))} of 8760 hours"
        )
    special = pd.Timestamp("2023-12-30T08:00:00Z")
    if raw.loc[special] != 73.50 or raw.loc[special - pd.Timedelta(hours=1)] != 73.50:
        raise ValueError("Original A03 December 30 constant-price block not reproduced")
    return {
        "schema_version": "pynexus.wp1.raw_a03_exact_equivalence.v1",
        "status": "PRIMARY_ORIGINAL_A44_XML_AND_PROCESSED_UTC_PRICES_EXACTLY_AGREE",
        "raw_monthly_xml_documents": len(entries),
        "raw_document_timeseries": period_count,
        "raw_a03_periods_with_omitted_positions": compressed_periods,
        "raw_a03_source_defined_inherited_constant_hours": inherited_hours,
        "observed_2023_hour_count": len(EXPECTED),
        "source_processed_price_mismatches": 0,
        "max_abs_eur_mwh_difference": float(differences.max()),
        "day_ahead_price_mean_eur_mwh": float(raw.mean()),
        "negative_price_hours": int((raw < 0).sum()),
        "december_30_2023_08utc_a03_published_eur_mwh": float(raw.loc[special]),
        "source_manifest_sha256": _sha256(raw_manifest_path),
        "processed_csv_sha256": pmanifest["csv_sha256"],
        "processed_manifest_sha256": _sha256(Path(processed_dir) / "manifest.json"),
        "wind_hourly_cds_source_alignment_verified": False,
        "storage_and_electrolyser_economics_fully_sourced": False,
        "raw_a03_expansion_is_source_semantics_not_time_interpolation": True,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    report = verify_primary_source(args.raw_dir, args.processed_dir)
    if args.output.exists():
        raise FileExistsError("Refuse to overwrite an existing evidence report")
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
