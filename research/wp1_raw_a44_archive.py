"""Archive *original* ENTSO-E NL A44 XML, not only entsoe-py parsed pandas data.

Credential stays local/in Actions secrets; error output never includes upstream
URL, HTTP response body or exception strings which might disclose the token.
This is a primary-source collection and curve-type audit, NOT a replacement
for the independently verified processed 2023 UTC market CSV.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

SCHEMA = "pynexus.wp1.entsoe_raw_a44.nl_2023.v1"
REGION = "NL"
TIMEZONE = "Europe/Amsterdam"


def months():
    """Bracket UTC 2023 by intact provider local months."""
    return [(2022, 12)] + [(2023, m) for m in range(1, 13)] + [(2024, 1)]


def xml_summary(raw):
    """Validate minimal A44 XML structure without inferring unreported price points."""
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not isinstance(raw, bytes) or not raw or len(raw) > 12_000_000:
        raise ValueError("Unexpected provider XML size/type")
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("Provider XML document/entity declarations prohibited")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        raise ValueError("ENTSO-E returned unparseable XML") from None
    rootname = root.tag.split("}")[-1]
    if rootname != "Publication_MarketDocument":
        raise ValueError("ENTSO-E did not return a Publication_MarketDocument")

    def texts(local_name):
        return [node.text for node in root.iter() if node.tag.split("}")[-1] == local_name]

    doc_types = texts("type")
    if "A44" not in doc_types:
        raise ValueError("ENTSO-E raw document does not declare A44")
    timeseries = [node for node in root.iter() if node.tag.split("}")[-1] == "TimeSeries"]
    if not timeseries:
        raise ValueError("No source price TimeSeries")
    curve_types = sorted(set(texts("curveType")))
    resolutions = sorted(set(texts("resolution")))
    point_positions = [
        elem.text
        for ts in timeseries
        for point in ts.iter()
        if point.tag.split("}")[-1] == "position"
        for elem in [point]
    ]
    if not point_positions or any(
        not isinstance(value, str) or not value.isdigit() for value in point_positions
    ):
        raise ValueError("Missing/non-numeric provider Point positions")
    return {
        "document_type": "A44",
        "root_type": rootname,
        "timeseries_count": len(timeseries),
        "curve_types": curve_types,
        "resolutions": resolutions,
        "source_point_count": len(point_positions),
        "compressed_curve_may_need_provider_spec_expansion": "A03" in curve_types,
        "no_price_interpolation_or_expansion_performed": True,
    }


def fetch_raw_year(output, *, api_key=None, client=None):
    """Record each unmodified provider XML response and a tamper-evident manifest."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refuse to replace primary provider evidence")
    token = api_key if api_key is not None else os.environ.get("ENTSOE_API_KEY", "")
    if client is None:
        if not token or not token.strip():
            raise ValueError("ENTSOE_API_KEY repository secret is required")
        from entsoe import EntsoeRawClient

        client = EntsoeRawClient(api_key=token)
    output.mkdir(parents=True, exist_ok=False)
    request_manifest = {
        "schema_version": SCHEMA,
        "status": "INCOMPLETE_RAW_SOURCE_COLLECTION",
        "source": "ENTSOE_TRANSPARENCY_PLATFORM_DIRECT_RAW_CLIENT_A44",
        "source_domain": REGION,
        "request_timezone": TIMEZONE,
        "retrieval_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "archive_contains_processed_prices": False,
        "raw_xml_is_original_provider_response": True,
        "contains_api_token": False,
        "segments": [],
    }
    try:
        for year, month in months():
            start = pd.Timestamp(year=year, month=month, day=1, tz=TIMEZONE)
            end = start + pd.DateOffset(months=1)
            try:
                response = client.query_day_ahead_prices(REGION, start=start, end=end)
            except Exception as exc:
                raise RuntimeError(
                    f"Official raw A44 request failed for {year}-{month:02d} "
                    f"({type(exc).__name__}); private provider details suppressed"
                ) from None
            if not isinstance(response, (bytes, str)):
                raise ValueError("Raw ENTSO-E response must be XML text/bytes")
            data = response.encode("utf-8") if isinstance(response, str) else response
            if token and token.encode("utf-8") in data:
                raise ValueError("Provider response echoed a secret token; evidence not stored")
            summary = xml_summary(data)
            name = f"NL_A44_{year}{month:02d}.xml"
            target = output / name
            target.write_bytes(data)
            request_manifest["segments"].append(
                {
                    "file": name,
                    "start_local": start.isoformat(),
                    "end_local_exclusive": end.isoformat(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                    **summary,
                }
            )
        request_manifest["status"] = "RAW_A44_SOURCE_ARCHIVED_WITHOUT_INTERPRETATION"
        return request_manifest
    finally:
        (output / "raw_manifest.json").write_text(
            json.dumps(request_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    record = fetch_raw_year(args.output)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
