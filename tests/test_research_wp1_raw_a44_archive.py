"""Hermetic strict A44 raw-source archival tests; no API key or remote calls."""

import hashlib
import json

import pytest

from research.wp1_raw_a44_archive import fetch_raw_year, months, xml_summary


XML = """<?xml version="1.0" encoding="utf-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <type>A44</type>
  <TimeSeries>
    <curveType>A03</curveType>
    <Period>
      <resolution>PT60M</resolution>
      <Point><position>1</position><price.amount>101.12</price.amount></Point>
      <Point><position>4</position><price.amount>110.00</price.amount></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>"""


class FakeClient:
    def __init__(self, payload=XML):
        self.payload = payload
        self.requests = []

    def query_day_ahead_prices(self, region, *, start, end):
        self.requests.append((region, start, end))
        return self.payload


def test_raw_document_marks_compressed_curve_without_guessing_hourly_values():
    metadata = xml_summary(XML)
    assert metadata["document_type"] == "A44"
    assert metadata["curve_types"] == ["A03"]
    assert metadata["resolutions"] == ["PT60M"]
    assert metadata["source_point_count"] == 2
    assert metadata["compressed_curve_may_need_provider_spec_expansion"] is True
    assert metadata["no_price_interpolation_or_expansion_performed"] is True


def test_strict_primary_archive_and_checksums(tmp_path):
    fake = FakeClient()
    folder = tmp_path / "primary"
    record = fetch_raw_year(folder, client=fake)
    assert record["status"] == "RAW_A44_SOURCE_ARCHIVED_WITHOUT_INTERPRETATION"
    assert len(record["segments"]) == len(months()) == 14
    assert len(fake.requests) == 14
    assert fake.requests[0][0] == "NL"
    assert record["segments"][0]["sha256"] == hashlib.sha256(XML.encode()).hexdigest()
    assert (folder / "NL_A44_202312.xml").read_text() == XML
    manifest = json.loads((folder / "raw_manifest.json").read_text())
    assert manifest["contains_api_token"] is False
    assert "securityToken" not in (folder / "raw_manifest.json").read_text()
    with pytest.raises(FileExistsError):
        fetch_raw_year(folder, client=fake)


def test_no_provider_url_or_secret_leak_on_exception(tmp_path):
    class LeakyClient:
        def query_day_ahead_prices(self, *args, **kwargs):
            raise RuntimeError("securityToken=SECRET_PRIVATE_VALUE")

    with pytest.raises(RuntimeError, match="private provider details suppressed") as exc:
        fetch_raw_year(tmp_path / "failure", client=LeakyClient())
    assert "SECRET_PRIVATE_VALUE" not in str(exc.value)
    assert "SECRET_PRIVATE_VALUE" not in (
        tmp_path / "failure" / "raw_manifest.json"
    ).read_text()


@pytest.mark.parametrize("raw", ["not-xml", "<Acknowledgement_MarketDocument/>",
                                "<!DOCTYPE boom><Publication_MarketDocument/>"])
def test_reject_non_price_or_active_xml(raw):
    with pytest.raises(ValueError):
        xml_summary(raw)
