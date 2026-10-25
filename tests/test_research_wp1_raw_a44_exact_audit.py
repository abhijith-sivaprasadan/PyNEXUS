"""Scientific source semantics: ENTSO-E A03 = variable-sized constant-price blocks."""

import pytest

from research.wp1_raw_a44_exact_audit import expand_official_a03_xml


def a03(start, end, points="1:5.0,4:-10.0", *, zone="10YNL----------L", curve="A03"):
    tags = "".join(
        f"<Point><position>{k}</position><price.amount>{v}</price.amount></Point>"
        for k, v in (part.split(":") for part in points.split(","))
    )
    return (
        f'<Publication_MarketDocument xmlns="urn:test"><type>A44</type>'
        f'<TimeSeries><in_Domain.mRID>{zone}</in_Domain.mRID>'
        f'<out_Domain.mRID>{zone}</out_Domain.mRID>'
        f'<currency_Unit.name>EUR</currency_Unit.name>'
        f'<price_Measure_Unit.name>MWH</price_Measure_Unit.name>'
        f'<curveType>{curve}</curveType><Period><timeInterval>'
        f'<start>{start}</start><end>{end}</end></timeInterval>'
        f'<resolution>PT60M</resolution>{tags}</Period></TimeSeries>'
        f'</Publication_MarketDocument>'
    ).encode("utf-8")


def test_source_defined_a03_constant_price_blocks_not_interpolation():
    series, meta = expand_official_a03_xml(
        a03("2023-12-29T23:00Z", "2023-12-30T23:00Z", "1:32.49,9:73.5,11:53.0")
    )
    assert len(series) == 24
    assert series.loc["2023-12-30T07:00:00Z"] == 73.50
    assert series.loc["2023-12-30T08:00:00Z"] == 73.50
    assert series.loc["2023-12-30T09:00:00Z"] == 53.00
    assert meta["source_a03_compressed_periods"] == 1
    assert meta["source_a03_implicit_constant_price_hours"] == 21


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2023-03-25T23:00Z", "2023-03-26T22:00Z", 23),
        ("2023-10-28T22:00Z", "2023-10-29T23:00Z", 25),
    ],
)
def test_dutch_dst_days_are_absolute_utc_intervals(start, end, expected):
    prices, _ = expand_official_a03_xml(a03(start, end, "1:100"))
    assert len(prices) == expected
    assert prices.index.is_unique
    assert (prices == 100).all()


@pytest.mark.parametrize(
    ("points", "message"),
    [
        ("2:4.0", "first market position"),
        ("1:5,1:5", "Invalid, unordered"),
        ("1:5,26:5", "Invalid, unordered"),
        ("1:5,0:5", "Invalid, unordered"),
        ("1:5,3:5,2:5", "Invalid, unordered"),
        ("1:nan", "Nonfinite"),
    ],
)
def test_reject_invalid_a03_positions_and_prices(points, message):
    with pytest.raises(ValueError, match=message):
        expand_official_a03_xml(a03("2023-12-29T23:00Z", "2023-12-30T23:00Z", points))


def test_reject_wrong_bidding_zone_curve_type_and_period():
    for xml in (
        a03("2023-12-29T23:00Z", "2023-12-30T23:00Z", zone="not_NL"),
        a03("2023-12-29T23:00Z", "2023-12-30T23:00Z", curve="A01"),
        a03("2023-12-29T23:00Z", "2023-12-31T23:00Z"),
    ):
        with pytest.raises(ValueError):
            expand_official_a03_xml(xml)


def test_reject_xml_entity_declarations():
    xml = a03("2023-12-29T23:00Z", "2023-12-30T23:00Z")
    with pytest.raises(ValueError, match="Unsafe"):
        expand_official_a03_xml(b"<!DOCTYPE anything>" + xml)
